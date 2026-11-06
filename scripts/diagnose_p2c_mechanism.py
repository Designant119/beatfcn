"""Reproduce rejected R12 exactly, then audit input dependence without promotion.

Archived model entries are restored only for this bounded run. The diagnostic
never changes the original fit contract, saves weights, or grants training access.
"""
import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import resource
import time

import numpy as np
import torch

from scripts.r09_local_data import ROOT, LocalPairs, file_sha, verify_files


VARIANTS = ('normal', 'zero_audio', 'reverse_local', 'reverse_all')


def perturb_inputs(inputs, variant):
    """Change feature values only; preserve chronology labels and padding masks."""
    if variant not in VARIANTS:
        raise ValueError(f'unknown intervention: {variant}')
    if variant == 'normal':
        return inputs
    local, song = inputs.local.clone(), inputs.song.clone()
    if variant == 'zero_audio':
        local.zero_()
        song.zero_()
    else:
        valid = torch.nonzero(inputs.local_valid, as_tuple=False).flatten()
        local[valid] = inputs.local[valid.flip(0)]
        if variant == 'reverse_all':
            song = song.flip(0)
    return replace(inputs, local=local, song=song)


def target_vectors(labels, device):
    """Audit the exact decisions used by the immutable R12 supervised loss."""
    counts = torch.as_tensor(labels.counts, device=device)
    n = torch.as_tensor(labels.multiplicity.reshape(-1), device=device)
    notes = torch.as_tensor(labels.notes.reshape(-1, 3), device=device)
    heads = torch.as_tensor(labels.heads.reshape(-1, 2), device=device)
    targets = {'block_count': (counts, torch.ones_like(counts, dtype=torch.bool))}
    if (n > 0).any():
        targets['member_count'] = n - 1, n > 0
        targets['head'] = torch.where(heads[:, 1] >= 0, 400 + heads[:, 1], heads[:, 0]), n > 0
    if (notes[:, 0] < 4).any():
        targets['kind'] = notes[:, 0], notes[:, 0] < 4
        if labels.mode == 'mania_4k':
            targets['lane'] = notes[:, 2], notes[:, 0] < 4
    hold = notes[:, 0] == 1
    if hold.any():
        targets.update(tail_block=(notes[:, 1] // 2000, hold),
                       tail_cell=(notes[:, 1] % 2000 // 40, hold),
                       tail_ms=(notes[:, 1] % 40, hold))
    return targets


@torch.no_grad()
def probability_audit(outputs, labels, reference, prefix, arrays):
    heads, total_nll, decisions = {}, 0., 0
    for name, (truth, mask) in target_vectors(labels, outputs['kind'].device).items():
        logits, truth = outputs[name][mask], truth[mask]
        logp = logits.log_softmax(-1)
        target_logp = logp.gather(-1, truth[:, None]).squeeze(-1)
        top = logits.topk(min(3, logits.shape[-1]), dim=-1)
        base_logits = reference[name][mask]
        base_logp = base_logits.log_softmax(-1)
        key = f'{prefix}__{name}'
        arrays[key + '__logits'] = logits.cpu().numpy()
        arrays[key + '__targets'] = truth.cpu().numpy()
        arrays[key + '__indices'] = torch.nonzero(mask).flatten().cpu().numpy()
        nll = -float(target_logp.sum())
        total_nll += nll
        decisions += len(truth)
        heads[name] = dict(support=len(truth), mean_ce=nll / len(truth),
            correct=int((logits.argmax(-1) == truth).sum()),
            target_probability_mean=float(target_logp.exp().mean()),
            target_probability_geometric_mean=float(target_logp.mean().exp()),
            target_probability_min=float(target_logp.min().exp()),
            mean_entropy=float(-(logp.exp() * logp).sum(-1).mean()),
            max_abs_logit_change=float((logits - base_logits).abs().max()),
            mean_kl_from_normal=float((base_logp.exp() * (base_logp - logp)).sum(-1).mean()),
            target_ids=truth.tolist(), argmax_ids=logits.argmax(-1).tolist(),
            top3_ids=top.indices.tolist(), top1_minus_top2=(top.values[:, 0] - top.values[:, 1]).tolist(),
            target_probabilities=target_logp.exp().tolist(), npz_prefix=key)
    return dict(heads=heads, decision_count=decisions, sum_decision_nll=total_nll,
                decision_weighted_ce=total_nll / decisions,
                sum_head_means=sum(h['mean_ce'] for h in heads.values()))


def generation_summary(generation, normal):
    finals = [c['rounds'][-1] for c in generation['cores']]
    baseline = [c['rounds'][-1] for c in normal['cores']]
    return dict(macro_full_note_f1=sum(r['full_note_f1'] for r in finals) / len(finals),
        macro_exact_group_recall=sum(r['exact_onset_group_recall'] for r in finals) / len(finals),
        predicted=sum(r['pred'] for r in finals), target=sum(r['true'] for r in finals),
        matched=sum(r['matched'] for r in finals),
        mania_lane_conflicts=sum(r['mania_lane_conflicts'] for r in finals),
        identical_core_events_to_normal=[a['events'] == b['events'] for a, b in zip(finals, baseline)])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    verify_files(plan)
    # This import is intentionally inside the entrypoint: archived entries must
    # be restored and hash-verified by the registered runner before execution.
    from scripts import p2c_candidate as candidate
    from scripts import probe_p2c_network as probe

    contract = json.loads((ROOT / plan['baseline_contract']).read_text())
    old = json.loads((ROOT / plan['baseline_report']).read_text())
    data_plan = json.loads((ROOT / contract['data_plan']).read_text())
    started = time.perf_counter()
    torch.set_num_threads(plan['threads'])
    torch.set_num_interop_threads(1)
    seed = contract['training_corruption']['seed']
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    data = LocalPairs(data_plan)
    items = [item for pair in data_plan['probe_pairs'] for item in pair]
    examples = [candidate.load_example(data, item, slots=contract['model']['slots']) for item in items]
    model = candidate.GroupModel(**contract['model']).eval()
    settings = contract['fit_if_health_passes']
    output = ROOT / plan['result']
    report = dict(schema='beatfcn.p2c_mechanism_diagnostic.v1', id='D30', complete=False,
        optimizer_updates=0, model_promotion=False, checkpoint_saved=False,
        original_R12_decision='failed_joint_gate_unchanged', new_paid_compute_usd=0,
        input_sha256=plan['input_sha256'], plan_sha256=file_sha(args.plan),
        initial_state_sha256=probe.state_sha(model), seed=seed, threads=plan['threads'],
        baseline_scope='four memorised training cores, two songs, one song per mode',
        limitations=['Feature reversal and zeroing are out-of-distribution dependence tests.',
            'Sensitivity does not prove correct music alignment.',
            'Two memorised fields cannot prove thirteen-condition generalisation.',
            'No held-out or full-song quality measurement is performed.'])

    def save():
        report['wall_s'] = time.perf_counter() - started
        report['peak_rss_gib'] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

    save()
    try:
        if report['initial_state_sha256'] != old['initial_state_sha256'] or \
                report['initial_state_sha256'] != plan['baseline_initial_state_sha256']:
            raise ValueError('initial weights differ from R12')
        health = []
        for inputs, labels, identity in examples:
            measured = {f.name: probe.stats(getattr(inputs, f.name)) for f in fields(inputs)
                        if isinstance(getattr(inputs, f.name), torch.Tensor)}
            for name in ('counts', 'heads', 'multiplicity', 'notes'):
                measured['target_' + name] = probe.stats(torch.from_numpy(getattr(labels, name)))
            if any(v['nonfinite'] for v in measured.values()):
                raise ValueError('nonfinite real input')
            if [e.__dict__ for e in candidate.decode_group_targets(labels)] != \
                    next(x['target_events'] for x in old['generation']['cores']
                         if x['identity']['chart_id'] == identity['chart_id']):
                raise ValueError('target alignment differs from R12')
            health.append(dict(identity=identity, tensors=measured, target_alignment_exact=True))
        report['tensor_health'] = health
        save()
        if settings['steps'] != plan['maximum_optimizer_updates']:
            raise ValueError('update bound differs from baseline')
        optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
        history = []
        fit_start = time.perf_counter()
        for step in range(1, settings['steps'] + 1):
            if time.perf_counter() - fit_start >= settings['fit_deadline_s']:
                raise TimeoutError('original fit deadline reached')
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for inputs, labels, _ in examples:
                view, kind = candidate.corruption(labels, inputs, step - 1, rng)
                loss, _, _ = candidate.supervised_loss(model(inputs, view), labels)
                if not torch.isfinite(loss):
                    raise ValueError('nonfinite fitting loss')
                (loss / len(examples)).backward()
                losses.append(float(loss.detach()))
            gradients = probe.gradient_report(model)
            if gradients['nonfinite']:
                raise ValueError('nonfinite fitting gradients')
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), settings['clip']))
            optimizer.step()
            mean_loss = sum(losses) / len(losses)
            expected = old['history'][step - 1]
            difference = max(abs(a - b) for a, b in zip(losses, expected['core_losses']))
            if difference > plan['reproduction_gate']['each_training_loss_max_abs_difference']:
                raise ValueError(f'R12 training trajectory changed at step {step}: {difference}')
            history.append(dict(step=step, view=kind, mean_loss=mean_loss, core_losses=losses,
                max_core_loss_difference=difference, grad_norm=grad_norm,
                gradient_groups=gradients['norms'], readout_gradients=gradients['readout_norms']))
            report.update(optimizer_updates=step, history=history)
            if step == 1 or step % 10 == 0:
                save()
                print(json.dumps(history[-1]), flush=True)
        report['fit_s'] = time.perf_counter() - fit_start
        report['final'] = probe.loss_report(model, examples)
        generation = probe.generation_report(model, examples, contract['decoding']['seed'])
        final_difference = abs(report['final']['mean_loss'] - old['final']['mean_loss'])
        same_events = all(a['events'] == b['events']
            for ca, cb in zip(generation['cores'], old['generation']['cores'])
            for a, b in zip(ca['rounds'], cb['rounds']))
        report['reproduction'] = dict(initial_state_exact=True,
            maximum_training_loss_difference=max(h['max_core_loss_difference'] for h in history),
            final_loss_difference=final_difference, all_round_events_exact=same_events,
            original_gates=old['gates'], final_state_sha256=probe.state_sha(model))
        save()
        if not same_events or final_difference > plan['reproduction_gate']['final_loss_max_abs_difference']:
            raise ValueError('final reproduction gate failed; interventions not run')
        report['reproduction']['passed'] = True
        with torch.no_grad():
            normal_logits = [model(inputs) for inputs, _, _ in examples]
        arrays, audits = {}, {}
        for variant in VARIANTS:
            changed = [(perturb_inputs(x, variant), y, ident) for x, y, ident in examples]
            generated = generation if variant == 'normal' else probe.generation_report(
                model, changed, contract['decoding']['seed'])
            probabilities = []
            with torch.no_grad():
                for i, (inputs, labels, identity) in enumerate(changed):
                    outputs = model(inputs)
                    if any(not torch.isfinite(v).all() for v in outputs.values()):
                        raise ValueError(f'nonfinite diagnostic logits: {variant}')
                    probabilities.append(dict(identity=identity, **probability_audit(
                        outputs, labels, normal_logits[i], f'{variant}__core{i}', arrays)))
            audits[variant] = dict(generation=generated, probabilities=probabilities,
                                  summary=generation_summary(generated, generation))
            report['interventions'] = audits
            save()
        np.savez_compressed(ROOT / plan['logits'], **arrays)
        report['logits'] = dict(path=plan['logits'], sha256=file_sha(ROOT / plan['logits']),
                               bytes=(ROOT / plan['logits']).stat().st_size)
        report['condition_binding_diagnostic'] = probe.binding_report(
            model, examples, generation, contract['decoding']['seed'])
        normal_f1 = audits['normal']['summary']['macro_full_note_f1']
        zero_f1 = audits['zero_audio']['summary']['macro_full_note_f1']
        report['diagnostic_findings'] = dict(
            exact_joint_reversal_invariance=all(audits['reverse_all']['summary']['identical_core_events_to_normal']),
            zero_content_retains_memorised_outputs=zero_f1 >= .9 and normal_f1 - zero_f1 <= .05,
            zero_content_f1_drop=normal_f1 - zero_f1,
            retrospective_R12_pass=False, held_out_quality=False, formal_training_ready=False)
        report['complete'] = True
        save()
        print(json.dumps(dict(reproduction=report['reproduction'],
                              findings=report['diagnostic_findings'], wall_s=report['wall_s'])), flush=True)
    except Exception as exc:
        report.update(complete=True, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    main()
