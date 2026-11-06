"""Bounded learning curve on a fixed, song-disjoint diagnostic partition."""
import argparse
from collections import Counter
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch

from scripts.acoustic_probe_metrics import event_metrics

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return sha256((ROOT / path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def load_model_module(path):
    spec = importlib.util.spec_from_file_location('r25_frozen_r24_model', ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def weight_hash(model):
    h = sha256()
    for name, value in model.state_dict().items():
        h.update(name.encode())
        h.update(value.detach().cpu().numpy().tobytes())
    return h.hexdigest()


def statistics(arrays):
    result = {}
    for key, value in arrays.items():
        x = value.double()
        result[key] = dict(shape=list(x.shape), dtype=str(value.dtype),
            nan=int(torch.isnan(x).sum()), inf=int(torch.isinf(x).sum()),
            min=float(x.min()), max=float(x.max()), mean=float(x.mean()),
            std=float(x.std(unbiased=False)))
    assert all(r['nan'] == r['inf'] == 0 for r in result.values())
    return result


def matching(pred, truth, allowed):
    edges = [[j for j, b in enumerate(truth) if allowed(a, b)] for a in pred]
    matched = {}

    def augment(i, seen):
        for j in edges[i]:
            if j in seen:
                continue
            seen.add(j)
            if j not in matched or augment(matched[j], seen):
                matched[j] = i
                return True
        return False

    for i in range(len(pred)):
        augment(i, set())
    return matched


def distribution_metrics(pred, truth):
    p = sorted(set(e['onset_us'] for e in pred))
    t = sorted(set(e['onset_us'] for e in truth))
    pairs = matching(p, t, lambda a, b: abs(a - b) <= 20_000)
    hp = [e for e in pred if e['type'] == 'hold']
    ht = [e for e in truth if e['type'] == 'hold']
    hold_pairs = matching(hp, ht, lambda a, b: a['lane'] == b['lane']
                         and abs(a['onset_us'] - b['onset_us']) <= 20_000)
    tails = [(hp[i]['end_us'] - ht[j]['end_us']) / 1000 for j, i in hold_pairs.items()]
    return dict(timing_point_f1=2 * len(pairs) / (len(p) + len(t)) if p or t else 1.,
        timing_point_matches=len(pairs), predicted_onset_groups=len(p), true_onset_groups=len(t),
        count_ratio=len(pred) / len(truth) if truth else None,
        largest_internal_gap_ms=max((b - a for a, b in zip(p, p[1:])), default=0) / 1000,
        hold_head_matched_tail_errors_ms=tails,
        hold_tail_median_abs_ms=float(np.median(np.abs(tails))) if tails else None,
        hold_tail_p90_abs_ms=float(np.percentile(np.abs(tails), 90)) if tails else None)


def aggregate(records):
    groups = {'all': records}
    for mode in sorted(set(r['mode'] for r in records)):
        groups[mode] = [r for r in records if r['mode'] == mode]
    result = {}
    for name, rows in groups.items():
        fields = {}
        for r in rows:
            for head, v in r['teacher']['fields'].items():
                if not v['n']:
                    continue
                x = fields.setdefault(head, dict(n=0, loss_sum=0., correct_sum=0.))
                x['n'] += v['n']
                x['loss_sum'] += v['loss'] * v['n']
                x['correct_sum'] += v['accuracy'] * v['n']
        for v in fields.values():
            v['loss'] = v.pop('loss_sum') / v['n']
            v['accuracy'] = v.pop('correct_sum') / v['n']
        types = {k: {f: sum(r['metrics']['per_type'][k][f] for r in rows)
                     for f in ['pred', 'true', 'matched']}
                 for k in ['tap', 'hold', 'flick', 'drag']}
        result[name] = dict(cores=len(rows), full_note_f1=float(np.mean([r['metrics']['full_note_f1'] for r in rows])),
            timing_point_f1=float(np.mean([r['distribution']['timing_point_f1'] for r in rows])),
            teacher_loss=sum(v['loss'] for v in fields.values()), teacher_fields=fields, per_type=types,
            complete_legal=sum(r['generation']['ended'] and r['generation']['failure'] is None
                and r['generation']['active_holds'] == 0 and r['metrics']['mania_lane_conflicts'] == 0 for r in rows),
            empty=sum(not r['generation']['events'] for r in rows))
    return result


def compare(baseline, final):
    a, b = baseline['all'], final['all']
    headroom = 1 - a['full_note_f1']
    return dict(gain_has_headroom=headroom > 1e-12,
        remaining_full_f1_error_reduced_10pct=b['full_note_f1'] >= a['full_note_f1'] + .1 * headroom,
        mode_protection=all(final[m]['full_note_f1'] >= baseline[m]['full_note_f1'] - .02
                            for m in ['mania_4k', 'phira']),
        timing_protection=b['timing_point_f1'] >= a['timing_point_f1'] - .02,
        complete_legal=b['complete_legal'] == b['cores'],
        no_applicable_type_disappeared=all(v['pred'] > 0 for m in ['mania_4k', 'phira']
            for v in final[m]['per_type'].values() if v['true']))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    for name, want in plan['input_sha256'].items():
        assert digest(name) == want, name
    destination = ROOT / plan['result']
    assert not destination.exists(), 'No retries or overwrite'
    assets = ROOT / plan['artifacts']
    assets.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(plan['seed'])
    module = load_model_module(plan['model_source'])
    panel_path = ROOT / plan['manifest']
    panel = json.loads(panel_path.read_text())
    with np.load(panel_path.parent / 'inputs.npz', allow_pickle=False) as a:
        inputs = {k: torch.from_numpy(a[k].copy()) for k in a.files}
    rows = {r['id']: r for r in panel['rows']}
    index = {r['id']: i for i, r in enumerate(panel['rows'])}
    train_ids, held_ids = plan['train_ids'], plan['heldout_ids']
    assert len(set(train_ids + held_ids)) == len(train_ids + held_ids)
    assert not ({rows[k]['audio_sha256'] for k in train_ids} & {rows[k]['audio_sha256'] for k in held_ids})
    assert not inputs['condition_known'].any()
    assert all(rows[k]['split'] == 'train' for k in train_ids + held_ids)
    assert Counter(k for b in plan['batch_cycle'] for k in b) == Counter(train_ids)

    def take(ids):
        return {k: v[[index[x] for x in ids]] for k, v in inputs.items()}

    started = time.time()
    report = dict(id='R25', complete=False, status='health', model_promotion=False,
        optimizer_updates=0, new_cloud_spend_usd=0, plan_sha256=digest(str(args.plan)),
        initial_weight_sha256=None, data_statistics=statistics(inputs), checkpoints={}, training=[],
        train_ids=train_ids, heldout_ids=held_ids, exposure={k: 0 for k in train_ids})

    def save():
        report['wall_s'] = time.time() - started
        write_json(destination, report)

    save()
    try:
        model = module.StateModel()
        report['initial_weight_sha256'] = weight_hash(model)
        assert report['initial_weight_sha256'] == plan['initial_weight_sha256']
        selected = train_ids + held_ids
        prefix = module.prefix_batch([rows[k] for k in selected])
        report['prefix_statistics'] = statistics(prefix)
        model.train()
        output = model.teacher(model.encode_audio(take(selected)), prefix, False)
        loss, met = module.loss_and_metrics(output, prefix, take(selected)['mode'])
        assert torch.isfinite(loss) and all(torch.isfinite(v).all() for v in output.values())
        loss.backward()
        gradients = {}
        for name in ['audio_projection', 'blocks', 'time_q', 'millisecond', 'microsecond',
                     'mania_head', 'phira_head', 'release_head']:
            grads = [p.grad for n, p in model.named_parameters() if n.startswith(name) and p.grad is not None]
            assert grads and all(torch.isfinite(g).all() for g in grads), name
            gradients[name] = sum(float(g.double().square().sum()) for g in grads) ** .5
            assert gradients[name] > 1e-12, name
        model.zero_grad(set_to_none=True)
        assert weight_hash(model) == report['initial_weight_sha256']
        report['health'] = dict(passed=True, gradients=gradients, initial_fields=met,
            actual_targets_supported=True, song_disjoint=True, heldout_has_low_mania=False,
            reused_real_overfit_evidence=plan['overfit_evidence'])
        del prefix, output, loss
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'],
            betas=(.9, .999), eps=1e-8, weight_decay=0.)
        batches = [(ids, take(ids), module.prefix_batch([rows[k] for k in ids])) for ids in plan['batch_cycle']]

        def evaluate(update):
            model.eval()
            checkpoint = assets / f'step-{update}.pt'
            torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict(),
                rng_state=torch.get_rng_state(), step=update, plan_sha256=report['plan_sha256'],
                model_source_sha256=digest(plan['model_source']), diagnostic_only=True), checkpoint)
            records = []
            logits = {}
            with torch.no_grad():
                for core_id in selected:
                    r = rows[core_id]
                    audio = take([core_id])
                    p = module.prefix_batch([r])
                    out = model.teacher(model.encode_audio(audio), p, False)
                    val, fields = module.loss_and_metrics(out, p, audio['mode'])
                    logits[core_id] = {k: v.cpu() for k, v in out.items()}
                    generation = module.generate(model, audio, False, maximum_actions=plan['maximum_actions'])
                    records.append(dict(core_id=core_id, chart_id=r['chart_id'], mode=r['mode'], tier=r['tier'],
                        partition='train' if core_id in train_ids else 'heldout', audio=r['audio'],
                        source_clip_start_us=r['source_clip_start_us'], truth=r['target_events'],
                        teacher=dict(loss=float(val), fields=fields), generation=generation,
                        metrics=event_metrics(generation['events'], r['target_events'], r['mode']),
                        distribution=distribution_metrics(generation['events'], r['target_events'])))
            logits_path = assets / f'teacher-logits-{update}.pt'
            torch.save(logits, logits_path)
            record_path = assets / f'predictions-{update}.json'
            write_json(record_path, records)
            grouped = {part: aggregate([r for r in records if r['partition'] == part]) for part in ['train', 'heldout']}
            saved = {str(p.relative_to(ROOT)): digest(p) for p in [checkpoint, logits_path, record_path]}
            report['checkpoints'][str(update)] = dict(summary=grouped, assets_sha256=saved,
                weight_sha256=weight_hash(model), exposure=dict(report['exposure']))
            print(json.dumps(dict(event='checkpoint', update=update,
                summary={part: grouped[part]['all'] for part in grouped})), flush=True)
            save()
            model.train()

        report['status'] = 'training'
        save()
        for step in range(1, plan['updates'] + 1):
            ids, audio, prefix = batches[(step - 1) % len(batches)]
            optimizer.zero_grad(set_to_none=True)
            output = model.teacher(model.encode_audio(audio), prefix, False)
            loss, fields = module.loss_and_metrics(output, prefix, audio['mode'])
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            optimizer.step()
            report['optimizer_updates'] = step
            for key in ids:
                report['exposure'][key] += 1
            if step == 1 or step % 10 == 0:
                record = dict(step=step, core_ids=ids, loss=float(loss.detach()), fields=fields,
                    gradient_norm_before_clip=float(norm))
                report['training'].append(record)
                save()
                print(json.dumps(dict(event='train', step=step, loss=record['loss'])), flush=True)
            del output, loss
            if step in plan['checkpoints']:
                evaluate(step)
        base = report['checkpoints']['100']['summary']['heldout']
        final = report['checkpoints']['1000']['summary']['heldout']
        gates = compare(base, final)
        report.update(complete=True, status='evaluated', gates=gates, diagnostic_horizon_benefit=all(gates.values()))
    except Exception as error:
        report.update(complete=True, status='execution_failed', error=dict(type=type(error).__name__,
            message=str(error), traceback=traceback.format_exc()))
    save()
    print(json.dumps({k: report.get(k) for k in ['status', 'optimizer_updates', 'gates', 'wall_s']}), flush=True)
    if report['status'] == 'execution_failed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
