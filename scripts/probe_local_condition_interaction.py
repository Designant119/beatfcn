"""Real paired-chart diagnostic: identical audio/context, different true labels.

D80 architecture and affine inputs are frozen; only the paired decision objective is added. Terminal states are
quarantined as diagnostic evidence, never promoted as quality checkpoints.
"""
from dataclasses import asdict, replace
from copy import deepcopy
import json
import importlib.util
import sys
import math
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.logit_audit import compare_valid_logits
from beatfcn.train.optimizer_restore import restore_named_optimizer_state
from beatfcn.train.condition_interaction import condition_interaction_loss
from beatfcn.train.event_conditions import CONTROL_FIELDS
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.local_shapes import pad_input_shapes
from scripts.run_local_song_baseline import sha, write_json

ROOT = Path(__file__).resolve().parents[1]


def write_failure_report(path, report):
    nonfinite_paths = []

    def finite(value, name='report'):
        if isinstance(value, float) and not math.isfinite(value):
            nonfinite_paths.append(name)
            return None
        if isinstance(value, dict):
            return {k: finite(v, f'{name}.{k}') for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v, f'{name}[{i}]') for i, v in enumerate(value)]
        return value

    result = finite(report)
    result['nonfinite_report_fields'] = nonfinite_paths
    result['passed'] = False
    write_json(path, result)


@torch.no_grad()
def cache_equivalence(model, inp, seq):
    """Compare only true queries; post-EOS padding cannot determine this gate."""
    if len(seq['valid']) != 1:
        raise ValueError('Cache probe requires one independent request')
    count = int(seq['valid'].sum())
    if count == 0 or not bool(seq['valid'][0, :count].all()) or bool(seq['valid'][0, count:].any()):
        raise ValueError('Expected a nonempty valid prefix')
    seq = {k: v[:, :count] for k, v in seq.items()}
    memory = model.encode(inp); full = model(inp, seq); cache = model.start_cache(memory)
    cached = torch.cat([model.decode(*(seq[k][:, i:i+1] for k in ('tokens','fields','times','valid')),
                        memory, cache=cache) for i in range(count)], 1)
    return compare_valid_logits(full, cached, seq['valid'], allowed=seq['allowed'])



def cpu_copy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k:cpu_copy(v) for k,v in value.items()}
    if isinstance(value, list):
        return [cpu_copy(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_copy(v) for v in value)
    return deepcopy(value)


def preserve_terminal(model, optimizer, report, plan, plan_path):
    destination = ROOT / plan['diagnostic_checkpoint']
    destination.parent.mkdir(parents=True,exist_ok=True)
    data = dict(diagnostic_only=True, eligible_for_promotion=False, auto_resume=False,
                plan_sha256=sha(plan_path), optimizer_updates=report['optimizer_updates'],
                config=asdict(model.c), model=cpu_copy(model.state_dict()),
                optimizer=cpu_copy(optimizer.state_dict()), optimizer_parameter_names=list(dict(model.named_parameters())),
                cpu_rng=torch.get_rng_state(), mps_rng=torch.mps.get_rng_state())
    with destination.open('xb') as file:
        torch.save(data,file)
    report['diagnostic_checkpoint'] = dict(path=plan['diagnostic_checkpoint'],sha256=sha(destination),
                                         bytes=destination.stat().st_size,diagnostic_only=True)


def initial_equivalence(original, model, inp, seq, *, known, index):
    original.zero_grad(set_to_none=True); model.zero_grad(set_to_none=True)
    before=original(inp,seq);old_loss=event_loss(before,seq);old_loss.backward()
    gradients={n:p.grad.detach().cpu().clone() for n,p in original.named_parameters() if p.grad is not None}
    old_logits=before.detach().cpu();old_ce=float(old_loss.detach())
    del before,old_loss
    after=model(inp,seq);new_loss=event_loss(after,seq);new_loss.backward()
    grad_delta=max(float((p.grad.detach().cpu()-gradients[n]).abs().max()) for n,p in model.named_parameters() if n in gradients)
    comparison=compare_valid_logits(old_logits,after.detach().cpu(),seq['valid'].cpu(),allowed=seq['allowed'].cpu(),tolerance=1e-6)
    adapter_grad=float(model.audio_key_conditioner.weight.grad.norm())
    proof=dict(index=index,known=known,loss_difference=abs(float(new_loss.detach())-old_ce),
               gradient_difference=grad_delta,adapter_gradient=adapter_grad,logit_comparison=comparison)
    proof['passed']=comparison['passed'] and grad_delta<1e-5 and math.isfinite(adapter_grad) and (adapter_grad>0 if known else adapter_grad==0)
    if not proof['passed']:
        raise ValueError('Real MPS zero/unknown gate equivalence failed: '+json.dumps(proof))
    return proof


def load_frozen_components(plan):
    """Load explicitly archived experiment code into this process, never edit production files."""
    modules=[]
    for name,key in [('beatfcn.model.conditioned_audio_keys','key_module'),
                     ('beatfcn.model._d79_full_event_model','model_module'),
                     ('beatfcn.train._d75_control_standardization','normalizer_module')]:
        source=ROOT/plan[key]
        if sha(source)!=plan['inputs_sha256'][plan[key]]:
            raise ValueError('Changed isolated experiment source: '+str(source))
        if name in sys.modules:
            existing=sys.modules[name]
            if Path(existing.__file__).resolve()!=source.resolve():
                raise ValueError('Conflicting experiment module name: '+name)
            modules.append(existing);continue
        spec=importlib.util.spec_from_file_location(name,source)
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(name,None)
            raise
        modules.append(module)
    return modules[1].FullEventConfig,modules[1].FullEventModel,modules[2].ControlStandardizer


def interaction_scores(model, device, cases, pair_start):
    rows=[]
    for i in (pair_start,pair_start+1):
        inp,seq=device[i]
        branch=cases[i]['divergence']
        prefix={k:v[:,:branch+1] for k,v in seq.items()}
        logits=model(inp,prefix)
        rows.append(logits[0,branch,cases[i]['pair_targets']])
    return torch.stack(rows)


def interaction_probe(model, device, cases):
    proofs=[]
    for start in (0,2):
        scores=interaction_scores(model,device,cases,start)
        loss,gap=condition_interaction_loss(scores)
        gscore=torch.autograd.grad(loss,scores,retain_graph=True)[0]
        grad=torch.autograd.grad(loss,model.audio_key_conditioner.weight)[0]
        cpu_scores=scores.detach().cpu().requires_grad_(True)
        cpu_loss,cpu_gap=condition_interaction_loss(cpu_scores)
        cpu_grad=torch.autograd.grad(cpu_loss,cpu_scores)[0]
        with torch.no_grad():
            full=torch.stack([model(*device[i])[0,cases[i]['divergence'],cases[i]['pair_targets']] for i in (start,start+1)])
        proof=dict(pair_start=start,loss=float(loss.detach()),gap=float(gap.detach()),
            gradient_norm=float(grad.norm()),prefix_full_max_error=float((scores.detach()-full).abs().max()),
            cpu_mps_loss_error=abs(float(cpu_loss.detach())-float(loss.detach())),
            cpu_mps_score_gradient_error=float((cpu_grad-gscore.detach().cpu()).abs().max()))
        proof['passed']=all(math.isfinite(proof[k]) for k in ('loss','gap','gradient_norm')) and proof['gradient_norm']>0 and proof['prefix_full_max_error']<=1e-4 and proof['cpu_mps_loss_error']<=1e-6 and proof['cpu_mps_score_gradient_error']<=1e-6
        if not proof['passed']:raise ValueError('Real interaction objective preflight failed: '+json.dumps(proof))
        proofs.append(proof)
    return proofs


def run():
    path = ROOT / 'bench/rhythm-v2/d82-condition-interaction-probe-v1.json'
    plan = json.loads(path.read_text()); output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['inputs_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Frozen probe input changed: ' + source)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    global FullEventConfig, FullEventModel
    FullEventConfig, FullEventModel, Standardizer = load_frozen_components(plan)
    normalizer = Standardizer.load(ROOT / plan['normalizer'])
    torch.set_num_threads(2); torch.mps.set_per_process_memory_fraction(.6)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    cases, indices, controls = [], [], []
    for pair in plan['pairs']:
        prepared = []
        for index in pair['indices']:
            row = panel.records[index]
            if row['split'] != 'train' or row['mode'] != 'phira':
                raise ValueError('Probe must use actual Phira training requests')
            control = np.asarray(row['controls'], np.float32).copy()
            field = CONTROL_FIELDS.index(pair['field'])
            control[[j for j in range(len(control)) if j != field], :3] = 0
            scope = control[field,4:]
            expected_scope = np.asarray([0., (row['request']['end_us']-row['request']['start_us'])/1_200_000_000], np.float32)
            if not np.array_equal(scope, expected_scope):
                raise ValueError('D82 probe accepts request-wide scope only')
            inp, seq = panel.batch([index], controls=[control])
            prepared.append((inp, seq)); indices.append(index); controls.append(control)
            cases.append(dict(index=index, chart_id=row['chart_id'], component=row['component'],
                              field=pair['field'], band=row['bands'][pair['field']],
                              request=row['request'], actual_events=row['event_count'], types=row['types']))
        if any(not torch.equal(value, prepared[1][0][key]) for key, value in prepared[0][0].items() if key != 'controls'):
            raise ValueError('Pair has different audio or visible context')
        ta, tb = [values[1]['targets'][0].tolist() for values in prepared]
        divergence = next(i for i, (a, b) in enumerate(zip(ta, tb)) if a != b)
        for key in ('tokens', 'fields', 'times', 'valid', 'allowed'):
            if not torch.equal(prepared[0][1][key][:, :divergence+1], prepared[1][1][key][:, :divergence+1]):
                raise ValueError('First branch has a non-condition discriminator')
        for case in cases[-2:]:
            case['divergence'] = divergence
            case['pair_targets'] = [ta[divergence], tb[divergence]]
            if not all(bool(prepared[j][1]['allowed'][0,divergence,t]) for j in (0,1) for t in case['pair_targets']):
                raise ValueError('Both real paired targets must be legal in each identical prefix')
    cpu_inp, cpu_seq = panel.batch(indices, controls=controls)
    raw_cpu_inp = pad_input_shapes(cpu_inp, plan['input_shapes'])
    cpu_inp = pad_input_shapes(normalizer(cpu_inp), plan['input_shapes'])
    raw_controls=raw_cpu_inp['controls'];changed_controls=cpu_inp['controls']
    known=(raw_controls[...,2]==1)&(raw_controls[...,3]==1)
    recovered=torch.where(known[...,None],changed_controls[...,:2]*normalizer.std[None,:,None]+normalizer.mean[None,:,None],changed_controls[...,:2])
    input_proof=dict(noncondition_equal=all(torch.equal(v,cpu_inp[k]) for k,v in raw_cpu_inp.items() if k!='controls'),
        flags_scope_equal=torch.equal(raw_controls[...,2:],changed_controls[...,2:]),
        unknown_equal=torch.equal(raw_controls[~known],changed_controls[~known]),
        inverse_max_error=float((recovered-raw_controls[...,:2]).abs().max()))
    if not all(input_proof[k] for k in ('noncondition_equal','flags_scope_equal','unknown_equal')) or input_proof['inverse_max_error']>2e-7:
        raise ValueError('Only the frozen known-endpoint affine transformation may change inputs')
    tensor_stats = {section: {k: tensor_health(v.numpy()) for k, v in arrays.items()}
                    for section, arrays in [('inputs', cpu_inp), ('sequence', cpu_seq)]}
    batch = [(pad_input_shapes(normalizer(panel.batch([i], controls=[c])[0]), plan['input_shapes']),
              panel.batch([i], controls=[c])[1]) for i, c in zip(indices, controls)]
    if any(not torch.equal(inp['controls'][0],cpu_inp['controls'][i]) for i,(inp,seq) in enumerate(batch)):
        raise ValueError('Training and generation condition transforms differ')
    device = [({k: v.to('mps') for k, v in inp.items()}, {k: v.to('mps') for k, v in seq.items()})
              for inp, seq in batch]
    saved = torch.load(ROOT / plan['checkpoint'], map_location='cpu', weights_only=False)
    report = dict(id=plan['id'], complete=False, passed=False, plan_sha256=sha(path), cases=cases,
                  tensor_stats=tensor_stats, input_alignment=input_proof, routes=[], optimizer_updates=0, quality_updates=0,
                  new_cloud_compute_calls=0, weights_discarded=False, weights_quarantined=True, promoted=False,
                  scope='D82 paired decision objective; exact D80 isolated model/inputs, common step2 initialization; training memory only')
    started = time.monotonic()
    try:
        for representation in ['frozen_d80_with_condition_interaction_objective']:
            config = replace(FullEventConfig(**saved['config']), audio_key_conditioning=True)
            model = FullEventModel(config).to('mps').train()
            loaded = model.load_state_dict(saved['model'], strict=False)
            if loaded.missing_keys != ['audio_key_conditioner.weight'] or loaded.unexpected_keys:
                raise ValueError('Unexpected candidate checkpoint keys')
            original = FullEventModel(FullEventConfig(**saved['config'])).to('mps').eval()
            original.load_state_dict(saved['model'])
            if sum(p.numel() for p in model.parameters()) != sum(p.numel() for p in original.parameters()) + 6144:
                raise ValueError('Expected exactly 6144 new parameters')
            if any(not torch.equal(model.state_dict()[n].detach().cpu(),v) for n,v in saved['model'].items()) or bool(model.audio_key_conditioner.weight.detach().count_nonzero()):
                raise ValueError('Old weights or new zero initializer differ from D79')
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=0.)
            restore_proof = restore_named_optimizer_state(optimizer, saved['optimizer'], list(dict(original.named_parameters())), list(dict(model.named_parameters())))
            if restore_proof['new_parameters'] != ['audio_key_conditioner.weight']:
                raise ValueError('Unexpected new optimizer parameters')
            expected_steps = [float(v['step']) for v in saved['optimizer']['state'].values()]
            initial_steps = [float(v['step']) for v in optimizer.state.values()]
            if initial_steps != expected_steps:
                raise ValueError('Optimizer initial steps differ')
            if any(a['step'] is b['step'] for a, b in zip(optimizer.state.values(), saved['optimizer']['state'].values())):
                raise ValueError('Optimizer state aliases common snapshot')
            group_proof = [dict(lr=g['lr'], betas=g['betas'], eps=g['eps'],weight_decay=g['weight_decay']) for g in optimizer.param_groups]
            if len(group_proof) != 1 or group_proof[0]['lr'] != .001:
                raise ValueError('Candidate must retain original single-group learning rate')
            torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
            route = dict(representation=representation, optimizer_restore=restore_proof, config=asdict(config), parameters=sum(p.numel() for p in model.parameters()),
                         curve=[], updates=0, learning_rate=optimizer.param_groups[0]['lr'], initial_optimizer_steps=sorted(set(initial_steps)), checkpoint_state_independent=True, optimizer_groups=group_proof, actual_update_samples=[])
            report['routes'].append(route)
            model.eval()
            route['initial_equivalence'] = []
            for index, (inp,seq) in enumerate(device):
                for known in (True, False):
                    controls=inp['controls'].clone()
                    if not known: controls[..., :3]=0
                    row={**inp,'controls':controls}
                    route['initial_equivalence'].append(initial_equivalence(original,model,row,seq,known=known,index=index))
            del original
            route['interaction_initial_probe']=interaction_probe(model,device,cases)
            model.zero_grad(set_to_none=True)
            route['initial_mps_cache_equivalence'] = cache_equivalence(model, *device[0])
            if not route['initial_mps_cache_equivalence']['passed']:
                raise ValueError('Real initial cached/full valid query equivalence failed')
            model.train()

            def measure(step):
                model.eval(); rows = []
                with torch.no_grad():
                    for case_index, (case, (inp, seq)) in enumerate(zip(cases, device)):
                        logits = model(inp, seq).masked_fill(~seq['allowed'], -1e9)
                        branch = case['divergence']
                        rows.append(dict(ce=float(event_loss(logits, seq)),
                            accuracy=float((logits.argmax(-1)[seq['valid']] == seq['targets'][seq['valid']]).float().mean()),
                            gain_min=float(1+model.audio_key_conditioner(inp['controls']).min()),
                            gain_max=float(1+model.audio_key_conditioner(inp['controls']).max()),
                            branch_correct=bool(logits.argmax(-1)[0, branch] == seq['targets'][0, branch]),
                            branch_target=int(seq['targets'][0, branch]), branch_prediction=int(logits.argmax(-1)[0, branch]),
                            branch_margin=float(logits[0,branch,int(seq['targets'][0,branch])] - logits[0,branch,int(device[case_index^1][1]['targets'][0,branch])]),
                            branch_logits={str(int(device[j][1]['targets'][0,branch])):float(logits[0,branch,int(device[j][1]['targets'][0,branch])]) for j in (case_index,case_index^1)}))
                matrices=torch.tensor([[[rows[i]['branch_logits'][str(t)] for t in cases[start]['pair_targets']] for i in (start,start+1)] for start in (0,2)])
                aux,gap=condition_interaction_loss(matrices)
                route['curve'].append(dict(step=step,rows=rows,interaction_loss=float(aux),interaction_gaps=gap.tolist())); model.train()
                report.update(wall_s=time.monotonic()-started)
                write_json(output, report)

            route['rng_before_training'] = dict(cpu_equal=torch.equal(torch.get_rng_state(),saved['cpu_rng']),mps_equal=torch.equal(torch.mps.get_rng_state(),saved['mps_rng']))
            if not all(route['rng_before_training'].values()):
                raise ValueError('Initial diagnostics unexpectedly consumed training RNG')
            measure(0)
            baseline=json.loads((ROOT/plan['baseline_report']).read_text())['routes'][0]['curve'][0]['rows']
            route['initial_ce_vs_d80']=[abs(x['ce']-y['ce']) for x,y in zip(route['curve'][0]['rows'],baseline)]
            if max(route['initial_ce_vs_d80'])>1e-6:raise ValueError('D82 starting CE differs from D80')
            for step in range(1, plan['steps']+1):
                if time.monotonic()-started > plan['max_wall_s']:
                    raise TimeoutError('Paired condition health time cap')
                optimizer.zero_grad(set_to_none=True)
                for inp, seq in device:
                    loss = event_loss(model(inp, seq), seq) / len(device)
                    loss.backward(); del loss
                # Additional supervised decisions, counted separately from original CE exposure.
                for pair_start in (0,2):
                    scores=interaction_scores(model,device,cases,pair_start)
                    interaction,_gap=condition_interaction_loss(scores)
                    (interaction/2).backward()
                    del interaction,_gap,scores
                route['auxiliary_pair_decision_exposures']=step*2
                grad = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                control_grad = float(model.control_projection.weight.grad.norm())
                watched = {n:p.detach().clone() for n,p in model.named_parameters()
                           if step in (1,25,50,100) and (n.startswith(('control_','audio_key_conditioner.')) or n=='output.weight')}
                optimizer.step()
                if watched:
                    route['actual_update_samples'].append(dict(step=step, parameters={
                        n:dict(update_norm=float((dict(model.named_parameters())[n].detach()-v).norm()),parameter_norm_before=float(v.norm()))
                        for n,v in watched.items()}))
                del watched
                route.update(updates=step, gradient_norm=float(grad), control_gradient_norm=control_grad, audio_key_gradient_norm=float(model.audio_key_conditioner.weight.grad.norm()))
                report['optimizer_updates'] += 1
                if step in (25, 50, 100):
                    measure(step)
            if [float(v['step']) for v in saved['optimizer']['state'].values()] != expected_steps:
                raise ValueError('Common optimizer snapshot mutated')
            route['common_optimizer_steps_after'] = sorted(set(expected_steps))
            unchanged = FullEventModel(FullEventConfig(**saved['config'])).to('mps').eval()
            unchanged.load_state_dict({n:p for n,p in model.state_dict().items() if n != 'audio_key_conditioner.weight'})
            inp,seq=device[0];unknown_controls=inp['controls'].clone();unknown_controls[...,:3]=0
            model.eval()
            route['learned_unknown_equivalence']=initial_equivalence(unchanged,model,{**inp,'controls':unknown_controls},seq,known=False,index=0)
            del unchanged
            model.zero_grad(set_to_none=True)
            model.eval()
            route['mps_cache_equivalence'] = cache_equivalence(model, *device[0])
            with torch.no_grad():
                mps_logits=[model(inp,seq).detach().cpu() for inp,seq in device]
            preserve_terminal(model, optimizer, report, plan, path)
            model = model.to('cpu').eval(); del optimizer
            route['cpu_cache_equivalence'] = cache_equivalence(model, *batch[0])
            with torch.no_grad():
                cross=[]
                for expected,(inp,seq) in zip(mps_logits,batch):
                    actual=model(inp,seq)
                    cross.append(compare_valid_logits(expected, actual, seq['valid'], allowed=seq['allowed']))
                route['cpu_mps_equivalence']=cross
            del mps_logits
            torch.mps.empty_cache()
            with torch.no_grad():
                requests = [EventRequest(**{**c['request'], 'copy_times': tuple(c['request']['copy_times']),
                                            'known_carry': tuple(c['request']['known_carry'])}) for c in cases]
                predictions = generate_batch(model, cpu_inp, requests, plan['token_cap'],
                                             deadline=started+plan['max_wall_s'])
            expected = [seq['targets'][0][seq['valid'][0]].tolist() for _, seq in batch]
            route['generation'] = predictions
            route['exact_memorized_outputs'] = [p['complete'] and p['tokens'] == t for p, t in zip(predictions, expected)]
            route['passed'] = (route['mps_cache_equivalence']['passed'] and route['cpu_cache_equivalence']['passed'] and all(x['passed'] for x in route['cpu_mps_equivalence']) and all(r['ce'] <= plan['maximum_ce'] and r['branch_correct'] for r in route['curve'][-1]['rows'])
                               and all(route['exact_memorized_outputs']))
            route['driver_bytes_end'] = torch.mps.driver_allocated_memory()
            del model
            write_json(output, report)
        report.update(complete=True, passed=report['routes'][-1]['passed'], wall_s=time.monotonic()-started)
        write_json(output, report)
        print(json.dumps(dict(passed=report['passed'], wall_s=report['wall_s'], routes=[
            {k: r[k] for k in ['representation', 'passed', 'exact_memorized_outputs', 'control_gradient_norm']} for r in report['routes']])))
    except Exception as exc:
        report.update(complete=True, error=f'{type(exc).__name__}: {exc}', wall_s=time.monotonic()-started)
        if report['optimizer_updates'] and 'diagnostic_checkpoint' not in report and 'optimizer' in locals() and 'model' in locals():
            preserve_terminal(model, optimizer, report, plan, path)
        write_failure_report(output, report)
        raise


if __name__ == '__main__':
    run()
