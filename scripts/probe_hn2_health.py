"""One authorized fresh full-size health run, never a quality warm start.

Fixed 100 applied AdamW updates, 600 seconds including preparation/evaluation,
10GiB own physical footprint, 15GiB disk reserve; no weights are persisted.
"""
from collections import Counter, defaultdict
from dataclasses import asdict
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import sys
import threading
import time

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.evaluation import score_events
from beatfcn.model.hn2_native import HN2Config, HN2Native
from beatfcn.model.local_song_rollout import canonical
from beatfcn.train.hn2_native import canonical_targets, loss_pass, sample_pass, two_pass, validate_targets
from beatfcn.train.modal_event_dataset import ModalEventDataset, collate_event_batch
from beatfcn.train.process_memory import physical_footprint
from scripts.check_experiment_storage import GIB, admission
from scripts.probe_hn2_native import tensor_digest


ROOT = Path(__file__).resolve().parents[1]
ART = ROOT/'artifacts/modal120m-local-health-v1'
MANIFEST = 'artifacts/modal120m-data-v1/prepared-v3/manifest.json'
PRIOR = 'artifacts/c17-masked-native-events-v1/priors.json'
OLD_PLAN = 'bench/rhythm-v2/c13-native-quality-v1.json'
SEED = 2026103001
FREE_SEEDS = (2026103002, 2026103003, 2026103004)
SOURCES = ('beatfcn/model/hn2_native.py', 'beatfcn/train/hn2_native.py',
           'tests/test_hn2_native.py', 'scripts/probe_hn2_native.py',
           'beatfcn/train/modal_event_dataset.py', 'scripts/probe_hn2_health.py')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def plain(value):
    if isinstance(value, dict):
        return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def write(name, value):
    path = ART/name
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(plain(value), ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


class Guard:
    def __init__(self):
        self.start = time.monotonic()
        self.done = threading.Event()
        self.rows = []
        self.phase = 'preparation'
        self.failure = None
        self.old_handler = signal.getsignal(signal.SIGUSR1)
        signal.signal(signal.SIGUSR1, self.interrupt)
        self.thread = threading.Thread(target=self.watch, daemon=True)
        self.thread.start()

    def interrupt(self, *unused):
        raise RuntimeError(self.failure or 'HN2 health watchdog')

    def watch(self):
        while not self.done.wait(.25):
            try:
                value = physical_footprint()
                self.rows.append(dict(**value, wall_s=time.monotonic()-self.start, phase=self.phase))
                if value['physical_footprint_bytes'] > 10*GIB:
                    raise MemoryError('physical10GiB cap')
                if time.monotonic()-self.start > 600:
                    raise TimeoutError('wall600s cap')
            except BaseException as exc:
                self.failure = f'{type(exc).__name__}: {exc}'
                os.kill(os.getpid(), signal.SIGUSR1)
                return

    def check(self, phase):
        self.phase = phase
        if self.failure:
            raise RuntimeError(self.failure)
        elapsed = time.monotonic()-self.start
        if elapsed >= 590:
            raise TimeoutError('590s safe stop before600s hard cap')
        if admission(ROOT, 64*2**20, 15*GIB)['status'] != 'admitted':
            raise MemoryError('Disk15GiB reserve')
        footprint = physical_footprint()['physical_footprint_bytes']
        if footprint >= 10*GIB:
            raise MemoryError('physical10GiB cap')
        return elapsed

    def close(self):
        self.done.set()
        self.thread.join(timeout=1)
        signal.signal(signal.SIGUSR1, self.old_handler)
        return dict(interval_s=.25, peak_physical_bytes=max((r['physical_footprint_bytes'] for r in self.rows), default=0),
                    wall_s=time.monotonic()-self.start, failure=self.failure, samples=self.rows)


def selected_batches(dataset):
    previous = json.loads((ROOT/OLD_PLAN).read_text())
    desired = previous['health_indices'] + [previous['empty_health_index']]
    lookup = {record['index']:i for i,record in enumerate(dataset.records)}
    selected = []
    for old in desired:
        if old not in lookup:
            # Only absent identities permit a public, deterministic fallback.
            public_order = sorted(range(len(dataset)), key=lambda i:(dataset.records[i]['chart_id'], dataset.records[i]['index']))
            used = {entry['dataset_index'] for entry in selected}
            candidate = next(i for i in public_order if i not in used)
            selected.append(dict(legacy_index=old, dataset_index=candidate, fallback='First unused public(chart_id,index)'))
        else:
            selected.append(dict(legacy_index=old, dataset_index=lookup[old], fallback=None))
    batches, identities = [], []
    totals = Counter()
    for item in selected:
        row = dataset[item['dataset_index']]
        batch = collate_event_batch([row])
        validate_targets(batch['inputs'], batch['targets'], batch['target_valid'])
        labels = batch['targets'][0, batch['target_valid'][0], 2].tolist()
        totals.update(labels)
        stats = {}
        for key,value in batch['inputs'].items():
            stats[key] = dict(shape=list(value.shape), dtype=str(value.dtype), nonfinite=int((~torch.isfinite(value)).sum()),
                min=float(value.min()) if value.numel() else None, max=float(value.max()) if value.numel() else None,
                mean=float(value.float().mean()) if value.numel() else None,
                std=float(value.float().std(unbiased=False)) if value.numel() else None)
        if any(v['nonfinite'] for v in stats.values()):
            raise ValueError('Nonfinite public tensor')
        identity = dict(**item, metadata=row['metadata'], targets=batch['targets'][0,batch['target_valid'][0]].tolist(),
            type_counts=dict(Counter(labels)), input_sha256=tensor_digest(batch['inputs']),
            target_sha256=tensor_digest(dict(targets=batch['targets'],valid=batch['target_valid'])), tensors=stats)
        batches.append(batch)
        identities.append(identity)
    if not all(totals[k] for k in range(4)) or not identities[-1]['targets'] == []:
        raise ValueError('Frozen panel must collectively cover four types/hold and end with a true empty window')
    return batches, identities, totals


def events_from_state(state):
    return [RhythmEvent(int(h), int(e), ('tap','hold','flick','drag')[int(k)], int(l))
            for h,e,k,l in state['events'][0, state['active'][0]].tolist()]


def scores(prediction, truth):
    pp, tt = canonical(prediction, 'phira'), canonical(truth, 'phira')
    result = score_events(pp, tt, 'phira')
    for ms in (10,50):
        tp = result[f'note_{ms}ms']['tp']-result[f'hold_head_{ms}ms']['tp']+result[f'hold_joint_{ms}ms']['tp']
        result[f'full_note_f1_{ms}ms'] = 2*tp/max(len(pp)+len(tt), 1)
    result['per_type'] = {}
    for kind in ('tap','hold','flick','drag'):
        item = score_events([e for e in pp if e.type == kind], [e for e in tt if e.type == kind], 'phira')
        result['per_type'][kind] = item['hold_joint_50ms' if kind == 'hold' else 'note_50ms']
    return result


def hold_errors(prediction, truth):
    pp, tt = [e for e in prediction if e.type == 'hold'], [e for e in truth if e.type == 'hold']
    pairs = sorted((abs(p.onset_us-t.onset_us),i,j) for i,p in enumerate(pp) for j,t in enumerate(tt)
                   if abs(p.onset_us-t.onset_us) <= 50_000)
    usedp, usedt, errors = set(), set(), []
    for _,i,j in pairs:
        if i not in usedp and j not in usedt:
            usedp.add(i);usedt.add(j)
            errors.append(abs(pp[i].end_us-tt[j].end_us)/1000)
    return dict(predicted=len(pp), true=len(tt), head_matches=len(errors), tail_abs_ms=errors,
                median_ms=statistics.median(errors) if errors else None, joint50=sum(e <= 50 for e in errors))


def proper_summary(output, detail, priors):
    valid = detail['target_valid'][0]
    labels = detail['targets'][0,valid,2]
    lp = detail['type_logits'][0,valid].float().log_softmax(-1)
    probability = lp.exp()
    result = dict(count=int(valid.sum()), first=None, by_type={}, type_nll=None)
    if len(labels):
        target_nll = -lp.gather(1, labels[:,None]).squeeze(1)
        result['type_nll'] = float(target_nll.mean())
        result['first'] = dict(kind=int(labels[0]), nll=float(target_nll[0]),
            priors={name:-math.log(p[int(labels[0])]) for name,p in priors.items()})
    for k,kind in enumerate(('tap','hold','flick','drag')):
        positive, negative = labels == k, labels != k
        other = lp.masked_fill(torch.arange(4)[None] == k, -torch.inf).logsumexp(-1)
        binary = torch.where(positive, -lp[:,k], -other)
        result['by_type'][kind] = dict(positive_n=int(positive.sum()), negative_n=int(negative.sum()),
            positive_nll=float(-lp[positive,k].mean()) if positive.any() else None,
            negative_binary_nll=float(binary[negative].mean()) if negative.any() else None,
            binary_nll=float(binary.mean()) if len(labels) else None,
            prior_binary_nll={name:float(torch.where(positive, torch.tensor(-math.log(p[k])),
                torch.tensor(-math.log1p(-p[k]))).mean()) if len(labels) else None for name,p in priors.items()})
    active_probability = output['active_logits'][0].float().sigmoid()
    result['occupancy'] = dict(expected=float(active_probability.sum()), marginal_mode_active=int((active_probability >= .5).sum()),
        true_slots_below_half=int((active_probability[valid] < .5).sum()),
        expected_on_NULL_slots=float(active_probability[~valid].sum()), marginal_mode_is_not_chart_MAP=True)
    holds = valid & (detail['targets'][0,:,2] == 1)
    result['hold_tail_nll'] = float(detail['tail_nll'][0,holds].mean()) if holds.any() else None
    result['head_nll'] = float(detail['head_nll'][0,valid].mean()) if valid.any() else None
    return result


@torch.no_grad()
def evaluate(model, batches, identities, prior, priors, step, seed, guard):
    records = []
    path = f'evaluation/step-{step:04d}-seed-{seed}.json'
    for i,(batch,identity) in enumerate(zip(batches,identities)):
        guard.check(f'evaluate-{step}-{i}-seed{seed}')
        inputs = batch['inputs']
        generator = torch.Generator().manual_seed(seed+identity['legacy_index']*17)
        out0 = model(inputs, pass_index=0)
        z = sample_pass(model, out0, prior, generator=generator)
        out1 = model(inputs, state=z, memory=out0['memory'], pass_index=1)
        y = sample_pass(model, out1, prior, generator=generator)
        truth = [RhythmEvent(int(h), int(e), ('tap','hold','flick','drag')[int(k)], int(l))
                 for h,e,k,l in batch['targets'][0,batch['target_valid'][0]].tolist()]
        record = dict(dataset_index=identity['dataset_index'], legacy_index=identity['legacy_index'],
            prediction=[asdict(e) for e in events_from_state(y)], provisional=[asdict(e) for e in events_from_state(z)],
            prediction_state=plain(y), provisional_state=plain(z), true=[asdict(e) for e in truth],
            scores=scores(events_from_state(y), truth), hold=hold_errors(events_from_state(y), truth))
        records.append(record)
        write(path, dict(step=step, seed=seed, complete=False, records=records,
                         free_predictions_persisted_before_target_proper=True))
        l0, d0 = loss_pass(model, out0, batch['targets'], batch['target_valid'], prior, details=True)
        l1, d1 = loss_pass(model, out1, batch['targets'], batch['target_valid'], prior, details=True)
        if not torch.isfinite(l0+l1) or not d0['supported'] or not d1['supported']:
            raise FloatingPointError('Health evaluation finite/support')
        record['loss'] = float(l0+l1)
        record['p0'] = proper_summary(out0,d0,priors)
        record['p1'] = proper_summary(out1,d1,priors)
        record['factors'] = {name:{key:float(details[key]) for key in ('occupancy','head','kind','tail')}
                             for name,details in (('p0',d0),('p1',d1))}
        write(path, dict(step=step, seed=seed, complete=False, records=records,
                         free_predictions_persisted_before_target_proper=True))
        del out0,out1,z,y,d0,d1,l0,l1
        gc.collect()
    summary = dict(mean_loss=statistics.mean(r['loss'] for r in records),
        mean_full50=statistics.mean(r['scores']['full_note_f1_50ms'] for r in records[:4]),
        minimum_full50=min(r['scores']['full_note_f1_50ms'] for r in records[:4]),
        mean_onset10=statistics.mean(r['scores']['onset_10ms']['f1'] or 0. for r in records[:4]),
        empty_events=len(records[-1]['prediction']), total_prediction=sum(len(r['prediction']) for r in records),
        scope='Training-health windows only; conditional proper is not free quality')
    write(path, dict(step=step,seed=seed,complete=True,records=records,summary=summary,
                     free_predictions_persisted_before_target_proper=True))
    return summary


def main():
    if torch.__version__ != '2.14.0' or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') == '1':
        raise ValueError('Installed isolatedtorch2.14/no fallback required')
    torch.set_num_threads(4)
    if (ART/'plan.json').exists():
        raise FileExistsError('This health identity is single-use; no automatic rerun/resume')
    ART.mkdir(parents=True,exist_ok=True)
    initial_disk = admission(ROOT,64*2**20,15*GIB)
    if initial_disk['status'] != 'admitted':
        raise MemoryError('Disk reserve')
    guard = Guard()
    report = dict(id='modal120m-local-health-v1',status='preparing',optimizer_updates=0,
                  previously_applied_updates=24,previous_updates_excluded_from_fresh_health=True,
                  cloud_calls=0,weights_saved=False,optimizer_saved=False)
    model = optimizer = initial = None
    try:
        command = [sys.executable,'-m','unittest','discover','-s','tests','-p','test_hn2_native.py','-v']
        test_start = time.monotonic()
        test = subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=60)
        (ART/'tests.log').write_text(test.stdout+test.stderr)
        write('tests.json',dict(command=command,returncode=test.returncode,wall_s=time.monotonic()-test_start,
                                log_sha256=sha(ART/'tests.log'),optimizer_updates=0))
        if test.returncode:
            raise RuntimeError('Frozen HN2 regression failed')
        dataset = ModalEventDataset(ROOT,MANIFEST,split='train',views=['draft'])
        batches,identities,counts = selected_batches(dataset)
        frequencies = np.full(4,.5)
        for record in dataset.records:
            a,b = record['target_slice']
            frequencies += np.bincount(dataset.arrays['targets'][a:b,2],minlength=4)
        full_prior = (frequencies/frequencies.sum()).tolist()
        frozen = json.loads((ROOT/PRIOR).read_text())
        duration = frozen['duration']
        panel = [(counts[k]+.5)/(sum(counts.values())+2) for k in range(4)]
        priors = dict(full_v3_train_draft_frequency=full_prior,health_panel_member_frequency=panel,
                      frozen_c17_train_frequency=frozen['global_type_probability'])
        write('data-probe.json',dict(identities=identities,panel_counts=dict(counts),priors=priors))
        source_identity = {name:dict(bytes=(ROOT/name).stat().st_size,sha256=sha(ROOT/name)) for name in SOURCES}
        plan = dict(id=report['id'],seed=SEED,free_primary_seed=FREE_SEEDS[0],final_diagnostic_seeds=FREE_SEEDS[1:],
            max_applied_updates=100,max_wall_s=600,safe_stop_wall_s=590,physical_cap_bytes=10*GIB,
            disk=initial_disk,source=source_identity,manifest_sha256=sha(ROOT/MANIFEST),prior_sha256=sha(ROOT/PRIOR),
            old_health_plan_sha256=sha(ROOT/OLD_PLAN),selection=[{k:v for k,v in x.items() if k not in ('tensors','targets')} for x in identities],
            rotation='Exact five recorded windows, batch1, (step-1)%5;no target-sized query/decode',
            optimizer=dict(name='AdamW',lr=.0002,weight_decay=.01,clip_grad_norm=1.),
            dropout=False,augmentation=False,weights_loaded=False,query_slots=512,
            caveat='Four windows collectively cover four types;drag has only one positive object. No independent quality evidence.')
        write('plan.json',plan)
        torch.manual_seed(SEED)
        model = HN2Native(HN2Config()).train()
        report['parameters'] = sum(p.numel() for p in model.parameters() if p.requires_grad)
        report['initial_tensor_sha256'] = tensor_digest(model.state_dict())
        initial = {name:p.detach().clone() for name,p in model.named_parameters()}
        optimizer = torch.optim.AdamW(model.parameters(),lr=.0002,weight_decay=.01)
        rng = torch.Generator().manual_seed(SEED+17)
        nonzero_updates = Counter()
        max_nonzero_scalars = defaultdict(int)
        report.update(status='running',curve=[],evaluation={},gradient_snapshots=[])
        report['evaluation']['0'] = evaluate(model,batches,identities,duration,priors,0,FREE_SEEDS[0],guard)
        write('report.json',report)
        for step in range(1,101):
            guard.check(f'update-{step}')
            batch_index = (step-1)%5
            batch = batches[batch_index]
            optimizer.zero_grad(set_to_none=True)
            loss,metrics,z,_ = two_pass(model,batch['inputs'],batch['targets'],batch['target_valid'],duration,generator=rng)
            if not torch.isfinite(loss) or not metrics['p0']['supported'] or not metrics['p1']['supported']:
                raise FloatingPointError('Training finite/support')
            loss.backward()
            current_gradients = {}
            for name,p in model.named_parameters():
                if p.grad is None:
                    current_gradients[name] = dict(present=False,finite=True,nonzero=0,norm=0.)
                    continue
                finite = bool(torch.isfinite(p.grad).all())
                if not finite:
                    raise FloatingPointError('Nonfinite gradient '+name)
                nonzero = int(torch.count_nonzero(p.grad))
                nonzero_updates[name] += bool(nonzero)
                max_nonzero_scalars[name] = max(max_nonzero_scalars[name],nonzero)
                current_gradients[name] = dict(present=True,finite=finite,nonzero=nonzero,norm=float(p.grad.float().norm()))
            gradient_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True))
            optimizer.step()
            report['optimizer_updates'] = step
            applied = dict(step=step,batch_index=batch_index,legacy_index=identities[batch_index]['legacy_index'],
                loss=float(loss.detach()),factors=plain(metrics),gradient_norm=gradient_norm,
                nonzero_gradient_tensors=sum(v['nonzero']>0 for v in current_gradients.values()),wall_s=time.monotonic()-guard.start)
            report['curve'].append(applied)
            with (ART/'updates.jsonl').open('a') as stream:
                stream.write(json.dumps(applied,allow_nan=False)+'\n')
            del loss,metrics,z
            if step%25 == 0:
                usage = {}
                for name,p in model.named_parameters():
                    state = optimizer.state.get(p,{})
                    usage[name] = dict(numel=p.numel(),**current_gradients[name],
                        nonzero_gradient_updates=nonzero_updates[name],max_nonzero_gradient_scalars=max_nonzero_scalars[name],
                        adam_step=int(state['step']) if state else 0,
                        delta_norm=float((p.detach()-initial[name]).float().norm()))
                write(f'usage/step-{step:04d}.json',dict(applied_updates=step,parameters=report['parameters'],tensors=usage,
                    note='Snapshot ends on real empty window;current type/tail gradients may legitimately be zero. Cumulative counts are separate.'))
                report['gradient_snapshots'].append(dict(step=step,tensor_sha256=tensor_digest(model.state_dict()),
                    tensors=len(usage),finite=all(v['finite'] for v in usage.values()),
                    tensors_with_nonzero_gradient_history=sum(v['nonzero_gradient_updates']>0 for v in usage.values())))
                report['evaluation'][str(step)] = evaluate(model,batches,identities,duration,priors,step,FREE_SEEDS[0],guard)
                report['evaluation'][str(step)]['last25_mean_training_loss'] = statistics.mean(r['loss'] for r in report['curve'][-25:])
                write('report.json',report)
                print(json.dumps(dict(step=step,wall_s=time.monotonic()-guard.start,evaluation=report['evaluation'][str(step)])),flush=True)
        report['final_diagnostic_seeds'] = {}
        for seed in FREE_SEEDS[1:]:
            if time.monotonic()-guard.start > 550:
                report['final_diagnostic_seeds'][str(seed)] = dict(attempted=False,reason='Preserve600s cap')
                continue
            report['final_diagnostic_seeds'][str(seed)] = evaluate(model,batches,identities,duration,priors,100,seed,guard)
        report['final_tensor_sha256'] = tensor_digest(model.state_dict())
        report['initial_to_final_mean_loss_ratio'] = report['evaluation']['100']['mean_loss']/report['evaluation']['0']['mean_loss']
        report['status'] = 'completed_health_only'
    except BaseException as exc:
        report['status'] = 'stopped'
        report['failure'] = f'{type(exc).__name__}: {exc}'
        if model is not None:
            report['final_tensor_sha256'] = tensor_digest(model.state_dict())
        raise
    finally:
        report['runtime'] = guard.close()
        report['source_unchanged'] = {name:sha(ROOT/name) == plan['source'][name]['sha256'] for name in SOURCES} if 'plan' in locals() else {}
        write('report.json',report)
        del optimizer,model,initial
        gc.collect()
    print(json.dumps({k:v for k,v in report.items() if k not in ('curve','runtime','source_unchanged')},ensure_ascii=False),flush=True)


if __name__ == '__main__':
    main()
