"""Matched fresh Transformer/full-stack Mamba-2 local quality experiment."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.fullstack_mamba import FullStackMamba, FullStackMambaConfig
from beatfcn.model.full_event_model import FullEventModel, FullEventConfig, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.train.reference_completion import completion_row
from scripts.probe_fullstack_mamba import make_pair
from scripts.run_completion_quality import batches, metrics
from scripts.run_reference_completion import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'bench/rhythm-v2/fullstack-mamba-quality-v2.json'
ARMS = ('transformer', 'fullstack_mamba')


def prepare():
    if (ROOT / PLAN).exists():
        raise FileExistsError(PLAN)
    health = read('reports/fullstack_mamba_health_v1.json')
    if not health['complete'] or not health['passed']:
        raise ValueError('Full-stack Mamba health not passed')
    if not read('reports/fullstack_mamba_backend_parity_v1.json')['passed']:
        raise ValueError('Actual CPU/MPS gradient equivalence is required')
    base = read('bench/rhythm-v2/completion-quality-v1.json')
    source = dict(base['source_sha256'])
    source.update(read('bench/rhythm-v2/fullstack-mamba-health-v1.json')['source_sha256'])
    for path in ('scripts/run_fullstack_mamba_quality.py', 'scripts/probe_fullstack_mamba.py',
                 'reports/fullstack_mamba_health_v1.json', 'design/FULLSTACK_MAMBA_v1.md',
                 'bench/rhythm-v2/fullstack-mamba-health-v1.json',
                 'reports/fullstack_mamba_backend_parity_v1.json',
                 'reports/fullstack_mamba_cpu_timing_v1.json', 'scripts/verify_fullstack_backend.py'):
        source[path] = sha(path)
    plan = {k: base[k] for k in ('seed', 'tape', 'features', 'steps', 'songs', 'validation_indices',
                                  'mode_exposure', 'task_exposure', 'hidden_type_exposure')}
    t, m, copied = make_pair()
    plan.update(id='fullstack-mamba-quality-v2', source_sha256=source,
        configs=dict(transformer=asdict(t.c), fullstack_mamba=asdict(m.c)),
        shared_initial_parameters=copied, parameters={a: sum(p.numel() for p in model.parameters())
                                                      for a, model in zip(ARMS, (t, m))},
        initialization='Fresh backbones seed20261026; identical compatible input/output weights; fresh AdamW for both. No health checkpoint or historical optimizer loaded.',
        learning_rate=.001, token_cap=4096, train_wall_s=3600, evaluation_wall_s=1800,
        deadline_unix=1793037775., checkpoints=[250, 500, 1000],
        report='reports/fullstack_mamba_quality_v2.json', artifacts='artifacts/fullstack-mamba-v1/quality-v2',
        quality_gates=dict(song_f1_gain=.05, task_degradation_max=.02, improving_songs=12,
                           complete_fraction=.95, close_interval_degradation_max=.05),
        early_stop='2 incomplete/illegal candidate songs; or first4 all empty; or first4 all tap/empty with >=8 hidden non-tap notes.',
        backend='FP32 Transformer MPS / Mamba CPU4; actual output and all-parameter gradient parity <=1e-4; no fused CUDA speed claim',
        devices=dict(transformer='mps', fullstack_mamba='cpu'), cpu_threads=4,
        new_cloud_spend_usd=0)
    write(PLAN, plan)
    print(json.dumps(dict(prepared=True, parameters=plan['parameters'], songs=len(plan['songs']))))


def factory(arm, config):
    return FullEventModel(FullEventConfig(**config)) if arm == 'transformer' else FullStackMamba(FullStackMambaConfig(**config))


def train(plan, report):
    torch.set_num_threads(plan['cpu_threads']); torch.mps.set_per_process_memory_fraction(.6)
    if not torch.backends.mps.is_available() or os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise RuntimeError('Explicit native MPS required')
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    t, m, shared = make_pair()
    if shared != plan['shared_initial_parameters']:
        raise ValueError('Initialization contract changed')
    for name in shared:
        if not torch.equal(t.state_dict()[name], m.state_dict()[name]):
            raise ValueError('Unmatched shared initialization: ' + name)
    report['shared_initial_parameters_equal'] = True
    models = dict(zip(ARMS, (t.to(plan['devices']['transformer']), m.to(plan['devices']['fullstack_mamba']))))
    opts = {a: torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.) for a, model in models.items()}
    started = time.monotonic()
    for step, items in enumerate(plan['steps'], 1):
        if time.monotonic() - started >= plan['train_wall_s'] or time.time() >= plan['deadline_unix']:
            raise TimeoutError('Frozen matched training bound')
        cpu = [batches(panel, item, plan)['reference'] for item in items]
        values = {}
        for arm in ARMS:
            model, opt = models[arm], opts[arm]
            model.train(); opt.zero_grad(set_to_none=True)
            total, tokens = 0., 0
            for pair in cpu:
                inp, seq = ({k: v.to(plan['devices'][arm]) for k, v in section.items()} for section in pair)
                loss = event_loss(model(inp, seq), seq) / len(cpu)
                if not torch.isfinite(loss):
                    raise ValueError('Nonfinite matched supervised loss')
                loss.backward(); total += float(loss.detach()); tokens += int(seq['valid'].sum())
                del inp, seq, loss
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
            opt.step(); report['updates'][arm] = step; report['tokens'][arm] += tokens
            values[arm] = dict(ce=total, preclip_norm=float(norm))
        report.update(status='matched_training', training_wall_s=time.monotonic() - started)
        if step % 25 == 0:
            report['curve'].append(dict(step=step, values=values, wall_s=report['training_wall_s']))
            write(plan['report'], report)
            print(json.dumps(dict(step=step, values=values, wall_s=report['training_wall_s'])), flush=True)
        if step in plan['checkpoints']:
            validation = {}
            with torch.no_grad():
                for arm in ARMS:
                    model = models[arm].eval(); rows = []
                    for index in plan['validation_indices']:
                        for task in ('middle', 'random_groups', 'continuation'):
                            pair = batches(panel, dict(index=index, task=task), plan)['reference']
                            inp, seq = ({k: v.to(plan['devices'][arm]) for k, v in section.items()} for section in pair)
                            rows.append(dict(index=index, task=task, ce=float(event_loss(model(inp, seq), seq))))
                    validation[arm] = rows
            report['validation'].append(dict(step=step, rows=validation))
            write(plan['report'], report)
    for arm in ARMS:
        path = ROOT / f"{plan['artifacts']}/{arm}/step-1000.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('xb') as stream:
            torch.save(dict(model={k: v.detach().cpu() for k, v in models[arm].state_dict().items()},
                       config=asdict(models[arm].c), optimizer=opts[arm].state_dict(),
                       family=arm, updates=1000, plan_sha256=sha(PLAN), auto_resume=False), stream)
        report['checkpoints'][arm] = dict(path=str(path.relative_to(ROOT)), sha256=sha(path.relative_to(ROOT)))
    del models, opts, t, m
    torch.mps.empty_cache()
    report.update(training_complete=True, status='trained', training_wall_s=time.monotonic() - started)
    write(plan['report'], report)


@torch.inference_mode()
def evaluate(plan, report):
    torch.set_num_threads(2)
    models = {}
    for arm in ARMS:
        entry = report['checkpoints'][arm]
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('Checkpoint identity changed')
        state = torch.load(ROOT / entry['path'], map_location='cpu', weights_only=False)
        model = factory(arm, state['config']).eval(); model.load_state_dict(state['model'])
        models[arm] = model; del state
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    started = time.monotonic()
    deadline = min(started + plan['evaluation_wall_s'], started + plan['deadline_unix'] - time.time())
    for song in plan['songs']:
        predictions, cores, failures = {a: [] for a in ARMS}, {a: [] for a in ARMS}, {a: None for a in ARMS}
        visible, hidden = [], []
        for index in song['indices']:
            _, request, meta = completion_row(panel, index, song['task'], reference=True, seed=plan['seed'])
            visible.extend(meta['visible']); hidden.extend(meta['hidden'])
            pair = batches(panel, dict(index=index, task=song['task']), plan)['reference']
            for arm in ARMS:
                if failures[arm] is not None:
                    continue
                pred = generate_batch(models[arm], pair[0], [request], plan['token_cap'], deadline=deadline)[0]
                predictions[arm].extend(pred['events'])
                cores[arm].append(dict(index=index, request=asdict(request), prediction=pred))
                if not pred['complete']:
                    failures[arm] = 'deadline' if pred['deadline_reached'] else 'token_cap'
        outputs = {}
        for arm in ARMS:
            score = metrics(predictions[arm], hidden, visible, failures[arm] is None, song['duration_us'])
            if score['legality']:
                failures[arm] = 'independent_legality'; score['complete'] = False
            path = f"{plan['artifacts']}/{arm}/songs/{song['chart_id'].replace(':', '-')}.json"
            write(path, dict(song=song, cores=cores[arm], visible=visible, hidden=hidden,
                    prediction=predictions[arm], merged_unmodified_visible_and_prediction=visible + predictions[arm],
                    metrics=score, failure=failures[arm]))
            with (ROOT / path).with_suffix('.csv').open('w') as stream:
                stream.write('source,onset_us,end_us,type\n')
                for source, items in (('visible', visible), ('hidden', hidden), ('predicted', predictions[arm])):
                    for e in items:
                        stream.write(f"{source},{e['onset_us']},{e['end_us']},{e['type']}\n")
            outputs[arm] = dict(artifact=path, failure=failures[arm], **{k: v for k, v in score.items() if k != 'scores'})
        report['songs'].append(dict(chart_id=song['chart_id'], stratum=song['stratum'], task=song['task'], outputs=outputs))
        report.update(status='whole_song_evaluation', evaluation_wall_s=time.monotonic() - started)
        write(plan['report'], report)
        print(json.dumps(dict(evaluated_songs=len(report['songs']), chart_id=song['chart_id'], outputs=outputs,
                              wall_s=report['evaluation_wall_s'])), flush=True)
        rows = report['songs']
        bad = sum(not r['outputs']['fullstack_mamba']['complete'] for r in rows)
        empty = len(rows) >= 4 and all(r['outputs']['fullstack_mamba']['pred_count'] == 0 for r in rows[:4])
        tap_only = (len(rows) >= 4 and all(set(r['outputs']['fullstack_mamba']['predicted_types']) <= {'tap'} for r in rows[:4])
                    and sum(r['outputs']['fullstack_mamba']['pred_count'] for r in rows[:4]) > 0
                    and sum(sum(e['type'] != 'tap' for e in read(r['outputs']['fullstack_mamba']['artifact'])['hidden']) for r in rows[:4]) >= 8)
        if bad >= 2 or empty or tap_only or time.monotonic() >= deadline:
            report['stop_reason'] = 'candidate_incomplete' if bad >= 2 else ('candidate_all_empty' if empty else ('candidate_all_tap' if tap_only else 'evaluation_deadline'))
            break
    rows = report['songs']; gains = [r['outputs']['fullstack_mamba']['full_note_f1'] - r['outputs']['transformer']['full_note_f1'] for r in rows]
    by_task = {task: float(np.mean([g for r, g in zip(rows, gains) if r['task'] == task]))
               for task in ('middle', 'random_groups', 'continuation') if any(r['task'] == task for r in rows)}
    types = Counter()
    for r in rows:
        types.update(r['outputs']['fullstack_mamba']['predicted_types'])
    short = float(np.mean([r['outputs']['fullstack_mamba']['short_interval_fraction'] - r['outputs']['transformer']['short_interval_fraction'] for r in rows]))
    complete = len(rows) == len(plan['songs'])
    complete_fraction = sum(r['outputs']['fullstack_mamba']['complete'] for r in rows) / len(plan['songs'])
    passed = (complete and np.mean(gains) >= .05 and sum(g > 0 for g in gains) >= 12 and len(by_task) == 3
              and min(by_task.values()) >= -.02 and complete_fraction >= .95 and len(types) >= 2
              and types['hold'] > 0 and short <= .05)
    report.update(evaluation_panel_complete=complete, quality_passed=bool(passed),
        summary=dict(observed_songs=len(rows), unattempted_songs=len(plan['songs']) - len(rows),
            paired_song_mean_full_note_f1_gain=float(np.mean(gains)), improving_songs=sum(g > 0 for g in gains),
            task_gains=by_task, candidate_types=dict(types), short_interval_degradation=short,
            complete_fraction_of_expected=complete_fraction), evaluation_wall_s=time.monotonic() - started)


def run():
    plan = read(PLAN)
    if (ROOT / plan['report']).exists():
        raise FileExistsError(plan['report'])
    started = time.monotonic()
    report = dict(id=plan['id'], plan_sha256=sha(PLAN), complete=False, training_complete=False,
        quality_passed=False, updates={a: 0 for a in ARMS}, tokens={a: 0 for a in ARMS},
        curve=[], validation=[], checkpoints={}, songs=[], new_cloud_spend_usd=0, status='verifying')
    try:
        for path, expected in plan['source_sha256'].items():
            if sha(path) != expected:
                raise ValueError('Frozen source/input changed: ' + path)
        train(plan, report); evaluate(plan, report)
        report.update(complete=True, status='quality_passed' if report['quality_passed'] else 'quality_failed')
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}'); raise
    finally:
        report['wall_s'] = time.monotonic() - started
        write(plan['report'], report)
    print(json.dumps(dict(status=report['status'], updates=report['updates'], summary=report['summary'], wall_s=report['wall_s'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args(); prepare() if args.prepare else run()
