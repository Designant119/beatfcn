"""Bounded inference-only review. No training function is registered on this app."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time
import modal

from beatfcn.cloud.train_app import image as base_image, REPO

app = modal.App('beatfcn-capability-review')
data = modal.Volume.from_name('beatfcn-data')
image = base_image.add_local_dir(str(REPO / 'bench/capability-v1'), '/bench/capability-v1')
ROOT = '/data/evolution/capability-evolution-20261024/evaluation-review-v1'
CHECKPOINTS = {'3500': '/data/runs/s1-v21/best.pt', '4500': '/data/runs/evo-r01-tape/last.pt'}


def write(path, value):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False))
    tmp.replace(p)


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(16384, 16384),
              volumes={'/data': data}, timeout=1200, retries=0, max_containers=1,
              scaledown_window=2)
def review_batch(batch_id: int, source_sha256: str, wall_s: float = 1080):
    import torch
    from beatfcn.events.requests import build_request
    from beatfcn.events.canonical import MODE_TYPES
    from beatfcn.model.p1 import P1Model, P1Config
    from beatfcn.model import features as F
    from beatfcn.model.evaluation import generate_song, score_events, quality_metrics
    from beatfcn.model.benchmark import audio_features, capability_metrics
    from beatfcn.model.input_diagnostics import without_recent_history, swap_reference, mismatch_audio
    from beatfcn.train.data import Corpus, selected_for_training
    from beatfcn.train.evolution import digest
    torch.set_num_threads(4)
    started = time.perf_counter()
    manifest = json.loads(Path('/bench/capability-v1/manifest.json').read_text())
    assert digest({k:v for k,v in manifest.items() if k != 'manifest_sha256'}) == manifest['manifest_sha256']
    output = Path(ROOT)
    output.mkdir(parents=True, exist_ok=True)
    models, checkpoints = {}, {}
    for label, path in CHECKPOINTS.items():
        ck = torch.load(path, map_location='cpu', weights_only=False)
        assert ck['step'] == int(label)
        model = P1Model(P1Config(**ck['config'])).cuda().eval()
        model.load_state_dict(ck['model'])
        model.requires_grad_(False)
        models[label] = model
        checkpoints[label] = {'path': path, 'step': ck['step'], 'config': ck['config'],
                              'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                              'data': ck['data'], 'manifest_name': ck['manifest_name']}
        del ck
    if digest(checkpoints['3500']['data']) != digest(checkpoints['4500']['data']):
        raise ValueError(f"checkpoint audio configurations differ: {checkpoints['3500']['data']} vs {checkpoints['4500']['data']}")
    corpus = Corpus(**checkpoints['3500']['data'])
    all_rows = [json.loads(s) for s in Path('/data/manifest', checkpoints['3500']['manifest_name']).read_text().splitlines() if s.strip()]
    heldout = [r for r in all_rows if r.get('split') == 'val' and r.get('kept') and selected_for_training(r)]
    corpus.train_rows(heldout)
    rows = manifest['rows']
    by_audio = {}
    for r in heldout:
        by_audio.setdefault((r['mode'], r['audio_sha256']), []).append(r)
    # Frozen prefix is the existing 8-song diagnostic panel, also in the 40-song suite.
    tasks = [(r, kind, label, 'baseline') for r in rows for kind in manifest['request_kinds'] for label in CHECKPOINTS]
    diagnostics = [(r, 'from_scratch', '4500', 'history_removed') for r in rows[:8] if r['mode'] == 'mania_4k']
    diagnostics += [(r, 'reference', label, 'reference_swap') for r in rows[:8] for label in CHECKPOINTS]
    diagnostics += [(r, 'from_scratch', label, 'audio_mismatch') for r in rows[:8] for label in CHECKPOINTS]
    # Complete the original paired smoke panel, then its diagnostics, then expand coverage.
    tasks = tasks[:48] + diagnostics + tasks[48:]
    protocol = {'schema': 'beatfcn.capability_review.v1', 'manifest_sha256': manifest['manifest_sha256'],
                'model_source_sha256': source_sha256, 'checkpoints': checkpoints,
                'baseline_requests': 240, 'diagnostic_requests': len(diagnostics),
                'history_scope': 'added K-group adapters only; original autoregressive history and legality preserved',
                'reference_scope': 'same-audio alternate chart, fixed spans, controls, observed events and truth',
                'audio_scope': 'nearest-duration held-out same-mode donor, all features length-resampled; metrics on target audio',
                'training_updates': 0, 'greedy': True, 'per_request_timeout_s': 90}
    protocol_path = output / 'protocol.json'
    if protocol_path.exists() and digest(json.loads(protocol_path.read_text())) != digest(protocol):
        raise ValueError('review protocol changed; cannot mix results')
    write(protocol_path, protocol)
    resident_key, resident, af = None, None, None
    completed_now = 0
    for row, kind, label, condition in tasks:
        task_id = digest([row['chart_id'], kind, label, condition])[:24]
        path = output / 'records' / (task_id + '.json')
        if path.exists():
            continue
        if time.perf_counter() - started > wall_s - 100:
            break
        begin = time.perf_counter()
        record = {'id': task_id, 'checkpoint': label, 'condition': condition, 'song': row['audio_sha256'],
                  'chart_id': row['chart_id'], 'mode': row['mode'], 'request_kind': kind,
                  'stratum': row['benchmark_stratum'], 'seed': row['benchmark_seed'],
                  'complete': False, 'legality': []}
        try:
            chart = corpus.chart(row['chart_id'])
            duration = round(row['audio_sec'] * 1e6)
            sample = build_request(chart, kind=kind, seed=row['benchmark_seed'], audio_id=row['audio_sha256'],
                                   audio_duration_us=duration, empty_overlay=None, control_dropout=False)
            if not sample.ok:
                raise ValueError('model_view_quarantine')
            sample = replace(sample, request={**sample.request, 'controls': {'allowed_types': list(MODE_TYPES[row['mode']])}})
            truth = [e for g in sample.model_view.target_groups for e in g.events]
            if resident_key != row['audio_sha256']:
                resident = corpus.song(row['audio_sha256'], in_memory=True)
                af = audio_features(resident.mel10, duration)
                resident_key = row['audio_sha256']
            input_song = resident
            if condition == 'reference_swap':
                candidates = sorted((r for r in by_audio.get((row['mode'], row['audio_sha256']), [])
                                     if r['chart_id'] != row['chart_id']),
                                    key=lambda r: (abs(r['n_events'] - row['n_events']), r['chart_id']))
                if not candidates:
                    raise ValueError('no_same_audio_alternate_chart')
                donor = candidates[0]
                original_evidence = sample.evidence
                sample = swap_reference(sample, corpus.chart(donor['chart_id']).events)
                record.update(donor_chart=donor['chart_id'], reference_a=original_evidence, reference_b=sample.evidence,
                              reference_changed=digest(original_evidence) != digest(sample.evidence))
            if condition == 'audio_mismatch':
                donor = min((r for r in rows if r['mode'] == row['mode'] and r['audio_sha256'] != row['audio_sha256']),
                            key=lambda r: (abs(r['audio_sec'] - row['audio_sec']), r['chart_id']))
                donor_song = corpus.song(donor['audio_sha256'], in_memory=True)
                input_song = mismatch_audio(resident, donor_song)
                record.update(donor_song=donor['audio_sha256'], duration_warp=row['audio_sec'] / donor['audio_sec'])
                del donor_song
            record.update(request=sample.request, truth_events=[e.to_json() for e in truth])
            context = without_recent_history(models[label]) if condition == 'history_removed' else nullcontext()
            with context, torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                rollout = generate_song(models[label], sample, input_song,
                                        line_index=F.line_index_of(chart.events), seed=row['benchmark_seed'],
                                        device='cuda', greedy=True, deadline=time.perf_counter() + 90)
            record.update({k:v for k,v in rollout.items() if k != 'events'})
            record['generated_events'] = [e.to_json() for e in rollout['events']]
            record['scores'] = score_events(rollout['events'], truth, row['mode'])
            record['quality'] = quality_metrics(rollout['events'], truth, sample.request['regions'], row['mode'], resident.summary)
            record['benchmark'] = capability_metrics(rollout['events'], truth, sample.request, af)
        except (ValueError, F.SkipExample, AssertionError, RuntimeError) as exc:
            record['failure'] = {'reason': f'{type(exc).__name__}:{exc}'}
            if isinstance(exc, RuntimeError):
                write(path, record)
                data.commit()
                raise  # infrastructure/programming errors must not silently consume the suite
        record['wall_s'] = time.perf_counter() - begin
        write(path, record)
        data.commit()
        completed_now += 1
        print(json.dumps({k:record.get(k) for k in ('id','checkpoint','condition','mode','request_kind','complete','failure','wall_s')}), flush=True)
    result = {'batch': batch_id, 'records_written': completed_now,
              'records_total': len(list((output/'records').glob('*.json'))),
              'expected': len(tasks), 'wall_s': time.perf_counter() - started, 'training_updates': 0}
    write(output / f'batch-{batch_id}.json', result)
    data.commit()
    return result


@app.local_entrypoint()
def run(batch_id: int = 1):
    ledger_path = REPO / 'reports/capability_review_budget.json'
    ledger = json.loads(ledger_path.read_text())
    key = str(batch_id)
    if key in ledger['stages']:
        raise ValueError('batch already reserved; inspect its outcome before any further dispatch')
    reserved = sum(s['reserved_usd'] for s in ledger['stages'].values())
    amount = ledger['per_batch_reserve_usd']
    if ledger['prior_cost_upper_usd'] + ledger['other_allowance_usd'] + reserved + amount > ledger['approved_total_usd']:
        raise ValueError('original $10 total budget would be exceeded')
    stage = {'reserved_usd': amount, 'status': 'dispatched', 'started_at': time.time(), 'app_id': app.app_id}
    ledger['stages'][key] = stage
    write(ledger_path, ledger)
    source = hashlib.sha256()
    for p in sorted((REPO / 'beatfcn').rglob('*.py')):
        source.update(str(p.relative_to(REPO)).encode())
        source.update(p.read_bytes())
    try:
        result = review_batch.remote(batch_id, source.hexdigest())
        stage.update(status='complete', result=result)
        print(json.dumps(result), flush=True)
    except BaseException as exc:
        stage.update(status='failed', error=f'{type(exc).__name__}:{exc}')
        raise
    finally:
        stage['ended_at'] = time.time()
        write(ledger_path, ledger)
