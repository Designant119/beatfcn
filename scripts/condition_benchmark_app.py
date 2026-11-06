"""Same repaired inputs for baseline and paired candidates, plus exact-scope controls.

Prepared independently of the frozen training source. Inference only; the local
entrypoint reserves its full resource envelope in the cumulative campaign ledger.
"""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import modal

from beatfcn.cloud.train_app import image as base_image, REPO
from beatfcn.cloud.rhythm_diagnostic_app import CHECKPOINTS, source_digest, save
from beatfcn.cloud.condition_training_app import ROOT as TRAIN_ROOT, RUNS

app = modal.App('beatfcn-condition-benchmark')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (base_image.add_local_file(str(REPO / 'bench/capability-v1/manifest.json'), '/bench/manifest.json')
         .add_local_file(str(REPO / 'scripts/condition_benchmark_app.py'), '/bench/evaluator.py'))
OUTPUT = TRAIN_ROOT + '/benchmark'
RESOURCES = {'cpu': 4, 'memory_gib': 16, 'gpu': 'L40S', 'timeout_s': 4500,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}


def key(label, chart, kind):
    return hashlib.sha256(json.dumps([label, chart, kind]).encode()).hexdigest()[:24]


def timing_baselines(audio, sample, truth, mode):
    """Two predeclared audio-only baselines; evaluate timing only, not invented note types."""
    from beatfcn.events.canonical import Event, LaneLoc
    from beatfcn.events.controls import contains
    from beatfcn.model.evaluation import editable_union
    from beatfcn.model.edit_utility import edit_utility
    runs = editable_union(sample.request['regions'])
    result = {}
    for field in ('onsets_us', 'strong_onsets_us'):
        points = [Event(f'audio:{i}', int(t), int(t), 'tap', LaneLoc(0))
                  for i, t in enumerate(audio[field]) if contains(runs, int(t))]
        result[field] = {f'{ms}ms': edit_utility(points, truth, mode, tolerance_us=ms * 1000)['timing_point']
                         for ms in (20, 50, 100)}
        result[field]['purpose'] = 'timing-only baseline; types, holds and placement are not predicted'
    return result


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(16384, 16384), volumes={'/data': data},
              timeout=4500, retries=0, max_containers=1, scaledown_window=2)
def evaluate(expected_source_sha256: str, evaluator_sha256: str):
    import numpy as np
    import torch
    import beatfcn
    from beatfcn.events.canonical import Event, MODE_TYPES
    from beatfcn.events.requests import build_request
    from beatfcn.events.controls import raw_controls
    from beatfcn.model import features as F
    from beatfcn.model.p1 import P1Model, P1Config
    from beatfcn.model.evaluation import generate_song, score_events, quality_metrics, editable_union
    from beatfcn.model.benchmark import audio_features, capability_metrics, metric_source_sha256, scorecard
    from beatfcn.model.condition_probe import control_views
    from beatfcn.model.edit_utility import edit_utility
    from beatfcn.train.density import complete_window, FIELD
    from beatfcn.train.data import Corpus
    torch.set_num_threads(4)
    started = time.perf_counter()
    if source_digest(Path(beatfcn.__file__).parent) != expected_source_sha256:
        raise ValueError('inference code identity changed')
    if hashlib.sha256(Path('/bench/evaluator.py').read_bytes()).hexdigest() != evaluator_sha256:
        raise ValueError('uploaded evaluator identity changed')
    training = json.loads((Path(TRAIN_ROOT) / 'training_result.json').read_text())
    if not training['complete'] or training['round_effective_updates'] != 1000:
        raise ValueError('paired 1000-update round is incomplete; needs separate diagnostic review')
    output = Path(OUTPUT)
    if output.exists():
        raise ValueError('benchmark already exists; inspect existing records, never silently repeat')
    output.mkdir(parents=True)
    manifest = json.loads(Path('/bench/manifest.json').read_text())
    from beatfcn.train.evolution import digest
    if digest({k: v for k, v in manifest.items() if k != 'manifest_sha256'}) != manifest['manifest_sha256']:
        raise ValueError('frozen development benchmark identity differs')
    paths = {'3500': {'path': CHECKPOINTS['3500'][0], 'sha256': CHECKPOINTS['3500'][1]}, **training['checkpoints']}
    models, configs = {}, {}
    for label, identity in paths.items():
        payload = Path(identity['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != identity['sha256']:
            raise ValueError('checkpoint fingerprint differs')
        import io
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        model = P1Model(P1Config(**ck['config'])).cuda().eval().requires_grad_(False)
        model.load_state_dict(ck['model'])
        models[label], configs[label] = model, ck['data']
        del ck, payload
    if len({json.dumps(x, sort_keys=True) for x in configs.values()}) != 1:
        raise ValueError('checkpoint audio configuration differs')
    corpus = Corpus(**configs['3500'])
    corpus.train_rows(manifest['rows'])
    whole_control_panel = {r['chart_id'] for mode in ('mania_4k', 'phira')
                           for r in [r for r in manifest['rows'] if r['mode'] == mode][:4]}
    expected = [{'id': key(label, row['chart_id'], kind), 'checkpoint': label, 'chart_id': row['chart_id'],
                 'song': row['audio_sha256'], 'mode': row['mode'], 'request_kind': kind}
                for row in manifest['rows'] for kind in
                (*manifest['request_kinds'], 'density_low', 'density_mid', 'density_high',
                 *(('density_song_low', 'density_song_mid', 'density_song_high') if row['chart_id'] in whole_control_panel else ()))
                for label in models]
    protocol = {'schema': 'beatfcn.condition_benchmark.v1', 'source_sha256': expected_source_sha256,
                'manifest_sha256': manifest['manifest_sha256'],
                'evaluator_sha256': evaluator_sha256, 'metric_source_sha256': metric_source_sha256(),
                'checkpoints': paths, 'expected': expected, 'input_policy': 'authorized_rhythm_prefix_v1',
                'density_scope': 'one complete 8s editable window; other numeric controls absent; fixed bands independent of truth',
                'whole_song_control_panel': sorted(whole_control_panel),
                'whole_song_controls': 'same fixed bands over the full editable song; own generated history; no oracle density',
                'baseline_scope': 'full-song requests without answer-derived numeric controls',
                'dataset_scope': '40-song development panel, not sealed final test', 'training_updates': 0}
    save(output / 'protocol.json', protocol)
    data.commit()
    records = []
    for row in manifest['rows']:
        if time.perf_counter() - started > 4230:
            break
        try:
            chart = corpus.chart(row['chart_id'])
            duration = round(row['audio_sec'] * F.SEC)
            song = corpus.song(row['audio_sha256'], in_memory=True)
            audio = audio_features(song.mel10, duration)
            samples = []
            for kind in manifest['request_kinds']:
                sample = build_request(chart, kind=kind, seed=row['benchmark_seed'], audio_id=row['audio_sha256'],
                                       audio_duration_us=duration, empty_overlay=None, control_dropout=False)
                sample = replace(sample, request={**sample.request, 'controls': {'allowed_types': list(MODE_TYPES[row['mode']])}})
                samples.append((kind, sample))
            c0 = min((duration // 2 // F.CORE_US) * F.CORE_US, max(0, (duration // F.CORE_US - 1) * F.CORE_US))
            window = complete_window(samples[0][1], chart, c0, duration, row['benchmark_seed'])
            samples += list(zip(('density_low', 'density_mid', 'density_high'), control_views(window, FIELD)))
            if row['chart_id'] in whole_control_panel:
                samples += list(zip(('density_song_low', 'density_song_mid', 'density_song_high'), control_views(samples[0][1], FIELD)))
        except (ValueError, F.SkipExample, AssertionError) as exc:
            # Keep the preregistered denominator when a source/request cannot be prepared.
            for task in expected:
                if task['chart_id'] == row['chart_id']:
                    failed = {**task, 'complete': False, 'legality': [],
                              'failure': {'stage': 'preparation', 'reason': f'{type(exc).__name__}: {exc}'}}
                    save(output / 'records' / (task['id'] + '.json'), failed)
                    records.append(failed)
            data.commit()
            print(json.dumps({'event': 'condition_benchmark_preparation_failed', 'chart': row['chart_id'],
                              'reason': f'{type(exc).__name__}: {exc}'}), flush=True)
            continue
        for kind, sample in samples:
            truth = [e for g in sample.model_view.target_groups for e in g.events]
            bases = timing_baselines(audio, sample, truth, row['mode'])
            for label, model in models.items():
                if time.perf_counter() - started > 4230:
                    break
                begin = time.perf_counter()
                record = {'id': key(label, row['chart_id'], kind), 'checkpoint': label, 'chart_id': row['chart_id'],
                          'song': row['audio_sha256'], 'mode': row['mode'], 'request_kind': kind,
                          'seed': row['benchmark_seed'],
                          'stratum': row['benchmark_stratum'], 'complete': False, 'legality': [],
                          'request': sample.request, 'truth_events': [e.to_json() for e in truth],
                          'audio_only_baselines': bases}
                try:
                    if not sample.ok:
                        raise ValueError('invalid quantized model view')
                    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                        rollout = generate_song(model, sample, song, seed=row['benchmark_seed'], device='cuda', greedy=True,
                                                deadline=min(started + 4300, begin + 75), stage_a_arm='legacy')
                    record.update({k: v for k, v in rollout.items() if k != 'events'})
                    pred = rollout['events']
                    record['generated_events'] = [e.to_json() for e in pred]
                    record['scores'] = score_events(pred, truth, row['mode'])
                    record['quality'] = quality_metrics(pred, truth, sample.request['regions'], row['mode'], song.summary)
                    record['benchmark'] = capability_metrics(pred, truth, sample.request, audio)
                    record['editing'] = {f'{ms}ms': edit_utility(pred, truth, row['mode'], tolerance_us=ms * 1000)
                                         for ms in (20, 50, 100)}
                    record['subsets'] = {subset: edit_utility([e for e in pred if (e.type == 'drag') == flag],
                                                             [e for e in truth if (e.type == 'drag') == flag], row['mode'])
                                         for subset, flag in (('drag', True), ('non_drag', False))}
                    if kind.startswith('density_'):
                        obs = [Event.from_json(e) for e in sample.request['observed_events']]
                        runs = editable_union(sample.request['regions'])
                        record.update(requested=sample.request['controls'][FIELD],
                                      achieved=raw_controls(row['mode'], obs + pred, runs)[FIELD],
                                      minimum_from_observed=raw_controls(row['mode'], obs, runs)[FIELD])
                except (ValueError, F.SkipExample, AssertionError) as exc:
                    record['complete'] = False
                    record['failure'] = {'reason': f'{type(exc).__name__}: {exc}'}
                record['wall_s'] = time.perf_counter() - begin
                save(output / 'records' / (record['id'] + '.json'), record)
                records.append(record)
                data.commit()
                print(json.dumps({'event': 'condition_benchmark', 'records': len(records), 'expected': len(expected),
                                  'checkpoint': label, 'kind': kind, 'complete': record['complete'], 'wall_s': record['wall_s']}), flush=True)
        del song
    present = {r['id'] for r in records}
    for task in expected:
        if task['id'] not in present:
            missing = {**task, 'complete': False, 'legality': [], 'failure': {'reason': 'not_executed_before_budgeted_deadline'}}
            save(output / 'records' / (task['id'] + '.json'), missing)
            records.append(missing)
    coverage = {label: {'requested': sum(r['checkpoint'] == label for r in expected),
                        'complete_legal': sum(r['checkpoint'] == label and r['complete'] and not r['legality'] for r in records)}
                for label in models}
    result = {'complete': all(v['requested'] == v['complete_legal'] for v in coverage.values()),
              'records': len(records), 'coverage': coverage, 'wall_s': time.perf_counter() - started,
              'remote_path': OUTPUT, 'checkpoints': paths,
              'scorecards': {label: scorecard([r for r in records if r['checkpoint'] == label and not r['request_kind'].startswith('density_')])
                            for label in models}, 'training_updates': 0}
    save(output / 'result.json', result)
    data.commit()
    return result


@app.local_entrypoint()
def run():
    from beatfcn.train.campaign_budget import reserve, update_stage
    ledger = REPO / 'reports/rhythm_v2_budget.json'
    source = source_digest(REPO / 'beatfcn')
    evaluator = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    reserve(ledger, 'paired_evaluation', RESOURCES, json.loads((REPO / 'reports/rhythm_v2_rates.json').read_text()),
            json.loads((REPO / 'reports/rhythm_v2_connection_status.json').read_text()),
            app_id=app.app_id, source_sha256=source)
    try:
        call = evaluate.spawn(source, evaluator)
        update_stage(ledger, 'paired_evaluation', status='running', function_call_id=call.object_id, evaluator_sha256=evaluator)
        result = call.get()
        save(REPO / 'reports/condition_v1_evaluation.json', result)
        update_stage(ledger, 'paired_evaluation', status='complete' if result['complete'] else 'incomplete', result=result, ended_at=time.time())
    except BaseException as exc:
        update_stage(ledger, 'paired_evaluation', status='observation_error', error=f'{type(exc).__name__}: {exc}')
        raise
