"""Bounded checkpoint diagnostics. This app registers no optimizer or training job."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import modal

from beatfcn.cloud.train_app import image as base_image, REPO

app = modal.App('beatfcn-rhythm-condition-diagnostic')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = base_image.add_local_file(str(REPO / 'bench/capability-v1/manifest.json'), '/bench/manifest.json')
RESOURCES = {'gpu': 'L40S', 'cpu': 4, 'memory_gib': 16, 'timeout_s': 600,
             'startup_allowance_s': 120, 'retries': 0, 'max_containers': 1}
CHECKPOINTS = {
    '3500': ('/data/runs/s1-v21/best.pt', '64a7797142fee342491ea269e4a105261f7399be32cfc19650a97395966ed1b4'),
    '4500': ('/data/runs/evo-r01-tape/last.pt', '6aa6f211911eed22002e7d22962c4df30b6050dc7e0a8edeef05114d05589a99'),
}


def source_digest(root):
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob('*.py')):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


@app.function(image=image, gpu='L40S', cpu=(4, 4), memory=(16384, 16384), volumes={'/data': data},
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def diagnose(stage: str, expected_source_sha256: str):
    import numpy as np
    import torch
    import beatfcn
    from beatfcn.events.canonical import Event, MODE_TYPES
    from beatfcn.events.controls import intersect, raw_controls
    from beatfcn.events.requests import build_request
    from beatfcn.model import features as F
    from beatfcn.model.batch import collate
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.model.history_probe import history_probe
    from beatfcn.model.condition_probe import control_views, condition_inventory
    from beatfcn.model.sample import generate_core, core_context
    from beatfcn.model.evaluation import groups_to_events, legality_issues, score_events
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    started = time.perf_counter()
    torch.set_num_threads(4)
    actual_source = source_digest(Path(beatfcn.__file__).parent)
    if actual_source != expected_source_sha256:
        raise ValueError('uploaded source differs from reserved source')
    output = Path('/data/evolution/capability-evolution-20261024/rhythm-diagnostics') / stage
    if output.exists():
        raise ValueError('diagnostic output already exists; never repeat a charged stage automatically')
    output.mkdir(parents=True)
    manifest = json.loads(Path('/bench/manifest.json').read_text())
    rows = [r for mode in ('mania_4k', 'phira') for r in [r for r in manifest['rows'] if r['mode'] == mode][:8]]
    result = {'stage': stage, 'source_sha256': actual_source, 'training_updates': 0, 'checkpoints': {},
              'inventory': condition_inventory(), 'history': [], 'controls': [], 'failures': [],
              'panel': [r['chart_id'] for r in rows], 'expected_control_records': len(rows) * 2 * 3,
              'expected_history_records': 8 * 2 * 2,
              'scope': 'Core diagnostic with human preceding context; not whole-song quality acceptance.',
              'control_field': 'density_notes_per_second', 'other_numeric_conditions': 'not_yet_measured'}
    save(output / 'protocol.json', result)
    data.commit()
    for label, (path, sha) in CHECKPOINTS.items():
        payload = Path(path).read_bytes()
        if hashlib.sha256(payload).hexdigest() != sha:
            raise ValueError('checkpoint fingerprint differs')
        import io
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        del payload
        if ck['step'] != int(label):
            raise ValueError('checkpoint step differs')
        model = P1Model(P1Config(**ck['config'])).cuda().eval().requires_grad_(False)
        model.load_state_dict(ck['model'])
        corpus = Corpus(**ck['data'])
        corpus.train_rows(rows)
        result['checkpoints'][label] = {'path': path, 'sha256': sha, 'step': ck['step'], 'config': ck['config']}
        del ck
        for row in rows:
            if time.perf_counter() - started > 500:
                break
            try:
                chart = corpus.chart(row['chart_id'])
                duration = round(row['audio_sec'] * F.SEC)
                sample = build_request(chart, kind='middle_holes', seed=row['benchmark_seed'],
                                       audio_id=row['audio_sha256'], audio_duration_us=duration,
                                       empty_overlay=None, control_dropout=False)
                if not sample.ok:
                    raise ValueError('quarantined model view')
                sample = replace(sample, request={**sample.request, 'controls': {'allowed_types': list(MODE_TYPES[row['mode']])}})
                c0 = F.choose_core(sample, np.random.default_rng(row['benchmark_seed']), duration)
                ex = F.build_core(sample, chart.events, c0, duration, stage_a_arm='legacy')
                if not ex.groups:
                    raise ValueError('diagnostic core has no targets')
                song = corpus.song(row['audio_sha256'], in_memory=True)
                identity = {'checkpoint': label, 'chart_id': row['chart_id'], 'song': row['audio_sha256'],
                            'mode': row['mode'], 'c0_us': c0, 'seed': row['benchmark_seed']}
                if row['mode'] == 'mania_4k':
                    bt = to_device(collate([ex], [song], model.c.taps, model.c.top_tap), torch.device('cuda'))
                    for fraction in (.33, .67):
                        index = min(len(ex.groups) - 1, max(0, int(len(ex.groups) * fraction)))
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            probe = history_probe(model, bt, index)
                        result['history'].append({**identity, **probe})
                    del bt
                observed = [Event.from_json(e) for e in sample.request['observed_events']]
                scope = intersect([(r['start_us'], r['end_us']) for r in sample.request['regions'] if r['state'] == 'editable'],
                                  [(c0, min(c0 + F.CORE_US, duration))])
                truth = [e for g in sample.model_view.target_groups for e in g.events if c0 <= e.onset_us < c0 + F.CORE_US]
                for view in control_views(sample, 'density_notes_per_second'):
                    begin = time.perf_counter()
                    core = F.build_core(view, chart.events, c0, duration, stage_a_arm='legacy')
                    holds, runs, heads = core_context(view, core, F.visible_line_index(observed, view.evidence))
                    diagnostics = {}
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        groups = generate_core(model, core, song, holds, runs, heads, device='cuda', greedy=True,
                                               seed=row['benchmark_seed'], diagnostics=diagnostics,
                                               deadline=min(started + 540, begin + 20))
                    events = groups_to_events(groups, c0, row['mode'], placement=False)
                    issues = legality_issues(view.request, events, placement=False)
                    complete = not diagnostics.get('deadline_reached') and not diagnostics.get('group_limit_reached')
                    record = {**identity, 'field': 'density_notes_per_second', 'core': row['chart_id'],
                              'requested': view.request['controls']['density_notes_per_second'],
                              'achieved': raw_controls(row['mode'], observed + events, scope)['density_notes_per_second'],
                              'complete': complete, 'legal': not issues, 'legality': issues,
                              'generated_events': [e.to_json() for e in events], 'scores': score_events(events, truth, row['mode']),
                              'wall_s': time.perf_counter() - begin, 'diagnostics': diagnostics}
                    if not complete:
                        record['achieved'] = None  # a timeout is not a sparse successful response
                    result['controls'].append(record)
                    if time.perf_counter() - started > 500:
                        break
                del song
            except (ValueError, F.SkipExample) as exc:
                result['failures'].append({'checkpoint': label, 'chart_id': row['chart_id'], 'reason': f'{type(exc).__name__}: {exc}'})
            result['wall_s'] = time.perf_counter() - started
            save(output / 'result.json', result)
            data.commit()
            print(json.dumps({'checkpoint': label, 'chart': row['chart_id'], 'history_cases': len(result['history']),
                              'control_cases': len(result['controls']), 'failures': len(result['failures'])}), flush=True)
        del model, corpus
        torch.cuda.empty_cache()
    result.update(wall_s=time.perf_counter() - started,
                  complete=(len(result['history']) == result['expected_history_records'] and
                            len(result['controls']) == result['expected_control_records'] and
                            all(r['complete'] and r['legal'] for r in result['controls']) and not result['failures']))
    save(output / 'result.json', result)
    data.commit()
    return {'remote_path': str(output / 'result.json'), 'complete': result['complete'], 'wall_s': result['wall_s'],
            'history_cases': len(result['history']), 'control_cases': len(result['controls']), 'training_updates': 0}


@app.local_entrypoint()
def run(stage: str = 'history-density-v1'):
    from beatfcn.train.campaign_budget import reserve, update_stage
    ledger = REPO / 'reports/rhythm_v2_budget.json'
    connection = json.loads((REPO / 'reports/rhythm_v2_connection_status.json').read_text())
    rates = json.loads((REPO / 'reports/rhythm_v2_rates.json').read_text())
    sha = source_digest(REPO / 'beatfcn')
    reserve(ledger, stage, RESOURCES, rates, connection, app_id=app.app_id, source_sha256=sha)
    try:
        call = diagnose.spawn(stage, sha)
        update_stage(ledger, stage, status='running', function_call_id=call.object_id)
        result = call.get()
        save(REPO / f'reports/{stage}_dispatch.json', result)
        update_stage(ledger, stage, status='complete' if result['complete'] else 'incomplete', result=result, ended_at=time.time())
        print(json.dumps(result))
    except BaseException as exc:
        update_stage(ledger, stage, status='observation_error', error=f'{type(exc).__name__}: {exc}', observed_at=time.time())
        raise  # the handle may still be alive: reconcile it before any restart
