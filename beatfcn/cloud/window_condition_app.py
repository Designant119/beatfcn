"""R03: one label-coverage intervention, with prerequisite probes and a cost ledger."""
from collections import Counter
from dataclasses import replace
import gzip
import hashlib
import io
import json
from pathlib import Path
import time

import modal

from beatfcn.cloud import train_app as training
from beatfcn.cloud.condition_training_app import POLICY as PREVIOUS_POLICY, VARIANT, checkpoint
from beatfcn.cloud.rhythm_diagnostic_app import CHECKPOINTS, source_digest, save
from beatfcn.train.control_panel import REQUEST_BANDS

REPO = training.REPO
ROOT = '/data/evolution/capability-evolution-20261024/window-conditions-v1'
PLAN = REPO / 'bench/rhythm-v2/r03-window-conditions-v1.json'
POLICY = {**PREVIOUS_POLICY, 'collect_full_window_controls': True}
RUNS = ['r03-window-density', 'r03-window-all']
STAGES = {p: f'r03-window-{p}' for p in ('prepare', 'preflight', 'train', 'evaluate')}
RESOURCES = {
    'prepare': dict(cpu=8, memory_gib=32, gpu=None, timeout_s=900),
    'preflight': dict(cpu=4, memory_gib=16, gpu='L40S', timeout_s=900),
    'train': dict(cpu=12, memory_gib=48, gpu='L40S', timeout_s=3000),
    'evaluate': dict(cpu=4, memory_gib=16, gpu='L40S', timeout_s=1800),
}
app = modal.App('beatfcn-window-condition-r03')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (training.image.add_local_file(str(REPO / 'bench/capability-v1/manifest.json'), '/bench/manifest.json')
         .add_local_file(str(PLAN), '/bench/r03-plan.json'))


def contract(expected_source, expected_plan):
    import beatfcn
    if source_digest(Path(beatfcn.__file__).parent) != expected_source:
        raise ValueError('uploaded source differs from reservation')
    payload = Path('/bench/r03-plan.json').read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_plan:
        raise ValueError('plan differs from reservation')
    plan = json.loads(payload)
    if plan['request_bands'] != REQUEST_BANDS or plan['training_policy'] != POLICY:
        raise ValueError('plan does not describe the implemented requests and training policy')
    return plan


def prepare_panel(corpus, rows):
    """Select before seeing predictions; no training audio and no boundary truncation."""
    from beatfcn.model import features as F
    from beatfcn.train.condition_pairs import blank_context_excerpt
    corpus.train_rows(rows)
    candidates = {'mania_4k': [], 'phira': []}
    rejected = Counter()
    for row in rows:
        duration = round(row['audio_sec'] * F.SEC)
        chart = corpus.chart(row['chart_id'])
        raw = Path(f'/data/canon/{row["source"]}/{row["native_id"]}.json').read_bytes()
        if json.loads(raw)['audio_sha256'] != row['audio_sha256']:
            raise ValueError('panel canonical audio identity differs')
        for c0 in range(0, duration - F.CORE_US + 1, F.CORE_US):
            events = [e for e in chart.events if c0 <= e.onset_us < c0 + F.CORE_US]
            if not 12 <= len(events) <= 56:
                continue
            try:
                sample = blank_context_excerpt(chart, c0, duration, row['audio_sha256'], 20261025)
                ex = F.build_core(sample, chart.events, c0, duration, stage_a_arm='legacy')
                if F.check_targets(ex):
                    raise ValueError('illegal targets')
                candidates[row['mode']].append(dict(row=row, c0_us=c0, duration_us=duration,
                    notes=len(events), types=dict(Counter(e.type for e in events)),
                    canon_sha256=hashlib.sha256(raw).hexdigest()))
            except (ValueError, F.SkipExample) as exc:
                rejected[str(exc)] += 1
    selected = []
    for mode, options in candidates.items():
        used = set()
        for index, target_count in enumerate((18, 40, 24, 48)):
            required = (('hold',) if mode == 'mania_4k' else
                        (('flick',) if index == 0 else ('drag', 'hold') if index == 1 else
                         ('hold',) if index == 2 else ('flick', 'drag')))
            eligible = [r for r in options if r['row']['audio_sha256'] not in used
                        and all(r['types'].get(t, 0) for t in required)]
            if not eligible:
                raise ValueError(f'held-out panel lacks {mode} case {index} with {required}')
            item = min(eligible, key=lambda r: (abs(r['notes'] - target_count), r['row']['chart_id'], r['c0_us']))
            selected.append(dict(case_id=f'{mode}:{index}', case_index=index, **item))
            used.add(item['row']['audio_sha256'])
    return dict(schema='beatfcn.fixed_user_control_panel.v1', cases=selected,
                request_bands=REQUEST_BANDS, eligible_windows={k: len(v) for k, v in candidates.items()},
                rejected_windows=dict(rejected), conditions_from_hidden_target=False,
                scope='Eight blank-canvas 8s held-out excerpts; not whole-song or human usability acceptance.')


@app.function(image=image, volumes={'/data': data}, cpu=(8, 8), memory=(32768, 32768),
              timeout=900, retries=0, max_containers=1, scaledown_window=2)
def prepare(expected_source: str, expected_plan: str):
    import torch
    from beatfcn.model.batch import unpack_batch
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import CoreSampler
    from beatfcn.train.tape import policy_digest
    torch.set_num_threads(1)
    contract(expected_source, expected_plan)
    started = time.perf_counter()
    root = Path(ROOT)
    if root.exists():
        raise ValueError('preparation already exists; inspect the original handle')
    (root / 'tape').mkdir(parents=True)
    ck = checkpoint()
    ca, cfg = ck['data'], ck['config']
    rows, val_rows, info = training._select_rows(ca['taps_key'], tuple(ca['layers']), ca['pca'], ca['whiten'],
                                                '', 0, '', ck['manifest_name'])
    if info['rows_sha256'] != ck['rows_sha256']:
        raise ValueError('training row selection changed')
    bench_rows = json.loads(Path('/bench/manifest.json').read_text())['rows']
    heldout = {r['audio_sha256'] for r in bench_rows + val_rows}
    if heldout & {r['audio_sha256'] for r in rows}:
        raise ValueError('held-out audio occurs in training')
    corpus = Corpus(**ca)
    panel = prepare_panel(corpus, bench_rows)
    save(root / 'panel.json', panel)
    sampler = CoreSampler(corpus.train_rows(val_rows), corpus.chart, corpus.song_info, 7,
                          salient=corpus.salient_us, stage_a_arm='legacy')
    val = training._fixed_val(sampler, 40, 8)
    if len(val) != 40:
        raise ValueError('validation preparation incomplete')
    prep_key = 'r03-window-conditions-v1'
    prep_path = Path('/data/prep') / f'{prep_key}.pt'
    if prep_path.exists():
        raise ValueError('validation preparation already exists')
    torch.save(dict(val=val, audit=None, info=info, policy=POLICY), prep_path)
    del ck, corpus, sampler, val
    loader = torch.utils.data.DataLoader(training._stream(rows, ca, 20261024, 8, tuple(cfg['taps']),
        cfg['top_tap'], pool=1, training_policy=POLICY), batch_size=None,
        num_workers=8, prefetch_factor=4, in_order=False)
    iterator = iter(loader)
    manifest = dict(schema='beatfcn.metadata_tape.v1', complete=False, rows_sha256=info['rows_sha256'],
        policy_sha256=policy_digest(POLICY), policy=POLICY, micro_batch=8, seed=20261024,
        checkpoint=CHECKPOINTS['3500'][0], source_sha256=expected_source, plan_sha256=expected_plan,
        parts=[], batches=0, prep_key=prep_key, panel_sha256=hashlib.sha256((root / 'panel.json').read_bytes()).hexdigest())
    exposure, bands, songs, charts = Counter(), Counter(), set(), set()
    rows_by_id = {r['chart_id']: r for r in rows}
    probe_saved = False
    with (root / 'tape/provenance.jsonl').open('w') as provenance:
        while manifest['batches'] < 4000:
            if time.perf_counter() - started > 820:
                raise TimeoutError('CPU preparation deadline')
            chunk = [next(iterator) for _ in range(min(32, 4000 - manifest['batches']))]
            for offset, (packed, _) in enumerate(chunk):
                records = packed['_provenance']
                provenance.write(json.dumps(records) + '\n')
                for r in records:
                    exposure[f"{r['mode']}:{r['kind']}:cores"] += 1
                    exposure[f"{r['mode']}:{r['kind']}:notes"] += r['notes']
                    songs.add(r['song']); charts.add(r['chart_id'])
                    if r['kind'] == 'density_window':
                        from beatfcn.model.features import CONTROL_FIELDS
                        for field, value in zip(CONTROL_FIELDS, r['full_window_controls']):
                            bands[f"{r['mode']}:{field}:{value}"] += 1
                if packed['_density_cf'] and not probe_saved and {r['mode'] for r in records} == {'mania_4k', 'phira'}:
                    bt = unpack_batch(packed)
                    valid = torch.arange(bt['note_type'].shape[-1])[None, None, :] < bt['grp_n'][:, :, None]
                    counts = torch.bincount(bt['note_type'][valid], minlength=4)
                    if all(counts >= 2) and 64 <= int(counts.sum()) <= 512 and 2 <= int(bt['hold_valid'].sum()) <= 64:
                        torch.save(packed, root / 'probe-batch.pt')
                        manifest['probe_batch_index'] = manifest['batches'] + offset
                        manifest['probe_target_type_counts'] = counts.tolist()
                        manifest['probe_rows'] = [rows_by_id[r['chart_id']] for r in records]
                        manifest['probe_canon_sha256'] = [hashlib.sha256(Path(
                            f'/data/canon/{r["source"]}/{r["native_id"]}.json').read_bytes()).hexdigest()
                            for r in manifest['probe_rows']]
                        probe_saved = True
            buffer = io.BytesIO(); torch.save(chunk, buffer)
            raw = buffer.getvalue(); encoded = gzip.compress(raw, compresslevel=1, mtime=0)
            path = root / 'tape' / f'part-{len(manifest["parts"]):04d}.pt.gz'
            path.write_bytes(encoded)
            manifest['parts'].append(dict(file=path.name, bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest(),
                raw_bytes=len(raw), raw_sha256=hashlib.sha256(raw).hexdigest(), codec='gzip', batches=len(chunk)))
            manifest['batches'] += len(chunk)
            if manifest['batches'] % 128 == 0:
                save(root / 'tape/progress.json', dict(manifest, wall_s=time.perf_counter() - started))
                print(json.dumps(dict(event='r03_metadata', batches=manifest['batches'], wall_s=time.perf_counter() - started)), flush=True)
    del iterator, loader
    window_count = sum(n for key, n in exposure.items() if key.endswith(':density_window:cores'))
    if not .18 <= window_count / 32000 <= .32 or len(songs) < 100 or not probe_saved:
        raise ValueError('unexpected exposure or no representative mixed-mode probe')
    manifest.update(complete=True, exposure=dict(exposure), full_window_bands=dict(bands),
        window_fraction=window_count / 32000, unique_songs=len(songs), unique_charts=len(charts),
        probe_batch_sha256=hashlib.sha256((root / 'probe-batch.pt').read_bytes()).hexdigest(),
        wall_s=time.perf_counter() - started)
    save(root / 'tape/manifest.json', manifest)
    data.commit()
    return manifest


def benchmark(model, corpus, panel, output, deadline):
    import torch
    from beatfcn.events import controls as C
    from beatfcn.model import features as F
    from beatfcn.model.evaluation import groups_to_events, legality_issues, score_events
    from beatfcn.model.sample import generate_core
    from beatfcn.model.reference_clock import benchmark_evidence
    from beatfcn.train.condition_pairs import blank_context_excerpt
    from beatfcn.train.control_panel import requested_view, response_gate, field_decision
    corpus.train_rows([c['row'] for c in panel['cases']])
    result = dict(complete=False, records=[], field_gates={}, field_decisions={},
                  request_bands=REQUEST_BANDS, scope=panel['scope'], conditions_from_hidden_target=False)
    model.eval().requires_grad_(False)
    for case in panel['cases']:
        row = case['row']; mode = row['mode']; c0 = case['c0_us']
        raw = Path(f'/data/canon/{row["source"]}/{row["native_id"]}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != case['canon_sha256']:
            raise ValueError('panel canonical identity differs')
        chart = corpus.chart(row['chart_id'])
        sample = blank_context_excerpt(chart, c0, case['duration_us'], row['audio_sha256'], 20261025)
        ex = F.build_core(sample, chart.events, c0, case['duration_us'], stage_a_arm='legacy')
        song = corpus.song(row['audio_sha256'], in_memory=True)
        truth = [e for group in sample.model_view.target_groups for e in group.events]
        def generate(field=None, band=None):
            if time.perf_counter() > deadline - 20:
                raise TimeoutError('benchmark deadline reached; partial results retained')
            requested = {'allowed_types': sample.request['controls']['allowed_types']}
            if field is not None:
                requested[field] = band
            request = {**sample.request, 'controls': requested}
            diag = {}; started = time.perf_counter()
            with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                generated = generate_core(model, requested_view(ex, field, band), song, [],
                    [(c0, c0 + F.CORE_US)], [], device='cuda', greedy=True, seed=20261025,
                    diagnostics=diag, deadline=min(deadline - 10, started + 15))
            events = groups_to_events(generated, c0, mode, placement=False)
            issues = legality_issues(request, events, placement=False)
            record = dict(case_id=case['case_id'], chart_id=row['chart_id'], mode=mode, c0_us=c0,
                variant='single_condition' if field else 'unspecified_conditions', field=field,
                requested=band, achieved=C.raw_controls(mode, events, [(c0, c0 + F.CORE_US)]),
                complete=bool(diag.get('terminated_by_end')), legal=not issues, legality=issues,
                diagnostics=diag, scores=score_events(events, truth, mode),
                evidence=benchmark_evidence(events, truth, mode, row),
                predicted_types=dict(Counter(e.type for e in events)), truth_types=dict(Counter(e.type for e in truth)),
                generated_events=[e.to_json() for e in events], truth_events=[e.to_json() for e in truth],
                wall_s=time.perf_counter() - started)
            result['records'].append(record)
            save(output, result)
            print(json.dumps(dict(event='r03_generate', case=case['case_id'], field=field, requested=band,
                                  achieved=record['achieved'].get(field), types=record['predicted_types'],
                                  complete=record['complete'], wall_s=record['wall_s'])), flush=True)
            return record
        generate()
        for field in C.MODE_FIELDS[mode]:
            key = f'{mode}:{field}'
            gates = result['field_gates'].setdefault(key, [])
            if field_decision(gates)['failed']:
                continue
            records = [generate(field, band) for band in REQUEST_BANDS[field]]
            gates.append(dict(case_id=case['case_id'], **response_gate(records, field)))
            result['field_decisions'][key] = field_decision(gates)
            save(output, result)
        del song
    result['complete'] = all(d['failed'] or d['passed'] for d in result['field_decisions'].values())
    result['passed_fields'] = sum(d['passed'] for d in result['field_decisions'].values())
    result['tested_fields'] = len(result['field_decisions'])
    save(output, result)
    return result


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(4, 4), memory=(16384, 16384),
              timeout=900, retries=0, max_containers=1, scaledown_window=2)
def preflight(expected_source: str, expected_plan: str):
    from types import SimpleNamespace
    import torch
    from beatfcn.events import controls as C
    from beatfcn.model.batch import unpack_batch, attach_audio
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus
    from beatfcn.train.loop import to_device
    from beatfcn.train.probe import tensor_summary, metrics_json, capture_decisions, assess_memorisation
    from beatfcn.train.window_controls import window_control_view, encode_window_controls
    plan = contract(expected_source, expected_plan)
    torch.set_num_threads(4); started = time.perf_counter(); root = Path(ROOT)
    output = root / 'preflight.json'
    if output.exists():
        raise ValueError('preflight already exists; do not repeat')
    manifest = json.loads((root / 'tape/manifest.json').read_text())
    if not manifest['complete'] or manifest['source_sha256'] != expected_source or manifest['plan_sha256'] != expected_plan:
        raise ValueError('preparation identity differs')
    report = dict(complete=False, probe_passed=False, optimizer_updates=0, source_sha256=expected_source,
                  plan_sha256=expected_plan, history=[], gradients=[], checkpoint_saved=False)
    def record():
        report['function_wall_s'] = time.perf_counter() - started
        save(output, report); data.commit()
    record()
    try:
        ck = checkpoint(); config = P1Config(**ck['config']); corpus = Corpus(**ck['data'])
        model = P1Model(config).cuda().eval(); model.load_state_dict(ck['model']); del ck
        panel_payload = (root / 'panel.json').read_bytes()
        if hashlib.sha256(panel_payload).hexdigest() != manifest['panel_sha256']:
            raise ValueError('panel fingerprint differs')
        baseline = benchmark(model, corpus, json.loads(panel_payload), root / 'baseline.json', started + 650)
        if not baseline['complete'] or baseline['tested_fields'] != 13:
            raise ValueError('current-protocol baseline incomplete')
        report['baseline'] = dict(complete=True, records=len(baseline['records']),
                                  passed_fields=baseline['passed_fields'], path=str(root / 'baseline.json'))
        record()
        payload = (root / 'probe-batch.pt').read_bytes()
        if hashlib.sha256(payload).hexdigest() != manifest['probe_batch_sha256']:
            raise ValueError('probe batch identity differs')
        bt = unpack_batch(torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False))
        del payload
        corpus.train_rows(manifest['probe_rows'])
        for i, (row, fingerprint) in enumerate(zip(manifest['probe_rows'], manifest['probe_canon_sha256'])):
            raw = Path(f'/data/canon/{row["source"]}/{row["native_id"]}.json').read_bytes()
            if hashlib.sha256(raw).hexdigest() != fingerprint or json.loads(raw)['audio_sha256'] != row['audio_sha256']:
                raise ValueError('probe canonical or audio identity differs')
            p = bt['_provenance'][i]
            if p['song'] != bt['_audio']['keys'][i] or p['chart_id'] != row['chart_id']:
                raise ValueError('probe metadata source alignment differs')
            values = C.raw_controls(row['mode'], corpus.chart(row['chart_id']).events, [p['density_scope_us']])
            if encode_window_controls(row['mode'], values) != bt['full_window_controls'][i].tolist():
                raise ValueError('window labels do not match the full editable scope')
        report['probe_provenance'] = bt['_provenance']
        songs = {}
        for sha in set(bt['_audio']['keys']):
            taps, summary, _ = corpus.load_numpy(sha, mel=False)
            songs[sha] = SimpleNamespace(taps={k: torch.from_numpy(v) for k, v in taps.items()},
                                        summary=torch.from_numpy(summary), mel10=None)
        bt = attach_audio(bt, songs.__getitem__, device='cpu', scales=corpus.scale)
        bt = window_control_view(to_device(bt, torch.device('cuda')), 'all_applicable')
        del songs, taps, summary
        stats = {k: tensor_summary(v) for k, v in bt.items() if torch.is_tensor(v)}
        report['data_health'] = dict(tensors=stats, passed=all(
            v['nan'] == v['posinf'] == v['neginf'] == 0 for v in stats.values()), label_source_alignment=True)
        if not report['data_health']['passed']:
            raise ValueError('nonfinite real input')
        model.requires_grad_(True).eval(); torch.manual_seed(20261025)
        def measure(step):
            with torch.no_grad(), capture_decisions() as decisions, torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = model(dict(bt))
            return dict(step=step, **metrics_json(loss, metrics), decisions=decisions)
        report['history'].append(measure(0)); record()
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0003, weight_decay=0.)
        for step in range(1, 101):
            if time.perf_counter() > started + 850:
                raise TimeoutError('preflight reserve reached')
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                loss, metrics = model(dict(bt))
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            finite = bool(torch.isfinite(norm)) and bool(torch.isfinite(loss)) and int(metrics['_illegal']) == 0
            report['gradients'].append(dict(step=step, finite=finite, norm=float(norm)))
            if not finite:
                raise ValueError('nonfinite gradient or illegal target')
            optimizer.step(); report['optimizer_updates'] = step
            if step in (1, 10, 25, 50, 75, 100):
                report['history'].append(measure(step)); record()
                print(json.dumps(dict(event='r03_memorisation', step=step, loss=report['history'][-1]['loss'])), flush=True)
        report['supervision_gate'] = assess_memorisation(report['history'], report['gradients'], plan['probe_gate'], 100)
        report['probe_passed'] = report['data_health']['passed'] and report['supervision_gate']['passed']
        report.update(complete=True, peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30,
                      peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30)
        record()
        return report
    except BaseException as exc:
        report['error'] = f'{type(exc).__name__}: {exc}'; record()
        raise


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(12, 12), memory=(49152, 49152),
              timeout=3000, retries=0, max_containers=1, scaledown_window=2)
def train(expected_source: str, expected_plan: str):
    import torch
    torch.set_num_threads(4)
    contract(expected_source, expected_plan)
    root = Path(ROOT)
    manifest = json.loads((root / 'tape/manifest.json').read_text())
    probe = json.loads((root / 'preflight.json').read_text())
    if not all(report['source_sha256'] == expected_source and report['plan_sha256'] == expected_plan
               and report['complete'] for report in (manifest, probe)) or not probe['probe_passed']:
        raise ValueError('same-source preparation and real-input preflight must pass before training')
    if any((Path('/data/runs') / run).exists() for run in RUNS):
        raise ValueError('run exists; inspect original handles, do not repeat or silently resume')
    result = training.train.local(
        run_id=RUNS[0], also=[dict(run_id=RUNS[1], variant=VARIANT, window_control_policy='all_applicable')],
        steps=8000, stop_at=4500, round_updates=1000, initialize_from=CHECKPOINTS['3500'][0],
        source_sha256=expected_source, micro_batch=8, accumulation=4, workers=8,
        val_cores=40, val_songs=8, lr=.0002, warmup=500, validate_every=250,
        taps_key='v4_i8_L9-16-24', taps=(9, 16, 24), pca='i8_9-16-24', tap_dim=1024,
        pool_songs=40, pool_per_worker=1, prefetch=16, loaders=4, graph=False,
        min_util=70., check_after_s=480., max_data_wait=.8, check_data_after_s=480., max_wall_s=2850.,
        manifest_name='charts.offset-v2-3d99ff6bd5e7.jsonl', variant=VARIANT,
        training_policy=POLICY, prep=manifest['prep_key'], review_start_step=3500,
        data_seed_step=3500, tape_start_step=3500, metadata_tape=ROOT + '/tape',
        trace_after_s=(), trace_updates=2, trace_detailed=False)
    result['checkpoints'] = {}
    for run in RUNS:
        path = Path('/data/runs') / run / 'last.pt'
        if path.exists():
            result['checkpoints'][run] = dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    result.update(source_sha256=expected_source, plan_sha256=expected_plan,
                  complete=result['round_effective_updates'] == 1000 and not result['aborted'])
    save(root / 'training.json', result); data.commit()
    return result


@app.function(image=image, volumes={'/data': data}, gpu='L40S', cpu=(4, 4), memory=(16384, 16384),
              timeout=1800, retries=0, max_containers=1, scaledown_window=2)
def evaluate(expected_source: str, expected_plan: str):
    import torch
    from beatfcn.model.p1 import P1Config, P1Model
    from beatfcn.train.data import Corpus
    contract(expected_source, expected_plan)
    torch.set_num_threads(4); started = time.perf_counter(); root = Path(ROOT)
    if (root / 'evaluation.json').exists():
        raise ValueError('evaluation already exists; inspect partial records')
    training_result = json.loads((root / 'training.json').read_text())
    if training_result['source_sha256'] != expected_source or not training_result['complete']:
        raise ValueError('paired round must finish before its registered comparison')
    manifest = json.loads((root / 'tape/manifest.json').read_text())
    panel_payload = (root / 'panel.json').read_bytes()
    if hashlib.sha256(panel_payload).hexdigest() != manifest['panel_sha256']:
        raise ValueError('panel fingerprint differs')
    result = dict(complete=False, source_sha256=expected_source, plan_sha256=expected_plan, runs={})
    for run in RUNS:
        entry = training_result['checkpoints'][run]
        payload = Path(entry['path']).read_bytes()
        if hashlib.sha256(payload).hexdigest() != entry['sha256']:
            raise ValueError('trained checkpoint fingerprint differs')
        ck = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
        model = P1Model(P1Config(**ck['config'])).cuda().eval(); model.load_state_dict(ck['model'])
        corpus = Corpus(**ck['data']); del payload, ck
        row = benchmark(model, corpus, json.loads(panel_payload), root / f'{run}.benchmark.json', started + 1700)
        result['runs'][run] = dict(path=str(root / f'{run}.benchmark.json'), complete=row['complete'],
                                  records=len(row['records']), passed_fields=row['passed_fields'])
        save(root / 'evaluation.json', result); data.commit()
        del model, corpus
    result.update(complete=all(r['complete'] for r in result['runs'].values()), function_wall_s=time.perf_counter() - started)
    save(root / 'evaluation.json', result); data.commit()
    return result


@app.local_entrypoint()
def main(phase: str = 'prepare'):
    raise RuntimeError(
        'R03 failed its frozen quality gates and its shared training changes were rolled back. '
        'Historical source and owned patch are in inputs/code_baselines/r03-all-conditions-rejected; '
        'do not dispatch this rejected experiment as the current training route.')
