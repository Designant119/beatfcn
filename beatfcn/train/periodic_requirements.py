"""Optimizer-boundary requirement evaluation, charged to the existing run clock."""
import json
from pathlib import Path
import time

from beatfcn.train.checkpoint_resume import file_sha
from beatfcn.train.requirement_screen import run_requirement_screen, write_result


def requirement_due(completed_updates, settings):
    interval=settings.get('every_optimizer_updates',250)
    if type(interval)is not int or interval<1 or type(completed_updates)is not int or completed_updates<0:
        raise ValueError('Requirement cadence must use positive integer optimizer updates')
    return completed_updates>0 and completed_updates%interval==0


def evaluate_training_checkpoint(model,panel,plan,report,*,root,device,remaining_wall_s):
    started=time.monotonic()
    settings=plan['requirement_eval'];root=Path(root);accum=int(plan['accum_steps'])
    if report['updates']%accum or report['updates']//accum!=report['optimizer_updates']:
        raise ValueError('Evaluation requires a complete actual optimizer boundary')
    checkpoint_path=root/report['checkpoint']
    expected_step=f'step-{report["updates"]:04d}.pt'
    if checkpoint_path.name!=expected_step or not report.get('checkpoint_sha256'):
        raise ValueError('Save the current complete checkpoint before evaluating')
    protocol_path=root/settings['protocol']
    if file_sha(protocol_path)!=settings['protocol_sha256']:
        raise ValueError('Frozen requirement protocol changed')
    protocol=json.loads(protocol_path.read_text())
    if protocol['tape_sha256']!=plan['inputs_sha256'][plan['tape']]:
        raise ValueError('Requirement protocol belongs to different data')
    protocol['_file_sha256']=settings['protocol_sha256']
    budget=min(float(settings.get('max_wall_s',150)),max(0.,remaining_wall_s-30.))
    if budget<float(settings.get('minimum_wall_s',90)):
        result=dict(skipped=True,reason='insufficient remaining original run time',
            optimizer_updates=report['optimizer_updates'],remaining_wall_s=remaining_wall_s)
        print(json.dumps(dict(phase='requirement_eval_skipped',**result)),flush=True)
        return result
    if not checkpoint_path.is_file() or file_sha(checkpoint_path)!=report['checkpoint_sha256']:
        raise ValueError('Current checkpoint SHA is not verified')
    checkpoint=dict(path=str(checkpoint_path),sha256=report['checkpoint_sha256'],
        optimizer_updates=report['optimizer_updates'],exposure_step=report['updates'])
    directory=root/settings['history_dir'];directory.mkdir(parents=True,exist_ok=True)
    output=directory/f'opt-{report["optimizer_updates"]:06d}-{report["checkpoint_sha256"][:12]}-{report["plan_sha256"][:8]}.json'
    print(json.dumps(dict(phase='requirement_eval_start',**checkpoint,max_wall_s=budget)),flush=True)
    from beatfcn.train.compiled_training import preparation_deadline
    remaining_budget=budget-(time.monotonic()-started)
    with preparation_deadline(max(1.,remaining_budget),label='Periodic requirement evaluation'):
        result=run_requirement_screen(model,panel,protocol,checkpoint=checkpoint,
            plan_sha256=report['plan_sha256'],output=output,device=device,max_wall_s=max(1.,remaining_budget-5),
            max_tokens=int(settings.get('max_tokens',2048)),batch_size=int(settings.get('batch_size',8)),
            compiled=bool(plan.get('compile_loss')))
    write_result(root/settings['output'],result)
    metadata=dict(path=str(output),raw_sha256=file_sha(output),protocol_sha256=settings['protocol_sha256'],
        checkpoint=checkpoint,summary=result['summary'],all_cases_measured=result['all_cases_measured'])
    print(json.dumps(dict(phase='requirement_eval_done',**metadata)),flush=True)
    return metadata


# --------------------------------------------------------------------------
# Benchmark v2 periodic screen (design/BENCHMARK_GUIDE_v2.md)
# --------------------------------------------------------------------------
PERIODIC_PLAN_KEYS = """plan['benchmark_v2'] keys (paths relative to root):
  every_optimizer_updates  int, cadence for benchmark_due (default 250)
  max_wall_s               float, total bound for one screen (default 300)
  minimum_wall_s           float, skip (and say so) below this remaining budget (default 60)
  teacher_fraction         float, share of max_wall_s for the teacher-forced screen (default 0.35)
  teacher_protocol         path to audio-dependence-panel-v1 JSON (scripts/freeze_benchmark_v2.py)
  teacher_protocol_sha256  sha256 of that file
  generation_protocol      path to benchmark-v2-generation-panel JSON
  generation_protocol_sha256
  history_dir              directory for immutable opt-XXXXXX-<sha12>.json outputs
  latest_output            path atomically replaced with the newest output (dashboard input)
  readouts                 optional, default [{name: sample_t1, temperature: 1.0, top_p: null,
                           seeds: [0,1,2,3]}, {name: greedy}]
  token_budget_factor      optional, default 2.0 (x longest teacher sequence of the panel)
  max_batch_rows           optional, default 32 (rows per generate_batch call; memory only)
  teacher_batch_size       optional, default 8
  base_seed                optional, default 20261101 (per-case generator seeds derive from it)
  weights                  optional 'ema' | 'raw'; default 'ema' when plan['ema'] is set, else 'raw'
  hard_timeout_grace_s     optional, default 60; SIGALRM guard at max_wall_s + grace (main thread)
  charter_timing           optional 'with' | 'without' (default 'with'); only used when the plan has
                           charter_timing and config.beat_features: windows get their own chart's beat
                           features and the output is labelled charter_timing_input
Optional top-level plan keys read: inputs_sha256/tape (tape identity), compile_loss (forces eager).
The runner (scripts/run_local_song_baseline.py) calls periodic_benchmark_step at every
optimizer boundary, after a due checkpoint save; time is charged to the attempt clock
(report['benchmark_v2_s'] / report['evaluation_s']) and a failure is recorded, never raised."""


def benchmark_due(completed_updates, settings):
    return requirement_due(completed_updates, dict(every_optimizer_updates=settings.get('every_optimizer_updates', 250)))


def _weights_fingerprint(model):
    import hashlib
    import torch
    digest = hashlib.sha256()
    with torch.no_grad():
        for name, parameter in model.state_dict().items():
            value = parameter.detach().float()
            digest.update(f'{name}:{tuple(value.shape)}:{float(value.sum()):.9e}:{float(value.abs().sum()):.9e}'.encode())
    return digest.hexdigest()


def _load_protocol(root, settings, key):
    path = root/settings[key]
    if file_sha(path) != settings[key+'_sha256']:
        raise ValueError(f'Frozen {key} changed')
    return json.loads(path.read_text())


def run_periodic_screen(model, panel, plan, *, root, device, update, checkpoint=None, remaining_wall_s=None):
    """Teacher-forced audio-dependence screen + batched generation benchmark.

    Call at an optimizer boundary (``benchmark_due(update, plan['benchmark_v2'])``).
    Runtime is bounded by max_wall_s (and remaining_wall_s - 30 when given). The
    RNG state, train/eval flags and weights are restored; no optimizer state is
    touched. Output is atomic and immutable per (update, checkpoint SHA); it holds
    checkpoint identity, decoding settings and every raw output row.
    See PERIODIC_PLAN_KEYS for the plan contract.
    """
    from contextlib import nullcontext
    import torch
    from beatfcn.train.audio_dependence import run_teacher_screen
    from beatfcn.train import learning_evaluation as L
    started = time.monotonic()
    settings = plan['benchmark_v2']
    root = Path(root)
    if type(update) is not int or update < 0:
        raise ValueError('update must be the completed optimizer-update count')
    budget = float(settings.get('max_wall_s', 300.))
    if remaining_wall_s is not None:
        budget = min(budget, max(0., remaining_wall_s-30.))
    if budget < float(settings.get('minimum_wall_s', 60.)) or budget <= 0:
        result = dict(skipped=True, reason='insufficient remaining run time', optimizer_updates=update,
                      remaining_wall_s=remaining_wall_s)
        print(json.dumps(dict(phase='benchmark_v2_skipped', **result)), flush=True)
        return result
    teacher_protocol = _load_protocol(root, settings, 'teacher_protocol')
    generation_protocol = _load_protocol(root, settings, 'generation_protocol')
    if plan.get('inputs_sha256') and plan.get('tape') in plan['inputs_sha256']:
        for protocol in (teacher_protocol, generation_protocol):
            if protocol.get('tape_sha256') and protocol['tape_sha256'] != plan['inputs_sha256'][plan['tape']]:
                raise ValueError('Benchmark protocol belongs to different data')
    identity = dict(checkpoint or {})
    if identity.get('path') and not identity.get('sha256'):
        path = Path(identity['path'])
        identity['sha256'] = file_sha(path if path.is_absolute() else root/path)
    if not identity.get('sha256'):
        identity['weights_fingerprint'] = _weights_fingerprint(model)
    tag = (identity.get('sha256') or identity['weights_fingerprint'])[:12]
    output = root/settings['history_dir']/f'opt-{update:06d}-{tag}.json'
    if output.exists():
        raise FileExistsError('Benchmark identity already exists; never overwrite')
    readouts = [dict(r) for r in settings.get('readouts', L.DEFAULT_READOUTS)]
    from beatfcn.train.charter_timing import evaluation_table
    timing = evaluation_table(plan, model, root=root, choice=settings.get('charter_timing'))
    import beatfcn.train.audio_dependence as A
    import beatfcn.train.requirement_diagnostics as RD
    import beatfcn.model.full_event_generate as G
    code = {Path(p).name: file_sha(p) for p in (L.__file__, A.__file__, RD.__file__, G.__file__, __file__)}
    from dataclasses import asdict
    result = dict(schema='benchmark-v2-periodic', status='running', optimizer_updates=update, checkpoint=identity,
        protocol_sha256=settings['generation_protocol_sha256'],
        teacher_protocol_sha256=settings['teacher_protocol_sha256'],
        plan_sha256=plan.get('plan_sha256'), config=asdict(model.c), code_sha256=code,
        decoding=dict(readouts=readouts, primary=next((r['name'] for r in readouts if r.get('temperature') is not None),
            'greedy'), token_budget_factor=float(settings.get('token_budget_factor', 2.)),
            max_batch_rows=int(settings.get('max_batch_rows', 32)), watchdog_wall_s=budget,
            postprocessing='none', kv_cache=True),
        meanings=L.MEANINGS, checkpoints=[])
    if timing is not None:
        result['charter_timing_input'] = timing.identity()
    L.write_atomic(output, result)
    device = torch.device(device)
    flags = [(module, module.training) for module in model.modules()]
    force_eager = torch.compiler.set_stance('force_eager') if plan.get('compile_loss') else nullcontext()
    devices = [device.index or 0] if device.type == 'cuda' else []
    log = lambda value: print(json.dumps(value), flush=True)
    log(dict(phase='benchmark_v2_start', optimizer_updates=update, max_wall_s=budget, output=str(output)))
    try:
        with torch.random.fork_rng(devices=devices), force_eager:
            model.eval()
            deadline = started+budget
            teacher = run_teacher_screen(model, panel, teacher_protocol, device=device,
                batch_size=int(settings.get('teacher_batch_size', 8)),
                deadline=started+budget*float(settings.get('teacher_fraction', .35)),
                **({} if timing is None else dict(charter_timing=timing)))
            records, benchmark = L.evaluate_model(model, panel, generation_protocol, device=device, deadline=deadline,
                readouts=readouts, token_budget_factor=float(settings.get('token_budget_factor', 2.)),
                max_batch_rows=int(settings.get('max_batch_rows', 32)), log=log,
                **({} if timing is None else dict(charter_timing=timing)))
    finally:
        for module, training in flags:
            module.training = training
    entry = dict(optimizer_updates=update, sha256=identity.get('sha256') or identity.get('weights_fingerprint'),
        path=identity.get('path'), records=records,
        summary=L.summarize(records, len(generation_protocol['samples'])), benchmark=benchmark, teacher_screen=teacher)
    result['checkpoints'] = [L.sanitize(entry)]
    result['complete'] = bool(teacher['summary']['complete'] and benchmark['complete_rows'] == benchmark['expected_rows'])
    result['status'] = 'done'
    result['elapsed_s'] = time.monotonic()-started
    L.write_atomic(output, result)
    if settings.get('latest_output'):
        L.write_atomic(root/settings['latest_output'], result)
    metadata = dict(path=str(output), sha256=file_sha(output), optimizer_updates=update, complete=result['complete'],
                    elapsed_s=result['elapsed_s'], headline=headline(result))
    log(dict(phase='benchmark_v2_done', **metadata))
    return metadata


def headline(result):
    """Compact primary numbers for logs; the JSON keeps everything."""
    entry = result['checkpoints'][0]
    readouts = entry['benchmark']['readouts']
    primary = result['decoding']['primary']
    summary = readouts.get(primary) or next(iter(readouts.values()), {})
    if summary.get('mean_sd_over_seeds'):
        values = {k: v['mean'] for k, v in summary['mean_sd_over_seeds'].items()}
    else:
        seed = next(iter(summary.get('per_seed', {}).values()), {})
        values = {f'{c}.{k}': seed.get(c, {}).get(k) for c, k in __import__(
            'beatfcn.train.learning_evaluation', fromlist=['HEADLINES']).HEADLINES}
    gaps = entry['teacher_screen']['summary'].get('headline', {})
    return dict(readout=summary.get('readout', {}).get('name'), **{k: values.get(k) for k in (
        'accuracy.note_f1_50ms', 'accuracy.note_f1_50ms_minus_null', 'accuracy.complete_fraction',
        'hacks.empty_positive_rate', 'condition_response.monotonic_fraction')},
        teacher_gap_foreign_head_time=gaps.get('foreign', {}).get('head_time'),
        teacher_gap_zero_head_time=gaps.get('zero_local_mel', {}).get('head_time'))


def _rng_states(device):
    import torch
    device = str(device)
    return dict(cpu=torch.get_rng_state(),
                mps=torch.mps.get_rng_state() if device.startswith('mps') else None,
                cuda=torch.cuda.get_rng_state_all() if device.startswith('cuda') else None)


def _restore_rng(states):
    import torch
    torch.set_rng_state(states['cpu'])
    if states['mps'] is not None:
        torch.mps.set_rng_state(states['mps'])
    if states['cuda'] is not None:
        torch.cuda.set_rng_state_all(states['cuda'])


def periodic_benchmark_step(model, panel, plan, report, *, root, device, ema=None, screen=None):
    """Runner hook: benchmark v2 at an optimizer boundary, never crashing training.

    Returns None when not due. Otherwise evaluates EMA weights (if plan EMA is
    on, unless ``weights='raw'``) or the live weights, restores weights, module
    train/eval flags and CPU/MPS/CUDA RNG, appends one entry to
    ``report['benchmark_v2_evaluations']`` (status done/skipped/failed, which
    weights, elapsed) and returns it. ``screen`` defaults to run_periodic_screen.
    """
    import threading
    import traceback
    from contextlib import nullcontext
    settings = plan['benchmark_v2']
    update = int(report['optimizer_updates'])
    if not benchmark_due(update, settings):
        return None
    screen = screen or run_periodic_screen
    started = time.monotonic()
    weights = settings.get('weights') or ('ema' if ema is not None else 'raw')
    entry = dict(optimizer_updates=update, exposure_step=report.get('updates'), weights=weights,
                 ema_updates=ema.updates if ema is not None and weights == 'ema' else None)
    # Identity: the just-saved checkpoint of this boundary when one exists
    # (file holds both raw and EMA weights), else a fingerprint of the
    # evaluated weights. ``weights`` always says which set was evaluated.
    checkpoint = dict(weights=weights, optimizer_updates=update, exposure_step=report.get('updates'))
    name = report.get('checkpoint') or ''
    if report.get('checkpoint_sha256') and Path(name).name == f'step-{report.get("updates", -1):04d}.pt':
        checkpoint.update(path=name, sha256=report['checkpoint_sha256'])
    flags = [(module, module.training) for module in model.modules()]
    states = _rng_states(device)
    max_wall = float(settings.get('max_wall_s', 300.))
    guard = nullcontext()
    if threading.current_thread() is threading.main_thread():
        from beatfcn.train.compiled_training import preparation_deadline
        guard = preparation_deadline(max_wall+float(settings.get('hard_timeout_grace_s', 60.)),
                                     label='Periodic benchmark v2')
    try:
        if weights == 'ema' and ema is None:
            raise ValueError("benchmark_v2.weights='ema' but the plan has no EMA")
        if weights not in ('ema', 'raw'):
            raise ValueError(f'Unknown benchmark_v2.weights {weights!r}')
        swap = ema.swapped() if weights == 'ema' else nullcontext()
        with guard, swap:
            result = screen(model, panel, plan, root=root, device=device, update=update, checkpoint=checkpoint)
        entry.update(status='skipped' if result.get('skipped') else 'done', **result)
    except Exception as exc:  # recorded, never raised into training
        entry.update(status='failed', error=f'{type(exc).__name__}: {exc}',
                     traceback=traceback.format_exc()[-2000:])
    finally:
        for module, training in flags:
            module.training = training
        _restore_rng(states)
    entry['elapsed_s'] = time.monotonic()-started
    report.setdefault('benchmark_v2_evaluations', []).append(entry)
    report['benchmark_v2_s'] = report.get('benchmark_v2_s', 0.)+entry['elapsed_s']
    print(json.dumps(dict(phase='benchmark_v2_step', **{k: v for k, v in entry.items() if k != 'traceback'}),
                     default=str), flush=True)
    return entry
