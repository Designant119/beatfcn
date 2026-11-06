"""Frozen local learning exposure followed by twenty actual whole-song outputs."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import collections
import json
import os
from pathlib import Path
import platform
import time
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.local_song_rollout import generate_complete_songs
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_request_panel import LocalRequestPanel, event_request
from beatfcn.train.density_curve import grid_density_curve
from beatfcn.train.local_panel import collate
from beatfcn.train.training_objective import training_loss, exposure_group_loss, update_learning_rate, gradient_norms
from beatfcn.train.exposure_loader import (make_vocabulary, is_default_vocabulary, request_sequence, onset_supervision,
    plan_loader_settings)
from beatfcn.train.counterfactual_exposure import FORMAT as COUNTERFACTUAL_FORMAT
from beatfcn.train.ema import ParameterEMA
from scripts.run_local_event_experiment import memory, sync

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*2**20), b''):
            h.update(chunk)
    return h.hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    tmp.replace(path)


def run(plan_path, resume=False, yield_after=None):
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['output']
    if output.exists() and not resume:
        raise FileExistsError('Existing run requires inspected terminal state and explicit --resume')
    for path, expected in plan['inputs_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen baseline input changed: '+path)
    for path, expected in plan['code_sha256'].items():
        if sha(ROOT/path) != expected:
            raise ValueError('Frozen implementation changed: '+path)
    health = json.loads((ROOT/plan['health']).read_text())
    if not health['passed'] or not json.loads((ROOT/plan['features']).read_text())['complete']:
        raise ValueError('Real implementation health and full feature admission are required')
    torch.set_num_threads(plan['cpu_threads'])
    torch.manual_seed(plan['seed'])
    from beatfcn.train.training_panel import build_training_panel
    panel = build_training_panel(ROOT, plan)
    exposure = json.loads((ROOT/plan['exposure']).read_text())
    if plan.get('exposure_slice'):
        exposure = dict(exposure)
        exposure['steps'] = exposure['steps'][:int(plan['exposure_slice'])]
    pool_id = hashlib.sha256(json.dumps(panel.records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if pool_id != exposure['pool_identity'] or len(exposure['steps']) != plan['steps']:
        raise ValueError('Frozen request exposure differs from the dataset')
    counterfactual = exposure.get('format') == COUNTERFACTUAL_FORMAT
    if counterfactual and not plan.get('multiprocess_loader'):
        raise ValueError('Counterfactual exposure is only built by the exposure loader; set multiprocess_loader')
    vocab = make_vocabulary(plan.get('vocabulary'))
    onset_settings = None
    if plan['config'].get('onset_head'):
        onset_settings = dict(plan.get('onset_supervision') or {})
        if not plan.get('loss') or not plan['loss'].get('onset_weight'):
            raise ValueError('onset_head needs loss.onset_weight > 0, else the head is untrained')
        if plan.get('cuda_graph'):
            raise ValueError('onset_head auxiliary loss is not wired into CUDA Graph capture')
    microbatch = plan.get('physical_microbatch_requests', 2)
    if microbatch not in (1, 2, 4, 8, 16) or any(len(step['indices']) != 4 for step in exposure['steps']):
        raise ValueError('Expected four effective requests and physical microbatch one or two')
    if microbatch > 4 and not plan.get('multiprocess_loader'):
        raise ValueError('Grouped physical microbatches require the exposure loader')
    exposure_stride = max(1, microbatch//4)
    if int(plan.get('accum_steps', 1)) % exposure_stride:
        raise ValueError('Physical microbatch must divide the effective request batch')
    # Startup identity verification is once per file, not inside the optimizer loop.
    identities = {}
    if not plan.get('mert_features'):
        identities = {p: h for r in panel.features.records.values() for p, h in r['sha256'].items()}
    for chart in panel.charts:
        identities.update({chart['target_path']: chart['target_sha256'], chart['token_path']: chart['token_sha256']})
    for path, expected in identities.items():
        if sha(ROOT/path) != expected:
            raise ValueError('Prepared cache changed: '+path)
    continuation = None
    if resume and plan.get('resume_checkpoint'):
        raise ValueError('Choose same-plan resume or execution-repair continuation')
    loader = None
    loader_iter = None
    loader_settings = plan_loader_settings(plan)
    if loader_settings['onset'] != onset_settings:
        raise ValueError('Loader onset settings disagree with the runner')
    if plan.get('multiprocess_loader'):
        # Worker processes must fork before CUDA; local CPU/MPS probes may use
        # the same dataset in-process (loader_workers=0).
        if plan.get('device') != 'cuda' and int(plan.get('loader_workers', 12)):
            raise ValueError('multiprocess_loader workers require an explicit cuda device; use loader_workers=0 locally')
        from beatfcn.train.exposure_loader import build_exposure_loader
        loader_start = 0
        if resume:
            loader_start = int(json.loads(output.read_text())['updates'])
        elif plan.get('resume_checkpoint'):
            loader_start = int(plan['resume_checkpoint']['updates'])
        elif plan.get('initialize_from'):
            proof_path = ROOT/plan['initialize_from']['driver_probe']
            loader_start = int(json.loads(proof_path.read_text())['updates'])
        # Start workers before any CUDA context exists in this process.
        loader = build_exposure_loader(panel, exposure,
            microbatch=microbatch, input_shapes=plan.get('input_shapes'),
            sequence_buckets=plan.get('sequence_buckets'),
            num_workers=int(plan.get('loader_workers', 12)),
            prefetch_factor=int(plan.get('loader_prefetch', 2)),
            start_update=loader_start,
            seed=plan['seed'],
            multiprocessing_context=plan.get('loader_context'),
            pin_memory=plan.get('device') == 'cuda', persistent_workers=True,
            **loader_settings)
        loader_iter = iter(loader)
    device = plan.get('device') or ('cuda' if torch.cuda.is_available()
        else 'mps' if torch.backends.mps.is_available() else 'cpu')
    if device == 'mps' and os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', '0') != '0':
        raise ValueError('Native MPS required')
    if device == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    if device == 'mps':
        torch.mps.manual_seed(plan['seed'])
        torch.mps.set_per_process_memory_fraction(.6)
    if plan.get('resume_checkpoint'):
        from beatfcn.train.checkpoint_resume import load_resume
        continuation = load_resume(ROOT, plan)
    model = FullEventModel(FullEventConfig(**plan['config']), vocab).to(device)
    autocast_dtype = {'bf16': torch.bfloat16, 'fp16': torch.float16}.get(plan.get('autocast'))
    scaler = torch.amp.GradScaler('cuda') if autocast_dtype == torch.float16 else None
    if scaler is not None:
        # The graph scales its loss with a static device scalar, so GradScaler's
        # lazy _scale initialization must happen before the first unscale_.
        scaler.scale(torch.ones((), device=device))
    if plan.get('compile') and device == 'cuda' and plan.get('cuda_graph'):
        raise ValueError('torch.compile and manual CUDA graph capture are mutually exclusive')
    if plan.get('compile') and device == 'cuda':
        model = torch.compile(model)
    opt_settings = plan.get('optimizer', {})
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (decay if parameter.ndim >= 2 and 'input_scale' not in name else no_decay).append(parameter)
    optimizer = torch.optim.AdamW([
        dict(params=decay, weight_decay=float(opt_settings.get('weight_decay', 0.))),
        dict(params=no_decay, weight_decay=0.)], lr=plan['learning_rate'],
        betas=tuple(opt_settings.get('betas', (.9, .999))))
    if plan['data_contract_version'] == 1:
        optimizer = torch.optim.AdamW(model.parameters(), lr=plan['learning_rate'], weight_decay=0.)
    # Optional EMA of the weights (created after every weight load below); when
    # absent the optimizer path is untouched.
    ema = None
    ema_state, ema_strict = None, False
    graph_step = None
    if plan.get('cuda_graph') and device == 'cuda':
        from beatfcn.train.cuda_graph import GraphStep
        graph_step = GraphStep(model, optimizer, autocast_dtype=autocast_dtype,
            warmup=int(plan.get('cuda_graph_warmup', 3)), loss_settings=plan.get('loss'))
    accum_steps = int(plan.get('accum_steps', 1))
    if accum_steps < 1 or len(exposure['steps']) % accum_steps:
        raise ValueError('Training exposure must end at a complete optimizer update')
    # The cosine schedule runs over OPTIMIZER updates (exposure steps / accum).
    # An explicit total_updates must agree with the frozen exposure, so a plan
    # whose exposure is far longer than the real budget cannot silently keep
    # the learning rate near its peak for the whole run.
    schedule_settings = dict(plan.get('learning_rate_schedule') or {})
    schedule_total = len(exposure['steps'])//accum_steps
    if 'total_updates' in schedule_settings:
        if int(schedule_settings.pop('total_updates')) != schedule_total:
            raise ValueError('learning_rate_schedule.total_updates disagrees with exposure steps / accum_steps')
    gpu_samples = []
    loss_window = []
    gpu_stop = threading.Event()

    def gpu_sampler():
        import subprocess as sp
        while not gpu_stop.is_set():
            try:
                out = sp.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used',
                              '--format=csv,noheader,nounits'], capture_output=True, text=True,
                             timeout=5).stdout.strip()
                util, mem = [int(float(x)) for x in out.splitlines()[0].split(',')]
                gpu_samples.append((time.monotonic(), util, mem))
            except Exception:
                pass
            gpu_stop.wait(2.0)
    report = dict(id=plan['id'], plan_sha256=sha(plan_path), complete=False, updates=0,
        optimizer_updates=0, curve=[],
        generation=[], new_cloud_spend_usd=0, quality_winner=None, quality_passed=False,
        parameters=sum(p.numel() for p in model.parameters()), valid_tokens=0, requests=0,
        physical_microbatch_requests=microbatch, effective_batch_requests=4*accum_steps,
        train_s=0., preparation_s=0., evaluation_s=0., gpu_work_s=0., song_exposure={}, mode_exposure={},
        software=dict(python=platform.python_version(), torch=torch.__version__,
                      cuda=torch.version.cuda,
                      gpu=(torch.cuda.get_device_name(0) if device == 'cuda' else device),
                      capability=(torch.cuda.get_device_capability(0) if device == 'cuda' else None)),
        tensor_stats=[], checkpoint=None, source_sha256=plan['code_sha256'],
        scope='100-song baseline; unknown reference, Phira 75%, full-song output at 1000 updates',
        remaining_quality_gates=['20-song review', 'all applicable conditions', 'authorized human reference continuation',
                                 'multiple checkpoints and convergence', 'second converged candidate'])
    if continuation is not None:
        model.load_state_dict(continuation['model'], strict=True)
        optimizer.load_state_dict(continuation['optimizer'])
        ema_state, ema_strict = continuation.get('ema'), True
        if scaler is not None:
            if continuation.get('scaler') is None:
                raise ValueError('Continuation lacks GradScaler state')
            scaler.load_state_dict(continuation['scaler'])
        torch.set_rng_state(continuation['cpu_rng'])
        if device == 'cuda':
            torch.cuda.set_rng_state_all(continuation['cuda_rng'])
        elif device == 'mps' and continuation.get('mps_rng') is not None:
            torch.mps.set_rng_state(continuation['mps_rng'])
        for key in ('updates', 'optimizer_updates', 'requests', 'valid_tokens',
                    'song_exposure', 'mode_exposure', 'curve', 'tensor_stats'):
            if key in continuation['progress']:
                report[key] = continuation['progress'][key]
        report['checkpoint'] = plan['resume_checkpoint']['path']
        report['checkpoint_sha256'] = plan['resume_checkpoint']['sha256']
        slot_steps = [float(state['step']) for state in optimizer.state.values() if 'step' in state]
        report['continuation'] = dict(source=plan['resume_checkpoint'],
            optimizer_slots=len(slot_steps), optimizer_step_min=min(slot_steps),
            optimizer_step_max=max(slot_steps), first_next_exposure=report['updates']+1)
        print(json.dumps(dict(phase='checkpoint_restored', **report['continuation'])), flush=True)
        del continuation
    initial = plan.get('initialize_from')
    if initial and not resume:
        saved = torch.load(ROOT/initial['checkpoint'], map_location='cpu', weights_only=False)
        prior = json.loads((ROOT/initial['report']).read_text())
        proof = json.loads((ROOT/initial['driver_probe']).read_text())
        prior_plan = json.loads((ROOT/initial['plan']).read_text())
        if (not proof['passed'] or saved['updates'] != proof['updates']
                or proof['checkpoint_sha256'] != sha(ROOT/initial['checkpoint'])
                or saved['plan_sha256'] != sha(ROOT/initial['plan'])
                or prior_plan['config'] != plan['config']
                or prior_plan['inputs_sha256'] != {k: plan['inputs_sha256'][k] for k in prior_plan['inputs_sha256']}):
            raise ValueError('Initial state does not match the verified driver probe and data contract')
        model.load_state_dict(saved['model']); optimizer.load_state_dict(saved['optimizer'])
        ema_state, ema_strict = saved.get('ema'), False
        torch.set_rng_state(saved['cpu_rng']); torch.mps.set_rng_state(saved['mps_rng'])
        report.update(updates=saved['updates'], optimizer_updates=saved['updates']//accum_steps,
            checkpoint=initial['checkpoint'],
            checkpoint_sha256=proof['checkpoint_sha256'],
            curve=[point for point in prior['curve'] if point['step'] <= saved['updates']],
            tensor_stats=prior['tensor_stats'], initialized_from=initial,
            train_s=proof['train_s'], preparation_s=proof['preparation_s'], evaluation_s=proof['evaluation_s'],
            latest_train_ce=proof['latest_train_ce'], gradient_norm=proof['gradient_norm'])
        report['timing_scope'] = proof.get('timing_scope', 'Verified driver and subsequent run time')
        for update in exposure['steps'][:saved['updates']]:
            for index in update['indices']:
                row = panel.records[index]
                report['requests'] += 1; report['valid_tokens'] += row['tokens']
                for field, key in [('song_exposure', row['component']), ('mode_exposure', row['mode'])]:
                    report[field][key] = report[field].get(key, 0)+1
    if resume:
        report = json.loads(output.read_text())
        if report['plan_sha256'] != sha(plan_path) or report.get('complete') or report.get('error'):
            raise ValueError('Run is complete, failed, or has another identity')
        checkpoint = ROOT/report['checkpoint']
        if sha(checkpoint) != report['checkpoint_sha256']:
            raise ValueError('Resume checkpoint changed')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['updates'] != report['updates']:
            raise ValueError('Checkpoint and report update counts differ')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        ema_state, ema_strict = saved.get('ema'), True
        if saved['updates'] % accum_steps:
            raise ValueError('Checkpoint contains an incomplete gradient accumulation; cannot resume')
        if scaler is not None:
            if saved.get('scaler') is None:
                raise ValueError('FP16 checkpoint lacks GradScaler state')
            scaler.load_state_dict(saved['scaler'])
        torch.set_rng_state(saved['cpu_rng'])
        if device == 'mps' and saved.get('mps_rng') is not None:
            torch.mps.set_rng_state(saved['mps_rng'])
        if device == 'cuda' and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
    init_weights = plan.get('init_weights')
    if init_weights and not resume:
        saved = torch.load(ROOT/init_weights, map_location='cpu', weights_only=False)
        current = model.state_dict()
        if set(saved['model']) != set(current):
            raise ValueError('init_weights does not match the model parameter names')
        model.load_state_dict({key: value.to(current[key].dtype)
                               for key, value in saved['model'].items()})
        report['initialized_weights'] = dict(path=init_weights,
            source_updates=saved.get('updates'))
    if plan.get('ema'):
        ema = ParameterEMA(model, float(plan['ema'].get('decay', .999)))
        if ema_state is not None:
            ema.load_state_dict(ema_state, model)
        elif ema_strict and report['optimizer_updates']:
            raise ValueError('Plan enables EMA but the resumed checkpoint has no EMA state')
        report['ema'] = dict(decay=ema.decay, bias_corrected=True, restored_updates=ema.updates)
    del ema_state
    GPU_NAME = torch.cuda.get_device_name(0) if device == 'cuda' else device
    started = time.monotonic()
    print(json.dumps(dict(phase='boot', revision=sha(Path(__file__)),
        gpu=GPU_NAME, micro=microbatch, accum=accum_steps,
        relative=bool(plan.get('counterfactual_relative'))), ensure_ascii=False), flush=True)
    if plan.get('mert_prefetch') and plan.get('mert_features'):
        # Warm the OS page cache for every song touched by the frozen exposure.
        # The window reads then become memory reads instead of Volume round trips.
        prefetch_tick = time.monotonic()
        seen = set()
        files = []
        for update in exposure['steps']:
            for index in update['indices']:
                audio = panel.records[index]['audio_sha256']
                if audio in seen:
                    continue
                seen.add(audio)
                folder = Path(plan['mert_features']) / audio
                for name in ('L12.npy', 'L23.npy', 'mel10_complete.npy', 'times_s.npy'):
                    path = folder / name
                    if path.is_file():
                        files.append(path)
        total_bytes = sum(path.stat().st_size for path in files)
        cap = float(plan.get('mert_prefetch_max_gb', 10.)) * 2**30
        if total_bytes <= cap:
            for path in files:
                with path.open('rb') as stream:
                    while stream.read(8*2**20):
                        pass
            report['mert_prefetch_s'] = time.monotonic() - prefetch_tick
            report['mert_prefetch_bytes'] = total_bytes
            report['mert_prefetch_songs'] = len(seen)
        else:
            report['mert_prefetch_skipped_bytes'] = total_bytes

    def panel_row(index, controls=None):
        row = panel.row(index, controls=controls)
        if is_default_vocabulary(vocab) and onset_settings is None:
            return row
        record = panel.records[index]
        request = event_request(record['request'])
        events = panel.targets(index)
        sequence = request_sequence(panel, index, request, vocab, events=events)
        if onset_settings is not None:
            sequence = dict(sequence, **onset_supervision(panel, record, request, events,
                row['inputs'], mel_bins=int(plan['config'].get('mel_bins', 0)),
                scope=onset_settings.get('scope', 'window')))
        return dict(row, sequence=sequence)

    def collate_cpu(indices, *, controls=None):
        inputs, sequence = collate([panel_row(i, None if controls is None else controls[j])
            for j, i in enumerate(indices)], input_shapes=plan.get('input_shapes'),
            sequence_buckets=plan.get('sequence_buckets'))
        if device == 'cuda':
            inputs = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in inputs.items()}
            sequence = {k: v.pin_memory() if torch.is_tensor(v) else v for k, v in sequence.items()}
        return inputs, sequence

    def to_device(inputs, sequence):
        return ({k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in inputs.items()},
                {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in sequence.items()})

    def batch(indices, *, controls=None):
        return to_device(*collate_cpu(indices, controls=controls))

    def save(phase):
        if graph_step is not None:
            report['cuda_graph'] = graph_step.summary()
        if gpu_samples:
            utils=[u for _,u,_ in gpu_samples]
            report['gpu_util_mean_percent']=round(sum(utils)/len(utils),2)
            report['gpu_util_max_percent']=max(utils)
            report['gpu_memory_max_mib']=max(m for _,_,m in gpu_samples)
        report.update(phase=phase, attempt_wall_s=time.monotonic()-started, memory=memory(device),
                      observed_at_unix=time.time())
        write_json(output, report)
        print(json.dumps(dict(phase=phase, step=report['updates'], elapsed=report['attempt_wall_s'])), flush=True)

    def checkpoint():
        if report['updates'] % accum_steps:
            raise ValueError('Checkpoint requires a complete optimizer update')
        path = ROOT/plan['checkpoint_dir']/f"step-{report['updates']:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError('Checkpoint already exists; inspect rather than overwrite')
        temporary = path.with_suffix('.partial')
        torch.save(dict(model={k: v.detach().cpu() for k, v in model.state_dict().items()},
            optimizer=optimizer.state_dict(), config=asdict(model.c), vocabulary=asdict(vocab),
            ema=ema.state_dict(model) if ema is not None else None, updates=report['updates'],
            progress=report.copy(),
            scaler=scaler.state_dict() if scaler is not None else None,
            plan_sha256=report['plan_sha256'], cpu_rng=torch.get_rng_state(),
            mps_rng=torch.mps.get_rng_state() if device == 'mps' else None,
            cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None), temporary)
        temporary.replace(path)
        report.update(checkpoint=str(path.relative_to(ROOT)), checkpoint_sha256=sha(path))
        keep = int(plan.get('checkpoint_keep', 0))
        if keep > 0:
            saved = sorted(path.parent.glob('step-*.pt'), key=lambda item: item.name)
            for old in saved[:-keep]:
                if old != path:
                    old.unlink(missing_ok=True)

    def light_checkpoint():
        path = ROOT/plan['checkpoint_dir']/f"light-step-{report['updates']:04d}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            return
        temporary = path.with_suffix('.partial')
        light_ema = None
        if ema is not None:
            light_ema = dict(decay=ema.decay, updates=ema.updates, bias_corrected=True,
                model={k: v.half() for k, v in ema.model_state_dict(model).items()})
        torch.save(dict(model={k: v.detach().cpu().half() for k, v in model.state_dict().items()},
            config=asdict(model.c), vocabulary=asdict(vocab), ema=light_ema, updates=report['updates'],
            plan_sha256=report['plan_sha256']), temporary)
        temporary.replace(path)
        report['light_checkpoint'] = str(path.relative_to(ROOT))
        keep = int(plan.get('light_checkpoint_keep', 2))
        saved = sorted(path.parent.glob('light-step-*.pt'), key=lambda item: item.name)
        for old in saved[:-keep]:
            if old != path:
                old.unlink(missing_ok=True)

    def teacher_inputs(index):
        row = panel.records[index]
        if row['split'] != 'val' or np.asarray(row['controls'])[:, 2].any():
            raise ValueError('Diagnostic validation contains training or hidden condition input')
        inp, sequence = batch([index])
        if 'intensity' not in inp:
            req = event_request(row['request'])
            inp = dict(inp)
            curve = grid_density_curve([], req.start_us, req.end_us,
                cell_us=int(plan.get('counterfactual_cell_us', 100_000)), scale=16.)
            inp.update({key: torch.as_tensor(value)[None].to(device)
                        for key, value in curve.items()})
        return row, inp, sequence

    @torch.no_grad()
    def evaluate():
        tick = time.monotonic()
        model.eval()
        values = []
        for index in plan['validation_indices']:
            row, inp, sequence = teacher_inputs(index)
            onset_auc = None
            if onset_settings is not None:
                from beatfcn.train.training_objective import onset_presence_auc
                logits, aux = model(inp, sequence, return_aux=True)
                onset_auc = onset_presence_auc(aux, sequence)
            else:
                logits = model(inp, sequence)
            loss = event_loss(logits, sequence)
            diagnostics = {} if onset_auc is None else dict(onset_auc=onset_auc)
            intensity = inp.get('intensity')
            if intensity is not None:
                zero_inp = dict(inp)
                zero_inp['intensity'] = torch.zeros_like(intensity)
                shuffled_inp = dict(inp)
                shuffled_inp['intensity'] = intensity.flip(1)
                zero_loss = float(event_loss(model(zero_inp, sequence), sequence))
                shuffled_loss = float(event_loss(model(shuffled_inp, sequence), sequence))
                diagnostics.update(zero_curve_ce=zero_loss, shuffled_curve_ce=shuffled_loss,
                    delta_zero=zero_loss-float(loss),
                    delta_shuffled=shuffled_loss-float(loss))
            values.append(dict(index=index, mode=row['mode'], component=row['component'],
                               chart_id=row['chart_id'], request_ce=float(loss), **diagnostics))
        if ema is not None and plan['ema'].get('evaluate', True):
            # Same unconditional teacher-forced requests with the EMA weights.
            with ema.swapped():
                for value in values:
                    _, inp, sequence = teacher_inputs(value['index'])
                    value['ema_request_ce'] = float(event_loss(model(inp, sequence), sequence))
        sync(device)
        report['evaluation_s'] += time.monotonic()-tick
        control = {}
        for key in ('delta_zero', 'delta_shuffled'):
            present = [r[key] for r in values if key in r]
            if present:
                control[key] = float(np.mean(present))
        report['curve'].append(dict(step=report['updates'], valid_tokens=report['valid_tokens'],
            requests=report['requests'], val=values,
            mode_val_ce={mode: float(np.mean([r['request_ce'] for r in values if r['mode'] == mode]))
                         for mode in ('phira', 'mania_4k') if any(r['mode'] == mode for r in values)},
            mode_ema_val_ce=({mode: float(np.mean([r['ema_request_ce'] for r in values if r['mode'] == mode]))
                              for mode in ('phira', 'mania_4k') if any(r['mode'] == mode for r in values)}
                             if values and 'ema_request_ce' in values[0] else None),
            onset_auc=(float(np.mean([r['onset_auc'] for r in values if r.get('onset_auc') is not None]))
                       if any(r.get('onset_auc') is not None for r in values) else None),
            control_response=control or None,
            scope='teacher-forced midpoint requests; control response via zero/shuffled curve; not product generation'))
        model.train()
        save('heldout_teacher_diagnostic')

    try:
        if not resume and not initial and not plan.get('resume_checkpoint'):
            save('actual_tensor_preflight')
            for index in plan['tensor_probe_indices']:
                row = panel_row(index)
                report['tensor_stats'].append(dict(index=index,
                    tensors={section: {k: tensor_health(v) for k, v in values.items()} for section, values in row.items()}))
            if not plan.get('skip_initial_eval'):
                evaluate()
        elif initial and not resume:
            save('initialized_from_verified_driver_checkpoint')
        if plan.get('learning_eval') and not plan.get('requirement_eval'):
            from beatfcn.train.learning_evaluation import run_checkpoint_screen
            from beatfcn.train.compiled_training import preparation_deadline
            with preparation_deadline(plan['learning_eval']['max_wall_s']+30, label='Held-out evaluation'):
                report['learning_evaluation'] = run_checkpoint_screen(model, panel, plan, root=ROOT, device=device)
            save('learning_evaluation_ready')
        if plan.get('requirement_eval') and plan['requirement_eval'].get('evaluate_at_start', True):
            from beatfcn.train.periodic_requirements import evaluate_training_checkpoint
            if report['updates'] and report.get('checkpoint_sha256'):
                tick_eval = time.monotonic()
                report.setdefault('requirement_evaluations', []).append(evaluate_training_checkpoint(
                    model, panel, plan, report, root=ROOT, device=device,
                    remaining_wall_s=plan['max_train_wall_s']-(time.monotonic()-started)))
                report['evaluation_s'] += time.monotonic()-tick_eval
                save('requirement_startup_evaluation_ready')
        model.train()
        lookahead = int(plan.get('prefetch_lookahead', 6))
        prefetch_pool = None
        if loader_iter is None:
            prefetch_pool = ThreadPoolExecutor(max_workers=int(plan.get('prefetch_workers', 2)),
                thread_name_prefix='panel-prep')

        def prepare_step(update):
            prepared = []
            for offset in range(0, 4, microbatch):
                ids = update['indices'][offset:offset+microbatch]
                controls = []
                for index, specified in zip(ids, update['specified_condition_fields'][offset:offset+microbatch]):
                    row = panel.records[index]
                    if row['split'] != 'train':
                        raise ValueError('Training exposure touches validation')
                    control = np.asarray(row['controls'], np.float32).copy()
                    hide = [i for i in range(len(control)) if i not in specified]
                    control[hide, :3] = 0
                    controls.append(control)
                inp, sequence = collate_cpu(ids, controls=controls)
                prepared.append((ids, inp, sequence))
            return prepared

        def objective(inp, sequence):
            if onset_settings is not None:
                logits, aux = model(inp, sequence, return_aux=True)
                return exposure_group_loss(logits, sequence, plan.get('loss'), aux=aux)
            return exposure_group_loss(model(inp, sequence), sequence, plan.get('loss'))

        def forward_loss(inp, sequence):
            if autocast_dtype is not None:
                with torch.autocast('cuda', dtype=autocast_dtype):
                    return objective(inp, sequence)
            return objective(inp, sequence)

        if plan.get('compile_loss'):
            if device != 'cuda' or graph_step is not None or plan.get('compile'):
                raise ValueError('Compiled loss requires CUDA without another graph/compile wrapper')
            from beatfcn.train.compiled_training import prepare_compiled_loss
            if loader is None:
                raise ValueError('Compiled loss preflight requires the production dataset')
            warm_item = loader.dataset[0]
            if warm_item['step'] != report['updates']+exposure_stride:
                raise ValueError('Cold preparation and resume exposure position disagree')
            warm_batches = warm_item['batches']
            if not warm_batches:
                raise ValueError('Compiled loss preflight requires a nonempty real batch')
            warm_inputs, warm_sequence = to_device(warm_batches[0][1], warm_batches[0][2])
            memory_callback = None
            if plan.get('memory_probe'):
                from beatfcn.train.compiled_training import probe_training_memory
                memory_settings = plan['memory_probe']
                memory_path = ROOT/memory_settings['path']
                if sha(memory_path) != memory_settings['sha256']:
                    raise ValueError('Memory probe identity changed')
                memory_protocol = json.loads(memory_path.read_text())
                if (memory_protocol['tape_sha256'] != plan['inputs_sha256'][plan['tape']]
                        or memory_protocol['exposure_sha256'] != plan['inputs_sha256'][plan['exposure']]):
                    raise ValueError('Memory probe data identity changed')
                probe_dataset = build_exposure_loader(panel, exposure, microbatch=microbatch,
                    num_workers=0, pin_memory=False, start_update=0, **loader_settings).dataset
                memory_callback = lambda fn: probe_training_memory(fn, model, optimizer,
                    probe_dataset, memory_protocol, ROOT/plan['resume_checkpoint']['path'], device)
            forward_loss, report['compile_preflight'] = prepare_compiled_loss(
                forward_loss, model, optimizer, warm_inputs, warm_sequence,
                output=ROOT/plan['compile_trace_dir'],
                options=plan['compile_loss'].get('options', {}),
                mode=plan['compile_loss'].get('mode', 'full'), memory_probe=memory_callback)
            del warm_inputs, warm_sequence, warm_batches
            save('compiled_training_ready')
        contrast_settings = plan.get('condition_contrast')
        if contrast_settings and graph_step is not None:
            raise ValueError('condition_contrast is not supported with CUDA Graph capture')
        if graph_step is not None:
            # Capture every sequence bucket before any prefetch thread exists.
            # This keeps cudaHostAlloc/pin_memory out of capture, and removes
            # warmup/capture stalls from the measured training window.
            capture_tick = time.monotonic()
            chosen = {}
            for update_index, update in enumerate(exposure['steps']):
                tokens = max(panel.records[i]['tokens'] for i in update['indices'])
                bucket = next((value for value in plan['sequence_buckets'] if value >= tokens), None)
                if bucket is None or bucket in chosen:
                    continue
                chosen[bucket] = (update_index, update)
                if len(chosen) == len(plan['sequence_buckets']):
                    break
            graph_step.set_scale(scaler.get_scale() if scaler is not None else 1.)
            for bucket in sorted(chosen):
                update_index, update = chosen[bucket]
                prepared = prepare_step(update)
                for ids, cpu_inputs, cpu_sequence in prepared:
                    inp, sequence = to_device(cpu_inputs, cpu_sequence)
                    graph_step.capture(inp, sequence)
                    del inp, sequence
            optimizer.zero_grad(set_to_none=False)
            sync(device)
            report['graph_precapture_s'] = time.monotonic()-capture_tick
            report['graph_precapture_buckets'] = sorted(chosen)
            report['cuda_graph'] = graph_step.summary()
        if device == 'cuda':
            threading.Thread(target=gpu_sampler, daemon=True).start()
        pending = collections.deque() if loader_iter is None else None
        next_to_prepare = report['updates'] if loader_iter is None else None
        accum_count = 0
        profile_next_micro = bool(plan.get('profile_first_micro'))
        resources = None
        resource_started = time.monotonic()
        if device == 'cuda' and plan.get('resource_guard'):
            from beatfcn.train.training_resources import TrainingResources
            resources = TrainingResources(plan['resource_guard'].get('minimum_util', 85.))
        try:
            for update_index in range(report['updates'], len(exposure['steps']), exposure_stride):
                update = exposure['steps'][update_index]
                if time.monotonic()-started > plan['max_train_wall_s'] and accum_count == 0:
                    if not report['checkpoint'] or not report['checkpoint'].endswith(f"{report['updates']:04d}.pt"):
                        checkpoint()
                    save('bounded_training_yield')
                    return
                if loader_iter is not None:
                    tick_wait = time.monotonic()
                    item = next(loader_iter)
                    report['preparation_s'] += time.monotonic()-tick_wait
                    prepared = item['batches']
                    update = dict(indices=item['indices'], effective_update=item['step'])
                else:
                    while len(pending) < lookahead and next_to_prepare < len(exposure['steps']):
                        pending.append(prefetch_pool.submit(prepare_step, exposure['steps'][next_to_prepare]))
                        next_to_prepare += 1
                    tick_wait = time.monotonic()
                    prepared = pending.popleft().result()
                    report['preparation_s'] += time.monotonic()-tick_wait
                step = update['effective_update']
                if step != report['updates']+exposure_stride:
                    raise ValueError('Exposure updates are discontinuous')
                if accum_count == 0:
                    if schedule_settings:
                        report['learning_rate'] = update_learning_rate(optimizer,
                            report['optimizer_updates'], schedule_total,
                            base_lr=plan['learning_rate'], **schedule_settings)
                        report['learning_rate_schedule_total_updates'] = schedule_total
                    optimizer.zero_grad(set_to_none=graph_step is None)
                    if graph_step is not None:
                        graph_step.set_scale(scaler.get_scale() if scaler is not None else 1.)
                    loss_sum_gpu = torch.zeros((), device=device)
                    unscaled_sum_gpu = torch.zeros((), device=device)
                    loss_batches = 0
                    replay_events = []
                step_tokens = 0
                step_frames = 0
                step_tok_ce = 0.
                step_tok_ce_weight = 0
                batch_composition = []
                for ids, cpu_inputs, cpu_sequence in prepared:
                    tick = time.monotonic()
                    if step == 1:
                        try:
                            print(json.dumps(dict(phase='shapes',
                                inputs={key: list(value.shape)
                                        for key, value in cpu_inputs.items()
                                        if hasattr(value, 'shape')},
                                sequence={key: list(value.shape)
                                          for key, value in cpu_sequence.items()
                                          if hasattr(value, 'shape')}),
                                ensure_ascii=False), flush=True)
                        except (AttributeError, TypeError):
                            pass
                    inp, sequence = to_device(cpu_inputs, cpu_sequence)
                    batch_composition.append(dict(
                        modes=[panel.records[i]['mode'] for i in ids],
                        valid_lengths=cpu_sequence['valid'].sum(-1).tolist(),
                        padded_tokens=int(cpu_sequence['targets'].shape[1])))
                    report['preparation_s'] += time.monotonic()-tick
                    tick = time.monotonic()
                    weight = len(ids)/4/accum_steps
                    scale = scaler.get_scale() if scaler is not None else 1.
                    graph_loss = None
                    loss = None
                    if device == 'cuda':
                        event_start = torch.cuda.Event(enable_timing=True)
                        event_end = torch.cuda.Event(enable_timing=True)
                        event_start.record()
                    if graph_step is not None:
                        graph_step.set_scale(scale)
                        graph_loss = graph_step.run(inp, sequence, weight,
                            allow_capture=False)
                        if graph_loss is not None:
                            loss_sum_gpu.add_(graph_loss.detach() / scale)
                            unscaled_sum_gpu.add_(graph_loss.detach() / (scale*weight))
                            loss_batches += 1
                    micro_profiler = None
                    if graph_loss is None and profile_next_micro:
                        micro_profiler = torch.profiler.profile(activities=[
                            torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
                        micro_profiler.start()
                    if graph_loss is None:
                        main_loss = forward_loss(inp, sequence)
                        if contrast_settings:
                            intensity = inp.get('intensity')
                            if intensity is None:
                                raise ValueError('condition_contrast requires an intensity input')
                            shuffled = intensity.clone()
                            if intensity.shape[-1] == 3:
                                shuffled[:, :, 0] = intensity[:, :, 0].flip(1)
                            else:
                                shuffled = intensity.flip(1)
                            contrast_inp = dict(inp)
                            contrast_inp['intensity'] = shuffled
                            shuffled_loss = forward_loss(contrast_inp, sequence)
                            margin = float(contrast_settings.get('margin', 0.))
                            contrast = torch.relu(margin + main_loss - shuffled_loss)
                            main_loss = main_loss + float(contrast_settings.get('weight', 1.))*contrast
                        loss = main_loss * weight
                        if scaler is not None:
                            scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        loss_sum_gpu.add_(loss.detach())
                        unscaled_sum_gpu.add_(loss.detach() / weight)
                        loss_batches += 1
                    if micro_profiler is not None:
                        micro_profiler.stop()
                        operators = [event.key for event in micro_profiler.key_averages()
                                     if 'attention' in event.key]
                        report['attention_operators'] = operators
                        print(json.dumps(dict(phase='attention_kernel_probe', operators=operators)), flush=True)
                        if not any('_efficient_attention' in op or '_flash_attention' in op
                                   for op in operators):
                            raise RuntimeError('Expected fused attention kernels were not selected')
                        profile_next_micro = False
                    if device == 'cuda':
                        event_end.record()
                        replay_events.append((event_start, event_end))
                    micro = (graph_loss.detach()/scale) if graph_loss is not None else loss
                    if micro is not None and weight:
                        micro_loss = float(micro.detach())/float(weight)
                        if micro_loss == micro_loss and abs(micro_loss) != float('inf'):
                            loss_window.append(micro_loss)
                            del loss_window[:-25]
                            try:
                                micro_tokens = int(sequence['valid'].sum())
                            except (KeyError, TypeError):
                                micro_tokens = 0
                            if micro_tokens:
                                step_tok_ce += micro_loss*micro_tokens
                                step_tok_ce_weight += micro_tokens
                    report['train_s'] += time.monotonic()-tick
                    # Actual (possibly edited) target tokens, not the native tape count.
                    step_tokens += int(cpu_sequence['valid'].sum())
                    try:
                        first = next(iter(cpu_inputs.values()))
                        if hasattr(first, 'ndim') and first.ndim >= 2:
                            step_frames = max(step_frames, int(first.shape[1]))
                    except (StopIteration, AttributeError, TypeError):
                        pass
                    del inp, sequence, loss, graph_loss
                tick = time.monotonic()
                accum_count += exposure_stride
                stepped = accum_count >= accum_steps
                if stepped:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    if plan.get('gradient_diagnostics'):
                        report['per_layer_preclip_norm'] = gradient_norms(model)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=True)
                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    if ema is not None:
                        ema.update()
                    accum_count = 0
                    report['optimizer_updates'] = report.get('optimizer_updates', 0)+1
                    count = max(loss_batches, 1)
                    report['latest_train_objective'] = float(loss_sum_gpu)
                    report['latest_train_ce'] = float(loss_sum_gpu) if not plan.get('loss') else None
                    report['latest_train_ce_unscaled'] = (float(unscaled_sum_gpu)/count
                        if not plan.get('loss') else None)
                    if (device == 'cuda' and plan.get('empty_cache_every', 5)
                            and step % int(plan.get('empty_cache_every', 5)) == 0):
                        # Reserved memory crept across changing sequence shapes
                        # until an 80 GB card ran out; release cached blocks.
                        torch.cuda.empty_cache()
                    if device == 'cuda' and replay_events:
                        report['gpu_work_s'] = report.get('gpu_work_s', 0.) + sum(
                            start.elapsed_time(end) / 1000. for start, end in replay_events)
                        replay_events.clear()
                else:
                    norm = None
                report['train_s'] += time.monotonic()-tick
                update_fields = dict(updates=step,
                    valid_tokens=report['valid_tokens']+step_tokens, requests=report['requests']+4*exposure_stride)
                if norm is not None:
                    update_fields['gradient_norm'] = float(norm)
                report.update(update_fields)
                for index in update['indices']:
                    row = panel.records[index]
                    for field, key in [('song_exposure', row['component']), ('mode_exposure', row['mode'])]:
                        report[field][key] = report[field].get(key, 0)+1
                if stepped and resources is not None:
                    try:
                        report['resource_guard'] = resources.observe(
                            elapsed_s=time.monotonic()-resource_started,
                            data_wait_s=report['preparation_s'])
                    except RuntimeError:
                        checkpoint()
                        save('resource_guard_stopped')
                        raise
                control_file = plan.get('yield_control_file')
                if stepped and control_file and (ROOT/control_file).is_file():
                    checkpoint()
                    save('operator_checkpoint_yield')
                    return
                light_every = int(plan.get('light_checkpoint_every', 0))
                if light_every and step % light_every == 0:
                    light_checkpoint()
                log_every = int(plan.get('log_every', 1))
                if log_every and step % log_every == 0:
                    sample = gpu_samples[-1] if gpu_samples else (0, None, None)
                    print(json.dumps(dict(phase='train', step=step,
                        updates=report.get('optimizer_updates'),
                        tok_ce=(round(step_tok_ce/step_tok_ce_weight, 5)
                                if step_tok_ce_weight else None),
                        loss_scope='composite objective; tok_ce is legacy naming, not pure CE',
                        composition=batch_composition,
                        source_exposures=[dict(step=source['effective_update'],rates=source.get('rates'))
                            for source in exposure['steps'][step-exposure_stride:step]],
                        tokens=step_tok_ce_weight,
                        loss=(round(sum(loss_window)/len(loss_window), 5) if loss_window else None),
                        loss_now=(round(loss_window[-1], 5) if loss_window else None),
                        objective=report.get('latest_train_objective'),
                        ce=report.get('latest_train_ce_unscaled'),
                        grad=report.get('gradient_norm'),
                        lr=report.get('learning_rate'),
                        gpu=GPU_NAME,
                        gpu_util=sample[1], gpu_mem_mib=sample[2],
                        gpu_util_interval=report.get('resource_guard', {}).get('util_mean'),
                        gpu_util_rolling=report.get('resource_guard', {}).get('util_rolling_mean'),
                        mem_alloc_mib=(round(torch.cuda.memory_allocated()/2**20)
                                       if device == 'cuda' else None),
                        mem_peak_mib=(round(torch.cuda.max_memory_allocated()/2**20)
                                      if device == 'cuda' else None),
                        frames=(step_frames or None),
                        elapsed=round(time.monotonic()-started, 2),
                        prep_s=round(report.get('preparation_s', 0.), 3),
                        train_s=round(report.get('train_s', 0.), 3)), ensure_ascii=False), flush=True)
                    if device == 'cuda':
                        torch.cuda.reset_peak_memory_stats()
                if stepped and (step % int(plan.get('checkpoint_every', 25)) == 0 or step in plan['checkpoints']):
                    checkpoint()
                    save('training')
                if step in plan['checkpoints']:
                    evaluate()
                if stepped and plan.get('requirement_eval'):
                    from beatfcn.train.periodic_requirements import requirement_due, evaluate_training_checkpoint
                    if requirement_due(report['optimizer_updates'], plan['requirement_eval']):
                        if not report.get('checkpoint') or not report['checkpoint'].endswith(f'step-{step:04d}.pt'):
                            checkpoint()
                        tick_eval = time.monotonic()
                        report.setdefault('requirement_evaluations', []).append(evaluate_training_checkpoint(
                            model, panel, plan, report, root=ROOT, device=device,
                            remaining_wall_s=plan['max_train_wall_s']-(time.monotonic()-started)))
                        report['evaluation_s'] += time.monotonic()-tick_eval
                        save('periodic_requirement_evaluation_ready')
                if stepped and plan.get('benchmark_v2'):
                    # After any due checkpoint save; EMA weights when EMA is on.
                    # Bounded by benchmark_v2.max_wall_s; failures are recorded
                    # in the entry and never stop training.
                    from beatfcn.train.periodic_requirements import periodic_benchmark_step
                    bench_entry = periodic_benchmark_step(model, panel, plan, report, root=ROOT,
                                                          device=device, ema=ema)
                    if bench_entry is not None:
                        report['evaluation_s'] += bench_entry['elapsed_s']
                        save('periodic_benchmark_v2_ready')
                        model.train()
                if yield_after is not None and step >= yield_after and stepped:
                    if not report['checkpoint'] or not report['checkpoint'].endswith(f"{step:04d}.pt"):
                        checkpoint()
                    save('explicit_driver_probe_yield')
                    return
        finally:
            if prefetch_pool is not None:
                prefetch_pool.shutdown(wait=False)
            if loader is not None:
                del loader_iter
                del loader
            gpu_stop.set()
            if resources is not None:
                resources.close()
        model.eval()
        if plan.get('skip_generation'):
            del optimizer
            report.update(complete=True, phase='complete_training_only',
                latest_train_ce=report.get('latest_train_ce'))
            save('complete_training_only')
            return
        del optimizer
        save('whole_song_generation')
        evaluation = json.loads((ROOT/plan['evaluation20']).read_text())['records']
        if len(evaluation) != 20 or len({r['component'] for r in evaluation}) != 20:
            raise ValueError('Twenty independent evaluation songs required')
        songs = [dict(chart_id=r['chart_id'], audio_sha256=r['audio_sha256'], mode=r['mode'],
                      duration_us=panel.features.records[r['audio_sha256']]['duration_us']) for r in evaluation]
        target = ROOT/plan['generation_output']
        gen = dict(id=plan['id']+'-whole-song-1000', complete=False, checkpoint=report['checkpoint'],
            checkpoint_sha256=report['checkpoint_sha256'], evaluation20=plan['evaluation20'],
            plan_sha256=report['plan_sha256'], song_files={}, song_progress={}, expected_songs=20, optimizer_updates=report['updates'])

        def progress(song):
            path = ROOT/plan['generation_dir']/(song['chart_id'].replace(':', '-')+'.json')
            write_json(path, song)
            gen['song_files'][song['chart_id']] = str(path.relative_to(ROOT))
            gen['song_progress'][song['chart_id']] = dict(complete=song['complete'], failure=song['failure'],
                generated_notes=len(song['events']), completed_until_us=song['next_start_us'], duration_us=song['duration_us'])
            gen['completed_songs'] = sum(s['complete'] for s in gen['song_progress'].values())
            gen['last_progress_at_unix'] = time.time()
            write_json(target, gen)

        intensity_provider = None
        if plan['config'].get('intensity_dim'):
            settings = plan.get('reference_intensity') or plan.get('intensity') or {}
            if plan['data_contract_version'] == 2:
                # Training renders every curve (and the unconditional request)
                # on the counterfactual_cell_us grid (100 ms); generation must
                # see the same grid, not requested_density_curve's 500 ms /
                # one-token-per-segment layout.
                curves = plan.get('generation_density_curves', {})
                render_cell = int(plan.get('counterfactual_cell_us', 100_000))
                intensity_provider = lambda audio, start, end: grid_density_curve(
                    curves.get(audio, []), start, end, cell_us=render_cell, scale=settings.get('scale', 16.))
            else:
                from beatfcn.train.intensity_curve import audio_intensity_curve
                intensity_provider = lambda audio, start, end: audio_intensity_curve(
                    panel.features, audio, start, end, cell_us=settings.get('cell_us', 500_000),
                    scale=settings.get('scale', 16.), mode=plan['generation_intensity'])
        generated = generate_complete_songs(model, panel.features, songs, device=device,
            batch_size=plan['generation_batch_size'], token_cap=plan['generation_token_cap'],
            max_wall_s=plan['max_generation_wall_s'], history_reference=False, on_progress=progress,
            input_shapes=plan.get('input_shapes'), intensity_provider=intensity_provider)
        gen.update(complete=generated['complete'], wall_s=generated['wall_s'],
            completed_songs=sum(s['complete'] for s in generated['songs']),
            failures={s['chart_id']: s['failure'] for s in generated['songs'] if s['failure']},
            history_reference=False, human_review_complete=False)
        write_json(target, gen)
        report['generation'].append(dict(step=report['updates'], report=plan['generation_output'],
            complete_songs=gen['completed_songs'], expected_songs=20, human_review_complete=False))
        report['complete'] = True
        save('awaiting_twenty_song_review')
    except Exception as exc:
        report.update(error=f'{type(exc).__name__}: {exc}', complete=False)
        save('failed')
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--yield-after', type=int)
    args = parser.parse_args()
    if args.yield_after is not None and args.yield_after < 1:
        parser.error('--yield-after must be positive')
    run(args.plan, args.resume, args.yield_after)
