"""Stage-0 local-panel CUDA training: frozen payload + panel data on the volume.

The frozen report/artifact payload is uploaded once as a tar (extracted to the
container-local disk).  The 33.6 GB panel taps, mel and summaries stay on the
volume and are symlinked into the repo layout, so reads are window-sized mmaps
instead of full copies.  Outputs are copied back to the volume and committed.
"""
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import modal

DATA = '/data'
ROOT = Path('/root/beatfcn')
DATA_LINKS = (
    'inputs/d66_local_cohort/features/taps',
    'inputs/d65_local/mel10',
    'inputs/d65_local/song_summary',
    'inputs/d66_local_cohort/features/pca',
)

data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.1', index_url='https://download.pytorch.org/whl/cu130')
         .pip_install('numpy==2.2.6')
         .pip_install('pyyaml==6.0.2')
         .add_local_python_source('beatfcn')
         .add_local_dir('scripts', '/root/beatfcn/scripts')
         .add_local_dir('bench', '/root/beatfcn/bench')
         .add_local_dir('design', '/root/beatfcn/design')
         .add_local_dir('tests', '/root/beatfcn/tests'))

app = modal.App('stage0-cuda-training')


REPO = Path('/root/repo')


def _prepare_root(plan=None):
    marker = Path('/tmp/stage0_repo.ready')
    if not marker.exists():
        import beatfcn as _pkg
        pkg_dir = Path(_pkg.__file__).resolve().parent  # /root/beatfcn
        if REPO.exists():
            shutil.rmtree(REPO)
        (REPO).mkdir(parents=True)
        shutil.copytree(pkg_dir, REPO / 'beatfcn',
                        ignore=shutil.ignore_patterns('__pycache__', 'scripts', 'bench',
                                                      'design', 'tests', 'reports', 'artifacts',
                                                      'inputs', 'experiments'))
        subprocess.run(['tar', '-xf', f'{DATA}/stage0/stage0_payload.tar', '-C', str(REPO),
                        '--exclude', 'scripts', '--exclude', 'scripts/*',
                        '--exclude', 'bench', '--exclude', 'bench/*',
                        '--exclude', 'design', '--exclude', 'design/*',
                        '--exclude', 'tests', '--exclude', 'tests/*'],
                       check=True)
        # Code always comes from the image (never the payload snapshot).
        for name in ('scripts', 'bench', 'design', 'tests'):
            for candidate in (pkg_dir / name, pkg_dir.parent / name):
                if candidate.is_dir():
                    shutil.copytree(candidate, REPO / name, dirs_exist_ok=True,
                                    ignore=shutil.ignore_patterns('__pycache__'))
                    break
        scripts_dir = REPO / 'scripts'
        if scripts_dir.is_dir() and not (scripts_dir / '__init__.py').exists():
            (scripts_dir / '__init__.py').write_text('')
        probe = subprocess.run([sys.executable, '-c', 'import scripts; print(scripts.__path__)'],
                               cwd=str(REPO), capture_output=True, text=True)
        print(json.dumps(dict(phase='import_probe', out=probe.stdout.strip()[-300:],
            err=probe.stderr.strip()[-300:], code=probe.returncode,
            py_files=len(list(scripts_dir.rglob('*.py'))) if scripts_dir.is_dir() else 0),
            ensure_ascii=False), flush=True)
        print(json.dumps(dict(phase='bootstrap', pkg=str(_pkg.__file__),
            pkg_dir=str(pkg_dir), repo=sorted(item.name for item in REPO.iterdir()),
            has_scripts=(REPO / 'scripts').is_dir(),
            has_init=(REPO / 'scripts' / '__init__.py').is_file(),
            candidates=[str(pkg_dir / name) for name in ('scripts',)]
                       + [str(pkg_dir.parent / name) for name in ('scripts',)]),
            ensure_ascii=False), flush=True)
        marker.write_text('ok')
    # MERT plans read absolute volume paths; only legacy Pupu panels need links.
    links = () if (plan or {}).get('mert_features') else DATA_LINKS
    for link in links:
        src = Path(DATA) / 'stage0' / link
        if not src.exists():
            raise FileNotFoundError(f'Volume data missing: {src}')
        dst = REPO / link
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.is_symlink():
            dst.unlink()
        elif dst.exists():
            continue
        dst.symlink_to(src)


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=7200, retries=0)
def build_full_native_tape(limit: int = 0) -> dict:
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    cmd = [sys.executable, '-m', 'scripts.build_full_native_tape',
           '--targets-root', f'{DATA}/full-database-v1/targets/targets',
           '--features-root', f'{DATA}/full-database-v1/mert-graph-diagnostic-v2/features',
           '--output', f'{DATA}/full-database-v1/native-v2-tape']
    if limit:
        cmd += ['--limit', str(limit)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True, file=sys.stderr)
    data.commit()
    return dict(returncode=proc.returncode, stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:])


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=1800, retries=0)
def rebuild_native_mert_tape() -> dict:
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    cmd = [sys.executable, '-m', 'scripts.rebuild_native_tape_mert',
           '--tape', f'{DATA}/native-v2-d94-580/tape-modal.json',
           '--features-root', f'{DATA}/full-database-v1/mert-graph-diagnostic-v2/features',
           '--output', f'{DATA}/native-v2-d94-580-mert']
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True, file=sys.stderr)
    data.commit()
    return dict(returncode=proc.returncode, stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:])


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8),
              memory=(16384, 16384), timeout=3600, retries=0)
def extract_native_tar() -> dict:
    target = Path(DATA)/'native-v2-d94-580'
    if target.exists():
        import shutil
        shutil.rmtree(target)
    target.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(['tar', '-xf', f'{DATA}/native580.tar', '-C', str(target)],
                          capture_output=True, text=True)
    data.commit()
    return dict(returncode=proc.returncode, stderr=proc.stderr[-1000:],
                files=sum(1 for _ in target.rglob('*')))


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=1800, retries=0)
def build_native_counterfactual_plan(tape: str, mel_stats: str, name: str,
                                     optimizer_updates: int, accum_steps: int = 16,
                                     extra_args_json: str = '[]') -> dict:
    """Run the native plan builder with the optimizer-update budget interface.

    ``optimizer_updates`` is the intended number of optimizer updates (the
    builder writes updates*accum exposure steps and an LR schedule over the
    real budget). ``extra_args_json`` passes further builder flags, e.g.
    ``["--vocabulary", "{\"relative_blocks\": true}"]``.
    """
    if type(optimizer_updates) is not int or optimizer_updates < 2:
        raise ValueError('optimizer_updates must be the intended optimizer-update count (>= 2)')
    extra = json.loads(extra_args_json)
    if '--steps' in extra:
        raise ValueError('Legacy --steps is retired here; pass optimizer_updates')
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    cmd = [sys.executable, '-m', 'scripts.prepare_native_counterfactual_training',
           '--tape', tape,
           '--mert-features', f'{DATA}/full-database-v1/mert-graph-diagnostic-v2/features',
           '--mel-stats', mel_stats, '--name', name,
           '--optimizer-updates', str(optimizer_updates), '--accum-steps', str(accum_steps),
           '--scale-config', '/root/beatfcn/bench/rhythm-v2/scale-600m-attention-v1.json',
           '--exposure', f'{DATA}/{name}/exposure.json',
           '--plan-output', f'{DATA}/{name}/plan.json',
           '--base-plan', f'{DATA}/data-contract-v2-probe-plan.json', *map(str, extra)]
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True, file=sys.stderr)
    data.commit()
    return dict(returncode=proc.returncode, stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:])


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8),
              memory=(16384, 16384), timeout=3600, retries=0)
def extract_counterfactual_tar() -> dict:
    target = Path(DATA)/'counterfactual-d94-580-v1'
    target.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(['tar', '-xf', f'{DATA}/cf580.tar', '-C', str(target)],
                          capture_output=True, text=True)
    data.commit()
    return dict(returncode=proc.returncode, stderr=proc.stderr[-1000:],
                files=sum(1 for _ in target.rglob('*')))


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=1800, retries=0)
def build_counterfactual_plan(tape: str, curves: str, mel_stats: str,
                              name: str, steps: int) -> dict:
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    cmd = [sys.executable, '-m', 'scripts.prepare_counterfactual_training',
           '--tape', tape, '--curves', curves, '--mel-stats', mel_stats,
           '--mert-features', f'{DATA}/full-database-v1/mert-graph-diagnostic-v2/features',
           '--name', name, '--steps', str(steps), '--device', 'cuda', '--loader',
           '--learning-rate', '0.0001', '--warmup', '500', '--min-lr', '0.00001',
           '--weight-decay', '0.1', '--max-wall', '12000', '--accum', '4',
           '--scale-config', '/root/beatfcn/bench/rhythm-v2/scale-600m-attention-v1.json',
           '--exposure-dir', f'{DATA}/counterfactual-d94-580-v1/exposure',
           '--plan-output', f'{DATA}/counterfactual-d94-580-v1/plan-600m.json']
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout[-4000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-4000:], flush=True, file=sys.stderr)
    data.commit()
    return dict(returncode=proc.returncode, stdout=proc.stdout[-2000:],
                stderr=proc.stderr[-2000:])


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=900, retries=0)
def bench_loader(plan_rel: str, workers: int, batches: int) -> dict:
    plan = json.loads((ROOT / plan_rel).read_text())
    _prepare_root(plan)
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    proc = subprocess.run([sys.executable, '-m', 'scripts.bench_exposure_loader',
                           '--plan', plan_rel, '--workers', str(workers),
                           '--batches', str(batches)], cwd=str(REPO), env=env,
                          capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-8000:], flush=True, file=sys.stderr)
    return dict(plan=plan_rel, workers=workers, batches=batches,
                returncode=proc.returncode, stdout=proc.stdout[-4000:],
                stderr=proc.stderr[-4000:])


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(16, 16),
              memory=(65536, 65536), timeout=7200, retries=0, max_containers=1,
              scaledown_window=60)
def benchmark_large(plan_rel: str, checkpoint_rel: str, out_rel: str,
                    limit: int = 4) -> dict:
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    cmd = [sys.executable, '-m', 'scripts.generate_requirement_benchmark',
           '--plan', plan_rel, '--checkpoint', checkpoint_rel,
           '--out', out_rel, '--limit', str(limit), '--device', 'cuda']
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
    print(proc.stdout[-8000:], flush=True)
    if proc.returncode:
        print(proc.stderr[-8000:], flush=True, file=sys.stderr)
    data.commit()
    return dict(returncode=proc.returncode, stdout=proc.stdout[-3000:],
                stderr=proc.stderr[-3000:])


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(16, 16),
              memory=(65536, 65536), timeout=14400, retries=0, max_containers=3,
              scaledown_window=60)
def train_large_plan(plan_rel: str, auto_resume: bool = False) -> dict:
    data.reload()  # a reused warm container must not read a stale plan/exposure
    plan = json.loads((ROOT / plan_rel).read_text())
    prep_start = time.monotonic()
    _prepare_root(plan)
    # Every spawn is a fresh run; a stale failed report from an earlier attempt
    # otherwise aborts the driver with FileExistsError.
    stale = REPO / plan['output']
    if stale.exists():
        stale.unlink()
    resume_args = []
    if auto_resume:
        resume_args = _restore_for_resume(plan)
    prep_s = time.monotonic() - prep_start
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=str(pkg_parent),
        PYTORCH_CUDA_ALLOC_CONF='expandable_segments:False,max_split_size_mb:64')
    if 'cuda_allocator_config' in plan:
        env.pop('PYTORCH_CUDA_ALLOC_CONF', None)
        env.pop('PYTORCH_ALLOC_CONF', None)
        if plan['cuda_allocator_config']:
            env['PYTORCH_ALLOC_CONF'] = plan['cuda_allocator_config']
    run_start = time.monotonic()
    stop_sync = threading.Event()

    def sync_outputs():
        while not stop_sync.is_set():
            try:
                for rel in (plan['output'], plan['checkpoint_dir'], plan.get('compile_trace_dir'), plan.get('learning_eval', {}).get('output'),
                            plan.get('requirement_eval', {}).get('output'), plan.get('requirement_eval', {}).get('history_dir')):
                    if not rel:
                        continue
                    src = REPO / rel
                    if not src.exists():
                        continue
                    dst = Path(DATA) / 'stage0' / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                data.commit()
            except Exception as exc:
                print(f'sync_outputs: {type(exc).__name__}: {exc}', flush=True)
            stop_sync.wait(60)

    sync_thread = threading.Thread(target=sync_outputs, daemon=True)
    sync_thread.start()
    _reap_stale_scripts()
    proc = subprocess.Popen([sys.executable, '-m', 'scripts.run_local_song_baseline',
                             '--plan', plan_rel, *resume_args], cwd=str(REPO), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    deadline_timer = None
    if plan.get('max_run_wall_s'):
        def stop_owned_process_group():
            if proc.poll() is None:
                print(json.dumps(dict(phase='owned_wall_deadline', seconds=plan['max_run_wall_s'])), flush=True)
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        deadline_timer = threading.Timer(float(plan['max_run_wall_s']), stop_owned_process_group)
        deadline_timer.daemon = True
        deadline_timer.start()
    lines = []
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
            lines.append(line)
            if len(lines) > 400:
                lines = lines[-200:]
        returncode = proc.wait()
    finally:  # cancellation / lost input / timeout: never leave the trainer running
        _kill_group(proc)
    if deadline_timer is not None:
        deadline_timer.cancel()
    run_s = time.monotonic() - run_start
    stop_sync.set()
    sync_thread.join(timeout=10)
    copied = []
    for rel in (plan['output'], plan.get('generation_output'),
                plan['checkpoint_dir'], plan.get('generation_dir'), plan.get('compile_trace_dir'), plan.get('learning_eval', {}).get('output'),
                plan.get('requirement_eval', {}).get('output'), plan.get('requirement_eval', {}).get('history_dir')):
        if not rel:
            continue
        src = REPO / rel
        if not src.exists():
            continue
        dst = Path(DATA) / 'stage0' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    data.commit()
    report = {}
    if (REPO / plan['output']).exists():
        report = json.loads((REPO / plan['output']).read_text())
    return dict(plan=plan_rel, returncode=returncode, copied=copied,
                prep_s=prep_s, run_s=run_s, complete=report.get('complete'),
                updates=report.get('updates'), parameters=report.get('parameters'),
                latest_train_objective=report.get('latest_train_objective'),
                stdout_tail=''.join(lines[-80:]), stderr_tail='')


@app.function(image=image, volumes={DATA: data}, gpu='T4', cpu=(8, 8),
              memory=(32768, 32768), timeout=2400, retries=0, max_containers=8,
              scaledown_window=60)
def run_plan(plan_rel: str) -> dict:
    plan = json.loads((ROOT / plan_rel).read_text())
    prep_start = time.monotonic()
    _prepare_root(plan)
    # Every spawn is a fresh run; a stale failed report from an earlier attempt
    # otherwise aborts the driver with FileExistsError.
    stale = REPO / plan['output']
    if stale.exists():
        stale.unlink()
    prep_s = time.monotonic() - prep_start
    import beatfcn as _pkg
    pkg_parent = Path(_pkg.__file__).resolve().parents[1]  # /root
    env = dict(os.environ, PYTHONPATH=str(pkg_parent))
    run_start = time.monotonic()
    stop_sync = threading.Event()

    def sync_outputs():
        while not stop_sync.is_set():
            try:
                for rel in (plan['output'], plan['checkpoint_dir']):
                    src = REPO / rel
                    if not src.exists():
                        continue
                    dst = Path(DATA) / 'stage0' / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if src.is_dir():
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                    else:
                        shutil.copy2(src, dst)
                data.commit()
            except Exception as exc:
                print(f'sync_outputs: {type(exc).__name__}: {exc}', flush=True)
            stop_sync.wait(60)

    sync_thread = threading.Thread(target=sync_outputs, daemon=True)
    sync_thread.start()
    _reap_stale_scripts()
    proc = subprocess.Popen([sys.executable, '-m', 'scripts.run_local_song_baseline',
                             '--plan', plan_rel, *resume_args], cwd=str(REPO), env=env,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1, start_new_session=True)
    lines = []
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
            lines.append(line)
            if len(lines) > 400:
                lines = lines[-200:]
        returncode = proc.wait()
    finally:  # cancellation / lost input / timeout: never leave the trainer running
        _kill_group(proc)
    run_s = time.monotonic() - run_start
    stop_sync.set()
    sync_thread.join(timeout=10)
    copy_start = time.monotonic()
    copied = []
    for rel in (plan['output'], plan['generation_output'],
                plan['checkpoint_dir'], plan['generation_dir']):
        src = REPO / rel
        if not src.exists():
            continue
        dst = Path(DATA) / 'stage0' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        copied.append(rel)
    data.commit()
    copy_s = time.monotonic() - copy_start
    report = {}
    if (REPO / plan['output']).exists():
        report = json.loads((REPO / plan['output']).read_text())
    return dict(plan=plan_rel, returncode=returncode, copied=copied,
                prep_s=prep_s, run_s=run_s, copy_s=copy_s,
                complete=report.get('complete'), updates=report.get('updates'),
                latest_train_ce=report.get('latest_train_ce'),
                stdout_tail=proc.stdout[-2500:], stderr_tail=proc.stderr[-2500:],
                val=[c.get('mode_val_ce') for c in report.get('curve', [])[-3:]])


@app.function(image=image, volumes={DATA: data}, gpu='T4', cpu=(8, 8),
              memory=(32768, 32768), timeout=900, retries=0, max_containers=4,
              scaledown_window=60)
def probe_panel(plan_rel: str, indices_json: str = '[0]') -> dict:
    """Build real panel rows from the MERT receipts and run one forward pass."""
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
    from beatfcn.train.exposure_loader import onset_supervision, request_sequence
    from beatfcn.train.local_panel import collate
    from beatfcn.train.local_request_panel import event_request
    from beatfcn.train.mert_panel import MertRequestPanel
    plan = json.loads((ROOT / plan_rel).read_text())
    _prepare_root(plan)
    panel = MertRequestPanel(REPO, plan['tape'], plan['mert_features'])
    indices = json.loads(indices_json)
    torch.manual_seed(plan['seed'])
    vocab = EventVocabulary(**(plan.get('vocabulary') or {}))
    model = FullEventModel(FullEventConfig(**plan['config']), vocab).to('cuda').eval()
    onset = plan['config'].get('onset_head')
    rows = []
    with torch.no_grad():
        for index in indices:
            record = panel.records[index]
            request = event_request(record['request'])
            events = panel.targets(index)
            row = panel.row(index)
            sequence = request_sequence(panel, index, request, vocab, events=events)
            if onset:
                sequence = dict(sequence, **onset_supervision(panel, record, request, events, row['inputs'],
                    mel_bins=int(plan['config'].get('mel_bins', 0)),
                    scope=(plan.get('onset_supervision') or {}).get('scope', 'window')))
            inp, seq = collate([dict(row, sequence=sequence)])
            inp = {k: v.to('cuda') for k, v in inp.items()}
            seq = {k: v.to('cuda') for k, v in seq.items()}
            logits = model(inp, seq)
            loss = float(event_loss(logits, seq))
            rows.append(dict(index=index, mode=panel.records[index]['mode'],
                loss=loss, finite=bool(torch.isfinite(logits).all()),
                shapes={k: list(v.shape) for k, v in inp.items() if torch.is_tensor(v)}))
    import beatfcn as pkg
    return dict(plan=plan_rel, parameters=sum(p.numel() for p in model.parameters()),
        rows=rows, package_file=str(Path(pkg.__file__).resolve()),
        root_entries=sorted(os.listdir('/root'))[:10],
        beatfcn_dir_entries=sorted(os.listdir(str(ROOT)))[:10] if ROOT.exists() else None)


def _restore_for_resume(plan):
    """After a platform restart (preemption), continue from the last full checkpoint.

    Only when the synced report is unfinished, error-free and its update count
    equals the referenced full checkpoint (reports are written at checkpoint and
    evaluation boundaries); otherwise start fresh and say why.
    """
    report_path = Path(DATA) / 'stage0' / plan['output']
    reason = None
    if not report_path.exists():
        reason = 'no synced report'
    else:
        report = json.loads(report_path.read_text())
        checkpoint = report.get('checkpoint')
        source = Path(DATA) / 'stage0' / checkpoint if checkpoint else None
        if report.get('complete') or report.get('error'):
            reason = 'report complete or failed'
        elif not report.get('updates') or not source or not source.exists():
            reason = 'no full checkpoint'
        elif not source.name.startswith('step-') or int(source.stem.split('-')[1]) != report['updates']:
            reason = f'report updates {report.get("updates")} != checkpoint {source.name}'
    if reason:
        print(json.dumps(dict(phase='auto_resume_skipped', reason=reason)), flush=True)
        return []
    (REPO / plan['output']).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(report_path, REPO / plan['output'])
    target = REPO / checkpoint
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    print(json.dumps(dict(phase='auto_resume', checkpoint=checkpoint, updates=report['updates'])), flush=True)
    return ['--resume']


def _kill_group(proc, grace_s=10):
    """Kill a child started with start_new_session=True (and its forked workers)."""
    if proc is None or proc.poll() is not None:
        return
    for sig, wait_s in ((signal.SIGTERM, grace_s), (signal.SIGKILL, 5)):
        try:
            os.killpg(proc.pid, sig)
        except ProcessLookupError:
            return
        try:
            proc.wait(timeout=wait_s)
            return
        except subprocess.TimeoutExpired:
            continue


def _reap_stale_scripts():
    """A reused warm container can still run a child of a cancelled/lost input
    (modal cancel does not kill children): it keeps billing and holds volume
    files open. Kill every leftover `-m scripts.*` process before starting."""
    me = os.getpid()
    for entry in os.listdir('/proc'):
        if not entry.isdigit() or int(entry) == me:
            continue
        try:
            cmd = open(f'/proc/{entry}/cmdline', 'rb').read().split(b'\0')
        except OSError:
            continue
        if b'-m' in cmd and any(part.startswith(b'scripts.') for part in cmd):
            print(json.dumps(dict(phase='reap_stale_child', pid=int(entry),
                                  cmd=b' '.join(cmd)[:160].decode(errors='replace'))), flush=True)
            try:
                os.killpg(int(entry), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                try:
                    os.kill(int(entry), signal.SIGKILL)
                except ProcessLookupError:
                    pass


def _run_repo_script(module, args, sync_rels):
    """Run scripts.<module> inside the prepared repo; copy outputs to /data/stage0."""
    data.reload()
    _prepare_root({'mert_features': True})
    import beatfcn as _pkg
    env = dict(os.environ, PYTHONPATH=str(Path(_pkg.__file__).resolve().parents[1]))
    _reap_stale_scripts()
    proc = subprocess.Popen([sys.executable, '-u', '-m', f'scripts.{module}', *map(str, args)],
                            cwd=str(REPO), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1, start_new_session=True)
    tail = []
    try:
        for line in proc.stdout:
            print(line, end='', flush=True)
            tail = (tail + [line])[-60:]
        code = proc.wait()
    finally:  # cancellation / lost input / timeout: never leave the child running
        _kill_group(proc)
    for rel in sync_rels:
        src = REPO / rel
        if not src.exists():
            continue
        dst = Path(DATA) / 'stage0' / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    data.commit()
    return dict(returncode=code, tail=''.join(tail)[-6000:])


# run_script_*: scaledown_window=2 -- a warm container of a PREVIOUS deploy picked up new inputs with old
# code twice (2026-11-02: preset CLI, timing BPM guard) and idle warm time bills; containers here are one-shot.
@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(12, 12),
              memory=(65536, 65536), timeout=5400, retries=0, max_containers=2,
              scaledown_window=2)
def run_script_gpu(module: str, args_json: str, sync_json: str = '[]') -> dict:
    """Evaluation entry (benchmark v2 / workflow benchmark) on one H100."""
    return _run_repo_script(module, json.loads(args_json), json.loads(sync_json))


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=3600, retries=0, max_containers=2,
              scaledown_window=2)
def run_script_cpu(module: str, args_json: str, sync_json: str = '[]') -> dict:
    """CPU-only repo script (panel freezing, human floor) next to the volume data."""
    return _run_repo_script(module, json.loads(args_json), json.loads(sync_json))


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(32, 32),
              memory=(98304, 98304), timeout=14400, retries=0, max_containers=3,
              scaledown_window=60)
def train_large_plan_cpu32(plan_rel: str) -> dict:
    """Same driver with 32 host cores: full-database loader work per request
    (conditions, onset targets, feature windows) starves models below ~600M on
    16 cores (measured ~40 requests/s at 12 workers)."""
    return train_large_plan.local(plan_rel)


FEATURE_FILES = ('L12.npy', 'L23.npy', 'times_s.npy', 'intervals_s.npy', 'mel10_complete.npy', 'receipt.json')
LOCAL_FEATURES = Path('/local-features')


def stage_features(plan, splits=('train', 'val', 'test')):
    """Bulk-copy the MERT/mel receipts of the plan's songs onto local NVMe.

    All splits by default: the request panel verifies a receipt for every
    song in the tape (test included) even though training never reads test.

    Network-volume small random reads made every model below ~600M loader-bound
    and dipped even 600M under the 70% guard when runs shared the volume; large
    sequential copies are fast, so pay a few minutes once instead.
    """
    from concurrent.futures import ThreadPoolExecutor
    tape = json.loads((Path('/') / plan['tape'].lstrip('/')).read_text())
    audios = sorted({c['audio_sha256'] for c in tape['charts'] if c['split'] in splits})
    source = Path(plan['mert_features'])
    started, copied = time.monotonic(), [0]

    def copy(audio):
        dst = LOCAL_FEATURES / audio
        dst.mkdir(parents=True, exist_ok=True)
        size = 0
        for name in FEATURE_FILES:
            src = source / audio / name
            if src.is_file():
                shutil.copyfile(src, dst / name)
                size += (dst / name).stat().st_size
        return size

    with ThreadPoolExecutor(max_workers=64) as pool:
        for index, size in enumerate(pool.map(copy, audios)):
            copied[0] += size
            if index % 1000 == 0:
                print(json.dumps(dict(phase='stage_features', songs=index + 1, of=len(audios),
                    gib=round(copied[0] / 2**30, 1), elapsed_s=round(time.monotonic() - started, 1))), flush=True)
    print(json.dumps(dict(phase='stage_features_done', songs=len(audios), gib=round(copied[0] / 2**30, 1),
                          elapsed_s=round(time.monotonic() - started, 1))), flush=True)
    return str(LOCAL_FEATURES)


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(32, 32),
              memory=(163840, 163840), ephemeral_disk=524288, timeout=30000, retries=0,
              max_containers=3, scaledown_window=60)
def train_staged(plan_rel: str) -> dict:
    """train_large_plan on local-NVMe features (32 cores)."""
    data.reload()
    plan = json.loads((ROOT / plan_rel).read_text())
    local_plan = dict(plan, mert_features=stage_features(plan),
                      staged_features=dict(source=plan['mert_features'], local=str(LOCAL_FEATURES)))
    path = Path('/tmp/plan-staged.json')
    path.write_text(json.dumps(local_plan, indent=1))
    # Execution-only: bound forked-worker panel copies (host OOM at 28 workers).
    os.environ.setdefault('BEATFCN_MAX_LOADER_WORKERS', '20')
    # A preempted input is restarted by the platform: continue, don't restart.
    return train_large_plan.local(str(path), auto_resume=True)


@app.function(image=image, volumes={DATA: data}, gpu='L40S', cpu=(8, 8),
              memory=(49152, 49152), timeout=7200, retries=0, max_containers=6,
              scaledown_window=2)
def run_script_l40s(module: str, args_json: str, sync_json: str = '[]') -> dict:
    """Evaluation on L40S: decoding is per-step-latency bound (~9 ms fixed per
    step on H100), so the cheaper GPU costs ~half per eval at similar speed."""
    return _run_repo_script(module, json.loads(args_json), json.loads(sync_json))

