"""Standalone CPU app: build the full-database native training tape on the volume.

Kept separate from the training app so data preparation can be dispatched
without redeploying training code. Resumable: verified per-chart sidecars are
reused, so a timeout or disconnect never forces a full rebuild.
"""
import os
import subprocess
import sys
from pathlib import Path

import modal

DATA = '/data'
ROOT = Path('/root/beatfcn')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.14.1', index_url='https://download.pytorch.org/whl/cpu')
         .pip_install('numpy==2.2.6')
         .add_local_python_source('beatfcn')
         .add_local_dir('scripts', '/root/beatfcn/scripts'))
app = modal.App('beatfcn-full-tape')


@app.function(image=image, volumes={DATA: data}, cpu=(16, 16),
              memory=(32768, 32768), timeout=7200, retries=0)
def build(limit: int = 0) -> dict:
    import beatfcn as _pkg
    env = dict(os.environ, PYTHONPATH=str(Path(_pkg.__file__).resolve().parents[1]))
    cmd = [sys.executable, '-u', '-m', 'scripts.build_full_native_tape',
           '--targets-root', f'{DATA}/full-database-v1/targets/targets',
           '--features-root', f'{DATA}/full-database-v1/mert-graph-diagnostic-v2/features',
           '--output', f'{DATA}/full-database-v1/native-v2-tape', '--workers', '16']
    if limit:
        cmd += ['--limit', str(limit)]
    proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    tail = []
    for line in proc.stdout:
        print(line, end='', flush=True)
        tail = (tail + [line])[-40:]
        if '"done"' in line:
            data.commit()  # persist sidecars progressively for resume
    code = proc.wait()
    data.commit()
    return dict(returncode=code, tail=''.join(tail)[-4000:])
