"""Cloud feature extraction for NEW songs (scripts/draft_song.py --extract-only) in the DATABASE MERT image.

Same image recipe as beatfcn/cloud/full_database_app.py (ffmpeg + torch/torchaudio 2.6.0 cu126 +
transformers 4.53.2 + the pinned MERT-v2 checkpoint on the volume), so a new song's features are produced by
the same decoder / MERT build / mel code as the training database. Generation and export then run on the
training image (stage0-cuda-training run_script_l40s 'draft_song --features-dir ...').

  modal deploy beatfcn/cloud/draft_extract_app.py
  modal.Function.from_name('beatfcn-draft-extract', 'extract').remote('stage0/drafts/in/song.mp3',
                                                                      'stage0/drafts/features')
Paths are relative to the volume root (/data). Never trains.
"""
import os
import signal
import subprocess
import time

import modal

DATA = '/data'
CHECKPOINT = '/data/full-database-v1/mert2-checkpoint'
app = modal.App('beatfcn-draft-extract')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .apt_install('ffmpeg')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('torchaudio==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('transformers==4.53.2', 'numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.2')
         .add_local_python_source('beatfcn')
         .add_local_dir('scripts', '/root/beatfcn/scripts'))


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(4, 8), memory=(16384, 32768),
              timeout=1800, retries=0, scaledown_window=2)
def extract(audio_rel: str, features_rel: str, extra_json: str = '[]') -> dict:
    """draft_song --extract-only for /data/<audio_rel> into /data/<features_rel>; returns rc + log tail."""
    import json
    started = time.monotonic()
    cmd = ['python', '-m', 'scripts.draft_song', '--extract-only', '--audio', f'{DATA}/{audio_rel}',
           '--features-dir', f'{DATA}/{features_rel}', '--out-dir', '/tmp/draft-extract',
           '--mert-source', CHECKPOINT, '--device', 'cuda'] + json.loads(extra_json)
    proc = subprocess.Popen(cmd, cwd='/root/beatfcn', stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, start_new_session=True)
    try:
        out, _ = proc.communicate(timeout=1700)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, _ = proc.communicate()
    finally:
        if proc.poll() is None:
            os.killpg(proc.pid, signal.SIGKILL)
    data.commit()
    return dict(returncode=proc.returncode, tail=(out or '')[-4000:], wall_s=round(time.monotonic() - started, 1))
