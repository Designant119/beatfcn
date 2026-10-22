"""Modal app: frozen PupuM2D weights, encoder throughput, tap extraction.

Run from the repo root, e.g.:
  .venv/bin/modal run beatfcn/cloud/encode_app.py::fetch_teacher
  .venv/bin/modal run beatfcn/cloud/encode_app.py::bench_encode
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import modal

_HERE = Path(__file__).resolve()
REPO = _HERE.parents[2] if len(_HERE.parents) > 2 else _HERE.parent  # container: /root/encode_app.py
TEACHER_SHA256 = '6a97605541ba9d3680f6758e2ae19019736f03e5db14e8bbfbd71746b19304f4'
TEACHER_BYTES = 1_612_597_160
WEIGHTS = '/weights'
DATA = '/data'

app = modal.App('beatfcn-encode')
weights = modal.Volume.from_name('beatfcn-weights', create_if_missing=True)
data = modal.Volume.from_name('beatfcn-data', create_if_missing=True)

cpu_image = (modal.Image.debian_slim(python_version='3.12').apt_install('curl')
             .add_local_dir(REPO / 'experiments/shared_kda_s0a/teacher_fetch', '/opt/teacher_fetch', copy=True))
gpu_image = (modal.Image.debian_slim(python_version='3.12').apt_install('ffmpeg')
             .pip_install('torch==2.7.1', index_url='https://download.pytorch.org/whl/cu126')
             .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3')
             .add_local_python_source('beatfcn'))


def _sha256(path, chunk=16 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


@app.function(image=cpu_image, volumes={WEIGHTS: weights}, cpu=1.0, memory=2048, timeout=1800)
def fetch_teacher():
    """Pinned-revision range download inside Modal; exact SHA256 or stop."""
    import shutil
    import subprocess
    dest = Path(WEIGHTS) / 'teacher_encoder.safetensors'
    if dest.exists() and dest.stat().st_size == TEACHER_BYTES and _sha256(dest) == TEACHER_SHA256:
        return {'status': 'present', 'sha256': TEACHER_SHA256}
    work = Path('/tmp/tf')
    shutil.copytree('/opt/teacher_fetch', work, dirs_exist_ok=True)
    (work / 'weights').mkdir(exist_ok=True)
    t = time.time()
    subprocess.run(['python', str(work / 'src/download_teacher.py')], check=True)
    got = work / 'weights/teacher_encoder.safetensors'
    digest = _sha256(got)
    if digest != TEACHER_SHA256 or got.stat().st_size != TEACHER_BYTES:
        raise RuntimeError(f'teacher mismatch {digest} {got.stat().st_size}; refusing')
    shutil.copy2(got, dest)
    shutil.copy2(work / 'reports/teacher_download_manifest.json', Path(WEIGHTS) / 'teacher_download_manifest.json')
    weights.commit()
    return {'status': 'downloaded', 'sha256': digest, 'seconds': round(time.time() - t, 1)}


def load_encoder(device='cuda'):
    import torch
    from beatfcn.audio.pupu import load_teacher
    from beatfcn.audio.memory import TapEncoder
    path = Path(WEIGHTS) / 'teacher_encoder.safetensors'
    if _sha256(path) != TEACHER_SHA256:
        raise RuntimeError('teacher checkpoint hash mismatch')
    return TapEncoder(load_teacher(path, dtype=torch.bfloat16, device=device))


@app.function(image=gpu_image, gpu='L40S', volumes={WEIGHTS: weights}, cpu=4.0, memory=16384, timeout=900)
def bench_encode(batches=(4, 8, 16, 32), repeats: int = 6):
    """Encoder throughput (windows/s) and peak memory on synthetic mel windows."""
    import torch
    t0 = time.time()
    enc = load_encoder()
    load_s = time.time() - t0
    rows = []
    for b in batches:
        x = torch.randn(b, 1, 1024, 128)
        torch.cuda.reset_peak_memory_stats()
        enc(x, [8, 16, 24])
        torch.cuda.synchronize()
        t = time.time()
        for _ in range(repeats):
            enc(x, [8, 16, 24])
        torch.cuda.synchronize()
        dt = (time.time() - t) / repeats
        rows.append({'batch': b, 'seconds_per_batch': round(dt, 4), 'windows_per_s': round(b / dt, 1),
                     'peak_allocated_gb': round(torch.cuda.max_memory_allocated() / 1e9, 2)})
    report = {'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__, 'load_seconds': round(load_s, 1),
              'layers': [8, 16, 24], 'dtype': 'bfloat16', 'rows': rows}
    print(json.dumps(report, indent=1))
    return report


PRICE_PER_HOUR = {'L4': 0.7992, 'A10': 1.1016, 'L40S': 1.9512, 'A100-40GB': 2.0988, 'A100-80GB': 2.4984, 'H100': 3.9492}


@app.local_entrypoint()
def compare_gpus(gpus: str = 'L4,A10,L40S,A100-40GB,H100'):
    """Same benchmark on several GPU types in parallel; rank by windows per dollar (GPU list price only)."""
    names = [g.strip() for g in gpus.split(',') if g.strip()]
    calls = {g: bench_encode.with_options(gpu=g).spawn() for g in names}
    table = []
    for g, call in calls.items():
        try:
            r = call.get()
        except Exception as exc:  # record, never retry automatically
            table.append({'gpu': g, 'error': f'{type(exc).__name__}: {exc}'[:200]})
            continue
        best = max(r['rows'], key=lambda x: x['windows_per_s'])
        table.append({'gpu': g, 'device': r['gpu'], 'best_batch': best['batch'], 'windows_per_s': best['windows_per_s'],
                      'peak_gb': best['peak_allocated_gb'],
                      'windows_per_dollar': round(best['windows_per_s'] * 3600 / PRICE_PER_HOUR[g])})
    out = REPO / 'reports' / 'gpu_encode_benchmark.json'
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(table, indent=1))
    print(json.dumps(table, indent=1))


GPU = 'H100'  # measured best windows per dollar (reports/gpu_encode_benchmark.json)


def _ap(scores, labels):
    import numpy as np
    if labels.sum() == 0:
        return None
    o = np.argsort(-scores, kind='stable')
    y = labels[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def _song_labels(row, n_cells):
    """Per 40 ms cell: any (model-view) head in the cell; plus flux-peak cells from mel10."""
    import numpy as np
    from beatfcn.train.data import Corpus
    from beatfcn.ingest import offset as OFF
    corpus = Corpus(DATA)
    corpus.train_rows([row])
    chart = corpus.chart(row['chart_id'])
    y = np.zeros(n_cells, np.float32)
    for e in chart.events:
        c = e.onset_us // 40_000
        if 0 <= c < n_cells:
            y[c] = 1
    mel = np.asarray(corpus.mel(row['audio_sha256']), np.float32)
    env = OFF.flux_envelope(mel)
    peak = np.zeros(n_cells, bool)
    pk = np.where((env[1:-1] > env[:-2]) & (env[1:-1] >= env[2:]) & (env[1:-1] > OFF.PEAK_MIN))[0] + 1
    peak[np.clip(pk // 4, 0, n_cells - 1)] = True
    return mel, y, peak


@app.function(image=gpu_image, gpu=GPU, volumes={WEIGHTS: weights, DATA: data}, cpu=4.0, memory=32768, timeout=3600)
def layer_probe(train_rows: list, val_rows: list, epochs: int = 2, batch: int = 16, lr: float = 1e-3):
    """S0b: one linear onset probe per layer (24) on frozen full-band features (8 x 1024 per 40 ms cell).

    Metrics on val songs: AP over all cells (onset), and AP restricted to cells holding a spectral-flux
    peak (selection proxy: was this salient sound charted?). Reported per mode.
    """
    import numpy as np
    import torch
    from torch.nn import functional as F
    from beatfcn.audio import memory as M
    t0 = time.time()
    enc = load_encoder()
    layers = list(range(1, 25))
    pad = M.silence_row()
    W = torch.zeros(24, 8 * 1024, device='cuda', requires_grad=True)
    b = torch.zeros(24, device='cuda', requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr)

    def windows(row):
        n_frames = np.load(f"{DATA}/mel10/{row['audio_sha256']}.npy", mmap_mode='r').shape[0]
        p = M.plan(n_frames)
        mel, y, peak = _song_labels(row, p.cells)
        for i in range(0, len(p.starts), batch):
            st = p.starts[i:i + batch]
            cells = np.concatenate([np.arange((s + M.EDGE) // 4, (s + M.EDGE) // 4 + M.HOP // 4) for s in st])
            ok = cells < p.cells
            taps = enc(M.window_batch(mel, st, pad), layers)
            feats = torch.stack([taps[l].reshape(-1, 8 * 1024) for l in layers])  # [24, N, 8192]
            yield feats[:, torch.from_numpy(ok).cuda()], y[cells[ok]], peak[cells[ok]]

    def logits(feats):
        x = F.layer_norm(feats.float(), (feats.shape[-1],))
        return torch.einsum('lnd,ld->ln', x, W) + b[:, None]

    rng = np.random.default_rng(0)
    pos = np.mean([r.get('n_events', 0) / max(r['audio_sec'] * 25, 1) for r in train_rows])
    pw = torch.tensor(min(max((1 - pos) / max(pos, 1e-3), 1.0), 20.0), device='cuda')
    n_train = 0
    for ep in range(epochs):
        for k in rng.permutation(len(train_rows)):
            for feats, y, _ in windows(train_rows[k]):
                yt = torch.from_numpy(y).cuda()[None].expand(24, -1)
                loss = F.binary_cross_entropy_with_logits(logits(feats), yt, pos_weight=pw)
                opt.zero_grad()
                (loss * 24).backward()
                opt.step()
                n_train += yt.shape[1]
    t_train = time.time() - t0
    scores, labels, peaks, modes = [], [], [], []
    with torch.no_grad():
        for row in val_rows:
            for feats, y, peak in windows(row):
                scores.append(logits(feats).cpu().numpy())
                labels.append(y)
                peaks.append(peak)
                modes.append(np.full(len(y), row['mode'], dtype=object))
    S, Y, Pk, Md = np.concatenate(scores, 1), np.concatenate(labels), np.concatenate(peaks), np.concatenate(modes)
    rows = []
    for li, l in enumerate(layers):
        r = {'layer': l, 'ap_onset': _ap(S[li], Y), 'ap_select_on_peaks': _ap(S[li][Pk], Y[Pk])}
        for mode in sorted(set(Md)):
            mm = Md == mode
            r[f'ap_onset_{mode}'] = _ap(S[li][mm], Y[mm])
            r[f'ap_select_{mode}'] = _ap(S[li][mm & Pk], Y[mm & Pk])
        rows.append(r)
    report = {'gpu': torch.cuda.get_device_name(), 'train_songs': len(train_rows), 'val_songs': len(val_rows),
              'train_cells_seen': n_train, 'val_cells': int(len(Y)), 'val_pos_rate': float(Y.mean()),
              'val_peak_cells': int(Pk.sum()), 'val_pos_rate_on_peaks': float(Y[Pk].mean()) if Pk.any() else None,
              'pos_weight': float(pw), 'epochs': epochs, 'seconds_total': round(time.time() - t0, 1),
              'seconds_train': round(t_train, 1), 'layers': rows}
    os.makedirs(f'{DATA}/reports', exist_ok=True)
    with open(f'{DATA}/reports/layer_probe.json', 'w') as f:
        json.dump(report, f, indent=1)
    data.commit()
    return report


@app.function(image=gpu_image, gpu=GPU, volumes={WEIGHTS: weights, DATA: data}, cpu=4.0, memory=32768, timeout=3600)
def encode_songs(shas: list, layers: list, taps_key: str, batch: int = 16):
    """Write taps/{key}/{sha}/L{l}.npy (float16 [cells,8,1024]) and S20.npy (2 s means of layer 20)."""
    import numpy as np
    import torch
    from beatfcn.audio import memory as M
    t0, done, windows = time.time(), 0, 0
    enc = load_encoder()
    pad = M.silence_row()
    want = sorted(set(layers) | {20})
    for sha in shas:
        base = f'{DATA}/taps/{taps_key}/{sha}'
        if all(os.path.exists(f'{base}/L{l}.npy') for l in layers) and os.path.exists(f'{base}/S20.npy'):
            continue
        mel = np.load(f'{DATA}/mel10/{sha}.npy').astype(np.float32)
        taps = M.encode_song(enc, mel, want, batch=batch, pad_row=pad)
        os.makedirs(base, exist_ok=True)
        for l in layers:
            np.save(f'{base}/L{l}.npy', taps[l].numpy())
        np.save(f'{base}/S20.npy', M.summary_means(taps[20]).numpy().astype(np.float16))
        done += 1
        windows += len(M.plan(mel.shape[0]).starts)
        if done % 20 == 0:
            data.commit()
    data.commit()
    return {'songs': done, 'windows': windows, 'seconds': round(time.time() - t0, 1), 'gpu': torch.cuda.get_device_name()}


@app.local_entrypoint()
def run_probe(rows_json: str, out: str = 'reports/layer_probe.json', epochs: int = 2):
    rows = json.loads(Path(rows_json).read_text())
    r = layer_probe.remote(rows['train'], rows['val'], epochs=epochs)
    Path(out).write_text(json.dumps(r, indent=1))
    best = sorted(r['layers'], key=lambda x: -(x['ap_onset'] or 0))[:5]
    print(json.dumps({k: r[k] for k in ('gpu', 'train_songs', 'val_songs', 'val_pos_rate', 'val_pos_rate_on_peaks', 'seconds_total')}))
    for x in r['layers']:
        print(x['layer'], round(x['ap_onset'] or 0, 3), round(x['ap_select_on_peaks'] or 0, 3))


@app.local_entrypoint()
def encode(shas_file: str, taps_key: str = 'v1_L8-16-24', layers: str = '8,16,24', per_container: int = 200):
    """Encode songs listed (one sha per line) in parallel H100 containers; idempotent."""
    shas = [s.strip() for s in Path(shas_file).read_text().split() if s.strip()]
    ls = [int(x) for x in layers.split(',')]
    chunks = [shas[i:i + per_container] for i in range(0, len(shas), per_container)]
    t = time.time()
    res = list(encode_songs.starmap([(c, ls, taps_key) for c in chunks]))
    print(json.dumps({'chunks': len(chunks), 'songs': sum(r['songs'] for r in res), 'windows': sum(r['windows'] for r in res),
                      'gpu_seconds': round(sum(r['seconds'] for r in res), 1), 'wall_s': round(time.time() - t, 1)}))
