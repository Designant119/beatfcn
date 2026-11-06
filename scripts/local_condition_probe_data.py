"""Rebuild the unchanged R03 probe locally, then select fixed whole cores."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from beatfcn.events.requests import build_request
from beatfcn.model import features as F
from beatfcn.model.batch import SongInfo, attach_audio, collate_meta, unpack_batch
from beatfcn.train.data import chart_from_canon, model_shift_us
from beatfcn.train.density import complete_window
from beatfcn.train.window_controls import encode_window_controls, window_control_view

ROOT = Path(__file__).resolve().parents[1]


def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return isinstance(b, (list, tuple)) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return a == b


def metadata(plan, *, with_generation=False):
    root = ROOT / 'reports/r03_window_raw'
    original = unpack_batch(torch.load(root / 'probe-batch.pt', map_location='cpu', weights_only=False))
    before = window_control_view(original, 'all_applicable')
    manifest = json.loads((root / 'manifest.json').read_text())
    examples, infos, rows, generation = [], [], [], []
    recipe = original['_audio']
    for i, (row, expected) in enumerate(zip(manifest['probe_rows'], manifest['probe_canon_sha256'], strict=True)):
        raw = (root / 'canon' / row['source'] / f'{row["native_id"]}.json').read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError('source canonical changed')
        doc = json.loads(raw)
        if doc['audio_sha256'] != row['audio_sha256'] or recipe['keys'][i] != row['audio_sha256']:
            raise ValueError('source audio differs')
        chart = chart_from_canon(doc, model_shift_us(row, True))
        p = original['_provenance'][i]
        c0, duration = p['c0_us'], round(row['audio_sec'] * F.SEC)
        base = build_request(chart, kind='from_scratch', seed=p['seed'],
            audio_id=row['audio_sha256'], audio_duration_us=duration)
        request = complete_window(base, chart, c0, duration, p['seed'])
        ex = F.build_core(request, chart.events, c0, duration, stage_a_arm='legacy')
        ex.controls = np.asarray(encode_window_controls(row['mode'], request.raw_controls))
        start, cells = recipe['crops'][i]
        if start != 0 or cells >= recipe['C']:
            raise ValueError('explicit complete song dimensions required')
        infos.append(SongInfo(row['audio_sha256'], cells, (cells + 49)//50))
        examples.append(ex)
        generation.append(dict(example=ex, request=request, chart=chart, row=row))
        rows.append(dict(chart_id=row['chart_id'], mode=row['mode'], c0_us=c0,
            targets=sum(len(g['notes']) for g in ex.groups), canonical_verified=True))
    def collate(indices):
        return collate_meta([examples[i] for i in indices], [infos[i] for i in indices],
            tuple(recipe['taps']), recipe['top'], recipe['Dt'], recipe['D_sum'], recipe['quant'])
    rebuilt = collate(range(len(examples)))
    checked = []
    for key, value in rebuilt.items():
        if torch.is_tensor(value):
            torch.testing.assert_close(before[key], value, atol=1e-6, rtol=1e-6, msg=lambda s: key+': '+s)
            checked.append(key)
        elif not equal(before[key], value):
            raise ValueError(f'independent metadata differs: {key}')
    indices = plan['exact_probe_core_indices']
    bt = collate(indices)
    bt['_density_cf'] = True
    bt['_provenance'] = [copy.deepcopy(original['_provenance'][i]) for i in indices]
    bt['full_window_controls'] = original['full_window_controls'][indices].clone()
    torch.testing.assert_close(bt['controls'], bt['full_window_controls'], rtol=0, atol=0)
    evidence = dict(passed=True, original_cores=len(examples), selected_indices=indices,
        selected_rows=[rows[i] for i in indices], compared_tensors=checked,
        unchanged_context_policy=True, whole_cores_selected=True)
    if with_generation:
        return bt, evidence, [generation[i] for i in indices]
    return bt, evidence


def attach(bt, plan):
    root = ROOT / plan['local_root']
    scale_data = np.load(root / 'pca/i8_9-16-24.npz')
    layers = bt['_audio']['taps']
    scales = {layer: scale_data[f'scale_{layer}'].astype(np.float32) for layer in layers}
    if any(not np.isfinite(x).all() or (x <= 0).any() for x in scales.values()):
        raise ValueError('invalid PCA scales')
    songs = {}
    for sha in set(bt['_audio']['keys']):
        path = root / 'taps/v4_i8_L9-16-24' / sha
        arrays = {layer: torch.from_numpy(np.load(path / f'L{layer}.npy')) for layer in layers}
        summary = torch.from_numpy(np.load(path / 'S20.npy'))
        if any(t.dtype != torch.int8 or t.shape[-2:] != (8,1024) for t in arrays.values()):
            raise ValueError('frozen feature shape or type differs')
        songs[sha] = SimpleNamespace(taps=arrays, summary=summary, mel10=None)
    return attach_audio(bt, songs.__getitem__, device='cpu', scales=scales)
