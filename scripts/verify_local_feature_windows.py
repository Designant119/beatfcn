"""Real-song window parity and masked-tail invariance; no training updates."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.local_audio_audit import tensor_health
from beatfcn.train.local_features import LocalFeatureStore
from beatfcn.train.local_panel import collate

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out = ROOT/'reports/d67_local_feature_window_probe_v1.json'
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    source = ROOT/'reports/d67_local_feature_audit_v1.json'
    snapshot = ROOT/'reports/d67_local_feature_audit_window_snapshot_v1.json'
    if snapshot.exists():
        raise FileExistsError(snapshot)
    snapshot.write_bytes(source.read_bytes())
    store = LocalFeatureStore(ROOT, str(snapshot.relative_to(ROOT)), allow_partial=True)
    clock = {r['audio_sha256']: r for r in json.loads((ROOT/'reports/d66_audio_clock_v1.json').read_text())['audio']}
    selected = [r for r in store.records.values() if r['invalid_padding_frames'] and r['duration_us'] > 20_000_000][:2]
    if len(selected) != 2:
        raise ValueError('Need two real admitted songs with partial mel padding')
    result = dict(id='D67-feature-window-probe-v1', records=[], optimizer_updates=0, cloud_compute_calls=0,
        feature_audit_snapshot=str(snapshot.relative_to(ROOT)), feature_audit_sha256=sha(snapshot),
        code_sha256={p: sha(ROOT/p) for p in ['scripts/verify_local_feature_windows.py',
            'beatfcn/train/local_features.py','beatfcn/train/local_panel.py','beatfcn/model/full_event_model.py']})
    rows = []
    for record in selected:
        audio = record['audio_sha256']
        layers, mel, summary = store._open(audio)
        c0 = 8_000_000
        actual = store.window(audio, c0, c0+8_000_000)
        legacy = np.load(ROOT/clock[audio]['mel_path'], mmap_mode='r', allow_pickle=False)
        lo, hi = max(0, c0//40_000-100), min(len(layers[0]), c0//40_000+300)
        old = dict(local=np.stack([np.asarray(a[lo:hi], np.float32)*store.scales[f'scale_{l}']
                    for a, l in zip(layers, (9, 16, 24))]),
            local_times=(np.arange(lo, hi)*.04+.02-c0/1e6).astype(np.float32),
            mel=np.ascontiguousarray(legacy[lo*4:hi*4], dtype=np.float32),
            mel_times=(np.arange(lo*4, hi*4)*.01+.005-c0/1e6).astype(np.float32))
        parity = {k: bool(np.array_equal(v, actual[k])) for k, v in old.items()}
        if not all(parity.values()):
            raise ValueError(f'Ordinary windows differ from D65: {parity}')
        duration = record['duration_us']
        start = (duration-1)//8_000_000*8_000_000
        tail = store.window(audio, start, duration)
        # Blank reference and unspecified conditions are the unchanged baseline.
        ref = np.zeros((1, 16), np.float32)
        ref[0, 1] = (duration-start)/1_200_000_000
        ref[0, 15] = 1
        inputs = dict(tail, reference=ref, reference_times=np.zeros(1, np.float32),
            reference_valid=np.ones(1, bool), controls=encode_controls('phira', {}, (start, duration), origin_us=start),
            mode=np.asarray(1, np.int64))
        request = EventRequest('phira', start, duration, duration)
        sequence = teacher_sequence(encode_events([], request), request)
        rows.append(dict(inputs=inputs, sequence=sequence))
        result['records'].append(dict(audio_sha256=audio, decoded_samples=record['decoded_samples'],
            ordinary_window_exact_parity=parity, tail_scope=[start, duration],
            missing_pupu_tail_samples=record['missing_pupu_tail_samples'],
            tensors={k: tensor_health(v) for k, v in inputs.items()},
            invalid_mel_frames=int((~tail['mel_valid']).sum())))
    inp, sequence = collate(rows)
    altered = {k: v.clone() for k, v in inp.items()}
    altered['local'].masked_fill_(~inp['local_valid'][:, None, :, None, None], 1000.)
    altered['mel'].masked_fill_(~inp['mel_valid'][..., None], -1000.)
    if torch.equal(inp['local'], altered['local']) and torch.equal(inp['mel'], altered['mel']):
        raise ValueError('Mask test changed no padding')
    config = FullEventConfig(**json.loads((ROOT/'bench/rhythm-v2/d65-local-curve-2m-v1.json').read_text())['config'])
    base = FullEventModel(config).eval()
    cpu_logits = None
    result['devices'] = []
    for device in ['cpu'] + (['mps'] if torch.backends.mps.is_available() else []):
        model = base.to(device)
        seq = {k: v.to(device) for k, v in sequence.items()}
        pair = [{k: v.to(device) for k, v in x.items()} for x in (inp, altered)]
        start = time.monotonic()
        with torch.no_grad():
            memories = [model.encode(x) for x in pair]
            logits = [model.decode(*(seq[k] for k in ('tokens', 'fields', 'times', 'valid')), m)
                      for m in memories]
        if device == 'mps':
            torch.mps.synchronize()
        valid_diff = (memories[0]['values']-memories[1]['values'])[memories[0]['valid']].abs().max().item()
        logit_diff = (logits[0]-logits[1])[seq['valid']].abs().max().item()
        current = logits[0][seq['valid']].cpu()
        backend_difference = None if cpu_logits is None else (current-cpu_logits).abs().max().item()
        if cpu_logits is None:
            cpu_logits = current
        passed = valid_diff <= 1e-5 and logit_diff <= 1e-5 and (backend_difference is None or backend_difference <= 1e-3)
        result['devices'].append(dict(device=device, valid_memory_max_difference=valid_diff,
            valid_logits_max_difference=logit_diff, cpu_max_difference=backend_difference,
            wall_s=time.monotonic()-start, passed=passed))
    result['passed'] = all(x['passed'] for x in result['devices'])
    result['limitations'] = ['No fitting or musical quality claim; this is a zero-update real-input wiring probe.',
        'Only currently admitted cache songs were used; the full panel remains pending.']
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(passed=result['passed'], devices=result['devices'], songs=len(selected))))
    if not result['passed']:
        raise RuntimeError('Feature wiring gate failed')


if __name__ == '__main__':
    main()
