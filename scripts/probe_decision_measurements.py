"""Read-only measurements for kind and condition, separate from the time probe.

Neither measurement trains the event model. A low flick recall here says the
10ms mel frame at the true onset does not linearly identify flick. The pair
count says whether same-audio charts actually disagree on one requested field.
"""
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from beatfcn.events.canonical import TYPES
from beatfcn.train.event_conditions import CONTROL_COLUMNS, CONTROL_FIELDS
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]
TAPE = 'reports/d94_expanded_target_tape_v1.json'
FEATURES = 'reports/d94_merged_feature_admission_v1.json'
OUTPUT = ROOT / 'reports/decision_measurements_v1.json'
SPECIFIED = CONTROL_COLUMNS.index('specified')
LOWER, UPPER = 0, 1
START, END = 4, 5


def condition_census(panel):
    groups = {}
    charts = {}
    for row in panel.records:
        if row['split'] != 'train' or row['mode'] != 'phira':
            continue
        audio = row['audio_sha256']
        charts.setdefault(audio, set()).add(row['chart_id'])
        controls = np.asarray(row['controls'], np.float32)
        specified = [i for i in range(len(CONTROL_FIELDS)) if controls[i, SPECIFIED] > 0]
        if len(specified) != 1:
            continue
        field = CONTROL_FIELDS[specified[0]]
        scope = (round(float(controls[specified[0], START]), 6),
                 round(float(controls[specified[0], END]), 6))
        band = (round(float(controls[specified[0], LOWER]), 6),
                round(float(controls[specified[0], UPPER]), 6))
        groups.setdefault((audio, scope, field), set()).add(band)
    fields = {}
    for field in CONTROL_FIELDS:
        if field == 'jack_ratio':
            continue
        audios = {audio for (audio, _scope, name), bands in groups.items()
                  if name == field and len(bands) >= 2}
        fields[field] = dict(audios_with_two_bands=len(audios), supports_eight=len(audios) >= 8)
    multi = sum(len(ids) >= 2 for ids in charts.values())
    return dict(train_phira_audios=len(charts), audios_with_two_charts=multi,
        single_field_contrast=fields,
        request_rows=sum(row['split'] == 'train' and row['mode'] == 'phira' for row in panel.records))


def mel_kind(panel, limit_requests=24):
    xs, ys, request_ids = [], [], []
    used = 0
    for index, row in enumerate(panel.records):
        if row['split'] != 'train' or row['mode'] != 'phira':
            continue
        sample = panel.row(index)
        mel, times = sample['inputs']['mel'], sample['inputs']['mel_times']
        start = row['request']['start_us']
        for event in panel.targets(index):
            when = (event.onset_us - start) / 1_000_000
            frame = int(np.argmin(np.abs(times - when)))
            if not sample['inputs']['mel_valid'][frame]:
                continue
            xs.append(mel[frame].copy())
            ys.append(TYPES.index(event.type))
            request_ids.append(used)
        used += 1
        if used >= limit_requests:
            break
    x, y, request_ids = np.stack(xs), np.asarray(ys), np.asarray(request_ids)
    cutoff = int(used * 0.7)
    train, test = request_ids < cutoff, request_ids >= cutoff
    features = torch.from_numpy(x)
    labels = torch.from_numpy(y)
    weight = torch.zeros(len(TYPES), features.shape[1], requires_grad=True)
    bias = torch.zeros(len(TYPES), requires_grad=True)
    optimizer = torch.optim.AdamW((weight, bias), lr=0.05, weight_decay=0.)
    train_x, train_y = features[train], labels[train]
    for _ in range(200):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(train_x @ weight.T + bias, train_y)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        pred = (features[test] @ weight.T + bias).argmax(1).numpy()
    truth = y[test]
    per_class = {}
    for index, name in enumerate(TYPES):
        mask = truth == index
        per_class[name] = dict(support=int(mask.sum()),
            recall=None if mask.sum() == 0 else float((pred[mask] == index).mean()))
    return dict(requests=used, frames=int(len(y)), train_frames=int(train.sum()),
        test_frames=int(test.sum()), train_loss=float(loss.detach()), per_class=per_class,
        flick_below_0_10=per_class['flick']['support'] >= 8 and per_class['flick']['recall'] < 0.10)


def main():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    panel = LocalRequestPanel(ROOT, TAPE, FEATURES)
    payload = dict(id='decision-measurements-v1', optimizer_updates=0, quality_updates=0,
        new_cloud_compute_calls=0, conditions=condition_census(panel), mel_kind=mel_kind(panel))
    OUTPUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(contrasts=payload['conditions']['single_field_contrast'],
        mel_kind=payload['mel_kind']['per_class']), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
