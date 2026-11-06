"""Hold readout from a fixed post-onset mel window. No event-model training."""
import json
from pathlib import Path

import numpy as np

from beatfcn.events.canonical import TYPES
from beatfcn.train.gesture_window import fit_linear, onset_and_window, recall_of, roll_mel
from beatfcn.train.local_request_panel import LocalRequestPanel

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / 'bench/rhythm-v2/gesture-window-v1.json'
HOLD = TYPES.index('hold')


def write(path, payload):
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def select(panel, skip, take):
    chosen = []
    seen = 0
    for index, row in enumerate(panel.records):
        if row['split'] != 'train' or row['mode'] != 'phira':
            continue
        seen += 1
        if seen <= skip:
            continue
        chosen.append(index)
        if len(chosen) == take:
            return chosen
    raise ValueError('Not enough later train Phira requests')


def collect(panel, indices, n_frames, fraction):
    onset, window, shifted, labels = [], [], [], []
    request_ids = []
    for request_id, index in enumerate(indices):
        row = panel.records[index]
        sample = panel.row(index)
        mel = sample['inputs']['mel']
        times = sample['inputs']['mel_times']
        valid = sample['inputs']['mel_valid']
        rolled = roll_mel(mel, fraction)
        start = row['request']['start_us']
        for event in panel.targets(index):
            when = (event.onset_us - start) / 1_000_000
            frame, mean = onset_and_window(mel, times, when, n_frames, valid)
            _, shifted_mean = onset_and_window(rolled, times, when, n_frames, valid)
            onset.append(frame)
            window.append(mean)
            shifted.append(shifted_mean)
            labels.append(TYPES.index(event.type))
            request_ids.append(request_id)
    return (np.stack(onset), np.stack(window), np.stack(shifted),
            np.asarray(labels), np.asarray(request_ids))


def main():
    plan = json.loads(PLAN_PATH.read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    panel = LocalRequestPanel(ROOT, plan['tape'], plan['features'])
    indices = select(panel, 24, 24)
    onset, window, shifted, labels, request_ids = collect(
        panel, indices, plan['n_frames'], plan['shift_fraction'])
    cutoff = int(24 * plan['train_fraction'])
    train, test = request_ids < cutoff, request_ids >= cutoff
    weight, bias, loss = fit_linear(
        window, labels, train, len(TYPES), plan['steps'], plan['seed'], plan['learning_rate'])
    onset_w, onset_b, _ = fit_linear(
        onset, labels, train, len(TYPES), plan['steps'], plan['seed'], plan['learning_rate'])
    hold = recall_of(window, labels, test, weight, bias, HOLD)
    onset_hold = recall_of(onset, labels, test, onset_w, onset_b, HOLD)
    shifted_hold = recall_of(shifted, labels, test, weight, bias, HOLD)
    per_class = {}
    for index, name in enumerate(TYPES):
        per_class[name] = recall_of(window, labels, test, weight, bias, index)
    gain = None if hold['recall'] is None or onset_hold['recall'] is None else hold['recall'] - onset_hold['recall']
    drop = None if hold['recall'] is None or shifted_hold['recall'] is None else hold['recall'] - shifted_hold['recall']
    support_ok = hold['support'] >= plan['min_test_holds']
    saturated = onset_hold['recall'] is not None and onset_hold['recall'] >= plan['onset_saturation']
    passed = bool(support_ok and not saturated and gain is not None and drop is not None
                  and hold['recall'] >= plan['hold_recall_min']
                  and gain >= plan['gain_over_onset_min']
                  and drop >= plan['shift_drop_min'])
    if not support_ok or saturated:
        gate = 'inconclusive_support_or_onset_saturation'
    elif gain is None or gain < plan['gain_over_onset_min'] or hold['recall'] < plan['hold_recall_min']:
        gate = 'hold_recall_or_gain'
    elif drop < plan['shift_drop_min']:
        gate = 'audio_time_shift'
    else:
        gate = 'no_held_out_song_outputs'
    payload = dict(
        id=plan['id'], complete=True, passed=passed, inconclusive=not support_ok or saturated,
        hypothesis=plan['hypothesis'],
        hold_recall_min=plan['hold_recall_min'],
        gain_over_onset_min=plan['gain_over_onset_min'],
        shift_drop_min=plan['shift_drop_min'],
        measured=dict(hold_recall=hold['recall'], onset_hold_recall=onset_hold['recall'],
                      shifted_hold_recall=shifted_hold['recall'], gain=gain, shift_drop=drop,
                      test_holds=hold['support']),
        per_class=per_class, train_loss=loss, requests=24,
        record_indices=indices, train_requests=int(cutoff),
        frames=int(len(labels)), train_frames=int(train.sum()), test_frames=int(test.sum()),
        optimizer_updates=0, quality_updates=0, new_cloud_compute_calls=0,
        song_outputs=0, confidence_reached=False, failed_gate=gate,
        failure_counters='unchanged; measurement only, not a renamed training retry',
        modal_rule=plan['modal_rule'], memorization_rule=plan['memorization_rule'])
    write(output, payload)
    print(json.dumps(dict(passed=passed, gate=gate, measured=payload['measured']),
                     ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
