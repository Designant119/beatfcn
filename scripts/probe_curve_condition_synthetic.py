"""Synthetic curve-only probe: can this model use the intensity stream at all?

Audio/context are constant; only the density curve changes, and the target note
counts are generated from that curve.  If the model still ignores the curve here,
the failure is architectural/optimization, not real-data redundancy.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, encode_events, teacher_sequence
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel, event_loss
from beatfcn.model.full_event_generate import generate_batch
from beatfcn.train.density_curve import requested_density_curve
from beatfcn.train.local_panel import collate

ROOT = Path(__file__).resolve().parents[1]
CELL_US = 500_000
WINDOW_US = 8_000_000


SIMPLE = False


def target_events(rate):
    events = []
    if SIMPLE:
        onset = (int(rate * 100_000) // 1000) * 1000
        return [RhythmEvent(onset, onset, 'tap', -1)] if onset < WINDOW_US else []
    for cell in range(WINDOW_US // CELL_US):
        count = int(round(rate * CELL_US / 1e6))
        for index in range(count):
            offset = int((index + .5) * CELL_US / max(count, 1))
            onset = cell * CELL_US + offset
            onset = (onset // 1000) * 1000
            if onset < (cell + 1) * CELL_US and onset < WINDOW_US:
                events.append(RhythmEvent(onset, onset, 'tap', -1))
    return events


def make_row(rate, config):
    request = EventRequest('phira', 0, WINDOW_US, WINDOW_US)
    events = target_events(rate)
    sequence = teacher_sequence(encode_events(events, request), request)
    curve = requested_density_curve([dict(start_us=0, end_us=WINDOW_US, density=float(rate))],
        0, WINDOW_US, scale=16.)
    frames, mel_frames, song_frames = 64, 128, 32
    inputs = dict(
        local=np.zeros((2, frames, 1, 1024), np.float32),
        local_times=(np.arange(frames) * 40_000 / 1e6).astype(np.float32),
        local_valid=np.ones(frames, bool),
        mel=np.zeros((mel_frames, 128), np.float32),
        mel_times=(np.arange(mel_frames) * 10_000 / 1e6).astype(np.float32),
        mel_valid=np.ones(mel_frames, bool),
        song=np.zeros((song_frames, 1024), np.float32),
        song_times=(np.arange(song_frames) * 500_000 / 1e6).astype(np.float32),
        song_valid=np.ones(song_frames, bool),
        reference=np.zeros((1, 16), np.float32),
        reference_times=np.zeros(1, np.float32),
        reference_valid=np.zeros(1, bool),
        controls=np.zeros((8, 6), np.float32),
        mode=np.asarray(1, np.int64),
    )
    inputs.update(curve)
    return dict(inputs=inputs, sequence=sequence)


def batch_rows(rates, config):
    return collate([make_row(rate, config) for rate in rates])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--steps', type=int, default=150)
    parser.add_argument('--train-rates', default='0,16,32')
    parser.add_argument('--test-rates', default='0,8,16,24,32')
    parser.add_argument('--simple', action='store_true')
    parser.add_argument('--contrast-weight', type=float, default=0.)
    parser.add_argument('--contrast-margin', type=float, default=0.)
    args = parser.parse_args()
    global SIMPLE
    SIMPLE = bool(args.simple)
    probe = json.loads((ROOT/'reports/data-contract-v2-final/probe-plan.json').read_text())
    config = FullEventConfig(**probe['config'])
    torch.set_num_threads(4)
    torch.manual_seed(20261101)
    model = FullEventModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    train_rates = [float(value) for value in args.train_rates.split(',')]
    model.train()
    losses = []
    for step in range(args.steps):
        rates = [train_rates[step % len(train_rates)] for _ in range(4)]
        inputs, sequence = batch_rows(rates, config)
        loss = event_loss(model(inputs, sequence), sequence)
        if args.contrast_weight:
            other = [32.0 if rate != 32.0 else 0.0 for rate in rates]
            wrong_rows = [make_row(rate, config) for rate in other]
            wrong_inputs, _ = collate(wrong_rows)
            for key in ('intensity', 'intensity_times', 'intensity_valid'):
                inputs = dict(inputs)
                wrong_inputs = dict(wrong_inputs)
                wrong_inputs[key] = wrong_inputs[key]
            shuffled_loss = event_loss(model(wrong_inputs, sequence), sequence)
            loss = loss + args.contrast_weight * torch.relu(
                args.contrast_margin + loss - shuffled_loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        if step % 25 == 0:
            print(json.dumps(dict(step=step, ce=losses[-1])), flush=True)
    model.eval()
    results = []
    with torch.no_grad():
        for rate in [float(value) for value in args.test_rates.split(',')]:
            inputs, sequence = batch_rows([rate], config)
            correct = float(event_loss(model(inputs, sequence), sequence))
            other = 32.0 if rate != 32.0 else 0.0
            wrong_inputs, wrong_sequence = batch_rows([rate], config)
            wrong_curve = requested_density_curve(
                [dict(start_us=0, end_us=WINDOW_US, density=other)], 0, WINDOW_US, scale=16.)
            wrong_inputs.update({k: torch.as_tensor(v)[None] for k, v in wrong_curve.items()})
            shuffled = float(event_loss(model(wrong_inputs, wrong_sequence), wrong_sequence))
            request = EventRequest('phira', 0, WINDOW_US, WINDOW_US)
            prediction = generate_batch(model, inputs, [request], max_tokens=4096)[0]
            counts = np.zeros(WINDOW_US // CELL_US, np.int64)
            for event in prediction['events']:
                counts[event['onset_us'] // CELL_US] += 1
            actual = counts / (CELL_US / 1e6)
            results.append(dict(rate=rate, correct_ce=correct, shuffled_ce=shuffled,
                delta=shuffled-correct, generated_events=len(prediction['events']),
                generated_heads_per_s=float(actual.mean()),
                generated_cell_counts=counts.tolist(),
                complete=prediction['complete']))
    report = dict(steps=args.steps, train_rates=train_rates, final_train_ce=losses[-1],
        results=results,
        interpretation='If correct_ce and shuffled_ce are indistinguishable, the intensity stream is ignored.')
    (ROOT/'reports/data-contract-v2-final/synthetic-curve-probe.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=1)+'\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
