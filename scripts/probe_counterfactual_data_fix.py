"""Data probe for the counterfactual-density data fix (data-fix-v1).

Targets only (no audio, no model): over random local train requests of the
native tape, apply every treatment through the real loader path
(CounterfactualExposureDataset.sample) and report, per treatment:
onset-gap tails vs native, achieved/requested density, cells left unchanged,
condition coverage after frozen dropout, curve == achieved density, mania
carry legality, and encode->replay round trips in the default and the
relative-block vocabularies. Also audits the realized exposure mix and the
retired v1 generator's slot parity for comparison.

  .venv/bin/python -m scripts.probe_counterfactual_data_fix \
      --tape inputs/native-v2-d94-580/tape.json --requests 600
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.model.full_event_codec import EventState
from beatfcn.train.counterfactual_density import CELL_US, real_density_curve
from beatfcn.train.counterfactual_exposure import (CONTROL_FIELDS, FORMAT, build_exposure,
    exposure_summary, normalize_mix, select_validation)
from beatfcn.train.density_curve import encode_density
from beatfcn.train.exposure_loader import CounterfactualExposureDataset, make_vocabulary
from beatfcn.train.local_request_panel import LocalRequestPanel
from beatfcn.events.controls import MODE_FIELDS

ROOT = Path(__file__).resolve().parents[1]


class TargetPanel:
    """Tape records + target arrays; audio inputs are zero placeholders with real clocks."""

    def __init__(self, root, tape):
        self.root = Path(root)
        self.manifest = json.loads((self.root/tape).read_text())
        self.charts = self.manifest['charts']
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            for request in chart['requests']:
                self.records.append(dict(request, chart_index=chart_index,
                    chart_id=chart['chart_id'], audio_sha256=chart['audio_sha256'],
                    component=chart['component'], split=chart['split'], mode=chart['mode'],
                    stratum=chart['stratum']))
        self._arrays = lru_cache(maxsize=8)(lambda i: LocalRequestPanel._read_arrays(self, i))
        self.sequence = lru_cache(maxsize=64)(lambda i: LocalRequestPanel._sequence(self, i))

    def targets(self, index):
        return LocalRequestPanel.targets(self, index)

    def row(self, index, controls=None):
        start = self.records[index]['request']['start_us']
        lo = max(0, start // 40_000 - 100)
        frames = np.arange(lo, start // 40_000 + 300)
        mel = np.arange(lo*4, (start // 40_000 + 300)*4)
        return dict(inputs=dict(
            local_times=(frames*.04 + .02 - start/1e6).astype(np.float32),
            local_valid=np.ones(len(frames), bool),
            mel_times=(mel*.01 + .005 - start/1e6).astype(np.float32),
            mel_valid=np.ones(len(mel), bool)), sequence=None)


def gaps(events):
    onsets = np.unique([e.onset_us for e in events])
    return np.diff(onsets) if len(onsets) > 1 else np.zeros(0, np.int64)


def replay(sequence, request, vocab):
    state = EventState(request, vocab)
    for token in np.asarray(sequence['targets'])[np.asarray(sequence['valid'])].tolist():
        state.apply(int(token))
    return sorted(state.events)


def held_lane_violations(events, carry):
    bad = 0
    for event in events:
        for hold in carry:
            if event.lane == hold.lane and event.onset_us < hold.end_us:
                bad += 1
    return bad


def old_v1_native_coverage(train_count, steps=300000):
    """Retired generator: train[(update*4+slot) % n]; slot 0 was the only native slot."""
    native = np.zeros(train_count, bool)
    for update in range(min(steps, 4*train_count)):
        native[(update*4) % train_count] = True
    return float(native.mean())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--tape', default='inputs/native-v2-d94-580/tape.json')
    parser.add_argument('--requests', type=int, default=600)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--exposure-updates', type=int, default=2000)
    parser.add_argument('--exposure-accum', type=int, default=16)
    parser.add_argument('--output', default='reports/data-fix-v1/probe.json')
    args = parser.parse_args(argv)
    started = time.monotonic()
    panel = TargetPanel(ROOT, args.tape)
    train = [i for i, r in enumerate(panel.records) if r['split'] == 'train']
    rng = np.random.default_rng(args.seed)
    chosen = sorted(int(i) for i in rng.choice(train, min(args.requests, len(train)), replace=False))

    mix = normalize_mix(None)
    exposure = build_exposure(train, steps=args.exposure_updates*args.exposure_accum, seed=args.seed,
                              pool_identity='probe')
    realized = exposure_summary(exposure)
    # Frozen dropout of each chosen request's first exposure, for coverage stats.
    first = {}
    for step in exposure['steps']:
        for k, request in enumerate(step['indices']):
            first.setdefault(request, (step['unconditional'][k], step['curve_specified'][k],
                                       step['specified_condition_fields'][k]))
    vocabularies = {'default': make_vocabulary({})}
    relative_error = None
    try:
        vocabularies['relative_blocks+tail_query'] = make_vocabulary(dict(relative_blocks=True, tail_query=True))
    except ValueError as exc:
        relative_error = str(exc)
    datasets = {name: CounterfactualExposureDataset(panel, dict(format=FORMAT, steps=[]),
        microbatch=4, vocabulary={} if name == 'default' else dict(relative_blocks=True, tail_query=True),
        onset=dict(scope='window'), mel_bins=128) for name in vocabularies}

    stats = defaultdict(lambda: defaultdict(list))
    coverage = defaultdict(Counter)
    failures = []
    old_curve_error = []
    for request_index in chosen:
        record = panel.records[request_index]
        native = panel.targets(request_index)
        unconditional, curve_specified, fields = first.get(request_index, (False, True, list(range(8))))
        for item in mix:
            name, rate = item['name'], item['multiplier']
            try:
                row, audit = datasets['default'].sample(request_index, rate, edit_seed=request_index,
                    curve_specified=True, specified_fields=list(range(8)))
            except Exception as exc:  # report, never hide
                failures.append(dict(index=request_index, treatment=name, error=f'{type(exc).__name__}: {exc}'))
                continue
            request, events = audit['request'], audit['events']
            s = stats[name]
            g, g0 = gaps(events), gaps(native)
            s['gaps'].extend(g.tolist())
            s['native_gaps'].extend(g0.tolist())
            s['heads'].append(len(events))
            s['native_heads'].append(len(native))
            if rate is not None and native:
                s['achieved_over_requested'].append(len(events) / (len(native)*rate) if rate else float(len(events) == 0))
            if rate is not None and rate > 1:
                cells = Counter((e.onset_us - request.start_us)//CELL_US for e in native)
                edited = Counter((e.onset_us - request.start_us)//CELL_US for e in events)
                s['cells_ge6_unchanged'].extend([edited[c] == n for c, n in cells.items() if n >= 6])
                s['cells_lt6_reached'].extend([edited[c] >= min(max(n, 6), int(np.floor(n*rate+.5)))
                                               for c, n in cells.items() if n < 6])
            if rate == .5:
                cells = Counter((e.onset_us - request.start_us)//CELL_US for e in native)
                edited = Counter((e.onset_us - request.start_us)//CELL_US for e in events)
                s['nonempty_cells_emptied'].extend([edited[c] == 0 for c in cells])
                old = real_density_curve(native, request.start_us, request.end_us)
                old_curve_error.extend(abs(cell['density']*rate - edited[k]/((cell['end_us']-cell['start_us'])/1e6))
                                       for k, cell in enumerate(old))
            # Curve == achieved coarse density (decoded from the model input).
            segments = audit['curve_segments']
            achieved = [sum(seg['start_us'] <= e.onset_us < seg['end_us'] for e in events)
                        / ((seg['end_us']-seg['start_us'])/1e6) for seg in segments]
            values = row['inputs']['intensity'][:, 0]
            centers = row['inputs']['intensity_times']*1e6 + request.start_us
            expected = np.asarray([encode_density([next(a for seg, a in zip(segments, achieved)
                                   if seg['start_us'] <= c < seg['end_us'])])[0] for c in centers])
            s['curve_matches_achieved'].append(bool(row['inputs']['intensity_valid'].all()
                                                     and np.allclose(values, expected, atol=1e-6)))
            # Conditions recomputed from the actual target (dropout applied separately below).
            bands = audit['bands']
            for field in MODE_FIELDS[record['mode']]:
                coverage[(name, record['mode'])][field] += field in bands
            coverage[(name, record['mode'])]['_n'] += 1
            if not unconditional:
                shown = {CONTROL_FIELDS[i] for i in fields} & set(bands)
                s['shown_fraction_of_defined'].append(len(shown)/max(1, len(bands)))
            s['carry_requests'].append(bool(request.known_carry))
            if record['mode'] == 'mania_4k':
                s['held_lane_violations'].append(held_lane_violations(events, request.known_carry))
            onset = row['sequence'].get('onset_targets')
            if onset is not None:
                s['onset_frames_positive'].append(float((onset[:, 0] == 1).sum()))
            for vname, vocab in vocabularies.items():
                try:
                    sequence = row['sequence'] if vname == 'default' else datasets[vname].sample(
                        request_index, rate, edit_seed=request_index, curve_specified=False,
                        specified_fields=[])[0]['sequence']
                    s[f'roundtrip_{vname}'].append(replay(sequence, request, vocab) == sorted(events))
                except Exception as exc:
                    s[f'roundtrip_{vname}'].append(False)
                    failures.append(dict(index=request_index, treatment=name, vocabulary=vname,
                                         error=f'{type(exc).__name__}: {exc}'))

    def tail(values, threshold):
        values = np.asarray(values)
        return float((values < threshold).mean()) if len(values) else None

    per_treatment = {}
    for item in mix:
        s = stats[item['name']]
        ratio = np.asarray(s['achieved_over_requested'], float)
        entry = dict(samples=len(s['heads']),
            gaps_lt_10ms=tail(s['gaps'], 10_000), gaps_lt_40ms=tail(s['gaps'], 40_000),
            native_gaps_lt_10ms=tail(s['native_gaps'], 10_000), native_gaps_lt_40ms=tail(s['native_gaps'], 40_000),
            mean_heads=float(np.mean(s['heads'])) if s['heads'] else None,
            mean_native_heads=float(np.mean(s['native_heads'])) if s['native_heads'] else None,
            achieved_over_requested_mean=float(ratio.mean()) if len(ratio) else None,
            achieved_over_requested_median=float(np.median(ratio)) if len(ratio) else None,
            achieved_over_requested_pooled=(float(sum(s['heads'])/max(1e-9, item['multiplier']*sum(s['native_heads'])))
                                            if item['multiplier'] else None),
            curve_matches_achieved=float(np.mean(s['curve_matches_achieved'])),
            shown_fraction_of_defined_controls=(float(np.mean(s['shown_fraction_of_defined']))
                                                if s['shown_fraction_of_defined'] else None),
            requests_with_carry=int(sum(s['carry_requests'])),
            mania_inserted_on_held_lane=int(sum(s['held_lane_violations'])),
            onset_positive_frames_mean=(float(np.mean(s['onset_frames_positive'])) if s['onset_frames_positive'] else None))
        for vname in vocabularies:
            entry[f'roundtrip_{vname}'] = float(np.mean(s[f'roundtrip_{vname}'])) if s[f'roundtrip_{vname}'] else None
        if s['cells_ge6_unchanged']:
            entry['dense_cells_ge6_unchanged'] = float(np.mean(s['cells_ge6_unchanged']))
            entry['cells_lt6_reached_capped_target'] = float(np.mean(s['cells_lt6_reached']))
        if s['nonempty_cells_emptied']:
            entry['nonempty_cells_emptied'] = float(np.mean(s['nonempty_cells_emptied']))
        per_treatment[item['name']] = entry
    condition_coverage = {f'{name}/{mode}': {field: count/counter['_n'] for field, count in counter.items()
                                             if field != '_n'} | dict(samples=counter['_n'])
                          for (name, mode), counter in sorted(coverage.items())}
    validation, validation_summary = select_validation(panel.records, count=200, seed=args.seed)
    result = dict(id='data-fix-v1-probe', tape=args.tape,
        tape_sha256=hashlib.sha256((ROOT/args.tape).read_bytes()).hexdigest(),
        seed=args.seed, requests=len(chosen), train_requests=len(train),
        modes=dict(Counter(panel.records[i]['mode'] for i in chosen)),
        scope='targets only, no audio/model; every chosen request x every treatment through '
              'CounterfactualExposureDataset.sample (curve and all controls shown for the audit; '
              'dropout coverage from the frozen exposure)',
        per_treatment=per_treatment, condition_coverage_before_dropout=condition_coverage,
        exposure=dict(updates=args.exposure_updates, accum=args.exposure_accum,
                      epochs=exposure['epochs'], complete_epochs=exposure['complete_epochs'],
                      assignment=exposure['treatment_assignment'], realized=realized),
        old_v1=dict(native_coverage_fraction=old_v1_native_coverage(len(train)),
                    train_mod_4=len(train) % 4,
                    x05_mean_abs_curve_error_nps=float(np.mean(old_curve_error)) if old_curve_error else None),
        validation_selection=validation_summary,
        relative_vocabulary_error=relative_error, failures=failures[:50], failure_count=len(failures),
        wall_s=time.monotonic()-started)
    output = ROOT/args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1, allow_nan=False)+'\n')
    print(json.dumps(dict(output=str(output.relative_to(ROOT)), failures=len(failures),
        per_treatment={k: {m: v[m] for m in ('gaps_lt_10ms', 'gaps_lt_40ms', 'achieved_over_requested_pooled',
            'curve_matches_achieved') if m in v} for k, v in per_treatment.items()}), indent=1))


if __name__ == '__main__':
    main()
