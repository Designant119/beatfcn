import hashlib
import json
import unittest
from collections import Counter

import numpy as np
import torch

from beatfcn.events.canonical import TYPES, Event, LaneLoc
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.counterfactual_exposure import build_exposure, exposure_summary
from beatfcn.train.event_reference import TIME_SCALE_US, VisibleReference
from beatfcn.train.exposure_loader import (CounterfactualExposureDataset, build_exposure_loader,
                                           make_vocabulary, plan_loader_settings, request_sequence)
from beatfcn.train.history_context import (check_no_leak, draw_history, history_sources,
                                           normalize_history_context)
from beatfcn.train.local_request_panel import event_request

AUDIO = ['%064x' % (i + 1) for i in range(2)]
WINDOW = 8_000_000
DURATION = 48_000_000


def _chart(mode):
    """Whole native chart on a 250 ms grid with holds crossing every window boundary."""
    rows = []
    for t in range(0, DURATION, 250_000):
        if mode == 'mania_4k':
            rows.append((t, t, 0, (t // 250_000) % 2))
        else:
            rows.append((t, t, 3 if t % 500_000 else 0, -1))
    for boundary in range(WINDOW, DURATION, WINDOW):
        # crosses into the window (carry) ...
        rows.append((boundary - 600_000, boundary + 1_400_000, 1, 2 if mode == 'mania_4k' else -1))
        # ... and starts inside the window, crossing out of it (a target, never a reference)
        rows.append((boundary - 700_000, boundary + 900_000, 1, 3 if mode == 'mania_4k' else -1))
    rows.append((DURATION - 300_000, DURATION - 300_000, 0, 1 if mode == 'mania_4k' else -1))
    rows.sort()
    return {key: np.asarray([r[i] for r in rows], np.int64)
            for i, key in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


class ChartPanel:
    """Two charts (phira, mania); every 8 s window of each is a train request."""

    def __init__(self):
        self.charts = [dict(chart_id='phira:1', mode='phira'), dict(chart_id='mania:2', mode='mania_4k')]
        self.arrays = [_chart(c['mode']) for c in self.charts]
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            a = self.arrays[chart_index]
            for start in range(0, DURATION, WINDOW):
                end = start + WINDOW
                inside = np.flatnonzero((a['onset_us'] >= start) & (a['onset_us'] < end))
                crossing = (a['kind'] == 1) & (a['onset_us'] < start) & (a['end_us'] > start)
                carry = [dict(onset_us=int(a['onset_us'][i]), end_us=int(a['end_us'][i]), type='hold',
                              lane=int(a['lane'][i])) for i in np.flatnonzero(crossing)]
                self.records.append(dict(split='train', chart_index=chart_index, chart_id=chart['chart_id'],
                    audio_sha256=AUDIO[chart_index], mode=chart['mode'], controls=np.zeros((8, 6), np.float32),
                    target_start=int(inside[0]), target_stop=int(inside[-1]) + 1,
                    request=dict(mode=chart['mode'], start_us=start, end_us=end, duration_us=DURATION,
                                 known_carry=carry)))

    def _arrays(self, chart_index):
        return None, self.arrays[chart_index]

    def targets(self, index):
        row = self.records[index]
        a = self.arrays[row['chart_index']]
        sl = slice(row['target_start'], row['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(a['onset_us'][sl], a['end_us'][sl], a['kind'][sl], a['lane'][sl])]

    def row(self, index, controls=None):
        local = 300
        times = np.arange(local) * .04 - 4 + .02
        mel = np.arange(local * 4) * .01 - 4 + .005
        return dict(inputs=dict(
            local=np.ones((2, local, 1, 4), np.float32), local_times=times.astype(np.float32),
            local_valid=np.ones(local, bool), mel=np.ones((local * 4, 2), np.float32),
            mel_times=mel.astype(np.float32), mel_valid=np.ones(local * 4, bool),
            song=np.ones((6, 4), np.float32), song_times=np.zeros(6, np.float32),
            song_valid=np.ones(6, bool), reference=np.zeros((1, 16), np.float32),
            reference_times=np.zeros(1, np.float32), reference_valid=np.ones(1, bool),
            controls=np.zeros((8, 6), np.float32),
            mode=np.asarray(0 if self.records[index]['mode'] == 'mania_4k' else 1, np.int64)), sequence=None)

    def sequence(self, index):
        request = event_request(self.records[index]['request'])
        return request_sequence(self, index, request, make_vocabulary({}), events=self.targets(index))


def _digest(items):
    h = hashlib.sha256()
    for item in items:
        for _, inputs, sequence in item['batches']:
            for section in (inputs, sequence):
                for key in sorted(section):
                    h.update(key.encode())
                    h.update(section[key].numpy().tobytes())
    return h.hexdigest()


def _slots(exposure):
    for step in exposure['steps']:
        for slot, index in enumerate(step['indices']):
            yield step, slot, index


class HistoryContextTest(unittest.TestCase):
    def dataset(self, history=True, seed=3, steps=60, **exposure_kw):
        panel = ChartPanel()
        exposure = build_exposure(list(range(len(panel.records))), steps=steps, seed=seed, **exposure_kw)
        settings = (dict(p=.8, p_infill=.5) if history is True else history)
        return CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=settings), exposure

    def sample(self, dataset, step, slot, index):
        key = step.get('occurrences', step['epochs'])[slot]
        return dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                              curve_specified=step['curve_specified'][slot],
                              specified_fields=step['specified_condition_fields'][slot], history_key=key)

    def test_references_only_outside_the_window_and_holds_follow_carry(self):
        dataset, exposure = self.dataset()
        applied = infill = crossing = 0
        for step, slot, index in _slots(exposure):
            row, audit = self.sample(dataset, step, slot, index)
            request, info = audit['request'], audit['history']
            if step['rates'][slot] is not None:
                self.assertFalse(info['applied'])
                self.assertEqual(info['reason'], 'edited_target_native_only')
                self.assertEqual(len(row['inputs']['reference']), 1)
                continue
            if not info['applied']:
                self.assertEqual(len(row['inputs']['reference']), 1)
                continue
            applied += 1
            infill += info['infill']
            reference = row['inputs']['reference']
            span = request.end_us - request.start_us
            carry = {(e.onset_us - request.start_us, e.end_us - request.start_us) for e in request.known_carry}
            events = reference[reference[:, 10] == 1]
            self.assertEqual(len(events), info['prefix_events'] + info['suffix_events'])
            for event in events:
                start_rel = float(event[0]) * TIME_SCALE_US
                end_rel = float(event[1]) * TIME_SCALE_US
                # No shown onset inside the target window (grid is 250 ms; window end is legal).
                self.assertTrue(start_rel < -100_000 or start_rel > span - 1_000, start_rel)
                if start_rel < 0 < end_rel:
                    crossing += 1
                    self.assertTrue(any(abs(start_rel - a) < 5 and abs(end_rel - b) < 5 for a, b in carry))
            # Region rows are disjoint from the window; the unknown row is the window itself.
            unknown = reference[reference[:, 15] == 1]
            self.assertEqual(len(unknown), 1)
            self.assertAlmostEqual(float(unknown[0, 1]) * TIME_SCALE_US, span, delta=5)
            for region in reference[reference[:, 14] == 1]:
                lo, hi = float(region[0]) * TIME_SCALE_US, float(region[1]) * TIME_SCALE_US
                self.assertTrue(hi <= 5 or lo >= span - 5, (lo, hi))
            if info['prefix_region_us']:
                lo, hi = info['prefix_region_us']
                self.assertEqual(hi, request.start_us)
                self.assertEqual(lo, max(0, request.start_us - int(info['prefix_s'] * 1e6)))
            else:
                self.assertEqual(request.start_us, 0)
            if info['suffix_region_us']:
                self.assertEqual(info['suffix_region_us'][0], request.end_us)
            # Target and supervision are the untouched native window.
            self.assertEqual(audit['events'], dataset.panel.targets(index))
        self.assertGreater(applied, 20)
        self.assertGreater(infill, 3)
        self.assertGreater(crossing, 3)

    def test_leak_guard_rejects_window_events_and_unauthorized_crossing_holds(self):
        request = event_request(dict(mode='mania_4k', start_us=8_000_000, end_us=16_000_000,
                                     duration_us=DURATION, known_carry=[]))
        def source(event, region=(0, 8_000_000)):
            return VisibleReference('s', 'mania_4k', AUDIO[0], 'human_input', (region,), (event,))
        with self.assertRaisesRegex(ValueError, 'crosses'):
            check_no_leak([source(Event('h', 7_000_000, 9_000_000, 'hold', LaneLoc(1)))], request)
        with self.assertRaisesRegex(ValueError, 'target-window event|overlaps'):
            check_no_leak([source(Event('t', 9_000_000, 9_000_000, 'tap', LaneLoc(1)), (0, 10_000_000))], request)
        carried = event_request(dict(mode='mania_4k', start_us=8_000_000, end_us=16_000_000, duration_us=DURATION,
            known_carry=[dict(onset_us=7_000_000, end_us=9_000_000, type='hold', lane=1)]))
        self.assertEqual(check_no_leak([source(Event('h', 7_000_000, 9_000_000, 'hold', LaneLoc(1)))], carried), 1)

    def test_deterministic_per_request_epoch_seed(self):
        a, exposure = self.dataset()
        b, _ = self.dataset()
        self.assertEqual(_digest([a[i] for i in range(6)]), _digest([b[i] for i in range(6)]))
        settings = normalize_history_context(dict(p=.8))
        draws = [draw_history(settings, 3, r, k) for r in range(20) for k in range(5)]
        self.assertEqual(draws, [draw_history(settings, 3, r, k) for r in range(20) for k in range(5)])
        self.assertNotEqual(draws, [draw_history(settings, 4, r, k) for r in range(20) for k in range(5)])
        # The audit choice equals the pure draw of (seed, request, epoch).
        step = exposure['steps'][0]
        _, audit = self.sample(a, step, 0, step['indices'][0])
        if step['rates'][0] is None:
            self.assertEqual(audit['history']['drawn'],
                             draw_history(a.history, exposure['seed'], step['indices'][0], step['epochs'][0]))

    def test_off_is_byte_identical(self):
        plain, exposure = self.dataset(history=None)
        panel = plain.panel
        legacy = CounterfactualExposureDataset(panel, exposure, microbatch=4)
        never = CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=dict(p=0.))
        reference = _digest([legacy[i] for i in range(len(legacy))])
        self.assertEqual(_digest([plain[i] for i in range(len(plain))]), reference)
        self.assertEqual(_digest([never[i] for i in range(len(never))]), reference)
        loader = build_exposure_loader(panel, exposure, microbatch=4, num_workers=0, pin_memory=False)
        self.assertEqual(_digest(list(loader)), reference)
        # Plans without the key keep the identical loader-settings dict.
        plan = dict(config=dict(mel_bins=0))
        self.assertNotIn('history_context', plan_loader_settings(plan))
        self.assertEqual(plan_loader_settings(dict(plan, history_context=dict(p=.5)))['history_context']['p'], .5)
        # build_exposure without mode_weights is unchanged (digest frozen before this feature).
        frozen = build_exposure(list(range(8)), steps=40, seed=5)
        self.assertEqual(hashlib.sha256(json.dumps(frozen, sort_keys=True).encode()).hexdigest(),
                         '8b4cd3f89f8661d8c27a2f401a30cbc75fe30225037e98bd22b2afbd24ad8826')
        self.assertEqual(build_exposure(list(range(8)), steps=40, seed=5, mode_weights=None), frozen)
        self.assertNotIn('mode_mix', exposure_summary(frozen))

    def test_history_probabilities_within_tolerance(self):
        settings = normalize_history_context({})
        draws = [draw_history(settings, 11, r, k) for r in range(500) for k in range(40)]
        n = len(draws)
        applied = [d for d in draws if d['applied']]
        self.assertAlmostEqual(len(applied) / n, .5, delta=.02)
        roles = Counter(d['role'] for d in applied)
        self.assertAlmostEqual(roles['human_input'] / len(applied), .7, delta=.02)
        self.assertAlmostEqual(sum(d['infill'] for d in applied) / len(applied), .25, delta=.02)
        lengths = Counter(d['prefix_s'] for d in applied)
        self.assertEqual(sorted(lengths), [4., 8., 16., 32.])
        for count in lengths.values():
            self.assertAlmostEqual(count / len(applied), .25, delta=.02)

    def test_prefix_clipped_at_song_start_and_capacity_fallback(self):
        panel = ChartPanel()
        settings = normalize_history_context(dict(max_reference_rows=20))
        request = event_request(panel.records[1]['request'])   # start 8 s
        draw = dict(applied=True, prefix_s=32., role='model_history', infill=False, suffix_s=4.)
        sources, info = history_sources(panel.arrays[0], request, draw, settings, chart_id='c', audio_sha256=AUDIO[0])
        # 32/16/8 s prefixes (clipped to [0, 8 s): 36 rows) exceed 20 rows -> falls back to 4 s (20 rows).
        self.assertTrue(info['applied'] and info['shrunk'])
        self.assertEqual(info['prefix_s'], 4.)
        self.assertEqual(sources[0].role, 'model_history')
        settings = normalize_history_context({})
        sources, info = history_sources(panel.arrays[0], request, draw, settings, chart_id='c', audio_sha256=AUDIO[0])
        self.assertEqual(info['prefix_region_us'], [0, 8_000_000])
        first = event_request(panel.records[0]['request'])
        _, info = history_sources(panel.arrays[0], first, dict(draw, infill=False), settings,
                                  chart_id='c', audio_sha256=AUDIO[0])
        self.assertFalse(info['applied'])
        self.assertEqual(info['reason'], 'song_start_no_prefix')

    def test_collate_and_model_forward_with_references_for_both_modes(self):
        from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
        dataset, exposure = self.dataset(history=dict(p=1., p_infill=.5))
        loader = build_exposure_loader(dataset.panel, exposure, microbatch=8, num_workers=0, pin_memory=False,
                                       history_context=dict(p=1., p_infill=.5))
        items = list(loader)
        config = FullEventConfig(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=1,
                                 tap_dim=4, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=2, intensity_dim=3)
        torch.manual_seed(0)
        model = FullEventModel(config).eval()
        modes = set()
        for item in items[:3]:
            _, inputs, sequence = item['batches'][0]
            self.assertEqual(inputs['reference'].shape[-1], 16)
            self.assertGreater(inputs['reference'].shape[1], 1)
            lengths = inputs['reference_valid'].sum(1)
            for row, n in enumerate(lengths.tolist()):
                self.assertFalse(inputs['reference_valid'][row, n:].any())
            modes.update(inputs['mode'].tolist())
            with torch.no_grad():
                logits = model(inputs, sequence)
            logits = logits['logits'] if isinstance(logits, dict) else logits
            self.assertTrue(torch.isfinite(logits).all())
        self.assertEqual(modes, {0, 1})


class ModeWeightsTest(unittest.TestCase):
    def test_oversampling_is_deterministic_and_realized_mix_recorded(self):
        train = list(range(300))
        mode_of = {i: 'phira' if i < 100 else 'mania_4k' for i in train}
        kw = dict(steps=1500, seed=9, mode_weights={'phira': 1.6, 'mania_4k': 1.0}, mode_of=mode_of)
        a, b = build_exposure(train, **kw), build_exposure(train, **kw)
        self.assertEqual(a, b)
        self.assertEqual(len(a['steps']), 1500)
        mix = a['mode_mix']
        self.assertAlmostEqual(mix['target_fraction']['phira'], 160 / 360)
        self.assertAlmostEqual(mix['exposure_fraction']['phira'], 160 / 360, delta=.02)
        self.assertEqual(exposure_summary(a)['mode_mix'], mix)
        # Draws key on the per-request occurrence: 0, 1, 2, ... in stream order.
        seen = Counter()
        for step in a['steps']:
            for request, occurrence in zip(step['indices'], step['occurrences']):
                self.assertEqual(occurrence, seen[request])
                seen[request] += 1
        # Repeats of one request inside an epoch get fresh treatment draws, still stratified.
        native = Counter(t for s in a['steps'] for t in s['treatments'])
        self.assertAlmostEqual(native['native'] / 6000, .5, delta=.03)
        self.assertNotEqual(a, build_exposure(train, **dict(kw, seed=10)))

    def test_subsampling_and_validation(self):
        train = list(range(200))
        mode_of = {i: 'phira' if i % 2 else 'mania_4k' for i in train}
        e = build_exposure(train, steps=500, seed=1, mode_weights={'phira': 1., 'mania_4k': .5}, mode_of=mode_of)
        self.assertAlmostEqual(e['mode_mix']['exposure_fraction']['mania_4k'], 1 / 3, delta=.03)
        with self.assertRaises(ValueError):
            build_exposure(train, steps=5, seed=1, mode_weights={'phira': 1.})
        with self.assertRaises(ValueError):
            build_exposure(train, steps=5, seed=1, mode_weights={'phira': 1.}, mode_of=mode_of)
        with self.assertRaises(ValueError):
            build_exposure(train, steps=5, seed=1, mode_weights={'phira': 0., 'mania_4k': 0.}, mode_of=mode_of)

    def test_loader_history_keys_on_occurrence(self):
        panel = ChartPanel()
        train = list(range(len(panel.records)))
        mode_of = {i: panel.records[i]['mode'] for i in train}
        exposure = build_exposure(train, steps=20, seed=2, mode_weights={'phira': 2., 'mania_4k': 1.},
                                  mode_of=mode_of)
        dataset = CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=dict(p=1.))
        step = exposure['steps'][0]
        for slot, index in enumerate(step['indices']):
            _, audit = dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                curve_specified=step['curve_specified'][slot],
                specified_fields=step['specified_condition_fields'][slot], history_key=step['occurrences'][slot])
            if step['rates'][slot] is None:
                self.assertEqual(audit['history']['drawn'],
                                 draw_history(dataset.history, 2, index, step['occurrences'][slot]))
        self.assertEqual(len(dataset[0]['batches']), 1)



class ApplyContextTrainingTest(unittest.TestCase):
    def test_rebuild_reproduces_then_reweights(self):
        import tempfile
        from pathlib import Path
        from scripts.apply_context_training import apply
        records = [dict(split='train' if i % 5 else 'val', mode='phira' if i % 3 else 'mania_4k',
                        audio_sha256=AUDIO[i % 2], request=dict(start_us=i)) for i in range(60)]
        pool = hashlib.sha256(json.dumps(records, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        train = [i for i, r in enumerate(records) if r['split'] == 'train']
        original = build_exposure(train, steps=30, seed=4, pool_identity=pool, song_chunk=2,
                                  song_of={i: records[i]['audio_sha256'] for i in train})
        with tempfile.TemporaryDirectory() as root:
            (Path(root)/'exposure.json').write_text(json.dumps(original))
            plan = dict(exposure='exposure.json', tape='unused', inputs_sha256={})
            same, exposure = apply(plan, root=root, history_context={}, records=records)
            self.assertIsNone(exposure)
            self.assertEqual(same['history_context'], normalize_history_context({}))
            new, exposure = apply(plan, root=root, mode_weights={'phira': 1.6, 'mania_4k': 1.}, records=records)
            self.assertEqual(len(exposure['steps']), 30)
            self.assertEqual(new['mode_weights'], {'mania_4k': 1., 'phira': 1.6})
            self.assertIn('mode_mix', new['counterfactual']['realized'])
            tampered = dict(original, seed=5)
            (Path(root)/'exposure.json').write_text(json.dumps(tampered))
            with self.assertRaisesRegex(ValueError, 'reproduce'):
                apply(plan, root=root, history_context={}, records=records)


if __name__ == '__main__':
    unittest.main()
