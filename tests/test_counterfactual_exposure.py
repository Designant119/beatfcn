import json
import math
import unittest
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.model.full_event_codec import EventRequest, EventState, encode_events
from beatfcn.train.counterfactual_density import (achieved_density_segments,
    edit_events_to_multiplier, scaled_count)
from beatfcn.train.counterfactual_exposure import (CONTROL_FIELDS, FORMAT, build_exposure,
    exposure_summary, normalize_mix, sample_conditions, select_validation, target_conditions,
    treatment_deck)
from beatfcn.train.density_curve import encode_density
from beatfcn.train.ema import ParameterEMA
from beatfcn.train.event_conditions import encode_controls
from beatfcn.train.exposure_loader import (CounterfactualExposureDataset, build_exposure_loader,
    make_vocabulary, request_sequence)
from beatfcn.train.local_panel import collate

ROOT = Path(__file__).resolve().parents[1]


class ExposureCoverageTest(unittest.TestCase):
    def test_every_request_gets_every_treatment_and_order_is_shuffled(self):
        train = list(range(10, 50))
        mix = normalize_mix(None)
        deck = treatment_deck(mix)
        self.assertEqual(len(deck), 20)
        steps = len(train) * len(deck) // 4  # exactly one deck cycle = 20 epochs
        exposure = build_exposure(train, steps=steps, seed=7)
        self.assertEqual(exposure['format'], FORMAT)
        self.assertEqual(exposure['complete_epochs'], 20)
        per_request = {}
        epochs = {}
        for step in exposure['steps']:
            self.assertEqual(len(step['indices']), 4)
            for k, request in enumerate(step['indices']):
                per_request.setdefault(request, Counter())[step['treatments'][k]] += 1
                epochs.setdefault(step['epochs'][k], []).append(request)
        expected = {item['name']: round(item['probability'] * 20) for item in mix}
        for request in train:
            self.assertEqual(dict(per_request[request]), expected)
        orders = [epochs[e] for e in sorted(epochs)]
        for order in orders:
            self.assertEqual(sorted(order), train)  # each epoch is a permutation
        self.assertGreater(len({tuple(order) for order in orders}), 15)
        self.assertNotEqual(orders[0], train)
        # Parity bug regression: windows land in every slot, native in every slot.
        native_slots = Counter(k for step in exposure['steps'] for k in range(4)
                               if step['rates'][k] is None)
        self.assertEqual(set(native_slots), {0, 1, 2, 3})

    def test_treatment_frequencies_and_dropout_rates_within_tolerance(self):
        exposure = build_exposure(list(range(2000)), steps=5000, seed=3)
        summary = exposure_summary(exposure)
        for item in normalize_mix(None):
            self.assertAlmostEqual(summary['treatment_fraction'][item['name']], item['probability'], delta=.01)
        self.assertAlmostEqual(summary['unconditional_fraction'], .2, delta=.015)
        self.assertAlmostEqual(summary['curve_hidden_given_conditional'], .3, delta=.015)
        self.assertAlmostEqual(summary['control_hidden_given_conditional'], .3, delta=.01)
        for step in exposure['steps']:
            for k in range(4):
                if step['unconditional'][k]:
                    self.assertFalse(step['curve_specified'][k])
                    self.assertEqual(step['specified_condition_fields'][k], [])

    def test_exposure_is_deterministic_and_seed_dependent(self):
        a = build_exposure(list(range(30)), steps=40, seed=1)
        b = build_exposure(list(range(30)), steps=40, seed=1)
        c = build_exposure(list(range(30)), steps=40, seed=2)
        self.assertEqual(a, b)
        self.assertNotEqual(a['steps'], c['steps'])

    def test_iid_fallback_for_irrational_mix(self):
        mix = [dict(name='native', multiplier=None, probability=.1234567),
               dict(name='x2', multiplier=2., probability=1-.1234567)]
        self.assertIsNone(treatment_deck(normalize_mix(mix)))
        exposure = build_exposure(list(range(100)), steps=500, seed=0, mix=mix)
        self.assertAlmostEqual(exposure_summary(exposure)['treatment_fraction']['native'], .1234567, delta=.03)


class RoundingAndCarryTest(unittest.TestCase):
    def test_round_half_up_keeps_one_head_in_nonempty_cells(self):
        self.assertEqual(scaled_count(1, .5), 1)
        self.assertEqual(scaled_count(3, .5), 2)
        self.assertEqual(scaled_count(5, .5), 3)
        self.assertEqual(scaled_count(4, .5), 2)
        self.assertEqual(scaled_count(0, 2.), 0)
        self.assertEqual(scaled_count(3, 0.), 0)
        events = [RhythmEvent(i*500_000 + 100_000, i*500_000 + 100_000, 'tap', 0) for i in range(4)]
        thin, _ = edit_events_to_multiplier(events, 0, 2_000_000, .5, 'mania_4k', 8_000_000)
        self.assertEqual(len(thin), 4)

    def test_mania_insertions_respect_incoming_holds(self):
        carry = tuple(RhythmEvent(100_000, 1_900_000, 'hold', lane) for lane in (2, 3))
        events = [RhythmEvent(500_000, 1_200_000, 'hold', 0), RhythmEvent(900_000, 900_000, 'tap', 1),
                  RhythmEvent(1_300_000, 1_300_000, 'tap', 1), RhythmEvent(1_500_000, 1_500_000, 'tap', 0)]
        request = EventRequest('mania_4k', 500_000, 2_000_000, 8_000_000, known_carry=carry)
        thick, _ = edit_events_to_multiplier(events, 500_000, 2_000_000, 2., 'mania_4k',
                                             8_000_000, seed=0, carry=carry)
        added = [e for e in thick if e not in events]
        self.assertTrue(added)
        self.assertFalse([e for e in added if e.lane in (2, 3) and e.onset_us < 1_900_000])
        encode_events(thick, request)  # legal with the carry kept in the request
        unaware, _ = edit_events_to_multiplier(events, 500_000, 2_000_000, 2., 'mania_4k',
                                               8_000_000, seed=0)
        self.assertTrue([e for e in unaware if e.lane in (2, 3) and e.onset_us < 1_900_000])
        with self.assertRaises(ValueError):
            encode_events(unaware, request)


class ConditionTest(unittest.TestCase):
    def setUp(self):
        self.request = EventRequest('mania_4k', 8_000_000, 16_000_000, 40_000_000,
            known_carry=(RhythmEvent(7_000_000, 9_000_000, 'hold', 3),))
        self.events = [RhythmEvent(t, t, 'tap', (t // 100_000) % 3)
                       for t in range(8_000_000, 12_000_000, 100_000)]
        self.events.append(RhythmEvent(13_000_000, 14_000_000, 'hold', 0))

    def test_curve_is_the_achieved_two_second_density(self):
        update, audit = sample_conditions('mania_4k', self.events, self.request,
            curve_specified=True, specified_fields=list(range(8)))
        segments = achieved_density_segments(self.events, 8_000_000, 16_000_000)
        self.assertEqual([s['density'] for s in segments], [10., 10., .5, 0.])
        self.assertEqual(audit['curve_segments'], segments)
        self.assertEqual(update['intensity'].shape, (80, 3))
        self.assertTrue(update['intensity_valid'].all())
        expected = np.repeat(encode_density([s['density'] for s in segments]), 20)
        np.testing.assert_allclose(update['intensity'][:, 0], expected)
        hidden, _ = sample_conditions('mania_4k', self.events, self.request,
            curve_specified=False, specified_fields=[])
        self.assertFalse(hidden['intensity_valid'].any())

    def test_controls_are_recomputed_from_the_actual_target(self):
        edited, _ = edit_events_to_multiplier(self.events, 8_000_000, 16_000_000, .5,
            'mania_4k', 40_000_000, seed=1, carry=self.request.known_carry)
        bands, raw = target_conditions('mania_4k', edited, self.request, carry=self.request.known_carry)
        self.assertAlmostEqual(raw['density_notes_per_second'], len(edited) / 8)
        self.assertGreater(raw['hold_occupancy_ratio'], 1_000_000 / 8_000_000)  # carry counted
        update, _ = sample_conditions('mania_4k', edited, self.request,
            curve_specified=True, specified_fields=list(range(8)))
        np.testing.assert_array_equal(update['controls'],
            encode_controls('mania_4k', bands, (8_000_000, 16_000_000), origin_us=8_000_000))
        native, _ = target_conditions('mania_4k', self.events, self.request, carry=self.request.known_carry)
        self.assertNotEqual(native['density_notes_per_second'], bands['density_notes_per_second'])

    def test_hidden_fields_keep_applicability_like_unconditional_eval_rows(self):
        keep = [CONTROL_FIELDS.index('density_notes_per_second')]
        update, audit = sample_conditions('mania_4k', self.events, self.request,
            curve_specified=False, specified_fields=keep)
        controls = update['controls']
        self.assertEqual(audit['shown_fields'], ['density_notes_per_second'])
        self.assertEqual(controls[:, 2].sum(), 1)
        unconditional = encode_controls('mania_4k', {}, (8_000_000, 16_000_000), origin_us=8_000_000)
        np.testing.assert_array_equal(np.delete(controls, keep, 0), np.delete(unconditional, keep, 0))
        np.testing.assert_array_equal(controls[:, 3:], unconditional[:, 3:])

    def test_native_controls_match_tape_builder_controls(self):
        tape = ROOT/'inputs/native-v2-d94-580/tape.json'
        if not tape.is_file():
            self.skipTest('local native tape absent')
        from beatfcn.train.local_request_panel import event_request
        charts = json.loads(tape.read_text())['charts']
        checked = 0
        for chart in charts:
            if chart['split'] != 'train' or checked >= 40:
                continue
            with np.load(ROOT/chart['target_path']) as arrays:
                values = {k: arrays[k] for k in arrays.files}
            for row in chart['requests'][:3]:
                request = event_request(row['request'])
                sl = slice(row['target_start'], row['target_stop'])
                from beatfcn.events.canonical import TYPES
                events = [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                          zip(*(values[key][sl] for key in ('onset_us', 'end_us', 'kind', 'lane')))]
                bands, _ = target_conditions(request.mode, events, request, carry=request.known_carry)
                np.testing.assert_allclose(encode_controls(request.mode, bands,
                    (request.start_us, request.end_us), origin_us=request.start_us),
                    np.asarray(row['controls'], np.float32))
                checked += 1
        self.assertGreater(checked, 20)


class FakePanel:
    """Minimal panel: real targets/requests, synthetic audio inputs."""

    def __init__(self):
        self.records = []
        self.events = []
        for index in range(8):
            mode = 'mania_4k' if index % 2 else 'phira'
            start = 8_000_000 * (index % 3)
            carry = []
            if mode == 'mania_4k' and index % 3:
                carry = [dict(onset_us=start - 500_000, end_us=start + 1_500_000, type='hold', lane=2)]
            request = dict(mode=mode, start_us=start, end_us=start + 8_000_000,
                           duration_us=40_000_000, known_carry=carry)
            if mode == 'mania_4k':
                events = [RhythmEvent(t, t, 'tap', (t // 250_000) % 2)
                          for t in range(start, start + 8_000_000, 250_000)]
            else:
                events = [RhythmEvent(t, t, 'tap' if t % 500_000 else 'drag', -1)
                          for t in range(start, start + 8_000_000, 250_000)]
            self.events.append(events)
            self.records.append(dict(split='train', request=request, mode=mode,
                                     chart_index=index, controls=np.zeros((8, 6), np.float32)))

    def targets(self, index):
        return list(self.events[index])

    def row(self, index, controls=None):
        local = 300 + index
        start = self.records[index]['request']['start_us']
        times = np.arange(local) * .04 - 4 + .02
        mel = np.arange(local * 4) * .01 - 4 + .005
        return dict(inputs=dict(
            local=np.ones((2, local, 1, 4), np.float32), local_times=times.astype(np.float32),
            local_valid=np.ones(local, bool), mel=np.ones((local * 4, 2), np.float32),
            mel_times=mel.astype(np.float32), mel_valid=np.ones(local * 4, bool),
            song=np.ones((6, 4), np.float32), song_times=np.zeros(6, np.float32),
            song_valid=np.ones(6, bool), reference=np.zeros((1, 16), np.float32),
            reference_times=np.zeros(1, np.float32), reference_valid=np.ones(1, bool),
            controls=np.zeros((8, 6), np.float32), mode=np.asarray(0, np.int64)),
            sequence=None, start=start)

    def sequence(self, index):
        from beatfcn.train.local_request_panel import event_request
        request = event_request(self.records[index]['request'])
        return request_sequence(self, index, request, make_vocabulary({}), events=self.targets(index))


class CounterfactualDatasetTest(unittest.TestCase):
    def dataset(self, **kwargs):
        exposure = build_exposure(list(range(8)), steps=40, seed=5)
        exposure['pool_identity'] = 'fake'
        return CounterfactualExposureDataset(FakePanel(), exposure, microbatch=4, **kwargs), exposure

    def test_items_condition_on_their_own_edited_target(self):
        dataset, exposure = self.dataset()
        seen = Counter()
        for index in range(len(dataset)):
            step = exposure['steps'][index]
            for slot, request_index in enumerate(step['indices']):
                row, audit = dataset.sample(request_index, step['rates'][slot],
                    edit_seed=step['edit_seeds'][slot], curve_specified=step['curve_specified'][slot],
                    specified_fields=step['specified_condition_fields'][slot])
                seen[step['treatments'][slot]] += 1
                request = audit['request']
                self.assertEqual(request.known_carry, tuple(RhythmEvent(**e) for e in
                    dataset.panel.records[request_index]['request']['known_carry']))
                if step['rates'][slot] == 0:
                    self.assertEqual(audit['events'], [])
                bands, _ = target_conditions(request.mode, audit['events'], request,
                                             carry=request.known_carry)
                self.assertEqual(audit['bands'], bands)
                if step['curve_specified'][slot]:
                    counts = [sum(lo <= e.onset_us < lo + 2_000_000 for e in audit['events'])
                              for lo in range(request.start_us, request.end_us, 2_000_000)]
                    self.assertEqual([s['density'] for s in audit['curve_segments']],
                                     [c / 2 for c in counts])
                else:
                    self.assertFalse(row['inputs']['intensity_valid'].any())
                state = EventState(request)
                for token in row['sequence']['targets'][row['sequence']['valid']].tolist():
                    state.apply(int(token))
                self.assertEqual(sorted(state.events), sorted(audit['events']))
        self.assertGreaterEqual(len(seen), 4)
        item = dataset[0]
        ids, inputs, sequence = item['batches'][0]
        self.assertEqual(tuple(inputs['controls'].shape), (4, 8, 6))
        self.assertEqual(inputs['intensity'].shape[:2], (4, 80))

    def test_relative_block_vocabulary_threads_through(self):
        try:
            vocab = make_vocabulary(dict(relative_blocks=True, tail_query=True))
        except ValueError:
            self.skipTest('relative_blocks vocabulary not available')
        dataset, exposure = self.dataset(vocabulary=dict(relative_blocks=True, tail_query=True))
        step = exposure['steps'][0]
        for slot, request_index in enumerate(step['indices']):
            row, audit = dataset.sample(request_index, step['rates'][slot], edit_seed=1,
                                        curve_specified=True, specified_fields=[])
            state = EventState(audit['request'], vocab)
            for token in row['sequence']['targets'][row['sequence']['valid']].tolist():
                state.apply(int(token))
            self.assertEqual(sorted(state.events), sorted(audit['events']))

    def test_onset_targets_follow_the_actual_target_and_pad_with_the_stream(self):
        try:
            from beatfcn.train.training_objective import onset_frame_targets  # noqa: F401
        except ImportError:
            self.skipTest('onset helper not available')
        dataset, exposure = self.dataset(onset=dict(scope='core'), mel_bins=128)
        row, audit = dataset.sample(1, 2.0, edit_seed=0, curve_specified=True, specified_fields=[])
        targets, valid = row['sequence']['onset_targets'], row['sequence']['onset_valid']
        self.assertEqual(len(targets), len(row['inputs']['mel']))
        times = row['inputs']['mel_times']
        self.assertFalse(valid[times < 0].any())
        # scope='core': only the core's actual target is passed, so every hit is one of its onsets.
        hits = sorted(set(np.round(times[targets[:, 0] == 1] * 1e6).astype(int)))
        self.assertEqual(len(hits), len({e.onset_us for e in audit['events']}))
        item = dataset[0]
        _, inputs, sequence = item['batches'][0]
        self.assertEqual(sequence['onset_targets'].shape[:2], inputs['mel'].shape[:2])
        self.assertEqual(sequence['onset_valid'].shape, inputs['mel_valid'].shape)
        for b in range(len(item['batches'][0][0])):
            n = int(inputs['mel_valid'][b].sum())
            self.assertFalse(sequence['onset_valid'][b, n:].any())
        local, _ = CounterfactualExposureDataset(FakePanel(), exposure, microbatch=4,
            onset=dict(scope='window'), mel_bins=0).sample(1, None, edit_seed=0,
            curve_specified=False, specified_fields=[])
        self.assertEqual(len(local['sequence']['onset_targets']), local['inputs']['local'].shape[1])

    def test_grouped_microbatch_keeps_onset_alignment(self):
        try:
            from beatfcn.train.training_objective import onset_frame_targets  # noqa: F401
        except ImportError:
            self.skipTest('onset helper not available')
        exposure = build_exposure(list(range(8)), steps=4, seed=2)
        loader = build_exposure_loader(FakePanel(), exposure, microbatch=8, num_workers=0,
            pin_memory=False, onset=dict(scope='window'), mel_bins=128)
        items = list(loader)
        self.assertEqual([item['step'] for item in items], [2, 4])
        _, inputs, sequence = items[0]['batches'][0]
        self.assertEqual(sequence['onset_targets'].shape[:2], inputs['mel'].shape[:2])
        self.assertEqual(len(items[0]['indices']), 8)

    def test_legacy_exposure_is_rejected(self):
        legacy = dict(steps=[dict(effective_update=1, indices=[0, 1, 2, 3], rates=[None]*4)])
        with self.assertRaisesRegex(ValueError, 'v2'):
            CounterfactualExposureDataset(FakePanel(), legacy, microbatch=4)
        with self.assertRaisesRegex(ValueError, 'retired'):
            build_exposure_loader(FakePanel(), legacy, microbatch=4, num_workers=0, pin_memory=False)


class CollateOnsetTest(unittest.TestCase):
    def test_onset_rows_pad_to_fixed_mel_shape(self):
        panel = FakePanel()
        rows = []
        for index in (0, 1):
            row = panel.row(index)
            frames = len(row['inputs']['mel'])
            sequence = dict(panel.sequence(index), onset_targets=np.ones((frames, 5), np.float32),
                            onset_valid=np.ones(frames, bool))
            rows.append(dict(inputs=row['inputs'], sequence=sequence))
        inputs, sequence = collate(rows, input_shapes=dict(mel=1300, local=330))
        self.assertEqual(tuple(sequence['onset_targets'].shape), (2, 1300, 5))
        self.assertEqual(int(sequence['onset_valid'][0].sum()), 1200)
        self.assertEqual(int(sequence['onset_valid'][1].sum()), 1204)
        rows[0]['sequence']['onset_valid'] = np.ones(7, bool)
        with self.assertRaises(ValueError):
            collate(rows)


class ValidationSelectionTest(unittest.TestCase):
    def test_stratified_over_modes_and_all_charts(self):
        records = []
        for chart in range(10):
            mode = 'phira' if chart < 6 else 'mania_4k'
            for window in range(30 if chart else 3):
                records.append(dict(split='val', mode=mode, chart_id=f'c{chart}',
                                    event_count=window % 4))
        records += [dict(split='train', mode='phira', chart_id='t', event_count=1)] * 5
        chosen, summary = select_validation(records, count=100, seed=1)
        self.assertEqual(chosen, select_validation(records, count=100, seed=1)[0])
        self.assertEqual(len(chosen), 100)
        self.assertEqual(Counter(records[i]['mode'] for i in chosen), {'phira': 50, 'mania_4k': 50})
        self.assertEqual(summary['charts_covered'], 10)
        self.assertTrue(all(records[i]['split'] == 'val' for i in chosen))
        per_chart = Counter(records[i]['chart_id'] for i in chosen)
        self.assertEqual(per_chart['c0'], 3)
        self.assertLessEqual(max(per_chart[f'c{c}'] for c in range(1, 6)) -
                             min(per_chart[f'c{c}'] for c in range(1, 6)), 1)
        self.assertGreater(summary['empty_windows'], 0)


class EmaTest(unittest.TestCase):
    def test_bias_corrected_average_matches_closed_form(self):
        model = torch.nn.Linear(3, 2)
        decay = .9
        ema = ParameterEMA(model, decay)
        history = []
        for step in range(1, 30):
            with torch.no_grad():
                for p in model.parameters():
                    p.add_(torch.randn_like(p))
            history.append([p.detach().clone() for p in model.parameters()])
            ema.update()
            weights = torch.tensor([decay ** (step - 1 - k) for k in range(step)])
            weights /= weights.sum()
            for i, shadow in enumerate(ema.shadow):
                expected = sum(w * h[i] for w, h in zip(weights, history))
                torch.testing.assert_close(shadow, expected, rtol=1e-5, atol=1e-5)
        self.assertAlmostEqual(ema.coefficient(1), 0.)
        self.assertAlmostEqual(ema.coefficient(10_000), decay)

    def test_state_roundtrip_and_swap_restores_live_weights(self):
        model = torch.nn.Linear(3, 2)
        ema = ParameterEMA(model, .99)
        with torch.no_grad():
            model.weight.add_(1.)
        ema.update(); ema.update()
        state = ema.state_dict(model)
        self.assertEqual(set(state['model']), set(model.state_dict()))
        live = model.weight.detach().clone()
        with ema.swapped():
            torch.testing.assert_close(model.weight, ema.shadow[0])
        torch.testing.assert_close(model.weight, live)
        other = ParameterEMA(model, .99)
        other.load_state_dict(state, model)
        self.assertEqual(other.updates, 2)
        torch.testing.assert_close(other.shadow[0], ema.shadow[0])
        with self.assertRaises(ValueError):
            ParameterEMA(model, .5).load_state_dict(state, model)


class PlanBudgetTest(unittest.TestCase):
    def test_budget_sets_steps_from_optimizer_updates(self):
        from types import SimpleNamespace
        from scripts.prepare_native_counterfactual_training import budget, every
        self.assertEqual(budget(SimpleNamespace(accum_steps=16, optimizer_updates=2000, steps=None)),
                         (2000, 32000))
        self.assertEqual(budget(SimpleNamespace(accum_steps=4, optimizer_updates=None, steps=400)),
                         (100, 400))
        with self.assertRaises(ValueError):
            budget(SimpleNamespace(accum_steps=4, optimizer_updates=None, steps=None))
        with self.assertRaises(ValueError):
            budget(SimpleNamespace(accum_steps=4, optimizer_updates=None, steps=402))
        self.assertEqual(every(250, 16), 4000)

    def test_cosine_reaches_minimum_within_the_declared_updates(self):
        from beatfcn.train.training_objective import update_learning_rate
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.)
        lrs = [update_learning_rate(optimizer, n, 2000, base_lr=1e-4, warmup_updates=100,
                                    minimum_lr=1e-5) for n in range(2000)]
        self.assertTrue(math.isclose(lrs[-1], 1e-5))
        self.assertTrue(math.isclose(max(lrs), 1e-4))


if __name__ == '__main__':
    unittest.main()


class SongChunkExposureTests(unittest.TestCase):
    def test_chunked_epoch_is_a_permutation_with_song_locality(self):
        from beatfcn.train.counterfactual_exposure import build_exposure
        train = list(range(200))
        song_of = {i: f'song{i // 25}' for i in train}
        exposure = build_exposure(train, steps=50, seed=3, song_of=song_of, song_chunk=8)
        seen = [i for step in exposure['steps'] for i in step['indices']]
        self.assertEqual(sorted(seen), train)
        items = [seen[k:k+8] for k in range(0, len(seen), 8)]
        songs_per_item = [len({song_of[i] for i in item}) for item in items]
        self.assertLessEqual(sum(songs_per_item) / len(items), 1.6)
        plain = build_exposure(train, steps=50, seed=3)
        plain_seen = [i for step in plain['steps'] for i in step['indices']]
        plain_items = [len({song_of[i] for i in plain_seen[k:k+8]}) for k in range(0, 200, 8)]
        self.assertGreater(sum(plain_items) / len(plain_items), 4)
        self.assertEqual(exposure['song_chunk'], 8)


class UnconditionalNativeTests(unittest.TestCase):
    def test_unconditional_samples_use_native_targets(self):
        from beatfcn.train.counterfactual_exposure import build_exposure
        train = list(range(400))
        exposure = build_exposure(train, steps=400, seed=5, unconditional_native=True)
        slots = [(s, k) for s in exposure['steps'] for k in range(len(s['indices']))]
        unconditional = [(s, k) for s, k in slots if s['unconditional'][k]]
        self.assertTrue(unconditional)
        self.assertTrue(all(s['rates'][k] is None for s, k in unconditional))
        conditional = [(s, k) for s, k in slots if not s['unconditional'][k]]
        self.assertTrue(any(s['rates'][k] is not None for s, k in conditional))
        self.assertTrue(exposure['unconditional_native'])
