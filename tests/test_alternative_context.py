import hashlib
import unittest

import numpy as np
import torch

from beatfcn.events.canonical import TYPES, Event, LaneLoc
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train.alternative_context import (ROLE, SiblingIndex, alternative_source, check_alternative,
                                               draw_alternative, normalize_alternative_context,
                                               reference_inputs_with_alternative)
from beatfcn.train.counterfactual_exposure import build_exposure
from beatfcn.train.event_reference import SOURCE_COLUMN, VisibleReference, encode_reference
from beatfcn.train.exposure_loader import (CounterfactualExposureDataset, build_exposure_loader,
                                           make_vocabulary, plan_loader_settings, request_sequence)
from beatfcn.train.local_request_panel import event_request

AUDIO = ['%064x' % (i + 1) for i in range(2)]
WINDOW = 8_000_000
DURATION = 32_000_000


def _mania(step_us, lane_shift=0):
    rows = [(t, t, 0, (t // step_us + lane_shift) % 3) for t in range(0, DURATION, step_us)]  # lane 3 is held
    rows.append((3_000_000, 9_500_000, 1, 3))  # a hold crossing the 8 s window boundary
    rows.sort()
    return {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


def _phira():
    rows = [(t, t, 0, -1) for t in range(0, DURATION, 250_000)]
    return {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


class SiblingPanel:
    """One mania song with dense / medium / sparse train charts, an exact duplicate of medium,
    a val chart on the same audio, and a lone Phira chart; every 8 s window of a train chart is a request."""

    def __init__(self):
        self.charts = [
            dict(chart_id='osu:dense', mode='mania_4k', audio_sha256=AUDIO[0], split='train', canonical_sha256='c-dense'),
            dict(chart_id='osu:medium', mode='mania_4k', audio_sha256=AUDIO[0], split='train', canonical_sha256='c-med'),
            dict(chart_id='osu:sparse', mode='mania_4k', audio_sha256=AUDIO[0], split='train', canonical_sha256='c-sparse'),
            dict(chart_id='osu:medium-dup', mode='mania_4k', audio_sha256=AUDIO[0], split='train', canonical_sha256='c-med'),
            dict(chart_id='osu:val', mode='mania_4k', audio_sha256=AUDIO[0], split='val', canonical_sha256='c-val'),
            dict(chart_id='phira:lone', mode='phira', audio_sha256=AUDIO[1], split='train', canonical_sha256='c-ph'),
        ]
        self.arrays = [_mania(125_000), _mania(250_000), _mania(500_000, 1), _mania(250_000), _mania(200_000), _phira()]
        self.records = []
        for chart_index, chart in enumerate(self.charts):
            if chart['split'] != 'train':
                continue
            a = self.arrays[chart_index]
            for start in range(0, DURATION, WINDOW):
                end = start + WINDOW
                inside = np.flatnonzero((a['onset_us'] >= start) & (a['onset_us'] < end))
                crossing = (a['kind'] == 1) & (a['onset_us'] < start) & (a['end_us'] > start)
                carry = [dict(onset_us=int(a['onset_us'][i]), end_us=int(a['end_us'][i]), type='hold',
                              lane=int(a['lane'][i])) for i in np.flatnonzero(crossing)]
                self.records.append(dict(split='train', chart_index=chart_index, chart_id=chart['chart_id'],
                    audio_sha256=chart['audio_sha256'], mode=chart['mode'], controls=np.zeros((8, 6), np.float32),
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


class AlternativeContextTest(unittest.TestCase):
    def dataset(self, alternative=None, history=None, steps=60, seed=3):
        panel = SiblingPanel()
        exposure = build_exposure(list(range(len(panel.records))), steps=steps, seed=seed)
        return CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=history,
                                             alternative_context=alternative), exposure

    def sample(self, dataset, step, slot, index):
        key = step.get('occurrences', step['epochs'])[slot]
        return dataset.sample(index, step['rates'][slot], edit_seed=step['edit_seeds'][slot],
                              curve_specified=step['curve_specified'][slot],
                              specified_fields=step['specified_condition_fields'][slot],
                              history_key=key if dataset.history is not None else None,
                              alternative_key=key)

    def test_settings(self):
        self.assertIsNone(normalize_alternative_context(None))
        self.assertIsNone(normalize_alternative_context(False))
        self.assertEqual(normalize_alternative_context(True)['p'], .35)
        for bad in (dict(p=1.5), dict(denser=-.1), dict(pad_s=-1), dict(max_reference_rows=2), dict(oops=1)):
            with self.assertRaises(ValueError):
                normalize_alternative_context(bad)

    def test_sibling_index_excludes_self_duplicates_val_and_other_modes(self):
        index = SiblingIndex(SiblingPanel())
        self.assertEqual(index.songs_with_siblings(), 1)                      # Phira song has no sibling
        names = lambda i: sorted(index.panel.charts[j]['chart_id'] for j in index.siblings(i))
        self.assertEqual(names(1), ['osu:dense', 'osu:sparse'])              # medium: not itself, not its duplicate
        self.assertEqual(names(0), ['osu:medium', 'osu:medium-dup', 'osu:sparse'])
        self.assertNotIn('osu:val', names(0) + names(1) + names(2))
        self.assertEqual(index.siblings(5), [])

    def test_draws_are_deterministic_and_respect_p(self):
        settings = normalize_alternative_context(dict(p=.35))
        a = [draw_alternative(settings, 3, r, k) for r in range(40) for k in range(25)]
        b = [draw_alternative(settings, 3, r, k) for r in range(40) for k in range(25)]
        self.assertEqual(a, b)
        self.assertAlmostEqual(np.mean([d['applied'] for d in a]), .35, delta=.04)
        self.assertNotEqual(a, [draw_alternative(settings, 4, r, k) for r in range(40) for k in range(25)])

    def test_direction_and_region(self):
        panel = SiblingPanel()
        index = SiblingIndex(panel)
        settings = normalize_alternative_context(dict(p=1., denser=1., pad_s=2.))
        medium = next(i for i, r in enumerate(panel.records) if r['chart_id'] == 'osu:medium' and r['request']['start_us'] == 8_000_000)
        request = event_request(panel.records[medium]['request'])
        source, info = alternative_source(index, request, panel.records[medium], dict(applied=True, denser=True, pick=.5), settings)
        self.assertEqual(info['sibling'], 'osu:dense')
        self.assertEqual(info['direction'], 'denser')
        self.assertEqual(info['region_us'], [6_000_000, 18_000_000])
        self.assertTrue(all(6_000_000 <= e.onset_us < 18_000_000 for e in source.events))
        self.assertGreater(info['window_events'], 0)
        dense = next(i for i, r in enumerate(panel.records) if r['chart_id'] == 'osu:dense')
        _, info = alternative_source(index, event_request(panel.records[dense]['request']), panel.records[dense],
                                     dict(applied=True, denser=True, pick=.0), settings)
        self.assertEqual(info['direction'], 'sparser_or_equal')               # densest chart falls back
        _, info = alternative_source(index, request, panel.records[medium], dict(applied=True, denser=False, pick=.0), settings)
        self.assertEqual(info['sibling'], 'osu:sparse')
        # nearest-difficulty preference: the sparse chart's denser side is {dense, medium, medium-dup}
        sparse = next(i for i, r in enumerate(panel.records) if r['chart_id'] == 'osu:sparse')
        req = event_request(panel.records[sparse]['request'])
        _, info = alternative_source(index, req, panel.records[sparse], dict(applied=True, denser=True, pick=.1), settings)
        self.assertEqual((info['choice'], info['sibling']), ('nearest', 'osu:medium'))
        _, info = alternative_source(index, req, panel.records[sparse], dict(applied=True, denser=True, pick=.65), settings)
        self.assertEqual((info['choice'], info['sibling']), ('uniform', 'osu:dense'))

    def test_encoder_overlap_only_for_human_alternative(self):
        events = (Event('x:1', 9_000_000, 9_000_000, 'tap', LaneLoc(0)),)
        for role, ok in (('human_input', False), ('model_history', False), (ROLE, True)):
            source = VisibleReference(source_id='alternative:osu:dense' if role == ROLE else 'native-prefix:x',
                                      mode='mania_4k', audio_sha256=AUDIO[0], role=role,
                                      regions=((6_000_000, 18_000_000),), events=events)
            call = lambda: encode_reference(mode='mania_4k', audio_sha256=AUDIO[0], duration_us=DURATION,
                                            origin_us=8_000_000, unknown_regions=((8_000_000, 16_000_000),),
                                            sources=(source,))
            if ok:
                reference, provenance = call()
                rows = reference['reference']
                self.assertTrue((rows[:, SOURCE_COLUMN[ROLE]] == 1).any())
            else:
                with self.assertRaises(ValueError):
                    call()
        bad = VisibleReference(source_id='alternative:osu:medium', mode='mania_4k', audio_sha256=AUDIO[0],
                               role=ROLE, regions=((6_000_000, 18_000_000),), events=events)
        with self.assertRaises(ValueError):                                    # never the target's own chart
            check_alternative((bad,), dict(chart_id='osu:medium'))

    def test_samples_show_siblings_never_the_target_and_edits_get_none(self):
        dataset, exposure = self.dataset(alternative=dict(p=1., denser=.7), history=dict(p=.8, p_infill=.5))
        shown = edited = 0
        for step, slot, index in _slots(exposure):
            row, audit = self.sample(dataset, step, slot, index)
            alt = audit['alternative']
            record = dataset.panel.records[index]
            if step['rates'][slot] is not None:
                self.assertFalse(alt['applied']); edited += 1
                continue
            if record['chart_id'] == 'phira:lone':
                self.assertEqual(alt['reason'], 'no_sibling')
                continue
            self.assertTrue(alt['applied'])
            self.assertNotEqual(alt['sibling'], record['chart_id'])
            self.assertNotIn(alt['sibling'], ('osu:val',))
            if record['chart_id'] in ('osu:medium', 'osu:medium-dup'):
                self.assertNotIn(alt['sibling'], ('osu:medium', 'osu:medium-dup'))
            ref = row['inputs']['reference']
            role = ref[:, SOURCE_COLUMN[ROLE]] == 1
            events = ref[:, 10] == 1
            rel = ref[:, 0] * 1_200_000_000 / 1e6                              # seconds from the window start
            inside = (rel >= 0) & (rel < 8)
            self.assertTrue((role & events & inside).any())                   # sibling notes inside the window
            self.assertFalse((~role & events & inside).any())                 # never same-chart notes inside
            shown += 1
        self.assertGreater(shown, 20)
        self.assertGreater(edited, 0)

    def test_off_and_p0_are_byte_identical(self):
        plain, exposure = self.dataset()
        panel = plain.panel
        legacy = CounterfactualExposureDataset(panel, exposure, microbatch=4)
        reference = _digest([legacy[i] for i in range(len(legacy))])
        self.assertEqual(_digest([plain[i] for i in range(len(plain))]), reference)
        never = CounterfactualExposureDataset(panel, exposure, microbatch=4, alternative_context=dict(p=0.))
        self.assertEqual(_digest([never[i] for i in range(len(never))]), reference)
        # history on: alternative p=0 goes through the merged encoder and must equal history-only
        history = dict(p=.8, p_infill=.5)
        h_only = CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=history)
        h_alt0 = CounterfactualExposureDataset(panel, exposure, microbatch=4, history_context=history,
                                               alternative_context=dict(p=0.))
        self.assertEqual(_digest([h_alt0[i] for i in range(len(h_alt0))]), _digest([h_only[i] for i in range(len(h_only))]))
        plan = dict(config=dict(mel_bins=0))
        self.assertNotIn('alternative_context', plan_loader_settings(plan))
        self.assertEqual(plan_loader_settings(dict(plan, alternative_context=dict(p=.5)))['alternative_context']['p'], .5)

    def test_loader_collate_and_model_forward(self):
        from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
        dataset, exposure = self.dataset(alternative=dict(p=1.), history=dict(p=1., p_infill=.5))
        loader = build_exposure_loader(dataset.panel, exposure, microbatch=8, num_workers=0, pin_memory=False,
                                       history_context=dict(p=1., p_infill=.5), alternative_context=dict(p=1.))
        items = list(loader)
        config = FullEventConfig(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=1,
                                 tap_dim=4, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=2, intensity_dim=3)
        torch.manual_seed(0)
        model = FullEventModel(config).eval()
        for item in items[:3]:
            _, inputs, sequence = item['batches'][0]
            self.assertTrue((inputs['reference'][..., SOURCE_COLUMN[ROLE]] == 1).any())
            with torch.no_grad():
                logits = model(inputs, sequence)
            logits = logits['logits'] if isinstance(logits, dict) else logits
            self.assertTrue(torch.isfinite(logits).all())


if __name__ == '__main__':
    unittest.main()


class RolloutProviderTest(unittest.TestCase):
    def test_provider_reaches_every_core_and_may_overlap_it(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        from beatfcn.model.local_song_rollout import generate_complete_songs
        from beatfcn.train.alternative_context import chart_arrays_from_notes, sibling_provider
        captured = []

        def fake_inputs(features, audio, request, controls, *, sources=()):
            reference, provenance = encode_reference(mode=request.mode, audio_sha256=audio,
                duration_us=request.duration_us, origin_us=request.start_us,
                unknown_regions=((request.start_us, request.end_us),), sources=sources)
            captured.append((request, sources))
            return dict(local=np.zeros((3, 1, 8, 1024), np.float32), mode=np.asarray(0), controls=controls,
                        **reference), provenance

        def predict(model, inputs, reqs, cap, **_):
            return [dict(complete=True, tokens=[], events=[]) for _ in reqs]
        notes = [(t, t, 'tap', (t // 250_000) % 4) for t in range(0, 17_000_000, 250_000)]
        provider = sibling_provider({'a': ('osu:mine', chart_arrays_from_notes(notes))}, pad_s=2.)
        songs = [dict(chart_id='a', audio_sha256='a' * 64, mode='mania_4k', duration_us=17_000_000),
                 dict(chart_id='b', audio_sha256='b' * 64, mode='mania_4k', duration_us=9_000_000)]
        with patch('beatfcn.model.local_song_rollout.request_inputs', fake_inputs):
            result = generate_complete_songs(SimpleNamespace(training=False), None, songs, device='cpu',
                                             generator=predict, alternative_provider=provider)
        cores = {s['chart_id']: s['cores'] for s in result['songs']}
        self.assertEqual(len(cores['a']), 3)
        self.assertTrue(all(c['alternative']['source_id'] == 'alternative:osu:mine' and c['alternative']['events'] > 0
                            for c in cores['a']))
        self.assertTrue(all(c['alternative'] is None for c in cores['b']))
        self.assertTrue(all(set(c['reference_provenance']) == {'rows', 'sources'} for c in cores['a']))
        for request, sources in captured:
            if request.duration_us == 17_000_000:
                (source,) = sources
                self.assertTrue(any(request.start_us <= e.onset_us < request.end_us for e in source.events))
                self.assertEqual(source.regions[0][0], max(0, request.start_us - 2_000_000))


class PreflightTest(unittest.TestCase):
    def test_preflight_covers_pairs_and_history_and_catches_bad_charts(self):
        from scripts.preflight_reference_paths import preflight
        panel = SiblingPanel()
        counts, failures = preflight(panel, alternative=dict(p=.7), history=dict(p=.5))
        self.assertEqual(failures, [])
        self.assertGreater(counts['pairs'], 0)
        self.assertGreater(counts['history_ok'], 0)
        bad = panel.arrays[0]
        bad['end_us'][5] = bad['onset_us'][5] - 1                      # a corrupted sibling event
        counts, failures = preflight(panel, alternative=dict(p=.7), history=None)
        self.assertTrue(any(f['path'] == 'alternative' for f in failures))
