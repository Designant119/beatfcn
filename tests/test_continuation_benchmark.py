import unittest

import numpy as np
import torch

from beatfcn.events.canonical import TYPES
from beatfcn.model.event_canvas import RhythmEvent
from beatfcn.train import continuation_benchmark as cb
from beatfcn.train.event_reference import TIME_SCALE_US

AUDIO = ['%064x' % (i + 1) for i in range(4)]
DURATION = 120_000_000
WINDOW = 8_000_000


class Features:
    def __init__(self):
        self.records = {a: dict(duration_us=DURATION) for a in AUDIO}

    def window(self, audio, start_us, end_us):
        n = 40
        return dict(local=np.ones((2, n, 1, 4), np.float32), local_times=np.linspace(-1, 9, n).astype(np.float32),
                    local_valid=np.ones(n, bool), mel=np.ones((n, 2), np.float32),
                    mel_times=np.linspace(-1, 9, n).astype(np.float32), mel_valid=np.ones(n, bool),
                    song=np.ones((3, 4), np.float32), song_times=np.zeros(3, np.float32), song_valid=np.ones(3, bool))


def _chart(mode, phase):
    rows = []
    for t in range(phase, DURATION, 250_000):
        rows.append((t, t, 0, (t // 250_000) % 3 if mode == 'mania_4k' else -1))
    for t in range(phase + 3_100_000, DURATION - 2_000_000, 9_000_000):
        rows.append((t, t + 1_000_000, 1, 3 if mode == 'mania_4k' else -1))
    rows.sort()
    return {k: np.asarray([r[i] for r in rows], np.int64) for i, k in enumerate(('onset_us', 'end_us', 'kind', 'lane'))}


class Panel:
    """Four val charts (two per mode, different songs); 8 s windows as records."""

    def __init__(self):
        self.features = Features()
        self.charts, self.arrays, self.records = [], [], []
        for i, mode in enumerate(('mania_4k', 'mania_4k', 'phira', 'phira')):
            self.charts.append(dict(chart_id=f'c{i}', audio_sha256=AUDIO[i], mode=mode, split='val',
                                    duration_us=DURATION, component=f'k{i}', stratum='x'))
            a = _chart(mode, 1000 * (i + 1))
            self.arrays.append(a)
            for start in range(0, DURATION, WINDOW):
                end = start + WINDOW
                inside = np.flatnonzero((a['onset_us'] >= start) & (a['onset_us'] < end))
                crossing = (a['kind'] == 1) & (a['onset_us'] < start) & (a['end_us'] > start)
                carry = [dict(onset_us=int(a['onset_us'][j]), end_us=int(a['end_us'][j]), type='hold',
                              lane=int(a['lane'][j])) for j in np.flatnonzero(crossing)]
                self.records.append(dict(split='val', chart_index=i, chart_id=f'c{i}', audio_sha256=AUDIO[i],
                    component=f'k{i}', mode=mode, target_start=int(inside[0]), target_stop=int(inside[-1]) + 1,
                    request=dict(mode=mode, start_us=start, end_us=end, duration_us=DURATION, copy_times=[],
                                 known_carry=carry, allowed_types=None, max_events=4096, max_tokens=32768)))

    def _arrays(self, chart_index):
        return self.arrays[chart_index]

    def targets(self, index):
        row = self.records[index]
        a = self.arrays[row['chart_index']]
        sl = slice(row['target_start'], row['target_stop'])
        return [RhythmEvent(int(h), int(t), TYPES[int(k)], int(l)) for h, t, k, l in
                zip(a['onset_us'][sl], a['end_us'][sl], a['kind'][sl], a['lane'][sl])]


def tiny_model():
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    torch.manual_seed(0)
    return FullEventModel(FullEventConfig(d=16, heads=2, kv_heads=1, ffn=32, encoder_layers=1, decoder_layers=1,
        tap_dim=4, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=2, intensity_dim=3)).eval()


class TeacherPanelTest(unittest.TestCase):
    def test_panel_windows_and_sources_never_show_the_target(self):
        panel = Panel()
        protocol = cb.freeze_continuation_panel(panel, windows=24, seed=1)
        self.assertEqual(len(protocol['windows']), 24)
        self.assertEqual(set(protocol['windows_by_mode']), {'mania_4k', 'phira'})
        for window in protocol['windows']:
            request = cb.event_request(window['request'])
            self.assertGreater(request.start_us, cb.MIN_START_US)
            self.assertFalse(request.known_carry)
            donor = panel.records[window['foreign_index']]
            self.assertNotEqual(donor['audio_sha256'], window['audio_sha256'])
            self.assertEqual(donor['mode'], window['mode'])
            for condition in cb.CONDITIONS:
                sources = cb.condition_sources(panel, window, condition, request)
                for source in sources:
                    for event in source.events:
                        self.assertFalse(request.start_us <= event.onset_us < request.end_us)
                        self.assertLessEqual(event.end_us, request.start_us if event.onset_us < request.start_us
                                             else request.duration_us)
                if condition == 'none':
                    self.assertEqual(sources, ())
                if condition.startswith('foreign'):
                    # Retimed foreign chart: the same relative pattern as the donor's own prefix.
                    length = int(condition[len('foreign'):]) * 1_000_000
                    donor_arrays = panel.arrays[donor['chart_index']]
                    s = window['foreign_start_us']
                    expected = sorted(int(t) - s for t in donor_arrays['onset_us'] if s - length <= t < s)
                    got = sorted(e.onset_us - request.start_us for e in sources[0].events)
                    self.assertEqual(got, expected)
                if condition == 'infill16':
                    self.assertEqual(sources[1].regions, ((request.end_us, request.end_us + cb.SUFFIX_US),))

    def test_teacher_run_and_none_equals_blank_reference(self):
        from beatfcn.train.audio_dependence import base_inputs
        panel = Panel()
        protocol = cb.freeze_continuation_panel(panel, windows=6, seed=2)
        model = tiny_model()
        _, _, _, variants, info = cb._prepare(panel, protocol['windows'][0], model.vocab, cb.CONDITIONS, 100_000)
        blank, _ = base_inputs(panel, protocol['windows'][0]['index'])
        for key, value in blank.items():
            np.testing.assert_array_equal(variants['none'][key], value)
        self.assertGreater(info['prefix32']['reference_rows'], info['prefix8']['reference_rows'])
        result = cb.run_continuation_teacher(model, panel, protocol, device='cpu', batch_size=4)
        summary = result['summary']
        self.assertTrue(summary['complete'])
        gaps = summary['by_mode']['phira']['gaps']
        self.assertIn('song_specific_prefix16', gaps)
        self.assertIsNotNone(gaps['prefix16_vs_none']['joint']['paired_mean_nats'])


class GenerationTest(unittest.TestCase):
    def test_jobs_hold_only_visible_events(self):
        panel = Panel()
        jobs, truths = cb.generation_jobs(panel, prefix_seconds=(30, 60), seed=3)
        self.assertEqual(len(jobs), 4 * 6)
        for job in jobs:
            lo, hi = job['fill']
            self.assertFalse(any(lo <= e.onset_us < hi for e in job['visible_events']))
            truth = truths[cb.truth_key(job)]
            self.assertTrue(truth and all(lo <= e.onset_us < hi for e in truth))
            if job['task'] == 'infill':
                self.assertTrue(8 <= job['gap_s'] <= 16)
                self.assertGreaterEqual(lo, cb.MIN_START_US)

    def test_rollout_history_roles_suffix_and_carry(self):
        panel = Panel()
        jobs, truths = cb.generation_jobs(panel, prefix_seconds=(30,), seed=3)
        seen = []

        def generator(model, inputs, requests, max_tokens, *, deadline=None):
            out = []
            for row, request in enumerate(requests):
                n = int(inputs['reference_valid'][row].sum())
                seen.append((request, inputs['reference'][row, :n].numpy().copy()))
                # A hold that crosses into the next window, plus a tap.
                end = min(request.end_us + 500_000, request.duration_us)
                out.append(dict(events=[dict(onset_us=request.start_us + 1000, end_us=request.start_us + 1000,
                                             type='tap', lane=1 if request.mode == 'mania_4k' else -1),
                                        dict(onset_us=request.end_us - 400_000, end_us=end, type='hold',
                                             lane=3 if request.mode == 'mania_4k' else -1)],
                                tokens=10, complete=True))
            return out
        states, _ = cb.rollout_jobs(tiny_model(), panel.features, jobs, device='cpu', generator=generator,
                                    batch_size=5)
        self.assertTrue(all(s['complete'] for s in states))
        for request, reference in seen:
            span = request.end_us - request.start_us
            events = reference[reference[:, 10] == 1]
            for event in events:
                rel = float(event[0]) * TIME_SCALE_US
                self.assertTrue(rel < -1 or rel > span - 1, rel)
        cond = [s for s in states if s['conditioned'] and s['task'] == 'continuation']
        for state in cond:
            roles = [c['reference_roles'] for c in state['cores']]
            self.assertIn('human_input', roles[0])
            self.assertNotIn('model_history', roles[0])
            self.assertIn('model_history', roles[1])
            # 32 s of history: after 4 windows past X the human prefix is out of reach.
            self.assertNotIn('human_input', roles[5])
            self.assertTrue(all(c['carry'] >= 1 for c in state['cores'][1:]))
        uncond = [s for s in states if not s['conditioned']]
        self.assertTrue(all(c['reference_rows'] == 1 for s in uncond for c in s['cores']))
        infill = [s for s in states if s['conditioned'] and s['task'] == 'infill']
        for state in infill:
            last = state['cores'][-1]
            self.assertEqual(last['end_us'], state['fill'][1])
            # Only the window that closes the gap sees the human suffix (more human rows than the first).
            self.assertGreaterEqual(last['reference_roles'].get('human_input', 0), 1)
        scored = [cb.score_job(s, truths[cb.truth_key(s)], s['visible_events']) for s in states]
        summary = cb.summarize_generation(scored)
        self.assertEqual({(g['task'], g['mode']) for g in summary},
                         {('continuation', 'mania_4k'), ('continuation', 'phira'), ('infill', 'mania_4k'),
                          ('infill', 'phira')})

    def test_perfect_draft_scores_cer_zero(self):
        panel = Panel()
        jobs, truths = cb.generation_jobs(panel, prefix_seconds=(30,), seed=3)
        job = jobs[0]
        state = dict(job, generated=list(truths[cb.truth_key(job)]), cores=[], complete=True, failure=None)
        row = cb.score_job(state, truths[cb.truth_key(job)], job['visible_events'])
        self.assertEqual(row['cer'], 0.)
        self.assertEqual(row['note_f1_50ms'], 1.)
        self.assertAlmostEqual(row['log_density_error'], 0.)
        self.assertAlmostEqual(row['density_ratio_model'], row['density_ratio_human'])
        self.assertEqual(row['ioi_js_vs_human'], 0.)


if __name__ == '__main__':
    unittest.main()
