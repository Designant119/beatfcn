"""Best-of-N whole-song readout + pilot kit export (scripts/run_workflow_benchmark.py)."""
import csv
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
from unittest.mock import patch
import unittest
import zipfile

import numpy as np
import torch

from beatfcn.model.rerank import score_song, select_best_song
import scripts.run_workflow_benchmark as wb


def fake_inputs(features, audio, request, controls, *, sources=()):
    return dict(local=np.zeros((3, 1, 8, 1024), np.float32), mode=np.asarray(1),
                controls=controls, reference=np.zeros((1, 16), np.float32)), []


def stub_generate_batch(model, inputs, requests, max_tokens, *, deadline=None, temperature=None, top_p=None,
                        generator=None, guidance=None, unconditional_inputs=None):
    """Draws a few taps per core from each row's own torch.Generator (seed-determined)."""
    out = []
    for request, g in zip(requests, generator):
        count = int(torch.randint(0, 6, (1,), generator=g))
        span = request.end_us-request.start_us
        onsets = sorted({request.start_us + int(torch.randint(0, span, (1,), generator=g)) // 1000 * 1000
                         for _ in range(count)})
        out.append(dict(complete=True, tokens=[0]*len(onsets), failure=None,
                        events=[dict(onset_us=t, end_us=t, type='tap', lane=-1) for t in onsets]))
    return out


class FakeOnsetModel:
    training = False
    c = SimpleNamespace(onset_head=True)

    def __init__(self):
        self.encoded = 0

    def encode(self, inputs):
        self.encoded += 1
        b = len(inputs['mode'])
        times = torch.arange(0, 8, .01)[None].repeat(b, 1)
        logits = torch.sin(times*7)[..., None].repeat(1, 1, 5)
        return dict(onset_logits=logits, onset_times=times, onset_valid=torch.ones_like(times, dtype=torch.bool))


SONGS = [dict(song_id='phira:aaaa', audio_sha256='a'*64, mode='phira', duration_us=17_000_000, references=[]),
         dict(song_id='phira:bbbb', audio_sha256='b'*64, mode='phira', duration_us=9_500_000, references=[])]


def args(**kw):
    base = dict(temperature='1', top_p=None, seed=5, device='cpu', batch_size=4, token_cap=64, max_wall_s=60.)
    return SimpleNamespace(**dict(base, **kw))


def run_pool(best_of, model=None, **kw):
    with patch('beatfcn.model.local_song_rollout.request_inputs', fake_inputs), \
         patch('beatfcn.model.full_event_generate.generate_batch', stub_generate_batch):
        return wb.sample_pool(args(**kw), dict(songs=SONGS), model or FakeOnsetModel(), None,
                              intensity_dim=0, cell_us=100_000, best_of=best_of)


def events_of(samples):
    return [[smp['by_id'][s['song_id']]['events'] for s in SONGS] for smp in samples]


def forbid_targets():
    def boom(*a, **k):
        raise AssertionError('reference-free selection touched a human target')
    return [patch.object(wb, 'reference_notes', boom), patch.object(wb, 'panel_references', boom),
            patch.object(wb, 'reference_grid', boom), patch('numpy.load', boom)]


class BestOfTests(unittest.TestCase):
    def test_sampling_and_selection_never_read_targets(self):
        patches = forbid_targets()
        for p in patches:
            p.start()
        try:
            samples, onset_term = run_pool(3)
            pick = wb.rerank_selection([smp['by_id'] for smp in samples], SONGS)
        finally:
            for p in patches:
                p.stop()
        self.assertTrue(onset_term)
        self.assertEqual(set(pick), {s['song_id'] for s in SONGS})
        self.assertTrue(all(0 <= v['index'] < 3 and len(v['scores']) == 3 for v in pick.values()))
        self.assertTrue(all(c.get('probe') for smp in samples for st in smp['result']['songs'] for c in st['cores']))

    def test_deterministic_given_seeds_and_seeds_are_consecutive(self):
        a, _ = run_pool(3)
        b, _ = run_pool(3)
        self.assertEqual([s['seed'] for s in a], [5, 6, 7])
        self.assertEqual(events_of(a), events_of(b))
        self.assertNotEqual(events_of(a)[0], events_of(a)[1])
        self.assertEqual(wb.rerank_selection([s['by_id'] for s in a], SONGS),
                         wb.rerank_selection([s['by_id'] for s in b], SONGS))

    def test_best_of_one_equals_single_sample(self):
        one, onset_term = run_pool(1)
        three, _ = run_pool(3)
        self.assertFalse(onset_term)  # probe only attached when there is something to choose
        self.assertEqual(len(one), 1)
        self.assertEqual(events_of(one)[0], events_of(three)[0])  # single_sample is seed --seed either way
        pick = wb.rerank_selection([one[0]['by_id']], SONGS)
        self.assertTrue(all(v['index'] == 0 for v in pick.values()))
        self.assertEqual(select_best_song([one[0]['by_id']['phira:aaaa']])[0], 0)

    def test_greedy_best_of_is_refused_and_no_onset_head_disables_term(self):
        with self.assertRaisesRegex(ValueError, 'greedy'):
            run_pool(2, temperature='none')
        model = FakeOnsetModel()
        model.c = SimpleNamespace(onset_head=False)
        samples, onset_term = run_pool(2, model=model)
        self.assertFalse(onset_term)
        self.assertEqual(model.encoded, 0)
        parts = wb.rerank_selection([s['by_id'] for s in samples], SONGS)['phira:aaaa']['parts']
        self.assertTrue(all(p['onset_term_cores'] == 0 for p in parts))

    def test_onset_probe_on_tiny_real_model(self):
        from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
        from test_full_event_model import inputs
        torch.manual_seed(0)
        config = FullEventConfig(d=32, heads=4, kv_heads=2, ffn=64, encoder_layers=1, decoder_layers=1, tap_dim=64,
                                 taps=2, bands=1, band_slots=1, dropout=0., mel_bins=0, onset_head=True)
        model = FullEventModel(config).eval()
        probe = wb.onset_probe(model)
        out = probe(model, inputs(config), [SimpleNamespace(start_us=0, end_us=150_000)], [None])
        self.assertEqual(len(out), 1)
        np.testing.assert_allclose(out[0]['onset_times_s'], [0., .04, .08, .12], atol=1e-6)  # valid, inside 0.15 s core
        self.assertTrue(all(0 < p < 1 for p in out[0]['onset_probs']))
        self.assertIsNone(wb.onset_probe(FullEventModel(FullEventConfig(d=32, heads=4, kv_heads=2, ffn=64,
            encoder_layers=1, decoder_layers=1, tap_dim=64, taps=2, bands=1, band_slots=1, mel_bins=0)).eval()))

    def test_song_score_prefers_complete_then_score_then_lowest_seed(self):
        core = lambda lo, hi, ok=True: dict(start_us=lo, end_us=hi, complete=ok)  # noqa: E731
        tap = lambda t: dict(onset_us=t, end_us=t, type='tap', lane=-1)  # noqa: E731
        full = dict(duration_us=16_000_000, complete=True, failure=None, events=[tap(0), tap(500_000)],
                    cores=[core(0, 8_000_000), core(8_000_000, 16_000_000)])
        cluster = dict(full, events=[tap(0), tap(10_000), tap(20_000)])
        broken = dict(full, complete=False, failure='core_token_limit', events=[tap(0)], cores=[core(0, 8_000_000, False)])
        best, scores, parts = select_best_song([broken, cluster, full])
        self.assertEqual(best, 2)
        self.assertEqual(parts[0]['missing_cores'], 1)
        self.assertLess(scores[0], scores[2]-10)
        self.assertEqual(select_best_song([full, dict(full)])[0], 0)
        self.assertEqual(score_song(full)[1]['onset_term_cores'], 0)

    def test_three_readouts_share_machinery_and_oracle_uses_cer(self):
        song = dict(song_id='mania_4k:x', mode='mania_4k', duration_us=10_000_000)
        ref = [(t, t, 'tap', t//500_000 % 4) for t in range(0, 10_000_000, 500_000)]
        refs = [dict(chart_id='osu:1', stratum='medium', coarse_stratum='medium', notes=ref)]
        drafts = [[], ref[:5], ref[:15]]
        evaluated = []
        for k, d in enumerate(drafts):
            r, row = wb.evaluate(d, dict(draft_id='model'), song, refs, grids={}, per_op=False, split='val')
            evaluated.append({song['song_id']: (r, row)})
        samples = [dict(seed=10+k) for k in range(3)]
        rerank = {song['song_id']: dict(index=1)}
        oracle = wb.oracle_selection(evaluated, [song])
        self.assertEqual(oracle[song['song_id']]['index'], 2)
        readouts = wb.build_readouts(samples, evaluated, [song], rerank=rerank, oracle=oracle, samples_n=50)
        self.assertEqual(list(readouts), ['single_sample', 'best_of_N_reranked', 'oracle_best_of_N'])
        self.assertEqual(readouts['single_sample']['seeds_used'], {song['song_id']: 10})
        self.assertIn('ORACLE', readouts['oracle_best_of_N']['label'])
        cer = {k: v['aggregate']['min_over_references']['val/mania_4k']['min_ref_cer']['mean'] for k, v in readouts.items()}
        self.assertEqual(cer['single_sample'], 1.)
        self.assertAlmostEqual(cer['best_of_N_reranked'], .75)
        self.assertAlmostEqual(cer['oracle_best_of_N'], .25)
        self.assertEqual(readouts['best_of_N_reranked']['agreement_with_oracle'], 0.)
        self.assertEqual(set(wb.build_readouts(samples[:1], evaluated[:1], [song], samples_n=50)), {'single_sample'})


# ------------------------------------------------------------------ export
def make_raw(root, songs):
    """Synthetic raw layout: osu/audio/<sha>.mp3, osu/charts/<id>.osu, phira/zips/<id>.zip (info.yml music)."""
    (root/'osu/audio').mkdir(parents=True)
    (root/'osu/charts').mkdir(parents=True)
    (root/'phira/zips').mkdir(parents=True)
    out = []
    for i, (mode, duration_s) in enumerate(songs):
        audio = f'audio-{i}'.encode()*100
        sha = hashlib.sha256(audio).hexdigest()
        chart = f'{"osu" if mode == "mania_4k" else "phira"}:{100+i}'
        if mode == 'mania_4k':
            (root/'osu/audio'/f'{sha}.mp3').write_bytes(audio)
            (root/'osu/charts'/f'{100+i}.osu').write_text(f'osu file format v14\nTitle:Song {i}\nArtist:Art {i}\n')
        else:
            with zipfile.ZipFile(root/'phira/zips'/f'{100+i}.zip', 'w') as z:
                z.writestr('info.yml', f'name: Song {i}\ncomposer: Comp {i}\nmusic: m{i}.ogg\n')
                z.writestr(f'm{i}.ogg', audio)
        notes = [(t, t, 'tap', (t//250_000) % 4 if mode == 'mania_4k' else -1) for t in range(0, duration_s*1_000_000, 250_000)]
        out.append(dict(song=dict(song_id=f'{mode}:{sha[:16]}', audio_sha256=sha, mode=mode, duration_us=duration_s*1_000_000,
                                  references=[dict(chart_id=chart, stratum='medium', coarse_stratum='medium')]),
                        notes=notes))
    return out


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.raw = self.root/'raw'
        self.items = make_raw(self.raw, [('mania_4k', 70), ('mania_4k', 80), ('mania_4k', 90), ('mania_4k', 60),
                                         ('phira', 75), ('phira', 85), ('mania_4k', 65)])
        self.panel = dict(songs=[i['song'] for i in self.items])
        self.refs = {i['song']['song_id']: [dict(i['song']['references'][0], notes=i['notes'])] for i in self.items}
        self.drafts = {i['song']['song_id']: dict(seed=3, complete=True, failure=None,
                       events=[dict(onset_us=t, end_us=t, type='tap', lane=l if i['song']['mode'] == 'mania_4k' else -1)
                               for t, _, _, l in i['notes'][::3]]) for i in self.items}

    def tearDown(self):
        self.tmp.cleanup()

    def test_audio_resolution_layout_and_sha_check(self):
        mania, phira = self.items[0]['song'], self.items[4]['song']
        a = wb.resolve_audio(self.raw, mania)
        self.assertTrue(a['ok'])
        self.assertIn('osu/audio', a['source'])
        b = wb.resolve_audio(self.raw, phira)
        self.assertTrue(b['ok'])
        self.assertTrue(b['source'].endswith('104.zip:m4.ogg'))
        self.assertEqual(wb.song_metadata(self.raw, phira), ('Song 4', 'Comp 4'))
        self.assertEqual(wb.song_metadata(self.raw, mania), ('Song 0', 'Art 0'))
        bad = dict(mania, audio_sha256='0'*64)
        self.assertFalse(wb.resolve_audio(self.raw, bad)['ok'])

    def test_missing_audio_still_writes_chart_with_reason(self):
        from beatfcn.events.osu_mania import parse_osu
        mania, phira = self.items[0]['song'], self.items[4]['song']
        missing = dict(ok=False, reason='nope')
        r = wb.export_song(self.root/'out'/'m', 'm', mania, self.drafts[mania['song_id']]['events'], missing,
                           title='t', artist='a', version='v')
        self.assertTrue(r['disk_roundtrip']['exact'])
        self.assertNotIn('osz', r)
        self.assertEqual(r['audio']['reason'], 'nope')
        self.assertFalse(parse_osu(Path(r['chart']['path']).read_bytes(), 'm.osu').quarantined)
        r = wb.export_song(self.root/'out'/'p', 'p', phira, self.drafts[phira['song_id']]['events'], missing,
                           title='t', artist='a', version='v')
        self.assertFalse(r['chart']['package'])
        self.assertTrue(Path(r['chart']['path']).is_file())

    def test_export_writes_the_playback_clock(self):
        from beatfcn.events.osu_mania import parse_osu
        mania = self.items[0]['song']
        events = [dict(onset_us=5_000, end_us=5_000, type='tap', lane=0)] + self.drafts[mania['song_id']]['events'][1:]
        r = wb.export_song(self.root/'clk'/'m', 'm', mania, events, dict(ok=False, reason='x'), title='t', artist='a',
                           version='v', clock_shift_us=14_000, clock_source='receipt')
        self.assertEqual((r['clock']['shift_us'], r['clock']['dropped_before_zero']), (14_000, 1))
        written = sorted(e.onset_us for e in parse_osu(Path(r['chart']['path']).read_bytes(), 'm.osu').events)
        self.assertEqual(written, sorted(e['onset_us'] - 14_000 for e in events[1:]))

    def test_pilot_kit_writes_drafts_on_the_playback_clock(self):
        from beatfcn.events.osu_mania import parse_osu
        a = SimpleNamespace(export_dir=self.root/'kshift', pilot_seed=7, pilot_segments=4, audio_dir=None,
                            clock_shifts=lambda chart_id: (14_000, 'receipt.json'))
        kit = wb.write_pilot_kit(a, self.drafts, self.panel, self.refs, self.raw, readout='consensus_8')
        seg = next(s for s in kit['segments'] if s['pair'] is not None and s['mode'] == 'mania_4k')
        self.assertEqual((seg['clock_shift_us'], seg['clock_shift_source']), (14_000, 'receipt'))
        model = sorted(int(e['onset_us']) for e in self.drafts[seg['song_id']]['events']
                       if seg['start_us'] <= int(e['onset_us']) < seg['end_us'])
        path = Path(seg['files']['draft']['chart']['path'])
        written = sorted(e.onset_us for e in parse_osu(path.read_bytes(), 'd.osu').events)
        self.assertEqual(written, [t - 14_000 for t in model])
        self.assertIn(f"Bookmarks: {(seg['start_us'] - 14_000)//1000},{(seg['end_us'] - 14_000)//1000}", path.read_text())

    def test_pilot_kit_is_counterbalanced_and_deterministic(self):
        def build(name):
            a = SimpleNamespace(export_dir=self.root/name, pilot_seed=7, pilot_segments=4, audio_dir=None)
            return wb.write_pilot_kit(a, self.drafts, self.panel, self.refs, self.raw, readout='best_of_N_reranked')
        kit = build('k1')
        again = build('k2')
        strip = lambda k: json.dumps([{x: v for x, v in s.items() if x != 'files'} for s in k['segments']], sort_keys=True)  # noqa: E731
        self.assertEqual(strip(kit), strip(again))
        paired = [s for s in kit['segments'] if s['pair'] is not None]
        self.assertEqual(len(paired), 4)
        self.assertEqual(len({s['song_id'] for s in kit['segments']}), len(kit['segments']))  # all different songs
        self.assertEqual(sorted({s['mode'] for s in paired}), ['mania_4k', 'phira'])  # pairs alternate modes
        for pair in kit['pairs']:
            self.assertEqual(len({s['mode'] for s in paired if s['segment_id'] in pair}), 1)
        self.assertEqual(kit['practice'], 'practice')
        for s in kit['segments']:
            self.assertTrue(30_000_000 <= s['end_us']-s['start_us'] <= 45_000_000)
            self.assertGreater(s['human_primary_notes_in_window'], 0)
        with open(self.root/'k1/pilot/assignment.csv', encoding='utf-8') as f:
            rows = list(csv.DictReader(f))
        analysed = [r for r in rows if r['pair'] != '']
        for seg in paired:
            conds = sorted(r['condition'] for r in analysed if r['segment_id'] == seg['segment_id'])
            self.assertEqual(conds, ['draft', 'scratch'])  # each segment once per condition across orders
        for order in wb.PILOT_ORDERS:
            for i in range(2):
                conds = sorted(r['condition'] for r in analysed if r['participant_order'] == order and r['pair'] == str(i))
                self.assertEqual(conds, ['draft', 'scratch'])
            seq = [r['condition'] for r in analysed if r['participant_order'] == order]
            self.assertEqual(seq[1], seq[2])  # ABBA over pairs
            self.assertTrue(all((self.root/'k1/pilot'/r['file']).is_file() for r in rows if r['participant_order'] == order))
        codes = [r['blind_code'] for r in analysed]
        self.assertEqual(len(set(codes)), len(codes))
        # The scratch template is empty, the draft holds only notes inside the window, timing identical.
        from beatfcn.events.osu_mania import parse_osu
        seg = next(s for s in paired if s['mode'] == 'mania_4k')
        d = parse_osu(Path(seg['files']['draft']['chart']['path']).read_bytes(), 'd.osu')
        e = Path(seg['files']['scratch']['chart']['path']).read_text()
        self.assertTrue(d.events and all(seg['start_us'] <= x.onset_us < seg['end_us'] for x in d.events))
        self.assertIn(f"Bookmarks: {seg['start_us']//1000},{seg['end_us']//1000}", e)
        self.assertEqual(e.split('[TimingPoints]')[1].split('[HitObjects]')[0],
                         Path(seg['files']['draft']['chart']['path']).read_text().split('[TimingPoints]')[1].split('[HitObjects]')[0])
        self.assertEqual(e.split('[HitObjects]')[1].strip(), '')


if __name__ == '__main__':
    unittest.main()
