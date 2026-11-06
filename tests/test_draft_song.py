"""scripts/draft_song.py: new audio -> database-layout features -> whole-song draft -> playback-clock export.

CPU plumbing with a tiny RANDOM-INIT FullEventModel (an END-token logit bias makes it place notes) and a mocked
MERT network inside the REAL FrozenMert2.encode (windows, tail padding, clocks, metadata). Decoding uses the real
ffmpeg (skipped without ffmpeg/ffprobe/libmp3lame). Says nothing about any trained checkpoint.
"""
import contextlib
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import wave
import zipfile

import numpy as np
import torch

from beatfcn.audio.mert2 import CELL_SAMPLES, REVISION, SR, FrozenMert2, Mert2Spec
from beatfcn.events.osu_mania import parse_osu
from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
import scripts.draft_song as ds
import scripts.run_workflow_benchmark as wb

ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILES = {'L12.npy', 'L23.npy', 'times_s.npy', 'intervals_s.npy', 'mel10_complete.npy', 'receipt.json'}
TIMING_LINES = ['437,400,4,2,1,60,1,0', '2000,-200,4,2,1,60,0,1', '3000.5,375.5,3,2,1,70,1,1']
TIMING_OSU = '\r\n'.join(['osu file format v14', '', '[General]', 'AudioFilename: song.mp3', 'Mode: 3', '',
                          '[Metadata]', 'Title:T', 'Artist:A', '', '[Difficulty]', 'CircleSize:4',
                          'OverallDifficulty:8', '', '[TimingPoints]', *TIMING_LINES, '', '[HitObjects]', ''])


def have_mp3_tools():
    if not (shutil.which('ffmpeg') and shutil.which('ffprobe')):
        return False
    out = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True)
    return 'libmp3lame' in out.stdout


class FakeNet(torch.nn.Module):
    """Deterministic stand-in for the MERT forward: 24 hidden states [1, cells, 1024] from per-cell RMS."""

    def forward(self, input_values, output_hidden_states=True):
        x = input_values[0]
        cells = x.shape[0]//CELL_SAMPLES
        rms = x[:cells*CELL_SAMPLES].reshape(cells, CELL_SAMPLES).pow(2).mean(1).sqrt()
        base = torch.arange(1024, dtype=torch.float32)[None]/1024
        return SimpleNamespace(hidden_states=tuple((rms[:, None]*(i+1)+base)[None].to(torch.bfloat16)
                                                   for i in range(24)))


class FakeMert(FrozenMert2):
    """The real FrozenMert2.encode over FakeNet (no weights, no transformers)."""

    def __init__(self):
        self.spec, self.device, self.revision, self.model = Mert2Spec(), 'cpu', REVISION, FakeNet()
        self.calls = 0

    def encode(self, wave_):
        self.calls += 1
        return super().encode(wave_)


def tiny_config(**kw):
    return FullEventConfig(**dict(dict(d=32, heads=4, kv_heads=2, ffn=48, encoder_layers=1, decoder_layers=2,
                                       tap_dim=1024, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=128,
                                       intensity_dim=3, onset_head=True, beat_features=True), **kw))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_export(path, stats_path, config=None, seed=0):
    torch.manual_seed(seed)
    config = config or tiny_config()
    model = FullEventModel(config)
    torch.save(dict(schema='beatfcn.final-model.v1', config=asdict(config), vocabulary={}, weights='ema',
                    model=model.state_dict(), parameters=sum(p.numel() for p in model.parameters()),
                    provenance=dict(plan_id='tiny-test', mel_normalization=dict(path=str(stats_path),
                                                                                sha256=sha(stats_path)))), path)
    return model


def end_bias(value):
    """Wrap ds.load_model: bias the grammar END token so the random model places notes."""
    real = ds.load_model

    def load(*args, **kwargs):
        model, identity, declared = real(*args, **kwargs)
        bias = torch.zeros(model.vocab.size)
        bias[model.vocab.token('group', 0)] = value
        model.output.register_forward_hook(lambda module, inputs, out: out+bias.to(out.device))
        return model, identity, declared
    return patch.object(ds, 'load_model', load)


def run_main(argv, **kw):
    with contextlib.redirect_stdout(io.StringIO()):
        return ds.main([str(a) for a in argv], **kw)


@unittest.skipUnless(have_mp3_tools(), 'needs ffmpeg + ffprobe with libmp3lame')
class DraftSongTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        t = np.arange(int(10.5*44100))/44100  # 10.5 s: two 8 s cores
        signal = .3*np.sin(2*np.pi*440*t)*(np.mod(t, .5) < .06)
        with wave.open(str(cls.root/'song.wav'), 'wb') as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(44100)
            w.writeframes((signal*32767).astype('<i2').tobytes())
        subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(cls.root/'song.wav'), '-c:a', 'libmp3lame',
                        '-b:a', '128k', str(cls.root/'song.mp3')], check=True)
        cls.audio = cls.root/'song.mp3'
        cls.stats = cls.root/'mel-stats.json'
        cls.stats.write_text(json.dumps(dict(schema='train-mel-channel-v1', train_audios=['f'*64], frames=1,
                                             mean=[0.]*128, std=[1.]*128)))
        cls.model = cls.root/'tiny.pt'
        save_export(cls.model, cls.stats)
        cls.timing = cls.root/'timing.osu'
        cls.timing.write_bytes(TIMING_OSU.encode())

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def out(self, name):
        return self.root/name

    def test_mania_draft_layout_roundtrip_clock_and_verbatim_timing(self):
        out, feat, mert = self.out('mania'), self.out('mania-features'), FakeMert()
        with end_bias(-2.):
            r = run_main(['--model', self.model, '--audio', self.audio, '--mode', 'mania_4k', '--out-dir', out,
                          '--timing', self.timing, '--device', 'cpu', '--temperature', '0.05', '--features-dir', feat],
                         extractor=mert)
        audio_sha = sha(self.audio)
        # Features: database layout, readable by the unchanged MertFeatureStore.
        folder = feat/'features'/audio_sha
        self.assertEqual({p.name for p in folder.iterdir()}, FEATURE_FILES)
        receipt = json.loads((folder/'receipt.json').read_text())
        samples = receipt['decoded_samples']
        frames = (samples+959)//960
        self.assertEqual(samples, round(10.5*SR))
        self.assertEqual(np.load(folder/'L12.npy').shape, (frames, 1024))
        self.assertEqual(np.load(folder/'L23.npy').dtype, np.float16)
        self.assertEqual(np.load(folder/'mel10_complete.npy').shape, (frames*4, 128))
        np.testing.assert_array_equal(np.load(folder/'times_s.npy'), (np.arange(frames)*960+360)/SR)
        try:
            from beatfcn.cloud.full_database_app import verify_feature_folder
        except ImportError:  # modal not installed: the explicit checks above stand
            verify_feature_folder = None
        if verify_feature_folder is not None:  # the database's own receipt/shape/clock verifier
            verify_feature_folder(folder, dict(audio_sha256=audio_sha, expected_mel_frames=4*(samples//960)), REVISION)
        self.assertEqual(receipt['backend'], 'injected')
        self.assertEqual(receipt['draft_song']['codec'], 'mp3')
        store = ds.open_features(feat, audio_sha, json.loads(self.stats.read_text()))
        window = store.window(audio_sha, 0, 8_000_000)
        self.assertEqual(window['local'].shape, (2, min(frames, 300), 1, 1024))  # [taps, frames, bands, dim]
        self.assertEqual(window['mel'].shape, (min(frames, 300)*4, 128))
        self.assertEqual(r['audio']['duration_us'], samples*1_000_000//SR)
        # A complete draft from complete samples.
        self.assertTrue(r['complete'])
        self.assertEqual(mert.calls, 1)
        self.assertTrue(all(s['complete'] and s['cores'] == 2 for s in r['samples']))
        self.assertEqual(len(r['samples']), 8)
        self.assertGreater(r['draft']['notes'], 0)
        self.assertEqual(r['draft']['legality'], [])
        # Export: .osz holds the .osu + the original audio; the repo parser round-trips the notes.
        osz = Path(r['export']['osz'])
        osu = Path(r['export']['chart']['path'])
        with zipfile.ZipFile(osz) as z:
            self.assertEqual(sorted(z.namelist()), sorted([osu.name, 'audio.mp3']))
            self.assertEqual(z.read('audio.mp3'), self.audio.read_bytes())
            self.assertEqual(z.read(osu.name), osu.read_bytes())  # rebuilt after the timing splice
        self.assertTrue(r['export']['disk_roundtrip']['exact'])
        parsed = parse_osu(osu.read_bytes(), osu.name)
        self.assertFalse(parsed.quarantined)
        # Clock: written = model - c, c(osu, mp3) = +14 ms.
        self.assertEqual((r['clock']['c_ms'], r['clock']['basis'], r['export']['clock']['shift_us']),
                         (14, 'source_codec', 14_000))
        model_events = json.loads((out/'draft-events.json').read_text())['events']
        kept = [e for e in model_events if e['onset_us'] >= 14_000]
        self.assertEqual(r['export']['clock']['dropped_before_zero'], len(model_events)-len(kept))
        self.assertEqual(sorted((e.onset_us, e.end_us, e.location.lane) for e in parsed.events),
                         sorted((e['onset_us']-14_000, e['end_us']-14_000, e['lane']) for e in kept))
        # --timing: the [TimingPoints] block is written unchanged; the model saw it shifted onto its clock (+c).
        self.assertEqual(ds.timing_section_lines(osu.read_bytes()), TIMING_LINES)
        self.assertEqual(ds._timing_rows(parsed), ds._timing_rows(parse_osu(TIMING_OSU.encode(), 't.osu')))
        beat = r['timing']['beat_features']
        self.assertTrue(beat['used'])
        self.assertEqual(beat['segments_model_clock'], [[451_000., 400_000., 4], [3_014_500., 375_500., 3]])
        self.assertTrue(all(s['charter_timing_cores'] == s['cores'] for s in r['samples']))  # every core got it
        self.assertTrue(all(s['conditioned_cores'] == 0 for s in r['samples']))  # no tier / density request
        self.assertEqual(r['export']['timing_points']['lines'], 3)
        # Receipt: model identity, decoding settings, timings.
        disk = json.loads((out/'draft-receipt.json').read_text())
        self.assertTrue(disk['complete'])
        self.assertEqual(disk['model']['sha256'], sha(self.model))
        self.assertEqual(disk['model']['kind'], 'exported')
        self.assertEqual(disk['features']['mel_normalization']['sha256'], sha(self.stats))
        self.assertEqual(disk['decoding']['seeds'], list(range(8)))
        self.assertEqual(disk['decoding']['field_temperature'], ds.PRODUCT_FIELD_TEMPERATURE)
        self.assertFalse(disk['decoding']['matches_product_default'])  # T = 0.05 here
        self.assertEqual(disk['decoding']['charter_timing_input']['label'], 'charter_timing_input')
        for key in ('decode_s', 'mert_s', 'mel_s', 'model_load_s', 'generation_s', 'export_s', 'total_s'):
            self.assertIn(key, disk['timings'])

    def test_phira_tier_preset_product_decoding_and_phira_constant(self):
        from beatfcn.events.phira import load_package
        out = self.out('phira')
        with end_bias(-3.):
            r = run_main(['--model', self.model, '--audio', self.audio, '--mode', 'phira', '--out-dir', out,
                          '--tier', 'medium', '--device', 'cpu'], extractor=FakeMert())
        self.assertTrue(r['complete'])
        self.assertTrue(r['decoding']['matches_product_default'])
        self.assertEqual(r['decoding']['condition']['label'], 'charter_difficulty_preset')
        self.assertEqual(r['decoding']['condition']['density'], ds.DIFFICULTY_PRESETS[('phira', 'medium')])
        self.assertTrue(all(s['conditioned_cores'] == s['cores'] == 2 for s in r['samples']))
        self.assertTrue(all(s['charter_timing_cores'] == 0 for s in r['samples']))
        self.assertGreater(r['draft']['notes'], 0)
        self.assertFalse(r['features']['temporary'] and Path(r['features']['dir']).exists())  # temp dir removed
        # c(phira, mp3) = -2 ms: written = model + 2 ms.
        self.assertEqual(r['clock']['c_ms'], -2)
        events = json.loads((out/'draft-events.json').read_text())['events']
        chart = load_package(r['export']['chart']['path'])
        self.assertFalse(chart.quarantined)
        self.assertEqual(sorted((e.onset_us, e.end_us, e.type) for e in chart.events),
                         sorted((e['onset_us']+2_000, e['end_us']+2_000, e['type']) for e in events))

    def test_two_step_reuse_skips_extraction_and_refuses_test_features_without_extractor(self):
        feat, mert = self.out('reuse-features'), FakeMert()
        r = run_main(['--extract-only', '--audio', self.audio, '--features-dir', feat, '--out-dir',
                      self.out('reuse-extract')], extractor=mert)
        self.assertTrue(r['extract_only'] and r['complete'])
        self.assertEqual(mert.calls, 1)
        with end_bias(-2.):
            again = run_main(['--model', self.model, '--audio', self.audio, '--mode', 'mania_4k', '--out-dir',
                              self.out('reuse-draft'), '--features-dir', feat, '--consensus', '1', '--device', 'cpu'],
                             extractor=mert)
        self.assertEqual(mert.calls, 1)
        self.assertTrue(again['features']['reused'] and again['complete'])
        self.assertEqual(again['audio']['codec_source'], 'feature receipt (ffprobe at extraction)')
        self.assertEqual(again['decoding']['aggregation'], 'single sample')
        with self.assertRaisesRegex(SystemExit, 'injected'):
            run_main(['--model', self.model, '--audio', self.audio, '--mode', 'mania_4k', '--out-dir',
                      self.out('reuse-refused'), '--features-dir', feat, '--device', 'cpu'])

    def test_incomplete_samples_are_never_exported_and_out_dir_never_overwritten(self):
        out = self.out('incomplete')
        with end_bias(-2.), self.assertRaisesRegex(SystemExit, 'Incomplete'):
            run_main(['--model', self.model, '--audio', self.audio, '--mode', 'mania_4k', '--out-dir', out,
                      '--token-cap', '1', '--device', 'cpu'], extractor=FakeMert())
        disk = json.loads((out/'draft-receipt.json').read_text())
        self.assertFalse(disk['complete'])
        self.assertTrue(all(s['failure'] == 'core_token_limit' for s in disk['samples']))
        self.assertFalse(list(out.glob('*.osz')))
        with self.assertRaisesRegex(SystemExit, 'not empty'):
            run_main(['--model', self.model, '--audio', self.audio, '--mode', 'mania_4k', '--out-dir', out,
                      '--device', 'cpu'], extractor=FakeMert())

    def test_model_files_exported_and_checkpoint_load_strictly(self):
        exported = ds.load_model(self.model, 'cpu')
        self.assertEqual((exported[1]['kind'], exported[1]['weights']), ('exported', 'ema'))
        self.assertEqual(exported[2], dict(path=str(self.stats), sha256=sha(self.stats)))
        with self.assertRaisesRegex(SystemExit, 'only its'):
            ds.load_model(self.model, 'cpu', 'raw')
        config = tiny_config()
        torch.manual_seed(1)
        raw, ema = FullEventModel(config), FullEventModel(config)
        path = self.root/'checkpoint.pt'
        torch.save(dict(config=asdict(config), vocabulary={}, model=raw.state_dict(), updates=7,
                        ema=dict(model=ema.state_dict(), decay=.999)), path)
        for weights, source in (('auto', ema), ('ema', ema), ('raw', raw)):
            model, identity, declared = ds.load_model(path, 'cpu', weights)
            self.assertEqual(identity['kind'], 'training_checkpoint')
            self.assertIsNone(declared)
            self.assertTrue(all(torch.equal(v, source.state_dict()[k]) for k, v in model.state_dict().items()))
        bad = self.root/'bad.pt'
        torch.save(dict(config=asdict(config), model={k: v for k, v in raw.state_dict().items()
                                                      if not k.startswith('output')}), bad)
        with self.assertRaises(RuntimeError):  # strict=True
            ds.load_model(bad, 'cpu')


class TableTests(unittest.TestCase):
    def test_codec_constants_match_the_database_table(self):
        stats = ROOT/ds.CODEC_TABLE_SOURCE['path']
        if not stats.is_file():
            self.skipTest('database manifest stats not local')
        self.assertEqual(sha(stats), ds.CODEC_TABLE_SOURCE['sha256'])
        table = json.loads(stats.read_text())['offset']
        got = {tuple(key.split('|')): (v['c_ms'], v['n_train'], v['basis']) for key, v in table.items()}
        self.assertEqual(got, ds.CODEC_CONSTANTS)

    def test_codec_constant_lookup_fallback_and_override(self):
        self.assertEqual(ds.codec_constant('mania_4k', 'vorbis')['c_ms'], 14)
        self.assertEqual(ds.codec_constant('phira', 'vorbis')['c_ms'], 10)
        pooled = ds.codec_constant('mania_4k', 'opus')
        self.assertEqual((pooled['c_ms'], pooled['n_train']), (14, 15212))
        self.assertTrue(pooled['basis'].startswith('source_pooled'))
        self.assertEqual(ds.codec_constant('phira', 'aac')['c_ms'], 0)
        self.assertEqual(ds.codec_constant('mania_4k', 'mp3', 9)['c_ms'], 9)

    def test_presets_match_report_and_render_like_the_benchmark_preset(self):
        report = ROOT/ds.PRESET_SOURCE['report']
        if report.is_file():
            presets = json.loads(report.read_text())['difficulty_presets']
            self.assertEqual({tuple(k.split('/')): v for k, v in presets.items()}, ds.DIFFICULTY_PRESETS)
        from beatfcn.model.full_event_codec import EventRequest
        panel = dict(songs=[dict(song_id='s', mode='mania_4k', references=[dict(chart_id='osu:1',
                                                                                stratum='density/high')])])
        theirs = wb.preset_condition_provider(panel, ds.DIFFICULTY_PRESETS, cell_us=100_000)
        ours = ds.constant_density_provider(ds.DIFFICULTY_PRESETS[('mania_4k', 'high')], cell_us=100_000,
                                            record=dict(label='charter_difficulty_preset'))
        request = EventRequest('mania_4k', 8_000_000, 13_500_000, 20_000_000)
        a, b = theirs(dict(chart_id='s', mode='mania_4k'), request), ours(dict(chart_id='s', mode='mania_4k'), request)
        self.assertEqual(set(a['inputs']), set(b['inputs']))
        for key in a['inputs']:
            np.testing.assert_array_equal(a['inputs'][key], b['inputs'][key])

    def test_mel_normalization_resolution(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'stats.json'
            path.write_text(json.dumps(dict(schema='train-mel-channel-v1', train_audios=['f'*64], mean=[0.]*128,
                                            std=[1.]*128)))
            stats, identity = ds.resolve_mel_normalization(dict(path=str(path), sha256=sha(path)))
            self.assertEqual(identity['sha256'], sha(path))
            with self.assertRaisesRegex(SystemExit, 'identity mismatch'):
                ds.resolve_mel_normalization(dict(path=str(path), sha256='0'*64))
            with self.assertRaisesRegex(SystemExit, 'does not declare'):
                ds.resolve_mel_normalization(None)
            self.assertEqual(ds.resolve_mel_normalization(None, mel_stats=path)[1]['path'], str(path))
        self.assertIsNone(ds.resolve_mel_normalization('none')[0])
        known = next(iter(ds.KNOWN_MEL_STATS))
        if (ROOT/ds.KNOWN_MEL_STATS[known]).is_file():  # volume path absent locally -> the repo copy
            _, identity = ds.resolve_mel_normalization(dict(path='/data/data-contract-v2-mel.json', sha256=known))
            self.assertEqual(identity['path'], str(ROOT/ds.KNOWN_MEL_STATS[known]))

    def test_splice_replaces_only_the_timing_block(self):
        from beatfcn.events.chart_export import osu_mania_bytes
        body, _ = osu_mania_bytes([dict(onset_us=1_000_000, end_us=1_000_000, type='tap', lane=1)])
        spliced = ds.splice_timing_points(body, TIMING_LINES)
        self.assertEqual(ds.timing_section_lines(body), ['0,500,4,1,0,100,1,0'])  # the exporter placeholder
        self.assertEqual(ds.timing_section_lines(spliced), TIMING_LINES)
        parts = lambda b: (b.split(b'[TimingPoints]')[0], b.split(b'[HitObjects]')[1])  # noqa: E731
        self.assertEqual(parts(body), parts(spliced))
        self.assertEqual([(e.onset_us, e.location.lane) for e in parse_osu(spliced, 'x.osu').events],
                         [(1_000_000, 1)])


if __name__ == '__main__':
    unittest.main()
