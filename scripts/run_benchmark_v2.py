"""Run benchmark v2 (teacher-forced audio-dependence screen + batched generation).

Checkpoint mode (any device; never schedules cloud work):
  .venv/bin/python -m scripts.run_benchmark_v2 --plan PLAN.json --checkpoint step-XXXX.pt \
      --teacher-protocol reports/benchmark-v2/<id>/teacher-panel.json \
      --generation-protocol reports/benchmark-v2/<id>/generation-panel.json \
      --out-dir reports/benchmark-v2/<id>/runs --device cuda --max-wall-s 300

Local plumbing smoke (tiny RANDOM-INIT model on locally cached MERT features of
the val songs; proves the plumbing only, says nothing about any checkpoint):
  .venv/bin/python -m scripts.run_benchmark_v2 --local-smoke --device cpu --out-dir /tmp/bench-smoke
"""
import argparse
import glob
import json
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]
TAPE = 'inputs/native-v2-d94-580/tape.json'
MERT_GLOB = 'inputs/d96_mert2_features_v2/worker*/{audio}'
MEL_DIR = 'inputs/d66_local_cohort/decoded/mel10_complete'


def local_mert_store():
    """MertFeatureStore over local MERT folders (no receipt.json) + the local complete mel."""
    import numpy as np
    from beatfcn.train.mert_features import MertFeatureStore

    class LocalMertFeatures(MertFeatureStore):
        def __init__(self):
            super().__init__(ROOT, ROOT)
            self.folders = {}

        def available(self, audio):
            folders = glob.glob(str(ROOT/MERT_GLOB.format(audio=audio)))
            ok = bool(folders) and (ROOT/MEL_DIR/f'{audio}.npy').is_file() and (
                Path(folders[0])/'L12.npy').is_file()
            if ok:
                self.folders[audio] = Path(folders[0])
            return ok

        def ensure(self, audio):
            if audio in self.records:
                return self.records[audio]
            metadata = json.loads((self.folders[audio]/'metadata.json').read_text())
            samples = int(metadata['real_samples'])
            self.records[audio] = dict(duration_us=samples*1_000_000//24000, real_samples=samples)
            return self.records[audio]

        def _open(self, audio):
            folder = self.folders[audio]
            self.ensure(audio)
            l12 = np.load(folder/'L12.npy', mmap_mode='r', allow_pickle=False)
            l23 = np.load(folder/'L23.npy', mmap_mode='r', allow_pickle=False)
            times = np.load(folder/'times_s.npy', allow_pickle=False)
            mel = np.load(ROOT/MEL_DIR/f'{audio}.npy', mmap_mode='r', allow_pickle=False)
            return l12, l23, times, mel
    return LocalMertFeatures()


def local_smoke(args):
    import torch
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from beatfcn.train.periodic_requirements import run_periodic_screen
    from beatfcn.train.checkpoint_resume import file_sha
    from scripts.freeze_benchmark_v2 import TapePanel, freeze
    store = local_mert_store()
    tape = json.loads((ROOT/TAPE).read_text())
    audios = {c['audio_sha256'] for c in tape['charts'] if c['split'] == 'val' and store.available(c['audio_sha256'])}
    if not audios:
        raise SystemExit('No local MERT+mel features for any val song')
    panel = TapePanel(ROOT, TAPE, features=store, audios=audios)
    for chart in panel.charts:
        store.ensure(chart['audio_sha256'])
        if abs(store.records[chart['audio_sha256']]['duration_us']-int(chart['duration_us'])) > 1:
            raise ValueError('Tape and local feature clocks disagree')
        store.records[chart['audio_sha256']]['duration_us'] = int(chart['duration_us'])
    teacher, generation = freeze(ROOT, TAPE, windows=args.windows, per_mode_positive=3, per_mode_silence=1,
                                 anchors_per_mode=1, audios=audios)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, value in (('teacher-panel.json', teacher), ('generation-panel.json', generation)):
        (out/name).write_text(json.dumps(value, indent=1)+'\n')
    torch.manual_seed(0)
    config = FullEventConfig(d=64, heads=4, kv_heads=2, ffn=128, encoder_layers=1, decoder_layers=2,
        tap_dim=1024, taps=2, bands=1, band_slots=1, dropout=0., mel_bins=128, intensity_dim=3)
    model = FullEventModel(config).to(args.device)
    plan = dict(benchmark_v2=dict(max_wall_s=args.max_wall_s, teacher_protocol=str(out/'teacher-panel.json'),
        teacher_protocol_sha256=file_sha(out/'teacher-panel.json'),
        generation_protocol=str(out/'generation-panel.json'),
        generation_protocol_sha256=file_sha(out/'generation-panel.json'),
        history_dir=str(out/'runs'), latest_output=str(out/'latest.json'),
        readouts=[dict(name='sample_t1', temperature=1., top_p=None, seeds=[0, 1, 2, 3]), dict(name='greedy')],
        max_batch_rows=args.max_batch_rows))
    started = time.monotonic()
    meta = run_periodic_screen(model, panel, plan, root='/', device=args.device, update=0,
                               checkpoint=dict(path=None, sha256=None, note='random-init smoke model'))
    meta['wall_s'] = time.monotonic()-started
    meta['val_songs_with_local_features'] = len(audios)
    meta['teacher_windows'] = len(teacher['windows'])
    print(json.dumps(meta, indent=1))


def extra_readouts(args):
    """Opt-in decoding readouts (D3 onset fusion, D4 CFG); none unless the flags are given.

    CFG runs on every case of the readout; it only differs from plain decoding where
    the conditional inputs specify something (the density anchors -- free cases and
    audio interventions have unspecified conditions, so their guided logits equal
    the conditional ones)."""
    out = []
    if getattr(args, 'onset_fusion', None):
        out.append(dict(name='onset_fusion', temperature=1., top_p=None, seeds=list(range(args.seeds)),
                        onset_fusion=dict(weight=float(args.onset_fusion))))
    for weight in getattr(args, 'guidance', None) or ():
        out.append(dict(name=f'cfg_w{weight:g}', temperature=1., top_p=None, seeds=list(range(args.seeds)),
                        guidance=float(weight)))
    return out


def checkpoint_run(args):
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    from beatfcn.train.checkpoint_resume import file_sha
    from beatfcn.train.periodic_requirements import run_periodic_screen
    from beatfcn.train.training_panel import build_training_panel
    plan = json.loads(args.plan.read_text())
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False, mmap=True)
    config = FullEventConfig(**saved['config'])
    vocab = EventVocabulary(**saved.get('vocabulary', {}))
    model = FullEventModel(config, vocab).to(args.device)
    weights = getattr(args, 'weights', 'raw')
    if weights == 'ema':
        if not saved.get('ema'):
            raise SystemExit('Checkpoint has no EMA state; use --weights raw')
        model.load_state_dict(saved['ema']['model'], strict=True)
    else:
        model.load_state_dict(saved['model'], strict=True)
    updates = saved['updates']//int(plan.get('accum_steps', 1))
    del saved
    panel = build_training_panel(args.root, plan)
    settings = dict(max_wall_s=args.max_wall_s, teacher_protocol=str(args.teacher_protocol.resolve()),
        teacher_protocol_sha256=file_sha(args.teacher_protocol),
        generation_protocol=str(args.generation_protocol.resolve()),
        generation_protocol_sha256=file_sha(args.generation_protocol),
        history_dir=str(args.out_dir.resolve()), latest_output=str((args.out_dir/'latest.json').resolve()),
        max_batch_rows=args.max_batch_rows, readouts=[dict(name='sample_t1', temperature=1., top_p=None,
            seeds=list(range(args.seeds))), dict(name='greedy')] + ([dict(name='sample_t1_fieldT', temperature=1., top_p=None,
            seeds=list(range(args.seeds)), field_temperature=args.field_temperature)] if args.field_temperature else [])
            + extra_readouts(args))
    if getattr(args, 'charter_timing', None):
        # Plans with charter_timing: 'without' = the cold readout (no beat features), 'with' = the
        # charter_timing_input readout. Absent = the screen default ('with'), settings unchanged.
        settings['charter_timing'] = args.charter_timing
    meta = run_periodic_screen(model, panel, dict(plan, benchmark_v2=settings, plan_sha256=file_sha(args.plan)),
        root='/', device=args.device, update=updates,
        checkpoint=dict(path=str(args.checkpoint), sha256=file_sha(args.checkpoint)))
    print(json.dumps(meta, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--local-smoke', action='store_true')
    p.add_argument('--plan', type=Path)
    p.add_argument('--checkpoint', type=Path)
    p.add_argument('--teacher-protocol', type=Path)
    p.add_argument('--generation-protocol', type=Path)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--root', type=Path, default=ROOT)
    p.add_argument('--device', default='cpu')
    p.add_argument('--max-wall-s', type=float, default=300.)
    p.add_argument('--max-batch-rows', type=int, default=32)
    p.add_argument('--seeds', type=int, default=4)
    p.add_argument('--windows', type=int, default=200)
    p.add_argument('--field-temperature', type=json.loads, default=None, help='extra readout with per-field temperatures')
    p.add_argument('--weights', choices=('raw', 'ema'), default='raw', help='checkpoint weights to evaluate')
    p.add_argument('--charter-timing', choices=('with', 'without'),
                   help="plans with charter_timing: 'without' = cold readout; default = the screen default ('with')")
    p.add_argument('--onset-fusion', type=float, help='extra readout onset_fusion (D3): T=1 + weight*log p_onset on '
                   'head-time logits, e.g. 0.5')
    p.add_argument('--guidance', type=float, nargs='+', help='extra CFG readouts cfg_w<W> (D4), T=1; differ from '
                   'sample_t1 only on the density anchors, e.g. --guidance 1.5 2.5')
    a = p.parse_args()
    if a.local_smoke:
        local_smoke(a)
    else:
        missing = [k for k in ('plan', 'checkpoint', 'teacher_protocol', 'generation_protocol') if getattr(a, k) is None]
        if missing:
            p.error('checkpoint mode needs --'+', --'.join(m.replace('_', '-') for m in missing))
        checkpoint_run(a)


if __name__ == '__main__':
    main()
