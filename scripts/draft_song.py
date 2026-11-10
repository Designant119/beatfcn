"""Draft an editor-openable chart for a NEW audio file: osu!mania 4K .osz or Phira preview zip.

Every stage reuses the database / benchmark code; nothing here re-implements a model input:
  1. decode      beatfcn.ingest.pipeline.decode -- ffmpeg -> 24 kHz mono float32 (the database decoder).
  2. features    beatfcn.train.full_database.feature_payloads + commit_feature (the database writer): frozen
                 MERT-v2 (m-a-p/MERT-v2-FullSong @ beatfcn.audio.mert2.REVISION, layers 12/23, bf16, 40 ms frames
                 centred at (k*960+360)/24000 s; > 360 s: 300 s windows, 100 s overlap) + the complete-tail 10 ms
                 mel (beatfcn.audio.local_mel_cache.complete_song_mel) into
                 <features-dir>/features/<audio_sha256>/{L12,L23,times_s,intervals_s,mel10_complete}.npy + receipt.json,
                 the layout of full-database-v1/mert-graph-diagnostic-v2/features.
  3. inputs      beatfcn.train.mert_features.MertFeatureStore (unchanged: the store of every MERT plan) wrapped in
                 beatfcn.train.mel_normalization.MelNormalizedFeatures with the model's train-only mel statistics
                 (sha256-checked), as beatfcn.train.training_panel.build_training_panel does.
  4. generation  beatfcn.model.local_song_rollout.generate_complete_song_samples (8 s cores, per-row seeds) with
                 scripts.run_workflow_benchmark.make_generator. Charter requests (labelled, never capability):
                 --tier (train-median heads/s of the tier, charter_difficulty_preset) or --density
                 (charter_intent_density), --timing (charter_timing_input beat features; config.beat_features models).
  5. aggregation scripts.run_workflow_benchmark.consensus_draft (D6: 30 ms onset clusters, majority vote).
  6. export      scripts.run_workflow_benchmark.export_song on the PLAYBACK clock: written time = model time - c,
                 c = the database's per-(source, codec) constant (osu: mp3/ogg +14 ms; Phira: mp3 -2, ogg +10 ms,
                 ...). With --timing the chart's [TimingPoints] block is written verbatim (it already is on the
                 playback clock) and the model sees it shifted by +c.
  7. receipts    <out-dir>/draft-receipt.json (decoding settings, model sha256, feature identity, clock shift,
                 timings) and <out-dir>/draft-events.json (the draft on the model clock).

Product decoding default (reports/goal-100usd-20261101/ft/DECODING_DECISION.md, PLAN.md E8): consensus over 8
samples at T=1 with field temperature {kind .3, head_block .7, head_cell .7, head_ms .7}.

Local (feature extraction needs ffmpeg/ffprobe, transformers==4.53.2 + torchaudio -- the database MERT image --
and the pinned MERT-v2 source: --mert-source DIR holding config.json, model.safetensors, configuration_mert2.py,
modeling_mert2.py; default /data/full-database-v1/mert2-checkpoint, else the repo's inputs/d96_mert2_source):
  .venv/bin/python -m scripts.draft_song --model beatfcn-ft-l322m.pt --audio song.mp3 --mode mania_4k \
      --out-dir drafts/song [--timing chart.osu] [--tier medium | --density 6.5] [--consensus 8] [--device mps]
Two steps (extract where MERT runs, draft anywhere; the feature dir keeps the probed codec):
  ... -m scripts.draft_song --extract-only --audio song.mp3 --features-dir F --out-dir drafts/song-features
  ... -m scripts.draft_song --features-dir F --audio song.mp3 --model M --mode phira --out-dir drafts/song

Modal: beatfcn/cloud/train_stage0_app.py run_script_l40s(module, args_json, sync_json) runs
`python -m scripts.<module> <args>` in /root/repo and copies each sync path to /data/stage0/<path>. Its image has
torch + numpy + pyyaml but no ffmpeg, transformers or torchaudio, so there only the generation/export half runs:
extract features elsewhere (--extract-only), put the feature dir and the audio on the volume, then
  modal.Function.from_name('stage0-cuda-training', 'run_script_l40s').remote('draft_song', json.dumps([
      '--model', '/data/stage0/inputs/main-l-v2-ft/checkpoints/step-80000.pt', '--plan', '/data/main-l-v2-ft/plan.json',
      '--weights', 'ema', '--audio', '/data/stage0/drafts/song.mp3', '--features-dir', '/data/stage0/drafts/song-features',
      '--mode', 'mania_4k', '--device', 'cuda', '--out-dir', 'reports/drafts/song']), json.dumps(['reports/drafts/song']))
(or --model <exported .pt>; its provenance mel statistics resolve to /data/data-contract-v2-mel.json on the volume).

Never trains, never schedules cloud work.
"""
import argparse
from dataclasses import asdict
import gc
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import scripts.run_workflow_benchmark as wb  # noqa: E402

SCHEMA = 'beatfcn.draft-song.v1'
MODES = ('mania_4k', 'phira')
PLATFORM_SOURCE = {'mania_4k': 'osu', 'phira': 'phira'}
PRODUCT_CONSENSUS = 8
PRODUCT_TEMPERATURE = 1.
PRODUCT_FIELD_TEMPERATURE = {'kind': 1., 'head_block': .7, 'head_cell': .7, 'head_ms': .7}
PRODUCT_VOTES = 3
PRODUCT_SOURCE = ('reports/goal-100usd-20261101/PLAN.md 2026-11-05: kind T 1, head T .7, anchor pattern, votes 3, fragment '
                  'guard; HOLD-100 and owner blind tests')

# Per-(source, codec) model-clock constant c_ms of the database: database targets = raw chart time +
# offset_decision shift_ms + c_ms (beatfcn.train.data.model_shift_us). c = train median of the chart-vs-audio
# offset diagnostic per (source, codec); < 20 train charts -> the source-pooled median
# (beatfcn.ingest.pipeline.finalize_rows, beatfcn.ingest.offset.train_constant).
# (source, codec): (c_ms, n_train, basis)
CODEC_CONSTANTS = {
    ('osu', 'mp3'): (14, 11773, 'source_codec'),
    ('osu', 'vorbis'): (14, 3439, 'source_codec'),
    ('phira', 'flac'): (0, 8, 'source_pooled'),
    ('phira', 'mp3'): (-2, 2308, 'source_codec'),
    ('phira', 'pcm_f32le'): (0, 2, 'source_pooled'),
    ('phira', 'pcm_s16le'): (29, 58, 'source_codec'),
    ('phira', 'pcm_s24le'): (0, 0, 'source_pooled'),
    ('phira', 'pcm_u8'): (0, 1, 'source_pooled'),
    ('phira', 'vorbis'): (10, 396, 'source_codec'),
}
# Source-pooled train medians (the finalize_rows fallback for a codec without its own constant), recomputed from
# the offset-v2 manifest rows (parsed, train, offset ok, no P0 quarantine).
SOURCE_POOLED_CONSTANTS = {'osu': (14, 15212), 'phira': (0, 2773)}
CODEC_TABLE_SOURCE = dict(
    path='inputs/full-database-cloud-v1/manifest/stats.json', key='offset',
    sha256='18d63e774dcda9bd862b3c8cd4af6b3ea4177e2b16a1591786d57441f2f4c861',
    pooled_from='inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl '
                '(sha256 676032329c43f829e0aa4439cdab6ac6fddd9c8361a5014e350c3e2e382e4652)',
    rule='c = round(median delta_ms) of parsed TRAIN charts with an ok offset diagnostic and no P0 quarantine per '
         '(source, codec); fewer than 20 -> the source-pooled median (beatfcn.ingest.pipeline.finalize_rows)')

# Tier -> constant heads/s request = median heads/s of TRAIN charts per (mode, density tertile) of the full-database
# native-v2 tape (scripts.run_workflow_benchmark.difficulty_presets), as recorded by the preset readout.
DIFFICULTY_PRESETS = {
    ('mania_4k', 'low'): 4.267, ('mania_4k', 'medium'): 8.494, ('mania_4k', 'high'): 13.178,
    ('phira', 'low'): 4.011, ('phira', 'medium'): 5.669, ('phira', 'high'): 7.452,
}
PRESET_SOURCE = dict(
    report='reports/goal-100usd-20261101/ft/wf_preset_cfg1/workflow-benchmark.json', key='difficulty_presets',
    sha256='e6e2544bd71239c51e559c8a2fe0fe92f019f2955790591b9078b4229728defc',
    tape='/data/full-database-v1/native-v2-tape/tape.json (sha256 0aaab6e82903ed19a5335f97f0dc5f1d37148a688530387e87e754bac893538f)',
    rule='median of sum(request event_count)/duration_s over TRAIN charts per (mode, density tertile); tiers are '
         'heads/s tertiles of the train split (assign_density_strata), not charter difficulty labels')

# Local copies of known train-only mel statistics (sha256 -> repo path); the exported model declares the sha256.
KNOWN_MEL_STATS = {
    'd87cad1d911c6001654e7fb2ca62e6eff8352e002aa602e24c3a36e32bb3e3c6':
        'reports/data-contract-v2-final/mel-statistics.json',
}
DATABASE_MERT_SOURCE = '/data/full-database-v1/mert2-checkpoint'  # beatfcn/cloud/full_database_app.py CHECKPOINT
# Repo-local pinned snapshot (scripts/cache_local_mert_features.py layout): weights and custom code split.
LOCAL_MERT_SPLIT = {'config.json': 'checkpoint/config.json', 'model.safetensors': 'checkpoint/model.safetensors',
                    'configuration_mert2.py': 'mert2_pinned/configuration_mert2.py',
                    'modeling_mert2.py': 'mert2_pinned/modeling_mert2.py'}
DATABASE_FEATURES = dict(
    root='/data/full-database-v1/mert-graph-diagnostic-v2/features (reports/full-database-cloud-v1/'
         'mert-full-corpus-completion.json: 10,897 receipts)',
    encoder='beatfcn.audio.mert2.FrozenMert2(device="cuda", local_source=/data/full-database-v1/mert2-checkpoint), '
            'bf16, torch 2.6.0 + transformers 4.53.2 (beatfcn/cloud/full_database_app.py image)',
    paths='formal eager runs: FrozenMert2.encode per song (windows above 360 s) + CPU complete_song_mel; '
          'graph runs (<= 360 s songs only): exact-shape CUDA graph of the same forward, gated on bitwise parity '
          'with FrozenMert2.encode, + complete_song_mel_cuda',
    decoder='ffmpeg 5.1.9-0+deb12u1 (inputs/full-database-cloud-v1/manifest/stats.json "ffmpeg")',
    backend='cuda_bf16')
CODE_FILES = ('scripts/draft_song.py', 'scripts/run_workflow_benchmark.py', 'scripts/build_chart_timing.py',
              'beatfcn/model/local_song_rollout.py', 'beatfcn/model/full_event_generate.py',
              'beatfcn/model/full_event_model.py', 'beatfcn/train/mert_features.py',
              'beatfcn/train/mel_normalization.py', 'beatfcn/train/full_database.py', 'beatfcn/audio/mert2.py',
              'beatfcn/audio/local_mel_cache.py', 'beatfcn/ingest/pipeline.py', 'beatfcn/events/chart_export.py')
TIMING_ISSUES = {'bad_timing_point', 'uninherited_nonpositive_beatlength', 'bad_uninherited_flag',
                 'no_uninherited_timing_point'}


def write_json(path, value):
    path = Path(path)
    tmp = path.with_name(path.name+'.partial')
    tmp.write_text(json.dumps(value, indent=1, ensure_ascii=False, default=str)+'\n')
    tmp.replace(path)


def auto_device():
    import torch
    return 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


def tool_version(name):
    try:
        out = subprocess.run([name, '-version'], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.splitlines()[0].strip() if out.returncode == 0 and out.stdout else None


# ------------------------------------------------------------------ clock
def codec_constant(mode, codec, override_ms=None):
    """Model clock -> playback clock for an export to ``mode``'s platform with audio ``codec``."""
    source = PLATFORM_SOURCE[mode]
    if override_ms is not None:
        return dict(source=source, codec=codec, c_ms=int(override_ms), n_train=None, basis='user override',
                    table=CODEC_TABLE_SOURCE)
    if (source, codec) in CODEC_CONSTANTS:
        c_ms, n_train, basis = CODEC_CONSTANTS[(source, codec)]
    else:
        c_ms, n_train = SOURCE_POOLED_CONSTANTS[source]
        basis = 'source_pooled (codec has no database constant)'
    return dict(source=source, codec=codec, c_ms=c_ms, n_train=n_train, basis=basis, table=CODEC_TABLE_SOURCE)


# ------------------------------------------------------------------ model
def load_model(path, device, weights='auto'):
    """Exported model (scripts/export_final_model.py) or training checkpoint -> (eval model, identity, declared
    mel normalization). Strict load; checkpoints use EMA weights when present unless --weights raw."""
    import torch
    from beatfcn.model.full_event_codec import EventVocabulary
    from beatfcn.model.full_event_model import FullEventConfig, FullEventModel
    path = Path(path)
    saved = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    if not isinstance(saved, dict) or 'config' not in saved or 'model' not in saved:
        raise SystemExit(f'{path}: neither an exported model nor a training checkpoint (needs config + model)')
    exported = saved.get('schema') == 'beatfcn.final-model.v1'
    if exported:
        chosen = saved.get('weights')
        if weights not in ('auto', chosen):
            raise SystemExit(f'{path} is an exported model holding only its {chosen!r} weights')
        state = saved['model']
    elif weights == 'raw' or (weights == 'auto' and not saved.get('ema')):
        chosen, state = 'raw', saved['model']
    else:
        if not saved.get('ema'):
            raise SystemExit(f'{path} has no EMA state; use --weights raw')
        chosen, state = 'ema', saved['ema']['model']
    config = FullEventConfig(**saved['config'])
    vocabulary = dict(saved.get('vocabulary') or {})
    model = FullEventModel(config, EventVocabulary(**vocabulary))
    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()
    provenance = saved.get('provenance') if exported else None
    declared = (provenance or {}).get('mel_normalization') if exported else None
    identity = dict(path=str(path), sha256=wb.sha_file(path), bytes=path.stat().st_size,
                    kind='exported' if exported else 'training_checkpoint', schema=saved.get('schema'),
                    weights=chosen, parameters=sum(p.numel() for p in model.parameters()), provenance=provenance,
                    updates=saved.get('updates'), plan_sha256=saved.get('plan_sha256'), config=asdict(config),
                    vocabulary=vocabulary)
    del saved, state
    return model, identity, declared


def resolve_mel_normalization(declared, *, mel_stats=None, root=ROOT):
    """(statistics or None, identity) for the model's declared train-only mel normalization.

    ``declared``: 'none' | {path, sha256} (exported provenance / plan key) | None (undeclared: --mel-stats).
    The statistics file is located at --mel-stats, the declared path (absolute or repo-relative) or a known local
    copy, and must match the declared sha256."""
    if declared == 'none':
        return None, dict(applied=False, declared='none')
    expected = None if declared is None else declared.get('sha256')
    if declared is None and mel_stats is None:
        raise SystemExit('The model does not declare its mel normalization: pass --plan (training checkpoints) or '
                         '--mel-stats')
    candidates = []
    if mel_stats is not None:
        candidates.append(Path(mel_stats))
    if declared is not None and declared.get('path'):
        declared_path = Path(declared['path'])
        candidates += [declared_path] if declared_path.is_absolute() else [root/declared_path]
    if expected in KNOWN_MEL_STATS:
        candidates.append(root/KNOWN_MEL_STATS[expected])
    tried = []
    for candidate in candidates:
        if not candidate.is_file():
            tried.append(f'{candidate} (missing)')
            continue
        digest = wb.sha_file(candidate)
        if expected is not None and digest != expected:
            tried.append(f'{candidate} (sha256 {digest[:12]} != {expected[:12]})')
            continue
        stats = json.loads(candidate.read_text())
        return stats, dict(applied=True, declared=declared, path=str(candidate), sha256=digest,
                           schema=stats.get('schema'), frames=stats.get('frames'),
                           train_audios=len(stats.get('train_audios', ())),
                           applied_by='beatfcn.train.mel_normalization.MelNormalizedFeatures '
                                      '(as beatfcn.train.training_panel.build_training_panel)')
    raise SystemExit('Mel normalization statistics not found / identity mismatch; tried: ' + '; '.join(tried))


# ------------------------------------------------------------------ features
def resolve_mert_source(arg, root=ROOT):
    """(directory for FrozenMert2(local_source=...) or None for the pinned Hugging Face revision, identity, temp)."""
    from beatfcn.audio.mert2 import LOCAL_SOURCE_SHA256, REPO, REVISION
    if arg == 'hf':
        return None, dict(kind='huggingface', repo=REPO, revision=REVISION), None
    candidates = [Path(arg)] if arg else [Path(DATABASE_MERT_SOURCE), root/'inputs/d96_mert2_source']
    for candidate in candidates:
        if all((candidate/name).is_file() for name in LOCAL_SOURCE_SHA256):
            return candidate, dict(kind='directory', path=str(candidate), expected_sha256=LOCAL_SOURCE_SHA256), None
        split = {name: candidate/rel for name, rel in LOCAL_MERT_SPLIT.items()}
        if all(p.is_file() for p in split.values()):
            # One directory as from_pretrained/validate_local_source expect: symlinks to the pinned files.
            assembled = Path(tempfile.mkdtemp(prefix='beatfcn-mert-source-'))
            for name, p in split.items():
                (assembled/name).symlink_to(p.resolve())
            return assembled, dict(kind='assembled_symlinks', path=str(candidate),
                                   files={k: str(v) for k, v in split.items()},
                                   expected_sha256=LOCAL_SOURCE_SHA256), assembled
    raise SystemExit('Pinned MERT-v2 source not found (tried ' + ', '.join(map(str, candidates)) + '); pass '
                     '--mert-source DIR (config.json, model.safetensors, configuration_mert2.py, modeling_mert2.py) '
                     "or --mert-source hf to download the pinned revision")


def make_extractor(device, source):
    """The database MERT encoder: FrozenMert2 on cuda (the database path) / cpu; on mps the local execution policy
    of scripts/cache_local_mert_features.py (CPU FP32 frontend, query-chunked SDPA, bf16 encoder on MPS)."""
    from beatfcn.audio.mert2 import FrozenMert2
    if device != 'mps':
        return FrozenMert2(device=device, local_source=source), f'{device}_bf16'
    from beatfcn.audio.local_mert import CpuMelTransfer, LocalFrozenMert2
    from beatfcn.audio.mert_attention import install_query_chunks
    base = FrozenMert2(device='cpu', local_source=source)  # pinned-identity load (validate_local_source)
    model = base.model.to('mps')
    model.feature_extractor = CpuMelTransfer(model.feature_extractor, 'mps')
    install_query_chunks(model, 512)
    return LocalFrozenMert2(model, base.spec), 'mps_bf16_cpu_fp32_frontend_query_chunk512'


def extractor_identity(extractor, backend, device, source_identity):
    import torch
    from beatfcn.audio.mert2 import CELL_SAMPLES, REPO, REVISION, SR, Mert2Spec
    from beatfcn.audio.preprocess import PREPROCESS_RECIPE
    try:
        import transformers
        transformers_version = transformers.__version__
    except ImportError:
        transformers_version = None
    spec = getattr(extractor, 'spec', None) or Mert2Spec()
    return dict(cls=f'{type(extractor).__module__}.{type(extractor).__qualname__}', backend=backend, device=device,
                repo=REPO, revision=REVISION, spec=asdict(spec), sample_rate=SR, cell_samples=CELL_SAMPLES,
                frame_clock='times_s = (k*960+360)/24000, intervals [k*960, min((k+1)*960, samples)]/24000',
                dtype='float16 (bf16 forward)', mel='beatfcn.audio.local_mel_cache.complete_song_mel (fp16, '
                '4 frames per 40 ms cell, complete zero-padded tail)', mel_recipe=PREPROCESS_RECIPE,
                torch=torch.__version__, transformers=transformers_version, source=source_identity,
                database=DATABASE_FEATURES)


def prepare_features(audio_path, audio_sha, features_dir, args, *, extractor=None, decoder=None):
    """Existing verified features for this audio, else decode + extract + write the database layout.

    Returns (receipt, record) where record holds the stage timings and whether features were reused."""
    from beatfcn.audio.mert2 import REVISION
    from beatfcn.train.full_database import existing_feature_receipt
    folder = Path(features_dir)/'features'/audio_sha
    if (folder/'receipt.json').is_file():
        receipt = existing_feature_receipt(folder, SimpleNamespace(revision=REVISION))  # revision + file sha256
        backend = str((receipt.get('draft_song') or {}).get('extractor', {}).get('backend', receipt.get('backend')))
        if receipt.get('complete') is not True or receipt.get('audio_sha256') != audio_sha:
            raise SystemExit(f'{folder}: incomplete feature receipt or a different audio')
        if backend.startswith('injected') and extractor is None:
            raise SystemExit(f'{folder} was written by an injected (test) extractor; refusing to draft from it')
        return receipt, dict(reused=True, folder=str(folder))
    import numpy as np
    from beatfcn.ingest.pipeline import decode
    from beatfcn.train.full_database import commit_feature, feature_payloads, write_json as db_write_json
    timings = {}
    codec = args.codec or probe(audio_path)
    tick = time.monotonic()
    wave = (decoder or decode)(str(audio_path))
    timings['decode_s'] = time.monotonic()-tick
    if not len(wave) or not np.isfinite(wave).all():
        raise SystemExit(f'{audio_path}: decoded audio is empty or not finite')
    source_identity, assembled = None, None
    tick = time.monotonic()
    try:
        if extractor is None:
            source, source_identity, assembled = resolve_mert_source(args.mert_source)
            mert_device = args.mert_device or args.device or auto_device()
            extractor, backend = make_extractor(mert_device, source)
        else:
            mert_device = getattr(extractor, 'device', None)
            backend = f'injected:{type(extractor).__name__}'
        timings['mert_load_s'] = time.monotonic()-tick
        identity = extractor_identity(extractor, backend, mert_device, source_identity)
        marks = {}
        tick = time.monotonic()
        features, mel, expected = feature_payloads(
            wave, extractor, phase_callback=lambda phase: marks.__setitem__(phase, time.monotonic()))
        done = time.monotonic()
    finally:
        if assembled is not None:
            shutil.rmtree(assembled, ignore_errors=True)
    timings['mert_s'] = marks['gpu_encode_end']-marks['gpu_encode_start']
    timings['mel_s'] = done-marks['gpu_encode_end']
    scratch = Path(tempfile.mkdtemp(prefix='beatfcn-draft-scratch-'))  # never contains the user's audio
    tick = time.monotonic()
    try:
        receipt = commit_feature(audio_sha, Path(audio_path), features_dir, scratch, features, mel, expected)
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    # commit_feature labels every receipt with the database backend; record the backend that actually ran.
    receipt['backend'] = backend.split(':')[0] if backend.startswith('injected') else backend
    receipt['draft_song'] = dict(schema=SCHEMA, audio=str(audio_path), codec=codec, codec_ext=Path(audio_path).suffix,
                                 decoder=dict(fn='beatfcn.ingest.pipeline.decode',
                                              command='ffmpeg -nostdin -v error -i <audio> -vn -f f32le -ac 1 -ar 24000',
                                              ffmpeg=tool_version('ffmpeg'), injected=decoder is not None),
                                 extractor=identity, timings=dict(timings))
    db_write_json(folder/'receipt.json', receipt)
    timings['features_write_s'] = time.monotonic()-tick
    del extractor, features, mel, wave
    gc.collect()
    _empty_device_cache()
    return receipt, dict(reused=False, folder=str(folder), timings=timings)


def probe(audio_path):
    """ffprobe codec_name of the first audio stream (beatfcn.ingest.pipeline.probe_codec)."""
    from beatfcn.ingest.pipeline import probe_codec
    try:
        return probe_codec(str(audio_path))
    except (OSError, subprocess.SubprocessError) as exc:
        raise SystemExit(f'ffprobe failed on {audio_path} ({type(exc).__name__}: {exc}); pass --codec') from exc


def _empty_device_cache():
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def open_features(features_dir, audio_sha, stats):
    """MertFeatureStore (unchanged) over the feature dir, mel-normalized like the training panel."""
    from beatfcn.train.mel_normalization import MelNormalizedFeatures
    from beatfcn.train.mert_features import MertFeatureStore
    store = MertFeatureStore(Path(features_dir), Path(features_dir)/'features')
    store.ensure(audio_sha)
    store._cache(audio_sha)  # shape / clock / coverage checks of MertFeatureStore._open
    if stats is None:
        return store
    # A new song is never a training audio; the check guards the statistics, not this song.
    return MelNormalizedFeatures(store, stats, train_audios=stats['train_audios'], forbidden_audios=())


# ------------------------------------------------------------------ timing
def timing_section_lines(data):
    """Non-empty lines of the [TimingPoints] section, verbatim (line endings removed)."""
    out, inside = [], False
    for index, raw in enumerate(bytes(data).splitlines()):
        line = raw.decode('utf-8')
        if index == 0:
            line = line.lstrip('﻿')
        stripped = line.strip()
        if stripped.startswith('[') and stripped.endswith(']'):
            inside = stripped == '[TimingPoints]'
        elif inside and stripped:
            out.append(line)
    return out


def _timing_rows(chart):
    side = chart.sidecar or {}
    strip = lambda rows: [{k: v for k, v in r.items() if k != 'line'} for r in rows or ()]  # noqa: E731
    return strip(side.get('timing_rows')), strip(side.get('sv_rows'))


def read_timing(path):
    """Charter timing from a 4K osu!mania .osu: verbatim [TimingPoints] lines, red points [(ms, bpm)] and beat
    segments [(start_us, beat_us, meter)] on the chart's (playback) clock."""
    from beatfcn.events.chart_export import timing_points_from_map
    from beatfcn.events.osu_mania import parse_osu
    from scripts.build_chart_timing import GimmickTiming, timing_segments
    path = Path(path)
    data = path.read_bytes()
    try:
        chart = parse_osu(data, path.name)
    except Exception as exc:  # noqa: BLE001 - surfaced as a CLI error
        raise SystemExit(f'--timing {path}: not a readable 4K osu!mania file ({type(exc).__name__}: {exc})')
    bad = sorted({i.code for i in chart.issues if i.code in TIMING_ISSUES and i.severity == 'quarantine'})
    if chart.timing is None or bad:
        raise SystemExit(f'--timing {path}: unusable timing points ({", ".join(bad) or "no timing"})')
    lines = timing_section_lines(data)
    try:
        segments, meter_source = timing_segments(chart)
        gimmick = None
    except GimmickTiming as exc:
        segments, meter_source, gimmick = None, None, str(exc)
    red, green = _timing_rows(chart)
    return dict(path=str(path), sha256=wb.sha_file(path), lines=lines, red_points=timing_points_from_map(chart.timing),
                segments=segments, meter_source=meter_source, gimmick=gimmick, red_lines=len(red),
                green_lines=len(green), first_bpm=round(60e6/segments[0][1], 4) if segments else None,
                other_quarantine=sorted({i.code for i in chart.quarantine_reasons()} - TIMING_ISSUES))


# ------------------------------------------------------------------ conditions
def constant_density_provider(density, *, cell_us, record):
    """condition_provider: every core requests one constant heads/s ``density`` -- the same rendering as
    scripts.run_workflow_benchmark.preset_condition_provider (counterfactual_exposure.condition_inputs)."""
    from beatfcn.train.counterfactual_exposure import condition_inputs

    def provider(state, request):
        segments = [dict(start_us=int(request.start_us), end_us=int(request.end_us), density=float(density))]
        update = condition_inputs(state['mode'], request, segments=segments, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(record))
    return provider


STYLE_FIELDS = ('hold_head_ratio', 'chord_ratio', 'hold_occupancy_ratio')
# Phira types are steered at decode time: the condition bands barely move flick and drag, a log-probability bias does.
# Measured 2026-11-06 on 30 songs (reports/evolve-20261105/g18): .8 x the logit change of the share lands close by.
KIND_BASE = dict(flick=.071, drag=.043)
KIND_GAIN = 1/.8


def kind_bias(args):
    out = {}
    for kind, field in (('flick', 'flick_ratio'), ('drag', 'drag_ratio')):
        value = getattr(args, field, None)
        if value is None:
            continue
        if args.mode != 'phira':
            raise SystemExit(f'{field} is a Phira control')
        if not 0 < value < 1:
            raise SystemExit(f'{field} must be between 0 and 1')
        base = KIND_BASE[kind]
        out[kind] = KIND_GAIN*(math.log(value/(1-value))-math.log(base/(1-base)))/1
    return out


def style_bands(args):
    """The charter's requested shares as table bands. A field this mode has no control for is an error."""
    from beatfcn.events import controls as C
    bands = {}
    for field in STYLE_FIELDS:
        value = getattr(args, field, None)
        if value is None:
            continue
        if field not in C.MODE_FIELDS[args.mode]:
            raise SystemExit(f'{field} is not a control of mode {args.mode}')
        if not 0 <= value <= 1:
            raise SystemExit(f'{field} must be within 0-1')
        edges = C.BAND_EDGES[field]
        bands[field] = next([lo, hi] for lo, hi in zip(edges, edges[1:]) if lo <= value < hi or (hi == edges[-1] and value == hi))
    return bands


def curve_style_provider(segments, bands, *, cell_us):
    """condition_provider: the charter's section densities (labelled charter_intent_density) plus style bands."""
    from beatfcn.train.counterfactual_exposure import condition_inputs

    def provider(state, request):
        parts = [dict(start_us=max(int(s['start_s']*1e6), request.start_us), end_us=min(int(s['end_s']*1e6), request.end_us),
                      density=float(s['density'])) for s in segments
                 if s['start_s']*1e6 < request.end_us and s['end_s']*1e6 > request.start_us]
        update = condition_inputs(state['mode'], request, bands=bands, segments=parts, render_cell_us=cell_us, scale=16.)
        return dict(inputs=update, record=dict(label='charter_intent_style', segments=parts, bands=bands))
    return provider


def condition_request(args, model, cell_us):
    """(condition_provider or None, guidance or None, record)."""
    level = getattr(args, 'phira_level', None), getattr(args, 'mania_star', None)
    bands = style_bands(args)
    curve = json.loads(args.density_curve.read_text()) if getattr(args, 'density_curve', None) else None
    if curve is not None or (bands and args.tier is None and args.density is None and level == (None, None)):
        if curve is not None and (args.tier is not None or args.density is not None or level != (None, None)):
            raise SystemExit('--density-curve replaces --tier / --density / level requests')
        guidance = None if args.guidance is None or float(args.guidance) == 1. else float(args.guidance)
        return curve_style_provider(curve or [], bands, cell_us=cell_us), guidance, dict(
            label='charter_intent_style', bands=bands, density_curve=curve, render_cell_us=cell_us, guidance=args.guidance)
    if args.tier is None and args.density is None and level == (None, None):
        return None, None, dict(label='unconditioned', rule='unspecified density curve + unconditional controls')
    if not getattr(model.c, 'intensity_dim', 0):
        raise SystemExit('--tier / --density / --phira-level / --mania-star need a model with a density curve input '
                         '(config.intensity_dim > 0)')
    if level != (None, None):
        from beatfcn.events.difficulty_presets import DifficultyPresets
        presets = DifficultyPresets.bundled()
        kind, value = ('phira', level[0]) if level[0] is not None else ('mania_4k', level[1])
        density = presets.heads_per_s(kind, value)
        record = dict(label='difficulty_level_preset', **{('phira_level' if kind == 'phira' else 'mania_star'): value},
                      density=round(density, 3))
        source = presets.source(kind)
    elif args.tier is not None:
        density = DIFFICULTY_PRESETS[(args.mode, args.tier)]
        record = dict(label='charter_difficulty_preset', tier=args.tier, density=density)
        source = PRESET_SOURCE
    else:
        density = float(args.density)
        record = dict(label='charter_intent_density', density=density)
        source = 'user: constant heads/s for every core'
    guidance = None if args.guidance is None or float(args.guidance) == 1. else float(args.guidance)
    provider = constant_density_provider(density, cell_us=cell_us, record=record)
    return provider, guidance, dict(record, unit='heads (notes) per second, constant over every 8 s core',
                                    render_cell_us=cell_us, source=source, guidance=args.guidance,
                                    cfg='off (plain conditioning)' if guidance is None else
                                        'logits_u + w*(logits_c - logits_u); unconditional = unspecified curve + '
                                        'controls')


# ------------------------------------------------------------------ main
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--audio', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, help='exported model (scripts/export_final_model.py) or training checkpoint')
    p.add_argument('--mode', choices=MODES)
    p.add_argument('--timing', type=Path, help="the charter's 4K osu!mania .osu: timing points written verbatim, "
                   'fed to beat_features models as charter_timing_input')
    level = p.add_mutually_exclusive_group()
    level.add_argument('--tier', choices=('low', 'medium', 'high'), help='difficulty tier preset (train-median heads/s)')
    level.add_argument('--density', type=float, help='requested constant heads (notes) per second')
    level.add_argument('--phira-level', type=float, metavar='LV',
                       help='Phira level preset, HD 8 .. AT 16 (database median density of charts at that level)')
    level.add_argument('--mania-star', type=float, metavar='STAR',
                       help='osu!mania 4K star preset, about 1 .. 7 (approximate star; database median density)')
    p.add_argument('--guidance', type=float, help='CFG weight for --tier/--density (default: plain conditioning)')
    p.add_argument('--consensus', type=int, default=None,
                   help='samples per draft (1 = single sample; default: mania 1 -- timing-only, no lane patterns '
                        'to vote on -- and Phira %d)' % PRODUCT_CONSENSUS)
    p.add_argument('--consensus-pattern', choices=('majority', 'anchor'), default='anchor',
                   help='pattern of each kept onset: per-onset majority, or the most central sample')
    p.add_argument('--consensus-votes', type=int, help=f'samples an onset needs (default: {PRODUCT_VOTES} of 8, scaled)')
    for field, name in (('hold_head_ratio', 'hold'), ('chord_ratio', 'chord'), ('drag_ratio', 'drag'), ('flick_ratio', 'flick'),
                        ('hold_occupancy_ratio', 'hold-occupancy')):
        p.add_argument(f'--{name}-ratio', type=float, dest=field, metavar='R', help=f'requested share, 0-1 ({field})')
    p.add_argument('--density-curve', type=Path, help='charter section densities: JSON list of {start_s, end_s, density}')
    p.add_argument('--profile', choices=('relaxed', 'standard', 'expert', 'none'), help="Phira playability limits (default "
                   "standard): fingers, holds down, close onsets. See beatfcn/events/phira_postprocess.py")
    p.add_argument('--consensus-min-jack-ms', type=float, default=100.,
                   help='mania: avoid same-key repeats under this gap when a voted pattern allows it (0 = off)')
    p.add_argument('--no-fragment-guard', dest='fragment_guard', action='store_false',
                   help='keep lone squeezed onsets that fewer than half of the samples wrote')
    p.add_argument('--scheduler', choices=('legacy', 'grouped', 'continuous'), default='continuous',
                   help='batch filling; token-identical, continuous is the fastest on CUDA')
    p.add_argument('--temperature', default=str(PRODUCT_TEMPERATURE), help="sampling temperature or 'none' (greedy)")
    p.add_argument('--field-temperature', type=json.loads, default=None,
                   help=f'JSON per-field temperatures (default {json.dumps(PRODUCT_FIELD_TEMPERATURE)})')
    p.add_argument('--top-p', type=float)
    p.add_argument('--hold-release-ms', type=float, default=40.,
                   help='mania: a hold releases >= MS before the next press of the same key, else it is shortened '
                        '(or becomes a tap); 0 disables (default 40: holds ending on the next note are unplayable)')
    p.add_argument('--seed', type=int, default=0, help='samples use seeds seed..seed+N-1')
    p.add_argument('--device', choices=('cuda', 'mps', 'cpu'), help='default: cuda, else mps, else cpu')
    p.add_argument('--batch-size', type=int, default=64, help='rows (8 s cores) per generation batch')
    p.add_argument('--token-cap', type=int, default=4096, help='token cap per 8 s core')
    p.add_argument('--max-wall-s', type=float, default=14400., help='wall limit of the whole rollout')
    p.add_argument('--history-reference', action='store_true',
                   help="each core sees the model's own previous 8 s (history_context-trained models only)")
    p.add_argument('--weights', choices=('auto', 'ema', 'raw'), default='auto', help='training checkpoints')
    p.add_argument('--plan', type=Path, help='training plan of a checkpoint (mel_normalization, counterfactual_cell_us)')
    p.add_argument('--mel-stats', type=Path, help='train-only mel statistics file (sha256 checked when declared)')
    p.add_argument('--features-dir', type=Path, help='persistent feature dir (reused when it holds this audio); '
                   'default: a temporary dir removed afterwards')
    p.add_argument('--extract-only', action='store_true', help='write the features (needs --features-dir) and stop')
    p.add_argument('--mert-source', help="pinned MERT-v2 directory, or 'hf' to download the pinned revision")
    p.add_argument('--mert-device', choices=('cuda', 'mps', 'cpu'), help='default: --device')
    p.add_argument('--codec', help='audio codec name (ffprobe codec_name) when ffprobe is unavailable')
    p.add_argument('--clock-constant-ms', type=int, help='override the model->playback clock constant c')
    p.add_argument('--title')
    p.add_argument('--artist', default='unknown')
    p.add_argument('--version', dest='chart_version', help='difficulty name written into the chart')
    a = p.parse_args(argv)
    if not a.extract_only and (a.model is None or a.mode is None):
        p.error('--model and --mode are required (or --extract-only)')
    if a.extract_only and a.features_dir is None:
        p.error('--extract-only needs --features-dir (a temporary feature dir is deleted)')
    if a.consensus < 1:
        p.error('--consensus must be >= 1')
    temperature = wb.parse_temperature(a.temperature)
    if a.consensus > 1 and temperature is None:
        p.error('--consensus > 1 needs sampling (identical greedy samples)')
    if a.field_temperature is not None and temperature is None:
        p.error('--field-temperature needs sampling')
    if a.field_temperature is None and temperature is not None:
        a.field_temperature = dict(PRODUCT_FIELD_TEMPERATURE)
    if a.phira_level is not None and a.mode not in (None, 'phira'):
        p.error('--phira-level is a Phira preset (use --mania-star for mania_4k)')
    if a.mania_star is not None and a.mode not in (None, 'mania_4k'):
        p.error('--mania-star is a mania_4k preset (use --phira-level for phira)')
    if a.phira_level is not None and not 6 <= a.phira_level <= 18:
        p.error('--phira-level must be within 6..18 (presets are fit on levels 6..18)')
    if a.mania_star is not None and not 1 <= a.mania_star <= 7:
        p.error('--mania-star must be within 1..7 (enough database charts only there)')
    if a.guidance is not None and (a.tier is None and a.density is None and a.phira_level is None
                                   and a.mania_star is None or not a.guidance > 0):
        p.error('--guidance needs --tier / --density / --phira-level / --mania-star and a positive weight')
    if a.density is not None and not a.density > 0:
        p.error('--density must be positive')
    return a


def stem_for(audio, mode):
    base = re.sub(r'[^A-Za-z0-9._-]+', '_', Path(audio).stem).strip('._') or 'draft'
    return f'{base[:80]}-{mode}'


def code_identity():
    return {name: wb.sha_file(ROOT/name) for name in CODE_FILES if (ROOT/name).is_file()}


def run(args, *, extractor=None, decoder=None):
    """Execute one draft (or extraction) and return the receipt; ``extractor`` / ``decoder`` are injectable."""
    started = time.monotonic()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    receipt_path = out/'draft-receipt.json'
    keep = Path(args.features_dir).resolve() if args.features_dir else None
    leftovers = [p for p in out.iterdir() if p.resolve() != keep]
    if leftovers:
        raise SystemExit(f'--out-dir {out} is not empty ({leftovers[0].name}); drafts are never overwritten')
    audio_path = Path(args.audio).resolve()
    if not audio_path.is_file():
        raise SystemExit(f'--audio {audio_path} not found')
    receipt = dict(schema=SCHEMA, created_unix=time.time(), command=list(getattr(args, 'argv', None) or sys.argv),
                   code=code_identity(), complete=False)
    timings = receipt['timings'] = {}
    temporary = keep is None
    features_dir = keep or Path(tempfile.mkdtemp(prefix='beatfcn-draft-'))
    try:
        tick = time.monotonic()
        audio_sha = wb.sha_file(audio_path)
        feature_receipt, feature_record = prepare_features(audio_path, audio_sha, features_dir, args,
                                                           extractor=extractor, decoder=decoder)
        timings.update(feature_record.get('timings') or {})
        timings['features_total_s'] = time.monotonic()-tick
        info = feature_receipt.get('draft_song') or {}
        codec = args.codec or info.get('codec') or probe(audio_path)
        receipt['audio'] = dict(path=str(audio_path), sha256=audio_sha, bytes=audio_path.stat().st_size,
                                codec=codec, codec_source='--codec' if args.codec else
                                'feature receipt (ffprobe at extraction)' if info.get('codec') else 'ffprobe',
                                decoded_samples=feature_receipt['decoded_samples'],
                                decoder=info.get('decoder'), database_decoder=DATABASE_FEATURES['decoder'])
        receipt['features'] = dict(dir=str(features_dir), temporary=temporary, reused=feature_record['reused'],
                                   folder=feature_record['folder'],
                                   store='beatfcn.train.mert_features.MertFeatureStore(root, <dir>/features) unchanged',
                                   layout='features/<audio_sha256>/{L12,L23,times_s,intervals_s,mel10_complete}.npy + '
                                          'receipt.json, written by beatfcn.train.full_database.commit_feature',
                                   receipt=feature_receipt)
        if args.extract_only:
            receipt.update(complete=True, extract_only=True)
            return receipt
        device = args.device or auto_device()
        tick = time.monotonic()
        model, model_identity, declared = load_model(args.model, device, args.weights)
        if args.consensus is None:
            args.consensus = 1 if args.mode == 'mania_4k' else PRODUCT_CONSENSUS
        elif args.mode == 'mania_4k' and args.consensus > 1:
            raise SystemExit('mania consensus voted on 4-lane patterns, which the grammar no longer predicts '
                             '(timing-only extraction); use --consensus 1 (the editor default)')
        plan = json.loads(Path(args.plan).read_text()) if args.plan else {}
        if plan:
            if declared is not None and plan.get('mel_normalization') not in (None, declared):
                raise SystemExit('--plan mel_normalization disagrees with the exported model provenance')
            declared = declared if declared is not None else plan.get('mel_normalization')
            model_identity['plan'] = dict(path=str(args.plan), sha256=wb.sha_file(args.plan), id=plan.get('id'))
        stats, normalization = resolve_mel_normalization(declared, mel_stats=args.mel_stats)
        cell_us = int(plan.get('counterfactual_cell_us', 100_000))
        timings['model_load_s'] = time.monotonic()-tick
        receipt['model'] = model_identity
        receipt['features']['mel_normalization'] = normalization
        features = open_features(features_dir, audio_sha, stats)
        duration_us = int(features.records[audio_sha]['duration_us'])
        receipt['audio']['duration_us'] = duration_us
        clock = codec_constant(args.mode, codec, args.clock_constant_ms)
        shift_us = clock['c_ms']*1000
        receipt['clock'] = dict(clock, shift_us=shift_us,
                                model_clock='decoded-audio clock (ffmpeg 24 kHz mono); database targets = raw chart '
                                            'time + model_shift_us (offset decision + c)',
                                rule='written (playback) time = model time - c; a new song has no offset decision, '
                                     'so the per-chart residual shift is 0')
        timing, timing_provider = None, None
        if args.timing:
            timing = read_timing(args.timing)
            beat_clock = codec_constant('mania_4k', codec, args.clock_constant_ms)  # the .osu's own platform
            beat = dict(used=False, shift_us=beat_clock['c_ms']*1000, clock=beat_clock,
                        meter_source=timing['meter_source'])
            if not getattr(model.c, 'beat_features', False):
                beat['reason'] = 'model has no config.beat_features (timing is written, not a model input)'
            elif timing['segments'] is None:
                beat['reason'] = f"gimmick timing, no usable beat grid: {timing['gimmick']} (absent state)"
            else:
                segments = [[round(s+beat['shift_us'], 3), round(b, 6), int(m)] for s, b, m in timing['segments']]
                beat.update(used=True, label='charter_timing_input', segments_model_clock=segments,
                            rule='per core: beatfcn.train.charter_timing.window_beat_features of the charter grid on '
                                 'the model clock (raw grid + c)')
                timing_provider = lambda state, request, segments=segments: segments  # noqa: E731
            timing['beat_features'] = beat
            receipt['timing'] = {k: v for k, v in timing.items() if k != 'segments'}
            receipt['timing']['segments_playback_clock'] = timing['segments']
        provider, guidance, condition = condition_request(args, model, cell_us)
        temperature = wb.parse_temperature(args.temperature)
        seeds = [args.seed+k for k in range(args.consensus)]
        song_id = f'draft:{audio_sha[:16]}'
        song = dict(chart_id=song_id, audio_sha256=audio_sha, mode=args.mode, duration_us=duration_us)
        args.kind_bias = kind_bias(args)
        if args.kind_bias and args.scheduler != 'continuous':
            raise SystemExit('--flick-ratio / --drag-ratio need --scheduler continuous')
        generator = wb.make_generator(temperature, args.top_p, seeds[0], args.field_temperature,
                                      wb.decode_options(args), guidance=guidance)
        intensity = None
        if model.c.intensity_dim:
            from beatfcn.train.density_curve import grid_density_curve
            intensity = lambda audio, start, end: grid_density_curve([], start, end, cell_us=cell_us, scale=16.)  # noqa: E731
        from beatfcn.model.full_event_generate import CUDA_GRAPH_DEFAULT, FAST_DEFAULT, fast_supported
        from beatfcn.model.local_song_rollout import generate_complete_song_samples
        tick = time.monotonic()
        results = generate_complete_song_samples(
            model, features, [song], seeds, device=device, batch_size=args.batch_size, token_cap=args.token_cap,
            max_wall_s=args.max_wall_s, history_reference=args.history_reference, intensity_provider=intensity,
            condition_provider=provider, timing_provider=timing_provider, generator=generator,
            scheduler=args.scheduler)
        timings['generation_s'] = time.monotonic()-tick
        states = [r['songs'][0] for r in results]
        receipt['decoding'] = dict(
            rule='greedy' if temperature is None else 'ancestral', temperature=temperature, top_p=args.top_p,
            field_temperature=args.field_temperature, seeds=seeds, samples=len(seeds),
            rng='per-row torch.Generator seeded from (seed, mode, duration, core start)',
            aggregation='single sample' if len(seeds) == 1 else f'consensus_{len(seeds)} (D6)',
            consensus=None if len(seeds) == 1 else dict(
                samples=len(seeds), window_ms=wb.CONSENSUS_WINDOW_US//1000, min_votes=math.ceil(len(seeds)/2),
                onset='lower median', pattern='majority (ties: earliest sample)',
                hold_tail='lower median of the voting members',
                overlap='mania: holds containing a later same-lane head dropped (head kept as tap)'),
            token_cap_per_core=args.token_cap, core_us=8_000_000, batch_size=args.batch_size,
            max_wall_s=args.max_wall_s, history_reference=bool(args.history_reference), device=device,
            generation_loop=dict(fast=bool(FAST_DEFAULT and fast_supported(model)),
                                 cuda_graph=bool(CUDA_GRAPH_DEFAULT and device == 'cuda')),
            condition=condition, density_curve_render_cell_us=cell_us,
            cell_us_source='plan counterfactual_cell_us' if 'counterfactual_cell_us' in plan else
                           'default 100000 (every recipe-v2 plan)',
            charter_timing_input=None if timing is None else timing['beat_features'],
            product_default=dict(consensus=PRODUCT_CONSENSUS, temperature=PRODUCT_TEMPERATURE,
                                 field_temperature=PRODUCT_FIELD_TEMPERATURE, source=PRODUCT_SOURCE),
            matches_product_default=bool(len(seeds) == PRODUCT_CONSENSUS and temperature == PRODUCT_TEMPERATURE
                                         and args.field_temperature == PRODUCT_FIELD_TEMPERATURE
                                         and args.top_p is None))
        receipt['samples'] = [dict(seed=seed, complete=st['complete'], failure=st['failure'], notes=len(st['events']),
                                   cores=len(st['cores']), tokens=sum(len(c['tokens']) for c in st['cores']),
                                   conditioned_cores=sum(c.get('condition') is not None for c in st['cores']),
                                   charter_timing_cores=sum(bool((c.get('charter_timing') or {}).get('present'))
                                                            for c in st['cores']),
                                   legality_issues=len(st.get('legality') or ()))
                              for seed, st in zip(seeds, states)]
        if not all(st['complete'] for st in states):
            raise SystemExit('Incomplete samples: ' + json.dumps(receipt['samples']))
        tick = time.monotonic()
        if len(seeds) == 1:
            notes = wb.draft_tuples(song, states[0])
            aggregation = dict(samples=1, notes=len(notes))
        else:
            notes, aggregation = wb.consensus_draft(
                [wb.draft_tuples(song, st) for st in states], args.mode, duration_us,
                min_jack_us=int(round(args.consensus_min_jack_ms*1000)) if args.consensus_min_jack_ms else None,
                pattern_source=args.consensus_pattern,
                min_votes=args.consensus_votes or max(1, round(len(seeds)*PRODUCT_VOTES/PRODUCT_CONSENSUS)),
                fragment_guard=args.fragment_guard)
        if args.hold_release_ms:
            if args.mode == 'mania_4k':
                aggregation['hold_release_repair'] = dict(skipped='mania carries no lane decision; same-key '
                                                               'release repair is the editor placement pass')
            else:
                from beatfcn.train.workflow_metrics import repair_hold_release
                notes, aggregation['hold_release_repair'] = repair_hold_release(
                    notes, args.mode, min_release_us=int(round(args.hold_release_ms * 1000)))
        aggregation['legality'] = wb.consensus_legality(notes, args.mode, duration_us)
        timings['consensus_s'] = time.monotonic()-tick
        if aggregation['legality']:
            raise SystemExit('Draft fails the independent legality check: ' + json.dumps(aggregation['legality'][:5]))
        events = [dict(onset_us=int(a), end_us=int(b), type=t, lane=int(lane)) for a, b, t, lane in notes]
        profile = args.profile or ('standard' if args.mode == 'phira' else 'none')
        if args.mode == 'phira' and profile != 'none':
            from beatfcn.events.phira_postprocess import playable
            events, aggregation['playability'] = playable(events, profile)
            aggregation['playability_profile'] = profile
        types = {}
        for e in events:
            types[e['type']] = types.get(e['type'], 0)+1
        receipt['draft'] = dict(aggregation, events_file='draft-events.json', clock='model',
                                notes_per_second=round(len(events)/max(duration_us/1e6, 1e-6), 4), types=types)
        write_json(out/'draft-events.json', dict(schema=SCHEMA, clock='model (decoded audio)', mode=args.mode,
                                                 audio_sha256=audio_sha, duration_us=duration_us, events=events))
        tick = time.monotonic()
        from beatfcn.ingest.pipeline import codec_ext
        stem = stem_for(audio_path, args.mode)
        audio = dict(ok=True, source=f'--audio {audio_path}', ext=audio_path.suffix.lower() or '.'+codec_ext(codec),
                     path=audio_path)
        label = args.tier or (f'{args.density:g} nps' if args.density is not None else
                              f'Phira Lv.{args.phira_level:g}' if getattr(args, 'phira_level', None) is not None else
                              f'{args.mania_star:g}* preset' if getattr(args, 'mania_star', None) is not None else
                              'unconditioned')
        version = args.chart_version or (f'BeatFCN draft ({label}, '
                                         + ('single sample' if len(seeds) == 1 else f'consensus {len(seeds)}') + ')')
        if args.mode == 'mania_4k':
            # 采音模式: the model is a timing extractor; lane placement is the chart editor's
            # job. draft-events.json above is the product (see editor/infer.py --raw).
            export = dict(skipped='mania .osu export needs lanes the grammar no longer predicts; '
                                  'use editor/infer.py --raw on draft-events.json')
            receipt['draft']['export'] = export['skipped']
        else:
            export = wb.export_song(out/stem, stem, dict(mode=args.mode, audio_sha256=audio_sha), events, audio,
                                    title=args.title or audio_path.stem, artist=args.artist, version=version,
                                    timing_points=timing['red_points'] if timing and args.mode == 'mania_4k' else None,
                                    clock_shift_us=shift_us,
                                    clock_source=f"database constant c({clock['source']}, {codec}) = {clock['c_ms']} ms "
                                                 f"[{clock['basis']}]")
        if export.get('failed'):
            raise SystemExit('Export failed: ' + export['failed'])
        if args.mode == 'phira' and timing is not None:
            export['timing_points'] = dict(written='none: the Phira preview has no charter timing (beat features only)')
        timings['export_s'] = time.monotonic()-tick
        receipt['export'] = export
        receipt['complete'] = True
        return receipt
    finally:
        timings['total_s'] = time.monotonic()-started
        if receipt.get('complete') or receipt.get('audio'):
            write_json(receipt_path, receipt)
        if temporary:
            shutil.rmtree(features_dir, ignore_errors=True)


def main(argv=None, *, extractor=None, decoder=None):
    args = parse_args(argv)
    args.argv = ['scripts.draft_song', *(sys.argv[1:] if argv is None else argv)]
    receipt = run(args, extractor=extractor, decoder=decoder)
    if receipt.get('extract_only'):
        print(json.dumps(dict(features=receipt['features']['folder'], audio_sha256=receipt['audio']['sha256'],
                              codec=receipt['audio']['codec'], timings=receipt['timings']), indent=1))
    else:
        export = receipt['export']
        print(json.dumps(dict(chart=export.get('osz') or export['chart'].get('path'), notes=receipt['draft']['notes'],
                              clock_shift_ms=receipt['clock']['c_ms'], receipt=str(Path(args.out_dir)/'draft-receipt.json'),
                              timings={k: round(v, 3) for k, v in receipt['timings'].items()}), indent=1))
    return receipt


if __name__ == '__main__':
    main()
