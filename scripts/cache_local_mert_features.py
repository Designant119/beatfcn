"""Bounded, per-song committed local MERT cache; no training or cloud calls."""
import argparse
import importlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

from scripts.probe_local_mert_full import ROOT, sha
from scripts.acquire_local_mert_weights import save


def run(plan_path):
    plan = json.loads(plan_path.read_text())
    output = ROOT/plan['report']
    if output.exists():
        raise FileExistsError('Inspect the existing cache job; never duplicate it')
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    gate = json.loads((ROOT/plan['resource_gate']).read_text())
    assert gate['complete'] and gate['same_model_export_parity']['passed']
    weight = ROOT/plan['checkpoint_dir']/'model.safetensors'
    assert sha(weight) == plan['weight_sha256']
    assert Path(sys.prefix).resolve() == (ROOT/plan['environment']).resolve()
    destination = ROOT/plan['output_root']
    if destination.exists():
        raise FileExistsError('First cache attempt requires a dedicated empty destination')
    assert shutil.disk_usage(ROOT).free >= plan['minimum_free_bytes']
    destination.mkdir(parents=True)
    import numpy as np
    import torch
    from safetensors import safe_open
    from beatfcn.audio.local_mert import CpuMelTransfer, LocalFrozenMert2
    from beatfcn.audio.mert_attention import install_query_chunks

    torch.set_num_threads(plan['cpu_threads'])
    torch.mps.set_per_process_memory_fraction(plan['memory_fraction'])
    sys.path.insert(0, str(ROOT/'inputs/d96_mert2_source'))
    official = importlib.import_module('mert2_pinned.modeling_mert2')
    configuration = importlib.import_module('mert2_pinned.configuration_mert2')
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        plan_sha256=sha(plan_path), started_at_unix=time.time(), records=[], committed_bytes=0,
        planned_songs=len(plan['audios']), optimizer_updates=0, new_cloud_compute_calls=0,
        new_cloud_spend_usd=0, scope='Frozen audio representation cache, no chart-quality or convergence result',
        frontend_placement='explicit_cpu_float32_then_mps', query_chunk=512,
        memory_fraction=plan['memory_fraction'], automatic_cpu_fallback=False)
    start = time.monotonic()

    def record():
        report.update(wall_s=time.monotonic()-start, committed_songs=len(report['records']),
                      observed_at_unix=time.time())
        save(output, report)

    def store(source, result, encode_s, provenance):
        audio = source['audio_sha256']
        layers = result['layers']
        frames = (source['decoded_samples']+959)//960
        if layers.shape != (2, frames, 1024) or not np.isfinite(layers).all():
            raise ValueError('Invalid feature shape or values: '+audio)
        starts = np.arange(frames, dtype=np.int64)*960
        expected_times = (starts+360)/24000
        expected_intervals = np.stack((starts, np.minimum(starts+960, source['decoded_samples'])), -1)/24000
        if not np.array_equal(result['times_s'], expected_times) or not np.array_equal(result['intervals_s'], expected_intervals):
            raise ValueError('Cache clock drift: '+audio)
        folder = destination/(audio+'.partial')
        folder.mkdir()
        values = [('L12', layers[0]), ('L23', layers[1]),
                  ('times_s', result['times_s']), ('intervals_s', result['intervals_s'])]
        for name, value in values:
            file = folder/(name+'.npy')
            np.save(file, np.ascontiguousarray(value))
            check = np.load(file, mmap_mode='r', allow_pickle=False)
            if not check.flags.c_contiguous or not np.array_equal(check, value):
                raise ValueError('Cache write mismatch: '+str(file))
        metadata = dict(result['metadata'], audio_sha256=audio, frontend_placement=report['frontend_placement'],
            query_chunk=512, official_source_revision=plan['revision'], checkpoint_sha256=plan['weight_sha256'],
            common_mel_reencoded=False, source_provenance=provenance)
        save(folder/'metadata.json', metadata)
        files = {p.name:dict(bytes=p.stat().st_size, sha256=sha(p)) for p in folder.iterdir()}
        byte_count = sum(f['bytes'] for f in files.values())
        if report['committed_bytes']+byte_count > plan['cache_bytes_limit']:
            raise ValueError('Cumulative cache byte limit exceeded')
        folder.rename(destination/audio)
        x = layers.astype(np.float32)
        report['records'].append(dict(audio_sha256=audio, modes=source['modes'], splits=source['splits'],
            samples=source['decoded_samples'], frames=frames, dtype=str(layers.dtype), finite=True,
            minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()), std=float(x.std()),
            path=str((destination/audio).relative_to(ROOT)), files=files, encode_s=encode_s, provenance=provenance))
        report['committed_bytes'] += byte_count
        record()

    record()
    try:
        sources = {x['audio_sha256']:x for x in plan['audios']}
        for prior in gate['songs'] if plan.get('reuse_resource_outputs', True) else []:
            source = sources[prior['audio']['audio_sha256']]
            file = ROOT/prior['file']
            assert sha(file) == prior['sha256']
            with np.load(file, allow_pickle=False) as saved:
                result = {k:saved[k] for k in saved.files}
            result['metadata'] = prior['metadata']
            store(source, result, 0., dict(kind='reuse_verified_full_resource_output', report=plan['resource_gate'], file=prior['file']))
        config = configuration.MERT2Config(**json.loads((ROOT/plan['config']).read_text()))
        config._attn_implementation = 'sdpa'
        tick = time.monotonic()
        model, info = official.MERT2Model.from_pretrained(str(ROOT/plan['checkpoint_dir']), config=config,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True, local_files_only=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        with safe_open(str(weight), framework='pt', device='cpu') as archive:
            for name, value in model.feature_extractor.named_buffers():
                assert value.dtype == torch.float32 and torch.equal(value, archive.get_tensor('feature_extractor.'+name))
        model.eval().requires_grad_(False).to('mps')
        model.feature_extractor = CpuMelTransfer(model.feature_extractor, 'mps')
        install_query_chunks(model, 512)
        encoder = LocalFrozenMert2(model)
        torch.mps.synchronize()
        report['load_and_place_s'] = time.monotonic()-tick
        if plan.get('parity_reference'):
            parity = plan['parity_reference']
            source = parity['source']
            raw = ROOT/source['raw_path']
            assert sha(raw) == source['audio_sha256']
            wave = np.frombuffer(subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw),
                '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'], timeout=30), dtype='<f4').copy()
            assert len(wave) == source['decoded_samples']
            same = encoder.encode(wave)['layers']
            original = np.stack([np.load(ROOT/parity['record']['path']/name, allow_pickle=False)
                                 for name in ('L12.npy', 'L23.npy')])
            report['fresh_process_feature_parity'] = dict(audio_sha256=source['audio_sha256'],
                exact=bool(np.array_equal(same, original)),
                max_abs=float(np.abs(same.astype(np.float32)-original.astype(np.float32)).max()))
            record()
            if not report['fresh_process_feature_parity']['exact']:
                raise ValueError('Fresh process changed stored audio features')
            del same, original, wave
            torch.mps.empty_cache()
        already = {x['audio_sha256'] for x in report['records']}
        for source in plan['audios']:
            if source['audio_sha256'] in already:
                continue
            if time.monotonic()-start >= plan['timeout_s']:
                raise TimeoutError('Bound reached; per-song committed caches are retained')
            if shutil.disk_usage(destination).free < plan['minimum_runtime_free_bytes']:
                raise ValueError('Local free-disk lower bound reached')
            report['active_audio_sha256'] = source['audio_sha256']
            record()
            raw = ROOT/source['raw_path']
            assert sha(raw) == source['audio_sha256']
            wave = np.frombuffer(subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw),
                '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'], timeout=30), dtype='<f4').copy()
            assert len(wave) == source['decoded_samples'] and np.isfinite(wave).all()
            tick = time.monotonic()
            result = encoder.encode(wave)
            torch.mps.synchronize()
            encode_s = time.monotonic()-tick
            store(source, result, encode_s, dict(kind='fresh_local_encoding'))
            report['last_memory_sample'] = dict(tensor_bytes=torch.mps.current_allocated_memory(),
                driver_bytes=torch.mps.driver_allocated_memory(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            del result, wave
            torch.mps.empty_cache()
            torch.mps.synchronize()
            report['post_cleanup_memory'] = dict(tensor_bytes=torch.mps.current_allocated_memory(),
                driver_bytes=torch.mps.driver_allocated_memory(), peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            record()
            print(json.dumps(dict(songs=len(report['records']), total=report['planned_songs'], encode_s=encode_s, cache_bytes=report['committed_bytes'])), flush=True)
        report.update(complete=True, execution_complete=True, active_audio_sha256=None)
        record()
    except Exception as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        record()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    run(parser.parse_args().plan)
