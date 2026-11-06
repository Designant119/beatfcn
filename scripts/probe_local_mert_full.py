"""Full pinned pretrained MERT resource probe, without training or quality claims."""
import argparse
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def run(path):
    plan = json.loads(path.read_text())
    output = ROOT / plan['report']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['source_sha256'].items():
        assert sha(ROOT / source) == expected, source
    download = json.loads((ROOT / plan['download_report']).read_text())
    assert download['complete'] and download['execution_complete']
    weight = ROOT / plan['checkpoint_dir'] / 'model.safetensors'
    assert weight.stat().st_size == plan['weight_bytes']
    assert sha(weight) == plan['weight_sha256']
    assert Path(sys.prefix).resolve() == (ROOT / plan['environment']).resolve()
    import numpy as np
    import torch
    from safetensors import safe_open
    from beatfcn.audio.local_mert import CpuMelTransfer
    from beatfcn.audio.mert2 import pad_tail

    assert torch.__version__.split('+')[0] == '2.6.0' and torch.backends.mps.is_available()
    torch.set_num_threads(plan['cpu_threads'])
    torch.mps.set_per_process_memory_fraction(plan['memory_fraction'])
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        optimizer_updates=0, new_cloud_compute_calls=0, new_cloud_spend_usd=0,
        music_quality_passed=False, full_library_encoding_allowed=False,
        plan_sha256=sha(path), weight_sha256=plan['weight_sha256'],
        automatic_cpu_fallback=False, frontend_placement='explicit_cpu_then_transfer',
        rows=[], memory_samples=[], dtype='bfloat16_encoder_float32_cpu_frontend',
        recommended_memory_bytes=torch.mps.recommended_max_memory(),
        memory_fraction=plan['memory_fraction'])
    started = time.monotonic()

    def save():
        report['wall_s'] = time.monotonic()-started
        temp = output.with_suffix('.tmp')
        temp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        temp.replace(output)

    def memory(label):
        torch.mps.synchronize()
        report['memory_samples'].append(dict(label=label,
            tensor_bytes=torch.mps.current_allocated_memory(),
            driver_bytes=torch.mps.driver_allocated_memory(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        save()

    def stats(values):
        x = np.asarray(values, dtype=np.float32)
        finite = bool(np.isfinite(x).all())
        return dict(shape=list(values.shape), dtype=str(values.dtype), finite=finite,
            nan=int(np.isnan(x).sum()), inf=int(np.isinf(x).sum()),
            minimum=float(x.min()) if finite else None, maximum=float(x.max()) if finite else None,
            mean=float(x.mean()) if finite else None, std=float(x.std()) if finite else None)

    save()
    try:
        sys.path.insert(0, str(ROOT / 'inputs/d96_mert2_source'))
        official = importlib.import_module('mert2_pinned.modeling_mert2')
        configuration = importlib.import_module('mert2_pinned.configuration_mert2')
        config = configuration.MERT2Config(**json.loads((ROOT / plan['config']).read_text()))
        config._attn_implementation = 'sdpa'
        tick = time.monotonic()
        model, info = official.MERT2Model.from_pretrained(str(ROOT / plan['checkpoint_dir']),
            config=config, torch_dtype=torch.bfloat16, attn_implementation='sdpa',
            output_loading_info=True, local_files_only=True)
        report['load_info'] = info
        if any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs')):
            raise ValueError('Pinned checkpoint did not load exactly')
        model.eval().requires_grad_(False)
        report['parameters'] = sum(p.numel() for p in model.parameters())
        assert report['parameters'] == 632295808
        if any(p.dtype != torch.bfloat16 for p in model.parameters()):
            raise ValueError('Unexpected encoder parameter dtype')
        # Verify checkpoint normalization and filters before and after placement.
        front_buffers = dict(model.feature_extractor.named_buffers())
        report['checkpoint_frontend_buffers'] = {}
        with safe_open(str(weight), framework='pt', device='cpu') as archive:
            for name, value in front_buffers.items():
                stored = archive.get_tensor('feature_extractor.'+name)
                same = value.dtype == torch.float32 and torch.equal(value, stored)
                report['checkpoint_frontend_buffers'][name] = same
                if not same:
                    raise ValueError('Checkpoint frontend precision changed: '+name)
        report['load_cpu_s'] = time.monotonic()-tick
        memory('loaded_cpu')
        tick = time.monotonic()
        model.to('mps')
        model.feature_extractor = CpuMelTransfer(model.feature_extractor, 'mps')
        torch.mps.synchronize()
        report['placement_s'] = time.monotonic()-tick
        memory('placed_mps_cpu_frontend')
        source = plan['audio']
        assert sha(ROOT / source['raw_path']) == source['audio_sha256']
        tick = time.monotonic()
        wave = np.frombuffer(subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(ROOT/source['raw_path']),
            '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'], timeout=30), dtype='<f4').copy()
        assert len(wave) == source['decoded_samples'] and np.isfinite(wave).all()
        report['decode_s'] = time.monotonic()-tick
        report['original_audio'] = source
        folder = ROOT / plan['feature_output']
        folder.mkdir(parents=True, exist_ok=False)
        with torch.inference_mode():
            for case in plan['cases']:
                signal = wave[:case['samples']]
                assert len(signal) == case['samples']
                padded = pad_tail(signal)
                tensor = torch.from_numpy(padded)[None]
                memory('before_'+case['name'])
                tick = time.monotonic()
                result = model(input_values=tensor, output_hidden_states=True)
                torch.mps.synchronize()
                forward_s = time.monotonic()-tick
                memory('after_'+case['name'])
                assert len(result.hidden_states) == 24
                expected = (len(signal)+959)//960
                assert tuple(result.last_hidden_state.shape) == (1, expected, 1024)
                assert tuple(result.feature_attention_mask.shape) == (1, expected)
                assert bool(result.feature_attention_mask.all())
                tick = time.monotonic()
                layers = np.stack([result.hidden_states[i-1][0].float().cpu().numpy().astype(np.float16) for i in plan['layers']])
                copy_s = time.monotonic()-tick
                if not np.isfinite(layers).all():
                    raise ValueError('Nonfinite stored feature values')
                start_samples = np.arange(expected, dtype=np.int64)*960
                times = (start_samples+360)/24000
                intervals = np.stack((start_samples, np.minimum(start_samples+960, len(signal))), axis=-1)/24000
                assert intervals[-1, 1] == len(signal)/24000
                tick = time.monotonic()
                file = folder / (case['name']+'.npz')
                np.savez(file, layers=layers, times_s=times, intervals_s=intervals)
                with np.load(file, allow_pickle=False) as check:
                    assert np.array_equal(check['layers'], layers)
                    assert np.array_equal(check['intervals_s'], intervals)
                write_s = time.monotonic()-tick
                report['rows'].append(dict(name=case['name'], real_samples=len(signal), padded_samples=len(padded),
                    audio_s=len(signal)/24000, frames=expected, waveform=stats(signal), features=stats(layers),
                    forward_s=forward_s, copy_s=copy_s, write_s=write_s,
                    cpu_frontend_included_in_forward=True, layers=plan['layers'],
                    file=str(file.relative_to(ROOT)), sha256=sha(file), bytes=file.stat().st_size,
                    scope='Full pretrained encoder resource/finite/clock evidence, no chart quality or full-model CPU-CUDA equivalence'))
                print(json.dumps({k: report['rows'][-1][k] for k in ('name', 'audio_s', 'forward_s', 'frames')}), flush=True)
                del result, layers, tensor
                torch.mps.empty_cache()
                save()
        report.update(complete=True, execution_complete=True,
            short_full_model_resource_passed=True, long_song_resource_passed=False,
            full_model_cross_device_equivalence_passed=False)
        memory('complete')
    except Exception as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    run(parser.parse_args().plan)
