"""Bounded CPU/MPS checks of pinned MERT components, without trained weights."""
import argparse
from copy import deepcopy
import hashlib
import importlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def run(path):
    plan = json.loads(path.read_text())
    runtime = json.loads((ROOT / plan['runtime_plan']).read_text())
    output = ROOT / plan['output']
    if output.exists():
        raise FileExistsError(output)
    for source, expected in plan['source_sha256'].items():
        if sha(ROOT / source) != expected:
            raise ValueError('Frozen source changed: ' + source)
    if Path(sys.prefix).resolve() != (ROOT / runtime['environment']).resolve():
        raise ValueError('Use the isolated runtime interpreter')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') not in (None, '0'):
        raise ValueError('Automatic CPU fallback is forbidden')
    # Before importing torch, pin offline behavior and explicitly disable fallback.
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    import numpy as np
    import torch
    from beatfcn.audio.mert2 import Mert2Spec, pad_tail, windows

    report = dict(id=plan['id'], complete=False, execution_complete=False,
        optimizer_updates=0, pretrained_model_forwards=0, weights_downloaded=False,
        new_cloud_compute_calls=0, new_cloud_spend_usd=0, pid=os.getpid(),
        plan_sha256=sha(path), rows=[], memory_samples=[],
        scope='Official untrained components only; not full632M resource acceptance or music/chart quality.')
    started = time.monotonic()

    def record():
        report['wall_s'] = time.monotonic() - started
        save(output, report)

    def memory(label):
        torch.mps.synchronize()
        report['memory_samples'].append(dict(label=label,
            tensor_bytes=torch.mps.current_allocated_memory(),
            driver_bytes=torch.mps.driver_allocated_memory(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))

    def stats(value):
        x = value.detach().float().cpu()
        return dict(shape=list(x.shape), device=str(value.device), dtype=str(value.dtype),
            nan=int(torch.isnan(x).sum()), inf=int(torch.isinf(x).sum()),
            minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()),
            std=float(x.std(correction=0)))

    def compare(label, expected, actual, *, low_precision=False, details=None):
        a, b = expected.detach().float().cpu(), actual.detach().float().cpu()
        if a.shape != b.shape:
            raise ValueError('Different output shapes: ' + label)
        finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
        delta = (a - b).abs()
        error = float(delta.max()) if finite else None
        l2 = float(torch.linalg.vector_norm(a - b) /
                   torch.linalg.vector_norm(a).clamp_min(1e-12)) if finite else None
        if low_precision:
            limits = runtime['bf16_tolerances']
            passed = finite and l2 <= limits['relative_l2_max'] and error <= limits['max_abs_error']
        else:
            limits = runtime['feature_tolerances']
            passed = finite and bool(torch.all(delta <= limits['atol'] + limits['rtol'] * a.abs()))
        report['rows'].append(dict(label=label, passed=passed, reference=stats(expected),
            actual=stats(actual), max_abs_error=error, relative_l2=l2,
            tolerances=limits, details=details))
        memory(label)
        record()
        if not passed:
            raise ValueError('Frozen numerical gate failed: ' + label)

    try:
        if torch.__version__.split('+')[0] != '2.6.0' or not torch.backends.mps.is_available():
            raise ValueError('Expected actual torch2.6 MPS runtime')
        torch.set_num_threads(runtime['cpu_threads'])
        torch.manual_seed(plan['seed'])
        torch.mps.set_per_process_memory_fraction(float(runtime['memory_fraction']))
        report.update(torch_version=torch.__version__, recommended_memory_bytes=torch.mps.recommended_max_memory(),
            memory_fraction=runtime['memory_fraction'], automatic_cpu_fallback=False)
        # Source files are exact copies of the already reviewed, pinned release.
        sys.path.insert(0, str(ROOT / plan['official_package_parent']))
        official = importlib.import_module('mert2_pinned.modeling_mert2')
        configuration = importlib.import_module('mert2_pinned.configuration_mert2')
        config = configuration.MERT2Config(**json.loads((ROOT / plan['config']).read_text()))
        config._attn_implementation = 'sdpa'
        front_cpu = official.MERT2MelFrontend(config).eval()
        # These deterministic buffers exercise arithmetic, not checkpoint normalization.
        front_cpu.mel_mean.copy_(torch.linspace(-10, 10, config.num_mel_bins))
        front_cpu.mel_std.copy_(torch.linspace(1, 3, config.num_mel_bins))
        placement = plan.get('frontend_placement', 'mps')
        if placement == 'explicit_cpu_then_transfer':
            from beatfcn.audio.local_mert import CpuMelTransfer
            front_mps = CpuMelTransfer(deepcopy(front_cpu), 'mps')
        elif placement == 'mps':
            front_mps = deepcopy(front_cpu).to(device='mps', dtype=torch.bfloat16).eval()
        else:
            raise ValueError('Unknown frontend placement')
        report['frontend_placement'] = placement
        if any(b.dtype != torch.float32 for b in front_mps.buffers() if b.is_floating_point()):
            raise ValueError('Official frontend did not preserve float32 buffers')
        report['frontend_normalization'] = 'Synthetic deterministic nonuniform buffers; pretrained normalization not loaded.'
        source = runtime['selected_training_audio']
        if sha(ROOT / source['raw_path']) != source['audio_sha256']:
            raise ValueError('Selected raw training audio changed')
        command = ['ffmpeg', '-nostdin', '-v', 'error', '-i', str(ROOT / source['raw_path']),
                   '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1']
        decoded = subprocess.run(command, capture_output=True, check=True, timeout=30).stdout
        wave = np.frombuffer(decoded, dtype='<f4').copy()
        if len(wave) != source['decoded_samples'] or not np.isfinite(wave).all():
            raise ValueError('Decoded clock or values changed')
        report['audio'] = dict(**source, decoder_command=command, waveform=stats(torch.from_numpy(wave)))
        impulse = np.zeros(48000, np.float32)
        impulse[12000] = 1.
        signals = [('silence_minimum', np.zeros(1025, np.float32)), ('impulse_exact', impulse),
            ('impulse_partial', np.pad(impulse, (0, 1))),
            ('real_10s_partial', wave[240000:288001]), ('real_tail_partial', wave[-48001:])]
        with torch.inference_mode():
            for label, signal in signals:
                padded = pad_tail(signal)
                x = torch.from_numpy(padded.copy())[None]
                expected = front_cpu(x)
                device_input = x if placement == 'explicit_cpu_then_transfer' else x.to('mps')
                actual = front_mps(device_input)
                if expected.shape != (1, len(padded) // 240, 128):
                    raise ValueError('Unexpected official frontend frame count')
                compare('frontend_' + label, expected, actual, details=dict(real_samples=len(signal),
                    padded_samples=len(padded), preserved_partial_cell=len(signal) % 960 != 0,
                    expected_encoder_frames=(len(signal) + 959) // 960))
                del actual, expected, device_input
            del front_mps, front_cpu
            torch.mps.empty_cache()
            # Actual official operators at real channel widths; no full encoder is created.
            modules = [('convnext128', official.ConvNextLayer(128, config.subsampling_layer_norm_eps), 128),
                ('conformer1024', official.ConformerBlock(config), 1024)]
            for label, module, width in modules:
                for child in module.modules():
                    if isinstance(child, (torch.nn.Linear, torch.nn.Conv1d)):
                        torch.nn.init.normal_(child.weight, std=config.initializer_range)
                        if child.bias is not None:
                            torch.nn.init.zeros_(child.bias)
                    if isinstance(child, official.GlobalResponseNorm):
                        child.weight.fill_(.2)
                        child.bias.fill_(.1)
                quantized = module.eval().to(dtype=torch.bfloat16)
                cpu = deepcopy(quantized).float()
                gpu = deepcopy(quantized).to('mps')
                x = torch.randn(1, 128, width).to(torch.bfloat16)
                device_input = x.to('mps')
                if label.startswith('conformer'):
                    rotary = official.RotaryEmbedding(config)
                    pos = rotary(x)
                    rotary_mps = deepcopy(rotary).to('mps')
                    positions_gpu = rotary_mps(device_input)
                    for index, name in enumerate(('cos', 'sin')):
                        compare('rotary_' + name, pos[index], positions_gpu[index],
                                low_precision=True)
                    expected = cpu(x.float(), tuple(v.float() for v in pos))
                    actual = gpu(device_input, positions_gpu)
                else:
                    expected, actual = cpu(x.float()), gpu(device_input)
                compare(label, expected, actual, low_precision=True, details=dict(
                    cpu_oracle='Same BF16-quantized input/weights promoted toFP32; no autocast',
                    trainable_parameters=sum(p.numel() for p in module.parameters()),
                    optimizer_updates=0, pretrained_weights=False))
                del cpu, gpu, quantized, actual, expected, device_input
                torch.mps.empty_cache()
            # Validate the existing long-window ownership clock without a large forward.
            spec = Mert2Spec()
            clock_rows = []
            for samples in plan['clock_samples']:
                chunks = windows(samples, spec)
                kept = []
                for i, (start, end) in enumerate(chunks):
                    ids = np.arange((end - start + 959) // 960) * 960 + start
                    centers = ids + 360
                    left = 0 if i == 0 else (chunks[i-1][1] + start) // 2
                    right = samples if i + 1 == len(chunks) else (end + chunks[i+1][0]) // 2
                    mask = (centers >= left) & ((centers < right) | (i + 1 == len(chunks))) & (ids < samples)
                    kept.append(ids[mask])
                ids = np.concatenate(kept)
                if not np.array_equal(ids, np.arange((samples + 959) // 960) * 960):
                    raise ValueError('Gap/overlap in long-window clock')
                clock_rows.append(dict(samples=samples, frames=len(ids), windows=chunks,
                    last_start_sample=int(ids[-1]), last_end_sample=samples, passed=True,
                    scope='CPU index audit only; not long-song encoder execution'))
            report['clock_rows'] = clock_rows
        report.update(complete=True, execution_complete=True,
            full_encoder_resource_passed=False, music_quality_passed=False)
        memory('end')
        record()
        print(json.dumps(dict(complete=True, rows=len(report['rows']), wall_s=report['wall_s'])))
    except Exception as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        record()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', required=True, type=Path)
    run(parser.parse_args().plan.resolve())
