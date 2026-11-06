"""Isolate D96 frontend device errors without changing the acceptance gate."""
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '0'
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'

import numpy as np
import torch
from copy import deepcopy
from beatfcn.audio.mert2 import pad_tail


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / 'reports/d96_frontend_isolation_v1.json'
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((root / 'bench/rhythm-v2/d96-local-mert-runtime-v1.json').read_text())
    assert Path(sys.prefix).resolve() == (root / plan['environment']).resolve()
    for source, expected in plan['source_sha256'].items():
        assert hashlib.sha256((root / source).read_bytes()).hexdigest() == expected
    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    sys.path.insert(0, str(root / 'inputs/d96_mert2_source'))
    official = importlib.import_module('mert2_pinned.modeling_mert2')
    config_module = importlib.import_module('mert2_pinned.configuration_mert2')
    config = config_module.MERT2Config(**json.loads((root / 'inputs/d64_implementation_sources/mert2-config.json').read_text()))
    cpu = official.MERT2MelFrontend(config).eval()
    cpu.mel_mean.copy_(torch.linspace(-10, 10, config.num_mel_bins))
    cpu.mel_std.copy_(torch.linspace(1, 3, config.num_mel_bins))
    gpu = deepcopy(cpu).to(device='mps', dtype=torch.bfloat16).eval()
    audio = plan['selected_training_audio']
    raw = root / audio['raw_path']
    assert hashlib.sha256(raw.read_bytes()).hexdigest() == audio['audio_sha256']
    wave = np.frombuffer(subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw), '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'], timeout=30), dtype='<f4').copy()
    assert len(wave) == audio['decoded_samples']
    report = dict(id='D96-frontend-isolation-v1', complete=False, optimizer_updates=0,
        new_cloud_compute_calls=0, automatic_cpu_fallback=False,
        hypotheses=['H1 spectrum differences amplified at low energy by log',
                    'H2 mel projection differs on identical spectrum',
                    'H3 normalization buffers differ after device/dtype conversion',
                    'H4 dB arithmetic differs on identical mel power'],
        tolerances=plan['feature_tolerances'], rows=[],
        buffers_equal={k: torch.equal(v, dict(gpu.named_buffers())[k].cpu()) for k, v in cpu.named_buffers()},
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    start = time.monotonic()

    def finish(front, power):
        db = front.amplitude_to_db(power)[..., :-1].transpose(-1, -2)
        return (db - front.mel_mean) / front.mel_std.clamp_min(1e-5)

    def measure(reference, actual):
        a, b = reference.cpu(), actual.cpu()
        delta = (a - b).abs()
        tol = report['tolerances']
        bad = delta > tol['atol'] + tol['rtol'] * a.abs()
        return dict(passed=not bool(bad.any()), violations=int(bad.sum()), elements=a.numel(),
            max_abs=float(delta.max()), relative_l2=float(torch.linalg.vector_norm(delta) / torch.linalg.vector_norm(a)),
            finite=bool(torch.isfinite(b).all()))

    with torch.inference_mode():
        for label, signal in [('real_start10s_duration2s', wave[240000:288001]), ('real_tail2s', wave[-48001:])]:
            x = torch.from_numpy(pad_tail(signal))[None]
            xm = x.to('mps')
            sp = cpu.spectrogram(x)
            mel = cpu.mel_scale(sp)
            ref = cpu(x)
            repeat = []
            for _ in range(2):
                repeat.append(measure(ref, gpu(xm)))
            sm = gpu.spectrogram(xm)
            # Change only the device used for each stage, keeping its input fixed.
            m_from_cpu_sp = gpu.mel_scale(sp.to('mps'))
            m_from_gpu_sp = cpu.mel_scale(sm.cpu())
            actual = gpu(xm)
            delta = (ref - actual.cpu()).abs()
            flat_indices = torch.topk(delta.flatten(), 10).indices.tolist()
            worst = []
            for flat in flat_indices:
                frame, band = divmod(flat, config.num_mel_bins)
                worst.append(dict(frame=frame, band=band, normalized_cpu=float(ref[0, frame, band]),
                    normalized_mps=float(actual.cpu()[0, frame, band]),
                    mel_power_cpu=float(mel[0, band, frame]),
                    mel_power_gpu_spectrum_cpu_projection=float(m_from_gpu_sp[0, band, frame])))
            report['rows'].append(dict(label=label, samples=len(signal), repeated_original=repeat,
                cpu_spectrum_mps_rest=measure(ref, finish(gpu, m_from_cpu_sp)),
                mps_spectrum_cpu_rest=measure(ref, finish(cpu, m_from_gpu_sp)),
                cpu_mel_mps_db_norm=measure(ref, finish(gpu, mel.to('mps'))),
                cpu_db_mps_norm=measure(ref, (cpu.amplitude_to_db(mel)[..., :-1].transpose(-1, -2).to('mps') - gpu.mel_mean) / gpu.mel_std.clamp_min(1e-5)),
                spectrum_max_abs=float((sp - sm.cpu()).abs().max()), worst=worst))
            torch.mps.synchronize()
    report.update(complete=True, wall_s=time.monotonic()-start,
                  scope='Real input arithmetic diagnosis; no pretrained weights, no quality result.')
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(json.dumps({**{k: report[k] for k in ('complete', 'wall_s', 'buffers_equal')},
        'rows': [{k: v for k, v in row.items() if k != 'worst'} for row in report['rows']]}))


if __name__ == '__main__':
    main()
