"""Bounded whole/long-song MERT execution using the unchanged window contract."""
import argparse
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

from scripts.probe_local_mert_full import ROOT, sha


def run(path):
    plan = json.loads(path.read_text())
    report_path = ROOT/plan['report']
    if report_path.exists():
        raise FileExistsError(report_path)
    for name, expected in plan['source_sha256'].items():
        assert sha(ROOT/name) == expected, name
    assert json.loads((ROOT/plan['short_report']).read_text())['complete']
    assert Path(sys.prefix).resolve() == (ROOT/plan['environment']).resolve()
    weight = ROOT/plan['checkpoint_dir']/'model.safetensors'
    assert sha(weight) == plan['weight_sha256']
    import numpy as np
    import torch
    from safetensors import safe_open
    from beatfcn.audio.mert2 import pad_tail
    from beatfcn.audio.local_mert import CpuMelTransfer, LocalFrozenMert2

    torch.set_num_threads(2)
    torch.mps.set_per_process_memory_fraction(.6)
    sys.path.insert(0, str(ROOT/'inputs/d96_mert2_source'))
    official = importlib.import_module('mert2_pinned.modeling_mert2')
    config_module = importlib.import_module('mert2_pinned.configuration_mert2')
    config = config_module.MERT2Config(**json.loads((ROOT/plan['config']).read_text()))
    config._attn_implementation = 'sdpa'
    report = dict(id=plan['id'], complete=False, execution_complete=False, pid=os.getpid(),
        plan_sha256=sha(path), optimizer_updates=0, new_cloud_compute_calls=0,
        memory_fraction=.6, cpu_threads=2, automatic_cpu_fallback=False,
        memory_samples=[], forward_windows=[], songs=[], quality_training_allowed=False,
        full_library_encoding_allowed=False, full_model_cross_device_equivalence_passed=False)
    started = time.monotonic()

    def save():
        report['wall_s'] = time.monotonic()-started
        tmp = report_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        tmp.replace(report_path)

    def memory(label):
        torch.mps.synchronize()
        report['memory_samples'].append(dict(label=label,
            tensor_bytes=torch.mps.current_allocated_memory(), driver_bytes=torch.mps.driver_allocated_memory(),
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
        save()

    def decode(source):
        raw = ROOT/source['raw_path']
        assert sha(raw) == source['audio_sha256']
        wave = np.frombuffer(subprocess.check_output(['ffmpeg', '-nostdin', '-v', 'error', '-i', str(raw),
            '-vn', '-f', 'f32le', '-ac', '1', '-ar', '24000', 'pipe:1'], timeout=30), dtype='<f4').copy()
        assert len(wave) == source['decoded_samples'] and np.isfinite(wave).all()
        return wave

    save()
    try:
        model, info = official.MERT2Model.from_pretrained(str(ROOT/plan['checkpoint_dir']), config=config,
            torch_dtype=torch.bfloat16, attn_implementation='sdpa', output_loading_info=True, local_files_only=True)
        assert not any(info.get(k) for k in ('missing_keys', 'unexpected_keys', 'mismatched_keys', 'error_msgs'))
        assert sum(p.numel() for p in model.parameters()) == 632295808
        with safe_open(str(weight), framework='pt', device='cpu') as archive:
            for name, value in model.feature_extractor.named_buffers():
                assert value.dtype == torch.float32 and torch.equal(value, archive.get_tensor('feature_extractor.'+name))
        model.eval().requires_grad_(False).to('mps')
        model.feature_extractor = CpuMelTransfer(model.feature_extractor, 'mps')
        encoder = LocalFrozenMert2(model)
        memory('loaded')
        # Verify only the newly introduced explicit-input dispatch, on real weights.
        wave = decode(plan['parity_audio'])
        with torch.inference_mode():
            direct = model(input_values=torch.from_numpy(pad_tail(wave))[None], output_hidden_states=True)
            expected = np.stack([direct.hidden_states[i-1][0].float().cpu().numpy().astype(np.float16) for i in encoder.spec.layers])
            del direct
            if plan.get('attention_query_chunk'):
                from beatfcn.audio.mert_attention import install_query_chunks
                install_query_chunks(model, plan['attention_query_chunk'])
                report['attention_query_chunk'] = plan['attention_query_chunk']
            exported = encoder.encode(wave)
        difference = float(np.max(np.abs(expected.astype(np.float32)-exported['layers'].astype(np.float32))))
        relative = float(np.linalg.norm(expected.astype(np.float32)-exported['layers'].astype(np.float32))/np.linalg.norm(expected.astype(np.float32)))
        report['same_model_export_parity'] = dict(exact=bool(np.array_equal(expected, exported['layers'])),
            max_abs=difference, relative_l2=relative, audio_sha256=plan['parity_audio']['audio_sha256'],
            source='Direct official model versus existing versioned encode with only CPU waveform dispatch override')
        if plan.get('attention_query_chunk'):
            limits = plan['chunk_tolerances']
            report['same_model_export_parity'].update(
                source='Same pretrained model: original full SDPA versus query-chunk SDPA, all keys retained',
                tolerances=limits, passed=difference <= limits['max_abs_error'] and relative <= limits['relative_l2_max'])
        else:
            report['same_model_export_parity']['passed'] = report['same_model_export_parity']['exact']
        save()
        if not report['same_model_export_parity']['passed']:
            raise ValueError('Explicit waveform dispatch changed actual stored features')
        del expected, exported
        torch.mps.empty_cache()
        active = {}

        def before(_module, args, kwargs):
            signal = kwargs.get('input_values', args[0] if args else None)
            active.update(start=time.monotonic(), padded_samples=int(signal.shape[1]))
            memory('before_window')

        def after(_module, _args, _kwargs, output):
            torch.mps.synchronize()
            report['forward_windows'].append(dict(padded_samples=active['padded_samples'],
                forward_s=time.monotonic()-active['start'], frames=int(output.last_hidden_state.shape[1])))
            memory('after_window')

        model.register_forward_pre_hook(before, with_kwargs=True)
        model.register_forward_hook(after, with_kwargs=True)
        folder = ROOT/plan['feature_output']
        folder.mkdir(parents=True, exist_ok=False)
        for source in plan['audios']:
            wave = decode(source)
            before_index = len(report['forward_windows'])
            tick = time.monotonic()
            result = encoder.encode(wave)
            torch.mps.synchronize()
            encode_s = time.monotonic()-tick
            features = result['layers']
            assert features.shape == (2, (len(wave)+959)//960, 1024)
            assert np.isfinite(features).all()
            assert result['intervals_s'][-1, 1] == len(wave)/24000
            file = folder/(source['audio_sha256']+'.npz')
            tick = time.monotonic()
            np.savez(file, layers=features, times_s=result['times_s'], intervals_s=result['intervals_s'])
            with np.load(file, allow_pickle=False) as check:
                assert np.array_equal(features, check['layers'])
            write_s = time.monotonic()-tick
            x = features.astype(np.float32)
            report['songs'].append(dict(audio=source, encode_s=encode_s, write_s=write_s,
                frames=features.shape[1], shape=list(features.shape), dtype=str(features.dtype),
                finite=True, minimum=float(x.min()), maximum=float(x.max()), mean=float(x.mean()), std=float(x.std()),
                metadata=result['metadata'], forward_window_indices=list(range(before_index, len(report['forward_windows']))),
                file=str(file.relative_to(ROOT)), bytes=file.stat().st_size, sha256=sha(file)))
            print(json.dumps(dict(audio_s=len(wave)/24000, encode_s=encode_s, frames=features.shape[1], windows=len(result['metadata']['windows']))), flush=True)
            del result, features, x
            torch.mps.empty_cache()
            memory('song_complete')
        report.update(complete=True, execution_complete=True, long_resource_passed=True)
        save()
    except Exception as exc:
        report.update(execution_complete=True, error=f'{type(exc).__name__}: {exc}')
        save()
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--plan', type=Path, required=True)
    run(parser.parse_args().plan)
