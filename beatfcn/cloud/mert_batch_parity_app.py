"""Small real-MERT parity probe; no stable-period or corpus training work."""
from pathlib import Path
import json

from .full_database_app import (
    CHECKPOINT, DATA, app, data, encoder, image, verify_beatfcn_sources)


@app.function(image=image, volumes={DATA: data}, gpu='H100', cpu=(8, 8),
              memory=(32768, 32768), timeout=300, retries=0,
              max_containers=1, scaledown_window=2)
def probe_window_batch(run_id, contract_json):
    from beatfcn.audio.mert2 import validate_local_source
    from beatfcn.audio.mert2_batch import encode_windows_batch, feature_parity
    from beatfcn.train.cuda_pipeline import decoded_audio_from_receipt, load_raw_index
    contract = json.loads(contract_json)
    verify_beatfcn_sources(contract)
    if contract.get('probe_window_batch') != 2 or not contract.get('parity_only'):
        raise ValueError('Parity probe identity changed')
    validate_local_source(CHECKPOINT)
    jobs = contract['jobs']
    if len(jobs) != 4:
        raise ValueError('Parity probe requires the same four preflight jobs')
    raw_index = load_raw_index(DATA, contract['raw_content_sha256'],
        contract['raw_manifest_sha256'])
    temporary_root = Path('/tmp') / 'mert-batch-parity-probe'
    temporary_root.mkdir(parents=True, exist_ok=True)
    prefetched = [decoded_audio_from_receipt(job, raw_index, DATA, temporary_root)
                  for job in jobs]
    waves = [value[2] for value in prefetched]
    model = encoder()
    candidates = encode_windows_batch(model, waves, window_batch=2)
    reports = []
    for job, wave, candidate in zip(jobs, waves, candidates):
        reference = model.encode(wave)
        report = feature_parity(reference['layers'], candidate['layers'])
        report['audio_sha256'] = job['audio_sha256']
        report['clock_exact'] = bool((reference['times_s'] == candidate['times_s']).all()
            and (reference['intervals_s'] == candidate['intervals_s']).all())
        report['passed'] = bool(report['finite'] and report['clock_exact']
            and report['max_abs'] <= .02 and report['float16_exact_fraction'] >= .95)
        reports.append(report)
    for _, original, _ in prefetched:
        original = Path(original)
        if original.is_relative_to(temporary_root):
            original.unlink()
    return dict(run_id=run_id, parity_only=True, probe_window_batch=2,
        reports=reports, passed=all(report['passed'] for report in reports),
        optimizer_updates=0, stable_period_started=False)
