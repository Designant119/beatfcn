"""In-cloud parallel census + resume audit (all processing stays in the cloud)."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json

from beatfcn.cloud.full_database_app import DATA, app, data, image


RECEIPTS = 48


@app.function(image=image, volumes={DATA: data}, cpu=(8, 8), memory=(4096, 4096),
              timeout=600, retries=0, max_containers=1, scaledown_window=2)
def census(run_id, contract_json):
    from beatfcn.audio.mert2 import REVISION
    from beatfcn.cloud.full_database_app import FEATURE_FILES
    contract = json.loads(contract_json)
    jobs = {job['audio_sha256']: job for job in contract['jobs']}
    features = Path(DATA) / 'full-database-v1/mert-graph-diagnostic-v2/features'
    entries = [entry for entry in features.iterdir() if entry.is_dir()]

    def audit(folder):
        receipt = folder / 'receipt.json'
        if not receipt.exists():
            return 'no_receipt'
        try:
            value = json.loads(receipt.read_text())
        except Exception:
            return 'unreadable'
        if not value.get('complete') or not value.get('files'):
            return 'incomplete'
        audio = folder.name
        job = jobs.get(audio)
        if not job:
            return 'unknown_sha'
        decoded = value.get('decoded_samples')
        if (type(decoded) is not int or decoded <= 0
                or value.get('audio_sha256') != audio
                or value.get('metadata', {}).get('revision') != REVISION
                or value.get('metadata', {}).get('real_samples') != decoded
                or value.get('frames') != (decoded + 959) // 960
                or job['expected_mel_frames'] != 4 * (decoded // 960)
                or set(value.get('files', {})) != FEATURE_FILES):
            return 'invalid_identity'
        return 'valid'

    counts = {'valid': 0, 'no_receipt': 0, 'unreadable': 0, 'incomplete': 0,
        'invalid_identity': 0, 'unknown_sha': 0}
    extra = {}
    with ThreadPoolExecutor(max_workers=RECEIPTS) as pool:
        for folder, outcome in zip(entries, pool.map(audit, entries)):
            counts[outcome] += 1
            if outcome == 'valid':
                extra.setdefault('valid_ids', []).append(folder.name)
    return dict(run_id=run_id, census='parallel_census_v1', total_folders=len(entries),
        counts=counts, valid_ids=extra.get('valid_ids', []),
        optimizer_updates=0, diagnostic_only=True, gpu_calls=0)