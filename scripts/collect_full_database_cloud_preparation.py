"""Collect small Modal JSON receipts after a verified preparation run.

No raw audio, canonical chart, target array, mel array, feature array or checkpoint
payload is downloaded to the Mac. The remote worker remains responsible for content
and numerical verification.
"""
import hashlib
import json
from pathlib import Path
import shutil

from beatfcn.audio.mert2 import LOCAL_SOURCE_SHA256, REVISION


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'reports/full-database-cloud-v1'
RUN = REPORTS / 'cloud-preparation-run-v9.json'
MANIFEST = ROOT / 'inputs/full-database-cloud-v1/manifest/charts.offset-v2-3d99ff6bd5e7.jsonl'
COLLECTION = REPORTS / 'cloud-preparation-collection-v9.json'
ARCHIVE = REPORTS / 'pre-v9-archive'
REMOTE = {
    'target_preparation': 'full-database-v1/reports/target-preparation-v1.json',
    'raw_content': 'full-database-v1/reports/raw-content-receipt-v1.json',
    'mel_content': 'full-database-v1/reports/mel-array-content-receipt-v1.json',
    'checkpoint_upload': 'full-database-v1/mert2-checkpoint/receipt.json',
}
LOCAL = {
    'target_preparation': REPORTS / 'target-preparation-v1.json',
    'raw_content': REPORTS / 'raw-content-receipt-v1.json',
    'mel_content': REPORTS / 'mel-array-content-receipt-v1.json',
    'checkpoint_upload': REPORTS / 'mert-checkpoint-upload-v1.json',
}


def sha_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.partial')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def validate_reports(values, manifest_sha, *, require_mel=True):
    target = values['target_preparation']
    if (target.get('complete') is not True or target.get('run_id') != 'full-database-cloud-preparation-v7'
            or target.get('manifest_sha256') != manifest_sha
            or target.get('parsed_rows') != 23211 or target.get('planned_batches') != 91
            or target.get('completed_batches') != list(range(91))
            or target.get('success_count') != 23211
            or target.get('permanent_failure_count') != 0):
        raise ValueError('Remote target report does not account for all 91 batches and 23,211 parsed charts')
    raw = values['raw_content']
    if (raw.get('complete') is not True or raw.get('records') != 10898
            or raw.get('verified') != 10897 or raw.get('unique_audio') != 10898
            or raw.get('failures') != [] or raw.get('cross_source_audio') != 16
            or raw.get('alternate_physical_sources_not_reread') != 189
            or raw.get('quarantined_content_count') != 1
            or raw.get('all_content_verified') is not False
            or raw.get('unquarantined_content_failures') != []
            or len(raw.get('quarantined_content', [])) != 1
            or raw['quarantined_content'][0].get('audio_sha256') != '31123bc5b0a08b31486386e55eb2214b887971d8bfdc27313efaf779d92b442f'
            or any(not item.get('manifest_quarantined') or item.get('kept') is not False
                   for item in raw.get('quarantined_content', []))
            or 'one selected physical source per unique audio' not in raw.get('verification_scope', '')):
        raise ValueError('Remote raw report does not verify the frozen multi-provenance audio/quarantine plan')
    if require_mel:
        mel = values['mel_content']
        if (mel.get('complete') is not True or mel.get('parsed_audio') != 10898
                or mel.get('records') != 10898 or mel.get('verified') != 10898
                or mel.get('failures') != []):
            raise ValueError('Remote mel report does not verify every parsed audio array')
    checkpoint = values['checkpoint_upload']
    if (checkpoint.get('complete') is not True or checkpoint.get('revision') != REVISION
            or checkpoint.get('execution') != 'modal_container'):
        raise ValueError('Remote checkpoint report is incomplete or has the wrong revision')
    files = checkpoint.get('files', {})
    for name, expected in LOCAL_SOURCE_SHA256.items():
        if files.get(name, {}).get('sha256') != expected or files[name].get('bytes', 0) <= 0:
            raise ValueError('Remote checkpoint pinned file identity differs: ' + name)
    return dict(target_charts=23211, target_batches=91, raw_audio=10898,
        raw_content_verified=10897, raw_content_quarantined=1,
        mel_audio=10898 if require_mel else None,
        mel_validation='complete' if require_mel else 'not_required_for_pilot',
        checkpoint_files={name: files[name] for name in LOCAL_SOURCE_SHA256})


def collect():
    run = json.loads(RUN.read_text())
    if run.get('status') != 'completed' or run.get('stop_verified') is not True:
        raise ValueError('Preparation must be completed with verified app stop before collection')
    import modal
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    payloads = {name: b''.join(volume.read_file(path)) for name, path in REMOTE.items()}
    values = {name: json.loads(payload) for name, payload in payloads.items()}
    summary = validate_reports(values, hashlib.sha256(MANIFEST.read_bytes()).hexdigest())
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name, path in LOCAL.items():
        if path.exists():
            archived = ARCHIVE / path.name
            if not archived.exists():
                shutil.copy2(path, archived)
    for name in ('target_preparation', 'raw_content', 'mel_content'):
        write(LOCAL[name], values[name])
    checkpoint = dict(values['checkpoint_upload'], status='already_present', complete=True,
        collection='small JSON receipt only; checkpoint bytes remain in Modal Volume',
        local_receipt_alias='mert-checkpoint-upload-v1.json')
    write(LOCAL['checkpoint_upload'], checkpoint)
    result = dict(schema='beatfcn.full-database-cloud-preparation-collection.v1',
        run_id=run['run_id'], complete=True, app_id=run.get('app_id'),
        run_receipt_sha256=hashlib.sha256(RUN.read_bytes()).hexdigest(),
        remote_payload_sha256={name: sha_bytes(payload) for name, payload in payloads.items()},
        local_payload_bytes=sum(len(payload) for payload in payloads.values()),
        raw_audio_downloaded=0, canonical_charts_downloaded=0, target_arrays_downloaded=0,
        mel_arrays_downloaded=0, feature_arrays_downloaded=0, checkpoint_bytes_downloaded=0,
        summary=summary, training_ready_rows_unlabelled=True,
        note='Remote content/numerical receipts only; train/val/test and quarantine flags are unchanged.')
    write(COLLECTION, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    collect()
