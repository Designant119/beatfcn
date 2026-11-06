"""Collect only target/raw/checkpoint JSON receipts needed by the MERT pilot.

The full mel verification may continue independently; the MERT worker recomputes
complete-song mel from decoded audio. No corpus arrays or checkpoint bytes are
 downloaded to the Mac.
"""
import hashlib
import json
from pathlib import Path
import shutil

from scripts.collect_full_database_cloud_preparation import (
    LOCAL, MANIFEST, REMOTE, REPORTS, sha_bytes, validate_reports, write)


RUN = REPORTS / 'checkpoint-parallel-run-v1.json'
DEST = REPORTS / 'mert-pilot-prerequisites-v1.json'
ARCHIVE = REPORTS / 'pre-pilot-archive'


def collect():
    run = json.loads(RUN.read_text())
    if run.get('status') != 'completed' or run.get('stop_verified') is not True:
        raise ValueError('Parallel checkpoint must be complete with a verified stopped app')
    import modal
    volume = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
    names = ('target_preparation', 'raw_content', 'checkpoint_upload')
    payloads = {name: b''.join(volume.read_file(REMOTE[name])) for name in names}
    values = {name: json.loads(payload) for name, payload in payloads.items()}
    summary = validate_reports(values, hashlib.sha256(MANIFEST.read_bytes()).hexdigest(),
        require_mel=False)
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = LOCAL[name]
        if path.exists():
            archived = ARCHIVE / path.name
            if not archived.exists():
                shutil.copy2(path, archived)
    for name in ('target_preparation', 'raw_content'):
        write(LOCAL[name], values[name])
    checkpoint = dict(values['checkpoint_upload'], status='already_present', complete=True,
        collection='small JSON receipt only; checkpoint bytes remain in Modal Volume',
        local_receipt_alias='mert-checkpoint-upload-v1.json')
    write(LOCAL['checkpoint_upload'], checkpoint)
    result = dict(schema='beatfcn.full-database-mert-pilot-prerequisites.v1', complete=True,
        checkpoint_run_id=run['run_id'], checkpoint_app_id=run.get('app_id'),
        checkpoint_run_receipt_sha256=hashlib.sha256(RUN.read_bytes()).hexdigest(),
        remote_payload_sha256={name: sha_bytes(payload) for name, payload in payloads.items()},
        local_payload_bytes=sum(len(payload) for payload in payloads.values()),
        raw_audio_downloaded=0, canonical_charts_downloaded=0, target_arrays_downloaded=0,
        mel_arrays_downloaded=0, feature_arrays_downloaded=0, checkpoint_bytes_downloaded=0,
        summary=summary, full_mel_verification_required_for_pilot=False,
        full_mel_verification_continues=True,
        full_corpus_encoding_authorized=False, training_authorized=False,
        note='Pilot prerequisites only; MERT worker computes complete mel from decoded audio.')
    write(DEST, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


if __name__ == '__main__':
    collect()
