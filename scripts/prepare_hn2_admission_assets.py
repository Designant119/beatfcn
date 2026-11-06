"""Reference only the audio needed for health/parity/public-shape admission.

No feature copies, NN calls, target filtering or paid operations.
"""
import hashlib
import json
from pathlib import Path

from beatfcn.train.modal_event_dataset import ModalEventDataset


ROOT=Path(__file__).resolve().parents[1]
MANIFEST='artifacts/modal120m-data-v1/prepared-v3/manifest.json'
ORIGINAL='artifacts/modal120m-data-v1/prepared-v3/upload-manifest.json'
OUTPUT='artifacts/modal120m-admission-assets-v1/upload-manifest.json'
HEALTH=[1736,455,3242,1770,1859]


def main():
    dataset=ModalEventDataset(ROOT,MANIFEST,split='train')
    largest=max(dataset.public_audio_bucket(i) for i in range(len(dataset)))
    longest=next(i for i in range(len(dataset)) if dataset.public_audio_bucket(i)==largest)
    reference=max(range(len(dataset)),key=lambda i:dataset.records[i]['reference_slice'][1]-dataset.records[i]['reference_slice'][0])
    draft=[i for i,r in enumerate(dataset.records) if r['view']=='draft']
    selected=sorted(set([0,longest,reference]+[draft[i] for i in HEALTH]))
    audio={dataset.records[i]['audio_sha256'] for i in selected}
    records=[r for r in dataset.manifest['audio_records'] if r['audio_sha256'] in audio]
    arrays=set(dataset.manifest['arrays'].values())|{MANIFEST,dataset.manifest['priors'],dataset.manifest['audit']}
    original=json.loads((ROOT/ORIGINAL).read_text())
    files={name:info for name,info in original['files'].items() if name in arrays
        or any(name==r['complete_mel'] or name.startswith(r['mert_path']+'/') for r in records)}
    if not arrays<=set(files):raise ValueError('Source upload closure misses shared arrays')
    result=dict(id='hn2-discarded-admission-assets-v1',files=files,
        bytes=sum(r['bytes'] for r in files.values()),audio_count=len(audio),
        source_manifest=MANIFEST,source_manifest_sha256=hashlib.sha256((ROOT/MANIFEST).read_bytes()).hexdigest(),
        original_upload_sha256=hashlib.sha256((ROOT/ORIGINAL).read_bytes()).hexdigest(),
        runtime_train_local_indices=selected,calibration_indices=[longest,reference],health_draft_indices=HEALTH,
        selection='Parity index0;two public whole-mel/reference maxima;fixedfive health;no targetN selection',
        records=[dict(train_local_index=i,array_index=dataset.records[i]['array_index'],
            source_index=dataset.records[i]['index'],audio_sha256=dataset.records[i]['audio_sha256']) for i in selected],
        neural_calls=0,optimizer_updates=0,cloud_calls=0,feature_bytes_copied=0)
    path=ROOT/OUTPUT;path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():raise FileExistsError('Preserve immutable admission assets')
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('bytes','audio_count','feature_bytes_copied','cloud_calls')}))


if __name__=='__main__':main()
