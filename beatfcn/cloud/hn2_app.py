"""HN2 Modal definitions; import only AFTER the local Owner/review reservation.

There is deliberately no local_entrypoint and no automatic upload or spawn.
The gated scripts/run_hn2_modal.py is the sole supported launcher.
"""
import hashlib
import json
from pathlib import Path

import modal

from beatfcn.train.hn2_runtime import rooted, sha, validate_envelope, write_json


REPO = Path(__file__).resolve().parents[2]
DATA = '/data'
app = modal.App('beatfcn-hn2-native-138m')
data = modal.Volume.from_name('beatfcn-data', create_if_missing=False)
image = (modal.Image.debian_slim(python_version='3.11')
         .pip_install('torch==2.6.0', index_url='https://download.pytorch.org/whl/cu126')
         .pip_install('numpy==2.2.6', 'safetensors==0.5.3', 'pyyaml==6.0.3', 'jsonschema==4.26.0')
         .add_local_python_source('beatfcn')
         .add_local_dir(REPO / 'scripts', '/opt/hn2-review-source/scripts')
         .add_local_dir(REPO / 'tests', '/opt/hn2-review-source/tests'))


def verify_remote_sources(contract):
    import beatfcn
    package_root = Path(beatfcn.__file__).resolve().parent.parent
    for name, expected in contract['source_sha256'].items():
        base = package_root if name.startswith('beatfcn/') else Path('/opt/hn2-review-source')
        if sha(rooted(base, name)) != expected:
            raise ValueError('Uploaded source differs from reviewed source: ' + name)


def envelopes(contract_json, authorization_json, review_json, local_gate_json):
    values = tuple(json.loads(s) for s in (contract_json,authorization_json,review_json,local_gate_json))
    validate_envelope(*values)
    verify_remote_sources(values[0])
    return values


@app.function(image=image,volumes={DATA:data},cpu=(4,4),memory=(4096,4096),
              timeout=1200,retries=0,max_containers=1,scaledown_window=2)
def verify_assets(contract_json,authorization_json,review_json,local_gate_json,asset_root):
    """CPU-only full payload hashing before allocating the H100 worker."""
    contract,_,_,_=envelopes(contract_json,authorization_json,review_json,local_gate_json)
    root=Path(asset_root)
    if not str(root).startswith('/data/hn2-assets/'):
        raise ValueError('Use an isolated immutable run asset prefix')
    upload=json.loads(rooted(root,contract['upload_manifest']).read_text())
    count=0;payload_bytes=0
    for name,info in upload['files'].items():
        path=rooted(root,name)
        if path.stat().st_size!=info['bytes'] or sha(path)!=info['sha256']:
            raise ValueError('Remote frozen payload differs: '+name)
        count+=1;payload_bytes+=info['bytes']
    for name,expected in contract['data_manifest_sha256'].items():
        if sha(rooted(root,name))!=expected:raise ValueError('Remote manifest/prior differs: '+name)
    receipt=dict(passed=True,candidate_id=contract['candidate_id'],
        contract_sha256=hashlib.sha256(contract_json.encode()).hexdigest(),
        source_sha256=contract['source_sha256'],data_manifest_sha256=contract['data_manifest_sha256'],
        files=count,payload_bytes=payload_bytes,gpu_calls=0,quality_updates=0)
    path=root/'remote-asset-receipt.json'
    if path.exists():raise FileExistsError('Preserve the prior asset-verification attempt')
    write_json(path,receipt);data.commit()
    return dict(receipt=receipt,path=str(path),sha256=sha(path))


@app.function(image=image,volumes={DATA:data},gpu='H100',cpu=(12,12),memory=(65536,65536),
              timeout=4*3600,retries=0,max_containers=1,scaledown_window=2)
def train(contract_json,authorization_json,review_json,local_gate_json,asset_root,
          asset_receipt_sha256,dispatch_deadline_unix):
    from beatfcn.train.hn2_runtime import run_cuda
    contract,_,_,_=envelopes(contract_json,authorization_json,review_json,local_gate_json)
    path=Path(asset_root)/'remote-asset-receipt.json'
    if sha(path)!=asset_receipt_sha256:raise ValueError('CPU asset admission identity changed')
    receipt=json.loads(path.read_text())
    if (not receipt['passed'] or receipt['candidate_id']!=contract['candidate_id']
            or receipt['source_sha256']!=contract['source_sha256']
            or receipt['data_manifest_sha256']!=contract['data_manifest_sha256']
            or receipt['contract_sha256']!=hashlib.sha256(contract_json.encode()).hexdigest()):
        raise ValueError('CPU admission is not for this exact frozen candidate')
    return run_cuda(contract,asset_root,Path(DATA)/'runs',
        dispatch_deadline_unix=dispatch_deadline_unix,commit=data.commit)
