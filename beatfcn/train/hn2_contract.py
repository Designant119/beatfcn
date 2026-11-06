"""The complete reviewed contract, excluding only its derived digest field."""
import hashlib
import json
from pathlib import Path
import time


def contract_digest(contract):
    value={k:v for k,v in contract.items() if k!='contract_sha256'}
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,
        separators=(',',':'),allow_nan=False).encode()).hexdigest()


def validate_connection(connection,root,*,now=None):
    now=time.time() if now is None else now;root=Path(root).resolve()
    if (connection.get('fresh_check_succeeded') is not True or
        not 0<=now-connection.get('checked_at_unix',0)<600 or
        connection.get('full_lifecycle_bound_verified') is not True):
        raise ValueError('Fresh verified account/lifecycle evidence required')
    results=connection.get('results',{});hashes=connection.get('evidence_sha256',{})
    for kind in ('apps','rates','billing'):
        record=results.get(kind,{})
        if record.get('ok') is not True or not 0<=now-record.get('checked_at_unix',0)<600:
            raise ValueError('Fresh nonempty '+kind+' result required')
        name=record.get('evidence_path','');path=(root/name).resolve()
        if not name or not path.is_relative_to(root) or not path.is_file():
            raise ValueError('Actual '+kind+' evidence file required')
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        if digest!=hashes.get(name):raise ValueError(kind+' evidence identity mismatch')
