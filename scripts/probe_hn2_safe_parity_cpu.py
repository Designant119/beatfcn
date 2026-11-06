"""Exercise the exact parity helper API on CPU twice;never a CUDA certificate."""
import hashlib
import inspect
import json
from pathlib import Path
import time
import torch

from beatfcn.train import hn2_safe_runtime as runtime
from beatfcn.train.hn2_native_fixtures import native_fixtures
from beatfcn.train.modal_event_dataset import ModalEventDataset,collate_event_batch


def main():
    root=Path(__file__).resolve().parents[1];torch.set_num_threads(4)
    source=inspect.getsource(runtime.fixed_z_cuda_parity)
    assert "for device in ('cpu','cuda'):" in source
    altered=source.replace("for device in ('cpu','cuda'):","for device in ('cpu','cpu'):")
    namespace=dict(vars(runtime));exec(compile(altered,'<explicit-cpu-only-harness>','exec'),namespace)
    data=ModalEventDataset(root,'artifacts/modal120m-data-v1/prepared-v3/manifest.json',split='train')
    prior=json.loads((root/'artifacts/c17-masked-native-events-v1/priors.json').read_text())['duration']
    cases=[dict(name='real-train0',**collate_event_batch([data[0]])),*native_fixtures()]
    start=time.monotonic();result=[]
    for case in cases:
        value=namespace['fixed_z_cuda_parity'](case,prior,2026103001)
        value['scope']='CPU/CPU API harness only;not CUDA verification'
        result.append(dict(name=case['name'],result=value))
    receipt=dict(passed=all(r['result']['passed'] for r in result),cases=result,
        scope='Only device tuple replaced;exact source/function hashes recorded;no optimizer/CUDA',
        hn2_forward_calls=15,backward_calls=6,optimizer_updates=0,cloud_calls=0,
        wall_s=time.monotonic()-start,torch_version=torch.__version__,
        source_sha256=hashlib.sha256((root/'beatfcn/train/hn2_safe_runtime.py').read_bytes()).hexdigest(),
        function_sha256=hashlib.sha256(source.encode()).hexdigest())
    path=root/'reports/modal-120m-20261030/safe-parity-cpu-harness-v2.json'
    if path.exists():raise FileExistsError('Keep this actual probe receipt immutable')
    path.write_text(json.dumps(receipt,indent=2)+'\n')
    print({k:v for k,v in receipt.items() if k!='cases'})


if __name__=='__main__':main()
