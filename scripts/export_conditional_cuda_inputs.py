"""Export verified native-prefix requests without local filesystem dependencies."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np

from beatfcn.train.local_conditional_inputs import FrozenLocalInputs
from scripts.check_experiment_storage import admission
from scripts.evolve_surgery import file_sha


def export(root, output):
    root, output = Path(root), Path(output)
    output.mkdir(exist_ok=False)
    receipt = admission(output, 2**30, 15*2**30)
    if receipt['status'] != 'admitted':
        raise RuntimeError('Storage admission failed')
    rows = []
    for split, name in [('train', 'local-history-probe/manifest.json'),
                        ('val', 'independent-val-manifest.json')]:
        manifest = root/'reports/workflow-pareto-20261105'/name
        provider = FrozenLocalInputs(root, manifest, None)
        for item in provider.spec['rows']:
            inputs, request, provenance = provider.build(item)
            path = output/(item['chart_id'].replace(':', '-')+'.npz')
            np.savez_compressed(path, **inputs)
            with np.load(path, allow_pickle=False) as loaded:
                if set(loaded.files) != set(inputs) or not all(
                    np.array_equal(loaded[k], v) for k, v in inputs.items()):
                    raise ValueError('Export changed input tensors')
            rows.append(dict(item=item, split=split, inputs=path.name,
                inputs_sha256=file_sha(path), request=asdict(request),
                provenance=provenance, source_manifest_sha256=file_sha(manifest),
                shapes={k:list(v.shape) for k,v in inputs.items()}))
            print(json.dumps(dict(exported=item['chart_id'], split=split, bytes=path.stat().st_size)), flush=True)
    if set(r['item']['component'] for r in rows if r['split']=='train') & set(
        r['item']['component'] for r in rows if r['split']=='val'):
        raise ValueError('Train/validation component overlap')
    result=dict(schema='conditional-portable-inputs.v1', rows=rows, storage=receipt,
                validation_training_forbidden=True, optimizer_updates=0)
    (output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();export(args.root,args.output)
