"""Count-preserving frozen frame-output null; no new model forward/update.

All fields of an output frame move together within equal legal head supports.
The exact original deterministic output must reproduce first. This breaks
output-time alignment, not a selective causal intervention on audio inputs.
"""
from collections import defaultdict
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np

from beatfcn.model.sample import match_f1
from beatfcn.train.relative_canvas import relative_batch
from scripts.audit_mamba_readout import decode
from scripts.run_frontend_canvas_comparison import setup
from scripts.run_local_portfolio_scale import row_factory,sha,write
from scripts.run_local_typed_canvas import score
from scripts.review_frontend_canvas_comparison import metrics


ROOT=Path(__file__).resolve().parents[1];IDENT='local-ssd-frame-permutation-v1'
PLAN=ROOT/f'bench/rhythm-v2/{IDENT}.json';REPORT=ROOT/'reports/local_ssd_frame_permutation_v1.json';ART=ROOT/f'artifacts/{IDENT}'


def main():
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve null audit')
    parent=json.loads((ROOT/'bench/rhythm-v2/local-mamba-readout-v1.json').read_text())
    paths=['artifacts/local-mamba-readout-v1/draft-expected_tail_rank.json','artifacts/local-mamba-readout-v1/empty-expected_tail_rank.json']
    plan=dict(id=IDENT,seed=20261065,optimizer_updates=0,model_forwards=0,device='cpu',wall_s=600,
        indices=parent['indices'],shifts_us=parent['shifts_us'],
        rule='Fixed RNG permutes complete40ms output frames within identical authorized head-support masks. Count multiset/global type counts preserved, no best permutation or target access. All raw outputs retained.',
        interpretation='Output-layer anti-prior null. It moves positional and acoustic decisions together; not input-audio causality. Original primary is unchanged.',
        source_sha256={n:sha(ROOT/n) for n in ['scripts/audit_ssd_frame_permutation.py','scripts/audit_mamba_readout.py']+paths},cloud_compute_calls=0)
    write(PLAN,plan);panel,store=setup(json.loads((ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json').read_text()));row=row_factory(panel,store)
    report=dict(id=IDENT,status='running',optimizer_updates=0,model_forwards=0,rows=[],source_probabilities={},cloud_compute_calls=0);write(REPORT,report);started=time.monotonic()
    tasks={}
    for task,path in zip(('draft','empty'),paths):
        baseline=json.loads((ROOT/path).read_text());records=[]
        for original in baseline:
            if time.monotonic()-started>plan['wall_s']:raise TimeoutError('Null wall')
            index=original['index'];e=row(index,'draft');inp,_=relative_batch([e],'cpu')
            path=ROOT/f'artifacts/local-mamba-readout-v1/probabilities/{index}.npz'
            report['source_probabilities'][str(path.relative_to(ROOT))]=sha(path)
            with np.load(path,allow_pickle=False) as saved:arrays={k:saved[k] for k in saved.files}
            same,_=decode(arrays,inp,e['request'],'expected_tail_rank',parent['seed']+17*index)
            if [asdict(v) for v in same]!=original['prediction']['events']:raise ValueError('Frozen primary does not reproduce')
            valid=inp['canvas_valid'][0].numpy();allowed=inp['canvas_head_allowed'][0].numpy()
            groups=defaultdict(list)
            for f in np.flatnonzero(valid):groups[allowed[f].tobytes()].append(int(f))
            permutation=np.arange(len(valid));rng=np.random.default_rng(plan['seed']+17*index)
            for values in groups.values():permutation[values]=rng.permutation(values)
            moved={k:v[:,permutation].copy() for k,v in arrays.items()}
            pred,detail=decode(moved,inp,e['request'],'expected_tail_rank',parent['seed']+17*index)
            if len(same)!=len(pred):raise ValueError('Count-preserving permutation changed count')
            if sorted(v.type for v in same)!=sorted(v.type for v in pred):raise ValueError('Permutation changed type marginal')
            start=e['request'].start_us;width=e['request'].end_us-start
            shifts=[match_f1([start+(v.onset_us-start+d)%width for v in pred],[v.onset_us for v in e['targets']],10000)[3] for d in plan['shifts_us']]
            record=dict(index=index,component=original['component'],request=asdict(e['request']),prediction=dict(complete=True,events=[asdict(v) for v in pred],count_decision=detail),
                target=original['target'],scores=score(pred,e['targets']),shifted_onset10=shifts,permutation=permutation.tolist(),original_count=len(same),count_and_type_marginals_exact=True)
            records.append(record);write(ART/f'{task}.json',records)
        tasks[task]=metrics(records)
    report.update(status='completed',results=tasks,wall_s=time.monotonic()-started);write(REPORT,report)


if __name__=='__main__':main()
