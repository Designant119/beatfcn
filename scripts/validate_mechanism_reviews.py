"""Read-only evidence/completeness gate; narrative judgment remains explicit."""
import argparse
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1]
INDEX=ROOT/'reports/local-portfolio-v1/mechanism-reviews-v2/index.json'


def load_registry(root=ROOT):
    rows=[]
    for name in ('reports/local-portfolio-v1/mechanism-reviews-v2/index.json',
                 'reports/local-evolution-v1/mechanism-reviews/index.json'):
        path=Path(root)/name
        if path.exists():rows.extend(json.loads(path.read_text())['candidates'])
    if len({r['id'] for r in rows})!=len(rows):raise ValueError('Duplicate mechanism-review candidate identity')
    return rows


def validate_review(review):
    errors=[]
    for name in ('candidate','analysis_status','identity','layers','mechanism_hypotheses','controls',
                 'missing_controls','lessons','successor_contract','closure_reason','source_record'):
        if not review.get(name):errors.append('Missing '+name)
    if review.get('analysis_status')!='complete':errors.append('Mechanism review not complete')
    for path,expected in review.get('identity',{}).items():
        p=ROOT/path
        if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=expected:errors.append('Evidence identity changed: '+path)
    if not (ROOT/review.get('source_record','')).is_file():errors.append('Missing narrative review')
    for layer in ('health','quality','structure_conditions'):
        if not review.get('layers',{}).get(layer):errors.append('Missing result layer: '+layer)
    if len(review.get('mechanism_hypotheses',[]))<2:errors.append('At least two discriminated mechanisms required')
    for hypothesis in review.get('mechanism_hypotheses',[]):
        for key in ('id','observation','mechanism','prediction','evidence','judgment'):
            if not hypothesis.get(key):errors.append('Incomplete mechanism '+hypothesis.get('id','?')+': '+key)
        for evidence in hypothesis.get('evidence',[]):
            if not (ROOT/evidence).exists():errors.append('Missing mechanism evidence: '+evidence)
    for control in review.get('missing_controls',[]):
        if not all(control.get(k) for k in ('question','comparison','decision_if')):errors.append('Missing discriminating control/action')
    if not review.get('missing_controls'):errors.append('Unmeasured alternatives must be explicit')
    for key in ('required_changes','forbidden_repeats'):
        if not review.get('successor_contract',{}).get(key):errors.append('Missing successor '+key)
    if len(review.get('closure_reason',''))<80:errors.append('Closure explanation is too short to state mechanism and scope')
    return errors


def validate_successor(contract,known):
    errors=[]
    if not contract.get('predecessors'):errors.append('Successor must identify predecessor reviews')
    for predecessor in contract.get('predecessors',[]):
        if predecessor not in known:errors.append('Unknown predecessor '+predecessor)
    for failure in contract.get('shared_failure_modes',[]):
        change=contract.get('mechanism_changes',{}).get(failure,{})
        if not all(change.get(k) for k in ('change','prediction','discriminating_control')):
            errors.append('Unaddressed shared failure: '+failure)
    probes=contract.get('pretraining_probe_receipts',[])
    if not probes:errors.append('Actual pretraining diagnostic receipts required')
    for probe in probes:
        if probe.get('status')!='completed' or not (ROOT/probe.get('evidence','')).is_file():
            errors.append('Unperformed pretraining diagnostic')
        if probe.get('scope')=='frozen_predecessor_diagnostic_only':
            errors.append('Predecessor diagnosis cannot certify a new implementation')
    if contract.get('execution_ready') is not True:errors.append('Successor contract not ready')
    return errors


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--candidate');parser.add_argument('--all',action='store_true')
    parser.add_argument('--successor',type=Path);args=parser.parse_args()
    if args.successor and args.candidate:parser.error('Successor dispatch requires all predecessor reviews, not a single-candidate bypass')
    registry_rows=load_registry();rows=registry_rows;errors=[];verified=[]
    if args.candidate:
        rows=[r for r in rows if r['id']==args.candidate]
        if len(rows)!=1:raise ValueError('Candidate not found')
    for r in rows:
        if r['analysis_status']!='complete':errors.append(r['id']+': mechanism review pending');continue
        path=ROOT/r['review'];review=json.loads(path.read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=r['review_sha256']:errors.append(r['id']+': review receipt drift')
        narrative=ROOT/r['markdown']
        if not narrative.is_file() or hashlib.sha256(narrative.read_bytes()).hexdigest()!=r.get('narrative_sha256'):
            errors.append(r['id']+': narrative receipt drift')
        errors.extend(r['id']+': '+e for e in validate_review(review));verified.append(r['id'])
    if args.successor:
        errors.extend(validate_successor(json.loads(args.successor.read_text()),{r['id'] for r in registry_rows}))
    print(json.dumps(dict(status='blocked' if errors else 'complete',verified=verified,errors=errors,
                         note='Checks artifact identity and required analysis branches; it does not prove a causal explanation or authorize cloud work.'),ensure_ascii=False,indent=2))
    if errors:raise SystemExit(1)


if __name__=='__main__':main()
