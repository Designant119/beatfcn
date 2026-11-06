"""Check frozen task coverage and input-intervention invariants on saved outputs."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def main():
    directory = ROOT/'reports/capability_review_raw/evaluation-review-v1'
    rows = [json.loads(p.read_text()) for p in (directory/'records').glob('*.json')]
    manifest = json.loads((ROOT/'bench/capability-v1/manifest.json').read_text())
    expected = {(q['chart_id'],q['kind'],ck) for q in manifest['requests'] for ck in ('3500','4500')}
    actual = {(r['chart_id'],r['request_kind'],r['checkpoint']) for r in rows if r['condition']=='baseline'}
    assert expected == actual and len(actual)==240
    keys = [(r['chart_id'],r['request_kind'],r['checkpoint'],r['condition']) for r in rows]
    assert len(set(keys)) == len(keys) == 276
    baselines = {(r['chart_id'],r['request_kind'],r['checkpoint']):r for r in rows if r['condition']=='baseline'}
    matched_targets = 0
    for (chart,kind,ck), r in baselines.items():
        if ck != '3500':continue
        other = baselines[(chart,kind,'4500')]
        assert r.get('request') == other.get('request')
        assert r.get('truth_events') == other.get('truth_events')
        matched_targets += 1
    checked_interventions = 0
    unavailable = []
    for r in rows:
        if r['condition']=='baseline':continue
        base = baselines[(r['chart_id'],r['request_kind'],r['checkpoint'])]
        if 'request' not in r:
            assert r.get('failure') and not r['complete']
            unavailable.append({'id':r['id'],'failure':r['failure']})
            continue
        assert r['truth_events']==base['truth_events']
        if r['condition']=='reference_swap':
            for k in base['request']:
                if k!='references':assert r['request'][k]==base['request'][k], k
        else:
            assert r['request']==base['request']
        checked_interventions += 1
    for r in rows:
        if r.get('complete'):assert r.get('failure') is None
        if 'scores' in r:
            assert r['scores']['pred']==len(r['generated_events'])
            assert r['scores']['true']==len(r['truth_events'])
    result={'baseline_tasks':len(actual),'paired_targets_identical':matched_targets,
            'diagnostic_inputs_and_targets_checked':checked_interventions,'unavailable_diagnostics':unavailable,
            'records_sha256':digest(sorted(rows,key=lambda r:r['id'])),
            'training_updates':0,'scope':'saved-record coverage and intervention invariants; not a spatial leakage or engine-playability certification'}
    path=ROOT/'reports/capability_review_integrity.json'
    path.write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    print(json.dumps({k:v for k,v in result.items() if k!='unavailable_diagnostics'}))


if __name__=='__main__':main()
