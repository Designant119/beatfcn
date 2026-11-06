"""Component and bad-case audit for the interrupted/resumed Mamba quality curve."""
import hashlib
import json
from pathlib import Path

from scripts.review_local_portfolio_data_scale import macro


ROOT = Path(__file__).resolve().parents[1]
PARENT = ROOT / 'reports/local_portfolio_mamba_quality_v1.json'
RESUME = ROOT / 'reports/local_portfolio_mamba_quality_resume_v1.json'
A1000 = ROOT / 'artifacts/local-portfolio-v1/mamba-quality'
A2000 = ROOT / 'artifacts/local-portfolio-v1/mamba-quality-resume'
DEST = ROOT / 'reports/local_portfolio_mamba_quality_review_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parent = json.loads(PARENT.read_text())
    resume = json.loads(RESUME.read_text())
    if parent['updates'] != 1775 or not resume['complete'] or resume['updates'] != 2000:
        raise ValueError('Unexpected Mamba parent/resume states')
    rows = {}
    for step, art in (('1000', A1000), ('2000', A2000)):
        rows[step] = json.loads((art / f'mamba/quality-{step}-draft.json').read_text())
    result = dict(id='local-portfolio-mamba-quality-review-v1', parent_sha256=sha(PARENT),
        resume_sha256=sha(RESUME), checkpoint_2000_sha256=sha(A2000 / 'mamba/quality-2000.pt'),
        steps={step: macro(values) for step, values in rows.items()}, comparisons={}, bad_cases=[],
        limitations=['One seed and 16 reused development components; confirmation remains sealed.',
            'Parent was interrupted after 1775; resume re-executed 1501–2000 from the exact 1500 checkpoint.',
            'The fixed 2000 endpoint is reported alongside, not replaced by, the better 1000 point.',
            'No human listening, edit timing or native play-test.'])
    a, b = result['steps']['1000'], result['steps']['2000']
    result['comparisons']['training_1000_to_2000'] = {key: b[key] - a[key]
        for key in ('full_note_macro', 'onset10_macro', 'acoustic_gain_macro')}
    small64 = json.loads((ROOT / 'artifacts/local-portfolio-v1/data-scale/small64/quality-2000-draft.json').read_text())
    small_macro = macro(small64)
    result['comparisons']['mamba_minus_small64_2000'] = {key: b[key] - small_macro[key]
        for key in ('full_note_macro', 'onset10_macro', 'acoustic_gain_macro')}
    for task in ('completion', 'reference_masked', 'density_low', 'density_high', 'empty'):
        values = json.loads((A2000 / f'mamba/quality-2000-{task}.json').read_text())
        result['comparisons'][task] = dict(predicted=sum(len(row['prediction']['events']) for row in values),
            target=sum(len(row['target']) for row in values),
            predicted_types={kind: sum(event['type'] == kind for row in values for event in row['prediction']['events'])
                for kind in ('tap', 'hold', 'flick', 'drag')})
    result['bad_cases'] = [dict(index=row['index'], component=row['component'], request=row['request'],
        scores=row['scores'], shifted_onset10=row['shifted_onset10'], prediction=row['prediction'], target=row['target'])
        for row in sorted(rows['2000'], key=lambda row: (((row['scores']['onset_10ms'] or {}).get('f1') or 0)
            - sum(row['shifted_onset10']) / len(row['shifted_onset10']), row['scores']['full_note_f1']))[:6]]
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(steps={step: {key: value for key, value in summary.items() if key != 'per_component'}
        for step, summary in result['steps'].items()}, comparisons=result['comparisons']), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
