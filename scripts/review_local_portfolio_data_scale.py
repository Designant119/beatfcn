"""Component-level audit of the completed train24/train64 quality contrast."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / 'reports/local_portfolio_data_scale_v1.json'
ART = ROOT / 'artifacts/local-portfolio-v1/data-scale'
DEST = ROOT / 'reports/local_portfolio_data_scale_review_v1.json'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def component_rows(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['component']].append(row)
    return grouped


def macro(rows):
    grouped = component_rows(rows)
    full = {component: sum(row['scores']['full_note_f1'] for row in values) / len(values)
        for component, values in grouped.items()}
    onset = {component: sum((row['scores']['onset_10ms'] or {}).get('f1') or 0 for row in values) / len(values)
        for component, values in grouped.items()}
    acoustic = {component: sum(((row['scores']['onset_10ms'] or {}).get('f1') or 0)
        - sum(row['shifted_onset10']) / len(row['shifted_onset10']) for row in values) / len(values)
        for component, values in grouped.items()}
    return dict(components=len(grouped), full_note_macro=sum(full.values()) / len(full),
        onset10_macro=sum(onset.values()) / len(onset), acoustic_gain_macro=sum(acoustic.values()) / len(acoustic),
        positive_acoustic_components=sum(value > 0 for value in acoustic.values()),
        per_component={component: dict(full_note=full[component], onset10=onset[component],
            acoustic_gain=acoustic[component]) for component in grouped})


def main():
    report = json.loads(REPORT.read_text())
    if report['status'] != 'completed' or report['updates'] != {'small24': 2000, 'small64': 2000}:
        raise ValueError('Need the completed 2000-step paired run')
    result = dict(id='local-portfolio-data-scale-review-v1', source_report_sha256=sha(REPORT),
        checkpoint_sha256={}, steps={}, comparisons={}, bad_cases={}, limitations=[
            'Development components are reused for model selection; no confirmation component was scored.',
            'One seed and 16 development components; this is not a scaling law or product-quality result.',
            'No human listening, edit timing or native play-test was performed.',
            'Post-hoc review does not retroactively create an investment gate.'])
    for arm in ('small24', 'small64'):
        checkpoint = ART / arm / 'quality-2000.pt'
        if sha(checkpoint) != report['final_weight_sha256'][arm]:
            raise ValueError(f'Checkpoint identity drift: {arm}')
        result['checkpoint_sha256'][arm] = report['final_weight_sha256'][arm]
    for step in ('1000', '2000'):
        result['steps'][step] = {}
        for arm in ('small24', 'small64'):
            rows = json.loads((ART / arm / f'quality-{step}-draft.json').read_text())
            result['steps'][step][arm] = macro(rows)
    for step in ('1000', '2000'):
        small, expanded = result['steps'][step]['small24'], result['steps'][step]['small64']
        per = {}
        for component in small['per_component']:
            a, b = small['per_component'][component], expanded['per_component'][component]
            per[component] = {key: b[key] - a[key] for key in ('full_note', 'onset10', 'acoustic_gain')}
        result['comparisons'][f'data_gain_{step}'] = dict(
            full_note_macro=expanded['full_note_macro'] - small['full_note_macro'],
            onset10_macro=expanded['onset10_macro'] - small['onset10_macro'],
            acoustic_gain_macro=expanded['acoustic_gain_macro'] - small['acoustic_gain_macro'],
            improved_full_components=sum(value['full_note'] > 0 for value in per.values()),
            improved_onset_components=sum(value['onset10'] > 0 for value in per.values()), per_component=per)
        result['comparisons'][f'longer_training_gain_{step}'] = {
            arm: {key: result['steps']['2000'][arm][key] - result['steps']['1000'][arm][key]
                for key in ('full_note_macro', 'onset10_macro', 'acoustic_gain_macro')}
            for arm in ('small24', 'small64')} if step == '2000' else None
    for arm in ('small24', 'small64'):
        rows = json.loads((ART / arm / 'quality-2000-draft.json').read_text())
        ranked = sorted(rows, key=lambda row: (((row['scores']['onset_10ms'] or {}).get('f1') or 0)
            - sum(row['shifted_onset10']) / len(row['shifted_onset10']), row['scores']['full_note_f1']))
        result['bad_cases'][arm] = [dict(index=row['index'], component=row['component'],
            request=row['request'], scores=row['scores'], shifted_onset10=row['shifted_onset10'],
            prediction=row['prediction'], target=row['target']) for row in ranked[:4]]
        completion = json.loads((ART / arm / 'quality-2000-completion.json').read_text())
        masked = json.loads((ART / arm / 'quality-2000-reference_masked.json').read_text())
        by_index = {row['index']: row for row in completion}
        effects = [masked_row['scores']['full_note_f1'] - by_index[masked_row['index']]['scores']['full_note_f1']
            for masked_row in masked]
        result['comparisons'][f'{arm}_reference_mask_minus_visible'] = dict(
            mean_full_note=sum(effects) / len(effects), positive_windows=sum(value > 0 for value in effects),
            windows=len(effects))
        low = json.loads((ART / arm / 'quality-2000-density_low.json').read_text())
        high = json.loads((ART / arm / 'quality-2000-density_high.json').read_text())
        result['comparisons'][f'{arm}_density_response'] = dict(
            low_predicted=sum(len(row['prediction']['events']) for row in low),
            high_predicted=sum(len(row['prediction']['events']) for row in high),
            target=sum(len(row['target']) for row in low),
            low_full_note=sum(row['scores']['full_note_f1'] for row in low) / len(low),
            high_full_note=sum(row['scores']['full_note_f1'] for row in high) / len(high))
    DEST.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(comparisons=result['comparisons'],
        steps={step: {arm: {key: value for key, value in summary.items() if key != 'per_component'}
            for arm, summary in arms.items()} for step, arms in result['steps'].items()}), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
