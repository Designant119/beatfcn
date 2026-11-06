"""D16: replay saved events through non-mutating benchmark evidence reporting."""
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

from beatfcn.events.canonical import Event
from beatfcn.model.reference_clock import benchmark_evidence
from scripts.audit_edit_cost_sensitivity import cost_proxy

ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT/'bench/rhythm-v2/d16-benchmark-evidence-v1.json'
    plan = json.loads(plan_path.read_text())
    for relative, expected in plan['offline_source_sha256'].items():
        if hashlib.sha256((ROOT/relative).read_bytes()).hexdigest() != expected:
            raise ValueError('frozen rescore source changed')
    panel = json.loads((ROOT/plan['panel']).read_text())
    metadata = {case['case_id']: case['row'] for case in panel['cases']}
    records, aggregates, sensitivity = [], [], []
    for source in plan['offline_sources']:
        payload = json.loads((ROOT/source['predictions']).read_text())
        review = json.loads((ROOT/source['review']).read_text())['results'][source['review_key']]
        expected = {r['case_id']: r for r in review['factual']}
        factual = [r for r in payload['records'] if r['field'] is None]
        if len(factual) != 8 or {r['case_id'] for r in factual} != set(metadata):
            raise ValueError('factual panel missing or changed')
        for row in factual:
            meta = metadata[row['case_id']]
            if row['chart_id'] != meta['chart_id']:
                raise ValueError('chart identity differs')
            pred = [Event.from_json(e) for e in row['generated_events']]
            truth = [Event.from_json(e) for e in row['truth_events']]
            evidence = benchmark_evidence(pred, truth, row['mode'], meta)
            prior = expected[row['case_id']]
            if not math.isclose(evidence['timing_f1_by_tolerance_ms']['20']['f1'],
                                prior['timing_point_f1'], abs_tol=1e-12):
                raise ValueError('primary timing score changed')
            full = evidence['reference_edit_proxy']['full_note']
            f1 = 2*full['matched']/(full['pred']+full['true']) if full['pred']+full['true'] else 1.
            if not math.isclose(f1, prior['full_note_f1'], abs_tol=1e-12):
                raise ValueError('primary full-note score changed')
            records.append(dict(model=source['name'], case_id=row['case_id'], mode=row['mode'],
                full_note_f1=f1, evidence=evidence))
            for attribute in plan['attribute_cost_sensitivity']:
                for review_cost in plan['review_cost_sensitivity']:
                    sensitivity.append(dict(model=source['name'], case_id=row['case_id'],
                        attribute_cost=attribute, review_cost=review_cost,
                        **cost_proxy(pred, truth, row['mode'], attribute, review_cost)))
        group = [r for r in records if r['model'] == source['name']]
        aggregates.append(dict(model=source['name'], windows=len(group),
            timing_f1_curve={str(t): mean(r['evidence']['timing_f1_by_tolerance_ms'][str(t)]['f1'] for r in group)
                             for t in plan['timing_tolerances_ms']},
            full_note_f1=mean(r['full_note_f1'] for r in group),
            positive_timing_edit_windows=sum(r['evidence']['reference_edit_proxy']['timing_point']['saved_fraction'] > 0 for r in group),
            positive_full_note_edit_windows=sum(r['evidence']['reference_edit_proxy']['full_note']['saved_fraction'] > 0 for r in group),
            condition_fields_passed=len(review['passed_fields']) if isinstance(review['passed_fields'], list) else review['passed_fields']))
    costs = []
    for source in plan['offline_sources']:
        for attribute in plan['attribute_cost_sensitivity']:
            for review_cost in plan['review_cost_sensitivity']:
                group = [r for r in sensitivity if (r['model'], r['attribute_cost'], r['review_cost']) ==
                         (source['name'], attribute, review_cost)]
                costs.append(dict(model=source['name'], attribute_cost=attribute, review_cost=review_cost,
                    mean_saved_fraction=mean(r['saved_fraction'] for r in group),
                    positive_windows=sum(r['saved_fraction'] > 0 for r in group), windows=len(group)))
    report = dict(schema='beatfcn.benchmark_evidence_review.v1', complete=True,
        plan_sha256=hashlib.sha256(plan_path.read_bytes()).hexdigest(), records=records,
        aggregates=aggregates, attribute_sensitivity_records=sensitivity, attribute_sensitivity_summary=costs,
        unchanged_primary_scores=len(records), new_model_inferences=0, training_updates=0,
        phase_adjustment_applied=False, model_acceptance_changed=False, product_promotion=False,
        limitations=plan['limitations'])
    path=ROOT/'reports/d16_benchmark_evidence.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(aggregates=aggregates, candidate_cost_sensitivity=[r for r in costs if r['model']=='R07B']), ensure_ascii=False))


if __name__ == '__main__':
    main()
