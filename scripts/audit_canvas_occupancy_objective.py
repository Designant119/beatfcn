"""No-model audit of sparse-slot loss weights and saved R10 false positives."""
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def quantiles(values):
    return dict(zip(('min', 'p25', 'median', 'p75', 'max'), map(float, np.quantile(values, [0, .25, .5, .75, 1]))))


def main():
    review = json.loads((ROOT / 'reports/p2b_probe_review.json').read_text())
    plan = json.loads((ROOT / 'bench/rhythm-v2/p2b-fit-v1.json').read_text())
    path = ROOT / 'inputs/r09_positive_pairs/catalog.pt'
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    if sha != plan['input_sha256'][str(path.relative_to(ROOT))]:
        raise ValueError('catalog identity changed')
    catalog = torch.load(path, map_location='cpu', weights_only=False)
    entries = {}
    for pair in catalog['train_pairs']:
        for entry in pair['entries']:
            if entry['row']['split'] != 'train':
                raise ValueError('non-training example')
            entries[(entry['row']['chart_id'], entry['example'].c0_us, entry['target_sha256'])] = entry
    modes = {}
    for mode in ('mania_4k', 'phira'):
        selected = [e for e in entries.values() if e['row']['mode'] == mode]
        count = np.array([len(e['truth_events']) for e in selected])
        if (count <= 0).any() or (count >= 640).any():
            raise ValueError('this audit formula requires nonempty, nonsaturated selected cores')
        # This is the necessary precision bound at perfect recall, not a new
        # model threshold. F1 = 2N / (2N + FP) >= .9 -> FP <= (2/.9-2)N.
        fp_budget = (2 / .9 - 2) * count
        modes[mode] = dict(windows=len(selected), event_count=quantiles(count),
            occupancy_fraction=quantiles(count / 640),
            per_positive_to_per_empty_loss_weight=quantiles((640 - count) / count),
            minimum_empty_accuracy_for_f1_09_at_perfect_recall=quantiles(1 - fp_budget / (640 - count)),
            source_audio=len({e['row']['audio_sha256'] for e in selected}))
    fixed = []
    for core in review['cores']:
        loss = core['occupancy_loss']
        n, negative = loss['positive_slots'], loss['empty_slots']
        fixed.append(dict(chart_id=core['identity']['chart_id'], notes=n,
            positive_to_empty_weight=negative / n,
            empty_accuracy=loss['empty_slot_accuracy'],
            required_empty_accuracy_for_f1_09_at_perfect_recall=1 - (2 / .9 - 2) * n / negative,
            maximum_integer_false_positives_at_f1_09=int(np.floor((2 / .9 - 2) * n)),
            measured_first_round_excess=core['rounds'][0]['excess_count']))
    result = dict(schema='beatfcn.canvas_occupancy_objective_audit.v1', complete=True,
        new_model_updates=0, new_model_inference=False, new_paid_compute_usd=0,
        catalog_sha256=sha, selected_train_windows=len(entries), modes=modes, fixed_probe=fixed,
        analytic_result={
            'scope': 'Derivation for constant positive/negative weights, not measured calibration of this neural model.',
            'weighted_ce_optimum': 'q = alpha*p / (alpha*p + beta*(1-p))',
            'argmax_q_above_half_corresponds_to': 'p > beta/(alpha+beta)',
            'if_alpha_equals_1_over_2Npositive_and_beta_1_over_2Nempty': 'threshold in p becomes Npositive/(Npositive+Nempty)',
            'limitation': 'R10 uses different target-dependent weights per core. No single deployable inverse mapping follows; inferring weights from hidden target counts at generation would leak labels.'},
        proposed_next_control={
            'dimension': 'objective',
            'change': 'Replace the two-stratum kind loss by the mean categorical log loss over all fixed slots; keep architecture, corruption, all other heads, decoding, optimizer, initialization, data, 100 updates and gates fixed.',
            'hypothesis': 'A proper occupancy likelihood may reduce false-positive empty slots while retaining already learned real notes.',
            'risk': 'Nonempty kind gradients become smaller; recall or rare types may collapse. Reject on original gates, not on a nicer count alone.',
            'status': 'proposal_only_not_registered_or_run',
            'no_hard_count_repair': True},
        limits=['Selected paired training windows are not representative of expanded corpus difficulty; no empty cores in this panel.',
                'This loss audit does not confirm causality or independent benchmark gains.',
                'Reference-free product acceptability still needs human calibration and held-out edit utility.'])
    (ROOT / 'reports/p2b_occupancy_objective_audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'windows':len(entries), 'modes':modes, 'fixed_probe':fixed}, ensure_ascii=False))


if __name__ == '__main__':
    main()
