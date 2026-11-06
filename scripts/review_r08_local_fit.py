"""Audit the frozen R08 CPU probe without inference or changing its gates."""
from collections import Counter
import hashlib
import json
from pathlib import Path

from beatfcn.events.controls import MODE_FIELDS
from beatfcn.train.probe import assess_generation, assess_memorisation


ROOT = Path(__file__).resolve().parents[1]


def main():
    plan_path = ROOT / 'bench/rhythm-v2/r08-local-cpu-fit-v1.json'
    plan = json.loads(plan_path.read_text())
    result_path = ROOT / plan['result']
    result = json.loads(result_path.read_text())
    resources = json.loads((ROOT / plan['resource_report']).read_text())
    if result['plan_sha256'] != hashlib.sha256(plan_path.read_bytes()).hexdigest():
        raise ValueError('the probe did not use this frozen plan')
    if not result['complete'] or not resources['complete']:
        raise ValueError('wait for the probe and its process supervisor to finish')
    if result['optimizer_updates'] != plan['steps']:
        raise ValueError('incomplete optimizer sequence; report execution failure separately')
    expected_fields = {f'{mode}:{field}' for mode, fields in MODE_FIELDS.items() for field in fields}
    before, after = result['condition_gaps_before'], result['condition_gaps_after']
    if set(before) != expected_fields or set(after) != expected_fields:
        raise ValueError('missing or unexpected condition fields')
    rows = []
    for field in sorted(expected_fields):
        left, right = before[field], after[field]
        identity = lambda x: (x['positive_band'], x['negative_band'], x['decisions'])
        if list(map(identity, left)) != list(map(identity, right)):
            raise ValueError('condition pairs changed between measurements')
        rows.append(dict(field=field, pairs=len(right),
            gaps_before=[r['gap'] for r in left], gaps_after=[r['gap'] for r in right],
            positive_nll_after=[r['positive_nll'] for r in right],
            negative_nll_after=[r['negative_nll'] for r in right],
            passed=all(r['gap'] + 1e-6 >= plan['margin_per_token'] for r in right)))
    supervised = assess_memorisation(result['history'], result['gradients'],
        plan['supervised_gate'], plan['steps'])
    generation = assess_generation(result['generation'],
        len(plan['exact_probe_core_indices']), plan['generation_f1_at_least'])
    binding = dict(passed=all(r['passed'] for r in rows),
        fields_passed=sum(r['passed'] for r in rows), fields_total=len(rows),
        margin_per_decision=plan['margin_per_token'], rows=rows)
    decision = all(g['passed'] for g in (supervised, binding, generation))
    if decision != result['probe_passed']:
        raise ValueError('independent gate recalculation disagrees with probe')
    predicted = Counter()
    truth = Counter()
    for row in result['generation']:
        predicted.update(row['predicted_types'])
        truth.update(row['truth_types'])
    summary = dict(schema='beatfcn.r08_local_fit_review.v1',
        complete=True, decision='probe_passed_not_product_acceptance' if decision else 'reject_before_formal_training',
        probe_passed=decision, model_promotion=False, cloud_functions_started=0,
        actual_optimizer_updates=result['optimizer_updates'],
        supervised_gate=supervised, binding_gate=binding, generation_gate=generation,
        first_loss=result['history'][0]['loss'], final_loss=result['history'][-1]['loss'],
        actual_wall_s=result['wall_s'], peak_rss_gib=result['peak_rss_gib'],
        resource_limit_reason=resources.get('terminated_by'),
        auxiliary_margin_hits=sum(r['margin_hits'] for r in result['auxiliary_steps']),
        auxiliary_pairs_seen=sum(r['pairs'] for r in result['auxiliary_steps']),
        generated_type_counts=dict(predicted), truth_type_counts=dict(truth),
        evidence={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (plan_path, result_path, ROOT / plan['resource_report'])},
        limitations=[
            'Two fixed training cores, with their original human context; not held-out quality.',
            'Training-derived all-field requests in fitted decoding; not a product control benchmark.',
            'Only one actual target per audio; fitting does not establish general conditional identifiability.',
            'No mania hold in this fixed batch; CPU FP32 does not establish CUDA equivalence.',
            'Final condition gradients were not measured; small likelihood gaps alone do not prove a gradient cause.',
        ])
    out = ROOT / 'reports/r08_local_cpu_fit_review.json'
    out.write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    text = [
        '# R08 真实小批拟合复盘', '',
        f"裁决：`{summary['decision']}`。没有正式训练或模型晋升。", '',
        f"本机 {result['optimizer_updates']} 次更新，{result['wall_s']:.1f} 秒，峰值 RSS {result['peak_rss_gib']:.3f} GiB。",
        f"监督损失 {summary['first_loss']:.6f} → {summary['final_loss']:.6f}；固定批监督关 {'通过' if supervised['passed'] else '失败'}。",
        f"正确/错误条件评分间隔通过 {binding['fields_passed']}/{binding['fields_total']}，冻结门槛为每决策 {plan['margin_per_token']}。", '',
        '| 条件 | 初始间隔范围 | 最终间隔范围 | 通过 |', '| --- | --- | --- | --- |',
    ]
    for row in rows:
        a, b = row['gaps_before'], row['gaps_after']
        text.append(f"| {row['field']} | {min(a):.4g} .. {max(a):.4g} | {min(b):.4g} .. {max(b):.4g} | {row['passed']} |")
    text += ['', '| 记忆样本自由生成 | 完整音符 F1 | 完成 / 合法 |', '| --- | --- | --- |']
    for row in generation['cores']:
        text.append(f"| {row['chart_id']} | {row['full_note_f1']:.4f} | {row['complete']} / {row['legal']} |")
    text += ['', '解释边界：这检验拟合和条件分辨的必要能力，不是独立泛化、生成控制或制谱实用性验收。最终条件梯度未测，不能把评分差缩小直接归因为梯度消失。']
    (ROOT / 'reports/r08_local_cpu_fit_review.md').write_text('\n'.join(text)+'\n')
    print(json.dumps({k:summary[k] for k in ('decision','actual_optimizer_updates','first_loss','final_loss','actual_wall_s','peak_rss_gib')}))


if __name__ == '__main__':
    main()
