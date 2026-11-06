"""Supplementary reconstruction-cost sensitivity; never a model acceptance gate."""
from collections import Counter
import hashlib
import json
from pathlib import Path
from statistics import mean

from beatfcn.events.canonical import Event

ROOT = Path(__file__).resolve().parents[1]


def minimum_assignment(costs):
    """Minimum-cost injective assignment, rows <= columns (Hungarian potentials)."""
    n = len(costs)
    if not n:
        return []
    m = len(costs[0])
    if m < n or any(len(row) != m for row in costs):
        raise ValueError('assignment matrix must be rectangular with rows <= columns')
    u, v, assigned, previous = [0.]*(n+1), [0.]*(m+1), [0]*(m+1), [0]*(m+1)
    for i in range(1, n+1):
        assigned[0] = i; j0 = 0
        distances, used = [float('inf')]*(m+1), [False]*(m+1)
        while True:
            used[j0] = True; i0 = assigned[j0]; delta = float('inf'); j1 = 0
            for j in range(1, m+1):
                if not used[j]:
                    current = costs[i0-1][j-1] - u[i0] - v[j]
                    if current < distances[j]:
                        distances[j], previous[j] = current, j0
                    if distances[j] < delta:
                        delta, j1 = distances[j], j
            for j in range(m+1):
                if used[j]:
                    u[assigned[j]] += delta; v[j] -= delta
                else:
                    distances[j] -= delta
            j0 = j1
            if assigned[j0] == 0:
                break
        while j0:
            j1 = previous[j0]; assigned[j0] = assigned[j1]; j0 = j1
    result = [-1]*n
    for j in range(1, m+1):
        if assigned[j]:
            result[assigned[j]-1] = j-1
    return result


def repair_fields(pred, truth, mode, tolerance_us=20_000):
    if abs(pred.onset_us-truth.onset_us) > tolerance_us:
        return None
    edits = []
    if pred.type != truth.type:
        edits.append('type')
    if mode == 'mania_4k' and pred.location.lane != truth.location.lane:
        edits.append('lane')
    if truth.is_hold and (not pred.is_hold or abs(pred.end_us-truth.end_us) > tolerance_us):
        edits.append('duration')
    return edits


def cost_proxy(pred, truth, mode, attribute_cost, review_cost):
    if mode not in ('mania_4k', 'phira') or attribute_cost < 0 or review_cost < 0:
        raise ValueError('invalid mode or costs')
    pairs = [[repair_fields(p, t, mode) for t in truth] for p in pred]
    # Start by inserting every truth event. Matching one prediction removes
    # that insertion; dummy columns delete the prediction. Truth columns are
    # used at most once, including for simultaneous chords and dense drags.
    costs = [[10. if fields is None else attribute_cost*len(fields)-1. for fields in row]
             + [1.]*len(pred) for row in pairs]
    assignment = minimum_assignment(costs)
    operations = Counter(insert=len(truth), delete=0, type=0, lane=0, duration=0)
    reused = 0
    for i, j in enumerate(assignment):
        if j < len(truth):
            if pairs[i][j] is None:
                raise AssertionError('an ineligible time pair was selected')
            operations['insert'] -= 1; operations.update(pairs[i][j]); reused += 1
        else:
            operations['delete'] += 1
    cost = operations['insert']+operations['delete']+attribute_cost*sum(operations[k] for k in ('type','lane','duration'))+review_cost*len(pred)
    return dict(pred=len(pred),truth=len(truth),reused=reused,operations=dict(operations),edit_cost=cost,
                saved_fraction=1-cost/len(truth) if truth else None)


def main():
    sources = {'3500': 'reports/r03_window_raw/baseline.json',
               '4500_density': 'reports/r03_window_raw/r03-window-density.benchmark.json',
               '4500_all': 'reports/r03_window_raw/r03-window-all.benchmark.json',
               'R04': 'reports/r04_evaluate.json', 'R05': 'reports/r05_evaluate.json'}
    records = []; fingerprints = {}
    for name, relative in sources.items():
        path = ROOT / relative; payload = json.loads(path.read_text())
        fingerprints[name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in payload['records']:
            if row['field'] is not None:
                continue
            pred = [Event.from_json(e) for e in row['generated_events']]
            truth = [Event.from_json(e) for e in row['truth_events']]
            for attribute in (.25, .5, 1.):
                for review in (0., .1):
                    result = cost_proxy(pred, truth, row['mode'], attribute, review)
                    records.append(dict(model=name,case_id=row['case_id'],mode=row['mode'],
                        attribute_cost=attribute,review_cost=review,**result))
    aggregate = []
    for name in sources:
        for attribute in (.25, .5, 1.):
            for review in (0., .1):
                subset = [r for r in records if (r['model'],r['attribute_cost'],r['review_cost'])==(name,attribute,review)]
                aggregate.append(dict(model=name,attribute_cost=attribute,review_cost=review,
                    macro_saved_fraction=mean(r['saved_fraction'] for r in subset),
                    positive_windows=sum(r['saved_fraction']>0 for r in subset),windows=len(subset)))
    report = dict(schema='beatfcn.edit_cost_sensitivity.v1',source_files_sha256=fingerprints,
        current_round_gate_unchanged='R06',model_acceptance=False,new_inference_calls=0,
        cost_assumptions=dict(insert=1,delete=1,per_attribute=[.25,.5,1.],review_each_prediction=[0,.1],
            eligible_onset_tolerance_us=20000,attributes=['type','mania_lane','hold_end'],
            phira_spatial_output_scored=False,real_human_times_measured=False),
        records=records,aggregate=aggregate)
    output=ROOT/'reports/edit_cost_sensitivity_2026-10-25.json'
    output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# 修改成本敏感性（补充诊断）','','所有值都是重建同一人谱的假设成本，不是实测制谱时间或音乐质量。',
        '只允许在 ±20ms 内复用音符；改类型、mania 键道、补/改 hold 尾点各计一次属性修改。',
        '插入和删除各为 1；同时列出便宜修改与额外逐音符复核成本。匹配为一对一最小成本，不重复复用同一音符。',
        '不改变 R06 冻结门槛，不替代类型控制、整曲、音乐对应与用户试听。','',
        '| 模型 | 每属性成本 | 每音符复核成本 | 平均节省比例 | 正收益窗口 |',
        '|---|---:|---:|---:|---:|']
    for row in aggregate:
        lines.append(f"| {row['model']} | {row['attribute_cost']:.2f} | {row['review_cost']:.2f} | "
                     f"{row['macro_saved_fraction']:.3f} | {row['positive_windows']}/{row['windows']} |")
    lines += ['', '空窗相对空白起稿的节省为 0，不能称为帮助。属性修改便宜时出现正值，仅表明插入/删除代理可能低估时间点复用；需要真实人评校准，不能据此撤销此前的用户否决。']
    output.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(aggregate,ensure_ascii=False))


if __name__=='__main__': main()
