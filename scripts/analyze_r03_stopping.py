"""Explain observed empty outputs without changing the decoder or reusing old probabilities."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    summaries = {}
    for run in ('baseline', 'r03-window-density', 'r03-window-all'):
        name = 'baseline_observed.json' if run == 'baseline' else f'{run}.benchmark.json'
        path = ROOT / 'reports/r03_window_raw' / name
        payload = json.loads(path.read_text())
        if run == 'baseline' and not payload.get('all_outputs_equal'):
            raise ValueError('baseline CUDA replay did not preserve original events')
        factual = [r for r in payload['records'] if r['variant'] == 'unspecified_conditions']
        rows = []
        for r in factual:
            obs = r['diagnostics']['time_pointer_observation']
            if obs['output_rule_changed'] or not obs['finite'] or not obs['calls']:
                raise ValueError('observer unavailable or generation protocol changed')
            first, last = obs['first'], obs['last']
            if first['greedy_end'] and r['generated_events']:
                raise ValueError('first END contradicts saved generated events')
            decisions = r['diagnostics']['note_decisions']
            count = decisions['type_decisions']
            rows.append(dict(case_id=r['case_id'], mode=r['mode'], notes=len(r['generated_events']),
                first_end_probability=first['end_probability'], first_any_note_probability=first['any_note_probability'],
                first_best_note_probability=first['best_note_probability'], first_greedy_end=first['greedy_end'],
                first_end_wins_individually_but_not_total_mass=first['end_beats_individual_but_not_total_note_mass'],
                last_end_probability=last['end_probability'],
                last_end_wins_individually_but_not_total_mass=last['end_beats_individual_but_not_total_note_mass'],
                type_decisions=count, mean_legal_type_probability=[p / count for p in decisions['legal_probability_sum']] if count else None,
                type_proposals=decisions['proposed_by_type'], types_emitted=decisions['emitted_by_type'],
                hold_to_tap=decisions['hold_to_tap'], hold_end_failures=decisions['hold_end_failures']))
        summaries[run] = dict(cases=rows, empty_cases=sum(r['notes'] == 0 for r in rows),
            empty_by_first_end=sum(r['first_greedy_end'] for r in rows),
            first_end_with_majority_non_end_mass=sum(r['first_end_wins_individually_but_not_total_mass'] for r in rows),
            endings_with_majority_non_end_mass=sum(r['last_end_wins_individually_but_not_total_mass'] for r in rows),
            hold_to_tap=sum(r['hold_to_tap'] for r in rows))
    report = dict(schema='beatfcn.r03_stopping_observation.v1', runs=summaries, changed_generation_rule=False,
        limitation='Applies only to these trained models and eight held-out blank-canvas excerpts. '
                   'Rank versus total mass is an observation, not proof that a different stopping rule helps. '
                   'Baseline probabilities come from a new observed CUDA replay with identical factual event outputs.')
    path = ROOT / 'reports/r03_stopping_review.json'
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    lines = ['# R03 空谱与类型消失观测', '',
        '| 模型 | 空谱 | 首步 END | 首步 END 但音符总概率 >50% | 结束时音符总概率 >50% | hold→tap |',
        '| --- | --- | --- | --- | --- | --- |']
    for run, s in summaries.items():
        lines.append(f"| {run} | {s['empty_cases']}/8 | {s['empty_by_first_end']}/8 | "
                     f"{s['first_end_with_majority_non_end_mass']}/8 | {s['endings_with_majority_non_end_mass']}/8 | {s['hold_to_tap']} |")
        lines += ['', '| 片段 | 输出数 | 首步 END 概率 | 首步最大单个音符候选概率 | 最终 END 概率 |',
                  '| --- | --- | --- | --- | --- |']
        for row in s['cases']:
            lines.append(f"| {row['case_id']} | {row['notes']} | {row['first_end_probability']:.4f} | "
                         f"{row['first_best_note_probability']:.4f} | {row['last_end_probability']:.4f} |")
        lines.append('')
    lines += ['没有改变解码规则。该观测不能单独证明提高停止阈值有效；任何解码干预仍须检查过量、完整性和修改成本。']
    path.with_suffix('.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({run: {k: v for k, v in s.items() if k != 'cases'} for run, s in summaries.items()}))


if __name__ == '__main__':
    main()
