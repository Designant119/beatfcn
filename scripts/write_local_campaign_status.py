"""Current campaign view from real reports; never decides or starts trials."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from scripts.portfolio_terminal_catalog import catalog


ROOT = Path(__file__).resolve().parents[1]


def latest(path):
    return {r['id']: r for r in map(json.loads, path.read_text().splitlines())}


def main():
    council = latest(ROOT / 'reports/council-archive/decisions.jsonl')
    research = latest(ROOT / 'reports/local-portfolio-v1/research-suggestions.jsonl')
    terminal = {'adopted', 'rejected', 'tested_supported', 'tested_failed'}
    decisions = {**council, **research}
    open_rows = [r for r in decisions.values() if r['status'] not in terminal]
    values = catalog()
    states = {}
    for name in ('remaining-pool-v4', 'research-then-finalists-v1',
                 'frozen-finalist-protocol-v2', 'terminal-closeout-watch-v1'):
        path = ROOT / f'reports/local-portfolio-v1/{name}.json'
        if path.exists():
            states[name] = json.loads(path.read_text())
    progress = {}
    for name in ('mert_context_pair', 'attention_control_quality', 'concat_control_quality',
                 'music_depth_pair', 'music_ssl_pair'):
        path = ROOT / f'reports/local_{name}_v1.json'
        if not path.exists():
            continue
        report = json.loads(path.read_text())
        progress[name] = dict(status=report['status'], arms={})
        for arm, row in report.get('arms', {}).items():
            item = {k: row[k] for k in ('status', 'health_updates', 'quality_updates',
                                         'health_wall_s', 'training_wall_s', 'evaluation_wall_s') if k in row}
            metrics = row.get('evaluations', {}).get('draft')
            if metrics:
                item['finished_primary_metrics'] = {k: metrics[k] for k in
                    ('full_note_f1', 'onset10', 'density_ratio', 'attempted', 'complete')}
            progress[name]['arms'][arm] = item
    snapshot = dict(
        checked_at=datetime.now(timezone.utc).isoformat(),
        status='in_progress' if open_rows else 'all_suggestions_have_terminal_dispositions',
        suggestions=len(decisions), council_suggestions=len(council), research_suggestions=len(research),
        statuses=dict(Counter(r['status'] for r in decisions.values())),
        open=[dict(id=r['id'], title=r['title'], status=r['status']) for r in open_rows],
        candidates=values['candidates'], runtime=states, actual_model_progress=progress,
        cloud_compute_calls=0, council_rounds_used=2, council_round_limit=2,
        note='Read-only aggregation. Prepared source, fit probes, policy adoption and real quality terminals are distinct. Human listening/editing/playtesting unexecuted. No candidate selected before all fixed terminals.')
    target = ROOT / 'reports/local-portfolio-v1/current-status-v1.json'
    target.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2) + '\n')
    state_path = ROOT / 'reports/local-portfolio-v1/state.json'
    old = json.loads(state_path.read_text())
    old.update(status=snapshot['status'], current_status_report=str(target.relative_to(ROOT)),
               live_verified_at=snapshot['checked_at'], current_open_ids=[r['id'] for r in open_rows],
               current_suggestion_counts=dict(council=len(council), research=len(research), total=len(decisions)),
               fixed_terminal_candidates=sum(r['status'] == 'completed_fixed_terminal' for r in values['candidates']),
               fixed_terminal_candidate_total=len(values['candidates']),
               historical_fields_note='Older experiments/candidates entries below are preserved history. current-status-v1.json is the refreshed runtime view, not a quality or method verdict.')
    state_path.write_text(json.dumps(old, ensure_ascii=False, indent=2) + '\n')
    lines = ['# 本地研究实际进度', '', f'核对时间：{snapshot["checked_at"]}', '',
             f'{len(decisions)} 条建议；开放 {len(open_rows)} 条。'
             f'{old["fixed_terminal_candidates"]}/{len(values["candidates"])} 个固定候选终点已完成。', '',
             '此视图读取真实报告，不会自动放行、结案、重训或选择检查点。'
             '不同候选的参数、步数、目标及读出可能不同，不能据此声称等算力效率。', '',
             '| 固定候选 | 实际状态 | 完整事件 F1@50ms | onset F1@10ms | 达到共同本地筛选门 |',
             '|---|---|---:|---:|---|']
    for r in values['candidates']:
        m = r.get('metrics')
        score = (f'{m["full_note_f1"]:.4f}', f'{m["onset10"]:.4f}') if m else ('—', '—')
        lines.append(f'| {r["id"]} | {r["status"]} | {score[0]} | {score[1]} | {"是" if r["eligible"] else "未通过或未完成"} |')
    lines += ['', '## 实际单臂进度', '',
              '配对尚在运行时，已完成的单臂结果也在这里显示；完整配对报告未结算前，'
              '上表不会把它登记为可选择的最终候选。', '',
              '| 实验 / 臂 | 阶段 | 健康 / 正式更新 | 已完成的主评测 |',
              '|---|---|---:|---|']
    for name, value in progress.items():
        for arm, row in value['arms'].items():
            m = row.get('finished_primary_metrics')
            score = f'full {m["full_note_f1"]:.4f} / onset10 {m["onset10"]:.4f}' if m else '尚无质量终点'
            lines.append(f'| {name} / {arm} | {row["status"]} | {row.get("health_updates", 0)} / '
                         f'{row.get("quality_updates", 0)} | {score} |')
    lines += ['', '开放建议：' + '、'.join(r['id'] for r in open_rows), '',
              '第二 seed、确认集、整曲/短上下文配对和研究配对必须等实际终态；'
              '未执行的人评、情绪听评、原生试玩和云训练仍标为未执行。', '']
    (ROOT / 'reports/local-portfolio-v1/CURRENT_STATUS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status=snapshot['status'], fixed_terminals=old['fixed_terminal_candidates'],
                         candidates=len(values['candidates']), open_ids=old['current_open_ids']), ensure_ascii=False))


if __name__ == '__main__':
    main()
