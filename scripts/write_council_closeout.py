"""Latest-state reading view, never an automatic decision/closure engine."""
from collections import Counter
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABEL = dict(adopted='采纳', rejected='明确否决', tested_failed='实测未过',
             tested_supported='有限实测支持', deferred='待完成', partial='未完整', gap='有缺口')


def main():
    events = [json.loads(v) for v in (ROOT / 'reports/council-archive/decisions.jsonl').read_text().splitlines()]
    research = ROOT / 'reports/local-portfolio-v1/research-suggestions.jsonl'
    if research.exists():
        events += [json.loads(v) for v in research.read_text().splitlines()]
    latest = {v['id']: v for v in events}
    count = Counter(v['status'] for v in latest.values())
    manifest = json.loads((ROOT / 'reports/council-archive/manifest.json').read_text())
    pending_reviews = sum(a['outcome'] == 'pending' for a in manifest['attempts'])
    open_rows = [v for v in latest.values() if v['status'] not in
                 ('adopted', 'rejected', 'tested_failed', 'tested_supported')]
    rows = ['# 本地建议逐项结案视图', '',
            '**仍在执行，尚未关池。**' if open_rows else
            ('**已有建议已逐项处置；新 Council 执行中，继续自由进化。**' if pending_reviews else
             '**已有建议已逐项处置；不停止自由进化。**'), '',
            f'{len(latest)} 条建议，{len(events)} 次追加决策；开放 {len(open_rows)} 条。', '',
            f'待回收独立评审 {pending_reviews} 份；未收到的意见不编造建议或结论。', '',
            f'其中 Council 行动建议 {sum(not k.startswith("R-LFM3-") for k in latest)} 条；'
            f'独立研究建议 {sum(k.startswith("R-LFM3-") for k in latest)} 条。历史两轮限制已由Owner撤销。', '',
            ' · '.join(f'{LABEL.get(k, k)} {v}' for k, v in sorted(count.items())), '',
            '原文、旧状态和回收历史均未改写：[全档案](COUNCIL_INDEX.md) · '
            '[全部决策历史](COUNCIL_DECISIONS.md) · [当前开放项](COUNCIL_BACKLOG.md)。', '',
            '采纳协议、预算模板与已经完成训练是不同事实；具体边界写在每项理由中。'
            '未执行的人评、试玩、情绪听评和云训练不得从“采纳”二字推断为完成。'
            '设计否决不冒充未训练模型的实测失败；具体小配方失败不扩大为方法族失败。', '']
    mechanism = ROOT / 'reports/local-portfolio-v1/mechanism-reviews-v2/index.json'
    if mechanism.exists():
        from scripts.validate_mechanism_reviews import load_registry
        registered = load_registry()
        completed = sum(r['analysis_status'] == 'complete' for r in registered)
        rows += [f'Owner 2026-10-29 补正及后续：{completed}/{len(registered)} 个登记终态已完成'
                 '[历史机制复盘](local-portfolio-v1/mechanism-reviews-v2/README.md)及'
                 '[自由进化复盘](local-evolution-v1/mechanism-reviews/index.json)，'
                 '逐项含三层数字、可检验解释、已有/缺失对照和后继动作。'
                 '[后继合同](local-portfolio-v1/successor-contracts-v2.md)与原训练结果分开；'
                 '结案不自动派发新训练。', '']
    evolution = ROOT / 'reports/local-evolution-v1/state.json'
    if evolution.exists():
        rows += ['当前 Owner 已授权[本地自由进化](../design/LOCAL_EVOLUTION.md)：'
                 '以上结案数是历史台账，不能据此停止后继。Council 不限轮，Argon 缺席；'
                 '最多6美元云小实验与最终scaling走[独立门槛](../design/CLOUD_EXPERIMENT_GATES.md)。'
                 '当前实验见[实时入口](local-evolution-v2/state.json)。', '']
    for prefix in sorted({ident.rsplit('-', 1)[0] for ident in latest}):
        rows += [f'## {prefix}', '', '| ID | 建议 | 当前处置 | 原始结果或实施证据 | 最新理由 |',
                 '|---|---|---|---|---|']
        for ident, value in sorted(latest.items()):
            if not ident.startswith(prefix):
                continue
            evidence = value.get('evidence', [])
            links = ' · '.join(f'[{Path(p).name}]({p.removeprefix("reports/") if p.startswith("reports/") else "../" + p})'
                               for p in evidence[:2])
            reason = value['rationale'].replace('|', '\\|').replace('\n', ' ')
            title = value['title'].replace('|', '\\|')
            rows.append(f'| {ident} | {title} | {LABEL.get(value["status"], value["status"])} | {links} | {reason} |')
        rows.append('')
    (ROOT / 'reports/COUNCIL_CLOSEOUT.md').write_text('\n'.join(rows) + '\n')
    print(dict(recommendations=len(latest), decision_events=len(events), open=len(open_rows), statuses=dict(count)))


if __name__ == '__main__':
    main()
