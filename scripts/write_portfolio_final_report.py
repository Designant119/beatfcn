"""Render verified final receipts. Refuses an open or incomplete campaign."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

from scripts.portfolio_terminal_catalog import catalog


ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads((ROOT / path).read_text())


def digest(path):
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def latest(path):
    events = list(map(json.loads, (ROOT / path).read_text().splitlines()))
    return events, {r['id']: r for r in events}


def type_counts(metric):
    """Pool fixed draws equally; never select the most successful draw."""
    rows = metric.get('samples', [metric])
    result = {}
    for kind in ('tap', 'hold', 'flick', 'drag'):
        if not all(kind in r.get('per_type', {}) for r in rows):
            continue
        values = {k: sum(r['per_type'][kind][k] for r in rows) for k in ('tp', 'pred', 'true')}
        values['f1'] = 2 * values['tp'] / max(1, values['pred'] + values['true'])
        result[kind] = values
    return dict(draws=len(rows), counts=result,
                interpretation='Pooled event-level TP/pred/true over every fixed draw, not song-macro F1. Repeated draw targets are intentionally repeated.')


def main():
    ce, council = latest('reports/council-archive/decisions.jsonl')
    re, research = latest('reports/local-portfolio-v1/research-suggestions.jsonl')
    closed = {'adopted', 'rejected', 'tested_failed', 'tested_supported'}
    pending = [r['id'] for r in {**council, **research}.values() if r['status'] not in closed]
    if pending:
        raise ValueError('Never call an open pool complete: ' + ','.join(pending))
    if len(council) != 116 or len(research) != 3:
        raise ValueError('The finite suggestion-pool identity changed')
    protocols = ['reports/local-portfolio-v1/remaining-pool-v4.json',
                 'reports/local-portfolio-v1/research-then-finalists-v1.json',
                 'reports/local-portfolio-v1/frozen-finalist-protocol-v2.json']
    if any(read(p)['status'] != 'completed' for p in protocols):
        raise ValueError('Actual experiment/replication pipelines are incomplete')
    values = catalog()
    if len(values['candidates']) != 23 or any(c['status'] != 'completed_fixed_terminal' for c in values['candidates']):
        raise ValueError('Missing actual fixed candidate terminal')
    selection = read('reports/local_portfolio_finalist_selection_v1.json')
    confirmation = read('reports/local_portfolio_confirmation_v1.json')
    if confirmation['status'] not in ('completed', 'not_opened_no_reproduced_finalist'):
        raise ValueError('No confirmation terminal')
    seeds = [read(f'reports/local_finalist_r{rank}_seed2_v1.json')
             for rank in range(1, len(selection['finalist_identities']) + 1)]
    diagnostics = ['reports/local_direct_song_routing_review_v1.json',
                   'reports/local_mert_context_pair_review_v1.json',
                   'reports/local_raw_audio_translation_v1.json',
                   'reports/local_music_depth_pair_v1.json',
                   'reports/local_music_ssl_pair_v1.json',
                   'reports/local_process_frame_sources_v1.json']
    if any(read(p)['status'] != 'completed' for p in diagnostics):
        raise ValueError('Required paired review or input diagnostic is missing')
    old_budget_digest = '7d3d79af750443fea25aa0b1824107d6b7300e289259fa70178c405030b127a6'
    if digest('reports/rhythm_v2_budget.json') != old_budget_digest:
        raise ValueError('The preserved legacy cloud ledger changed; inspect explicitly')
    for r in [*council.values(), *research.values()]:
        for path in r.get('evidence', []):
            if not (ROOT / path).exists():
                raise FileNotFoundError(path)
    support = [r['candidate'] for r in confirmation.get('finalists', []) if r['both_seeds_support']]
    # Keep the frozen development ranking; never rank by confirmation scores.
    preferred = next((r['id'] for r in selection['finalist_identities'] if r['id'] in support), None)
    product_diagnostics = {}
    for item in confirmation.get('finalists', []):
        seed_rows = list(item['seeds'].values())
        per_seed = {}
        for name, outcome in item['seeds'].items():
            folder = ROOT / f'artifacts/local-portfolio-confirmation-v1/rank{item["rank"]}/{name}'
            rates = {}
            for task in ('density_low', 'density_high'):
                records = json.loads((folder / f'{task}.json').read_text())
                rates[task] = [len(r['prediction']['events']) / ((r['request']['end_us'] - r['request']['start_us']) / 1e6)
                               for r in records]
            low, high = rates['density_low'], rates['density_high']
            per_seed[name] = dict(reference_full_gain=outcome['completion']['full_note_f1'] - outcome['reference_masked']['full_note_f1'],
                seen_empty_false_events=sum(outcome['seen_empty_diagnostic']['predicted_types'].values()),
                density_notes_per_second=rates, density_direction_positive=sum(h > l for l, h in zip(low, high)),
                density_both_bands=sum(2 <= l <= 3 and 6 <= h <= 8 for l, h in zip(low, high)))
        product_diagnostics[item['candidate']] = dict(
            pooled_types=type_counts(dict(samples=[s['draft'] for s in seed_rows])),
            per_seed=per_seed,
            mean_reference_full_gain=sum(s['reference_full_gain'] for s in per_seed.values()) / len(per_seed),
            interpretation='Confirmation type counts and reference/condition diagnostics are separate from the investment gate. Empty windows are the previously seen development diagnostics, not fresh held-out empty examples.')
    result = dict(id='local-portfolio-final-review-v1', status='completed',
        written_at=datetime.now(timezone.utc).isoformat(),
        council_suggestions=len(council), research_suggestions=len(research),
        decision_events=len(ce) + len(re), open_suggestions=[],
        dispositions=dict(Counter(r['status'] for r in {**council, **research}.values())),
        candidates=values['candidates'], type_counts={c['id']: type_counts(c['metrics']) for c in values['candidates']},
        selection=selection, second_seeds=seeds, confirmation=confirmation,
        product_diagnostics=product_diagnostics,
        paired_results={p: read(p) for p in diagnostics},
        bounded_h100_research_signal=bool(support), supported_candidates=support,
        preferred_by_frozen_development_rank=preferred,
        cloud_training_authorized_now=False, new_cloud_compute_calls=0,
        legacy_budget_sha256=old_budget_digest,
        council_rounds_used=2, council_round_limit=2,
        human_listening=False, human_editing_time_measured=False, native_phira_playtest=False,
        limitations=['Confirmation components were held out only from this local campaign, not all project history.',
                    'Development data was repeatedly inspected; selection policy was frozen late and is not whole-campaign preregistration.',
                    'Different recipes have different objectives, effective batches, warmstarts and schedules; ranking is not equal-compute efficiency.',
                    'Native output fields and chart-reference matching do not establish subjective quality or a unique correct chart.',
                    'Neither motif/emotion understanding nor playable spatial design was demonstrated.'],
        receipts={p: digest(p) for p in [*protocols, *diagnostics,
            'reports/council-archive/decisions.jsonl', 'reports/local-portfolio-v1/research-suggestions.jsonl',
            'reports/local_portfolio_finalist_selection_v1.json', 'reports/local_portfolio_confirmation_v1.json']})
    report = ROOT / 'reports/local_portfolio_final_review_v1.json'
    if report.exists():
        raise FileExistsError('Preserve the first complete final review')
    report.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    lines = ['# BeatFCN 本地建议池结案', '',
        f'119 条建议已逐项结案；{len(ce) + len(re)} 次追加决策，开放 0 条。'
        '23 个固定候选终点、最多两个入围者第二 seed 和条件确认协议均已有实际终态。', '',
        '结论：' + (f'以下候选支持一次有界约 0.25B/H100 工程研究：{", ".join(support)}。优先级沿用冻结开发集排名，默认 {preferred}。'
                   if support else '本批没有候选同时复现并通过确认门；当前证据不支持直接选择一个方案投入约 0.25B/H100 试训。'), '',
        '**本次云调用 0；用户暂停上云的指令继续生效。**'
        '本地信号不是产品通过，也不是大模型训练收益保证。', '',
        '## 固定开发终点', '',
        '完整事件 F1 使用 50ms 容差，onset F1 使用 10ms；主分数沿用固定读出，随机配方等权平均所有冻结样本。'
        '各行预算和目标不同；参数数目不含另行冻结的音乐编码器或提议模型成本。', '',
        '| 候选 | 模型参数 M | 更新 | 完整 F1@50ms | onset F1@10ms | 数量比 | 共同筛选门 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for c in values['candidates']:
        m = c['metrics']
        lines.append(f'| [{c["id"]}]({Path(c["report"]).name}) | {c["parameters"]/1e6:.3f} | {c["steps"]} | '
                     f'{m["full_note_f1"]:.4f} | {m["onset10"]:.4f} | {m["density_ratio"]:.3f} | {"过" if c["eligible"] else "未过"} |')
    context = result['paired_results']['reports/local_mert_context_pair_review_v1.json']['comparison']
    routing = result['paired_results']['reports/local_direct_song_routing_review_v1.json']
    depth = result['paired_results']['reports/local_music_depth_pair_v1.json']['comparison']
    ssl = result['paired_results']['reports/local_music_ssl_pair_v1.json']['comparison']
    lines += ['', '## 不只比较横向扩容：四个配对假说', '',
        '| 配对假说 | 完整 F1 增益 | onset10 增益 | 冻结联合假说门 |',
        '|---|---:|---:|---|',
        f'| 同编码器整曲 vs 32 秒提取 | {context["gains"]["full_note_f1"]:.4f} | {context["gains"]["onset10"]:.4f} | {context["supported"]} |',
        f'| 整曲物理读出 vs 原串接路由 | {routing["gains"]["full_note_f1"]:.4f} | {routing["gains"]["onset10"]:.4f} | {routing["hypothesis_supported"]} |',
        f'| 近似等参数深窄 vs 浅宽 | {depth["full_gain"]:.4f} | {depth["onset_gain"]:.4f} | {depth["hypothesis_supported"]} |',
        f'| 同模型上下文匹配目标 vs 纯监督 | {ssl["full_gain"]:.4f} | {ssl["onset_gain"]:.4f} | {ssl["hypothesis_supported"]} |', '',
        f'深度配对另报固定 600 秒包络内完整 F1 增益 {depth["within_600s_gain"]:.4f}；'
        f'自监督配对的有效粗上下文收益增益 {ssl["useful_context_gain"]:.4f}。'
        '联合假说门还包含组件胜出、声学对照或结构条件；单一正增益不改判整项成功。'
        '这些是有界小模型假说，不能推出全局最优、情绪理解或大模型收益。', '']
    lines += ['', '## 复验与确认', '',
              '| 入围者 | 第二 seed 状态 | 开发完整 F1 / onset10 | 复现门 |',
              '|---|---|---|---|']
    for s in seeds:
        m = s.get('development', {})
        scores = f'{m["full_note_f1"]:.4f} / {m["onset10"]:.4f}' if m else '没有质量终点'
        lines.append(f'| {s["candidate"]} | {s["status"]} | {scores} | {s.get("reproduced", False)} |')
    lines += ['', f'确认协议：{confirmation["status"]}；实际打开评分：{confirmation["confirmation_scored"]}。', '']
    for item in confirmation.get('finalists', []):
        lines += [f'### {item["candidate"]}', '', '| 权重 | 完整 F1 | onset10 | 数量比 | 支持门 |',
                  '|---|---:|---:|---:|---|']
        for name, s in item['seeds'].items():
            m = s['draft']
            lines.append(f'| {name} | {m["full_note_f1"]:.4f} | {m["onset10"]:.4f} | {m["density_ratio"]:.3f} | {s["support"]} |')
        lines += ['', f'两 seed 同时通过：{item["both_seeds_support"]}。', '']
    lines += ['## 仍存在的实际能力差距', '',
              '两条路线的确认收益主要来自 tap 与起音定位。下面是两个 seed 合并的事件级类型 F1，'
              '不是上表的歌曲宏平均；重复 seed 的目标计数也重复入账。', '',
              '| 路线 | tap | hold（头尾联合） | flick | drag | 参考收益 Δ完整F1 |',
              '|---|---:|---:|---:|---:|---:|']
    for candidate, d in product_diagnostics.items():
        t = d['pooled_types']['counts']
        lines.append(f'| {candidate} | {t["tap"]["f1"]:.4f} | {t["hold"]["f1"]:.4f} | '
                     f'{t["flick"]["f1"]:.4f} | {t["drag"]["f1"]:.4f} | {d["mean_reference_full_gain"]:+.4f} |')
    lines += ['', '参考收益比较同一权重、同一补全请求的真实参考与遮罩参考；没有稳定正收益。'
              '控制密度的方向与进入指定区间也分开：', '',
              '| 路线 / seed | 既见两个空窗的误发数 | 高档 > 低档 | 两档同时入区间 |',
              '|---|---:|---:|---:|']
    for candidate, d in product_diagnostics.items():
        for name, row in d['per_seed'].items():
            lines.append(f'| {candidate} / {name} | {row["seen_empty_false_events"]} | '
                         f'{row["density_direction_positive"]}/4 | {row["density_both_bands"]}/4 |')
    lines += ['', '这些是下一阶段要解决的具体失败；没有因本地投资门通过而改写为产品成功。', '',
              '## 下一阶段投入建议', '',
              '默认保留复验通过的 Mamba-2/SSD 配方；从零 mel 的混合算子路线作为独立备选。'
              '已给出约 0.25B 的静态参数方案：[246,280,967 参数 SSD 试训提案](../bench/rhythm-v2/local-portfolio-250m-ssd-proposal-v1.json)。'
              '这只做了 meta 参数核算，未做 CUDA 前反向、运行速度或显存验证；未上云。'
              '按原七个双向 SSD 块扩大宽度是一次可解释的规模控制，不代表已找到最优容量分配。'
              '未来新确认组件、运行环境与全生命周期费用仍需在用户恢复云工作后落实。', '']
    lines += ['## 范围与可回看资料', '',
        '所有原文、旧决策、暂缓和回收历史均保留；设计否决与实测失败分开，完整组合失败不归因为每一个模块失效。'
        '采纳的协议/预算文档也与实际训练分开。', '',
        '- [119 条逐项结案](COUNCIL_CLOSEOUT.md)',
        '- [Council 原文及全历史](COUNCIL_INDEX.md)',
        '- [结构化完整结果](local_portfolio_final_review_v1.json)',
        '- [同编码器上下文配对](local_mert_context_pair_review_v1.json)',
        '- [同权重整曲路由配对](local_direct_song_routing_review_v1.json)',
        '- [等参数深度分配](local_music_depth_pair_v1.json)',
        '- [固定模型自监督目标配对](local_music_ssl_pair_v1.json)',
        '- [真实波形平移和重新提取](local_raw_audio_translation_v1.json)', '',
        '未执行真人听评、编辑计时或 Phira 原生试玩；未证明情绪理解和空间可玩性。'
        '人工后续协议已经交付，但不能算成人体实验已经发生。'
        '确认集仅对本轮封存；本轮反复使用的开发集不再宣称盲测。', '']
    (ROOT / 'reports/LOCAL_PORTFOLIO_FINAL.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='completed', open=0, support=support, preferred=preferred), ensure_ascii=False))


if __name__ == '__main__':
    main()
