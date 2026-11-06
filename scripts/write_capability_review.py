"""Render the finished checkpoint review from auditable saved measurements."""
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.summarize_capability_review import paired, style


def read(name):
    return json.loads((ROOT / name).read_text())


def num(x, scale=1, digits=3):
    return '未定义' if x is None else f'{x * scale:.{digits}f}'


def point(group, field):
    return group.get(field, {}).get('mean_complete_only')


def change(entry, field):
    d = entry[field]
    ci = d['song_bootstrap_95']
    interval = f"[{num(ci[0], 100, 2)}, {num(ci[1], 100, 2)}]" if ci else '不可估'
    return f"{num(d['delta_b_minus_a'], 100, 2)} pp；95% 区间 {interval}；n={d['independent_songs']}"


def main():
    summary = read('reports/capability_review_summary.json')
    drag = read('reports/capability_drag_review.json')
    budget = read('reports/capability_review_budget.json')
    raw_dir = ROOT / 'reports/capability_review_raw/evaluation-review-v1'
    rows = [json.loads(p.read_text()) for p in sorted((raw_dir / 'records').glob('*.json'))]
    assert len(rows) == 276 and all(c['missing'] == 0 for c in summary['coverage'].values()), 'Main panel remains incomplete'
    labels = {'mania_4k': 'mania 四键', 'phira': 'Phira'}
    kinds = {'from_scratch': '从零', 'middle_holes': '填空', 'reference': '参考'}
    lines = ['# 3500 / 4500 检查点整曲复盘', '',
             '2026-10-24。仅使用已有权重推理，训练更新数为 0。原总预算上限 $10；训练与自动续训仍暂停。', '',
             '## 主要结论', '',
             '4500 没有获得全面晋升的证据。mania 从零生成的音符 F1 从 16.25% 提高到 20.22%，配对增幅 3.97 个百分点，歌曲 bootstrap 95% 区间为 [1.57, 6.89]；其余五个模式/请求组合的音符 F1 变化区间均跨零。多项比较未作校正，这是一项局部改善证据，不能据此宣布整体质量通过。', '',
             'mania 从零生成更积极：音符 precision 18.87% → 20.07%，recall 20.44% → 28.95%，密度比 1.55 → 2.39。precision 点估计也略升，但未单独检验；召回与放音数量的变化更大，难度和可玩性仍需裁决。匹配起音误差中位数的配对均值反而增加约 2.97 ms，同时覆盖率提高，不能把本轮认定为软时间目标已经降低整曲误差。', '',
             'Phira 从零/参考的音频锚定率分别约下降 7.95/8.08 个百分点，相应探索性区间不跨零。音符 F1 没有形成清楚整体改善。锚定率偏向瞬态，无法完整评价连续 drag；但完整面板中生成 drag 平均占比也没有增加，不能简单以“多写 drag”解释这些变化。', '',
             '音频错配使对原曲的匹配明显下降，支持模型确实使用音频；移除新增历史通路的 4 曲诊断尚不能判定模块收益。参考交换结果与限制见下文。两版各完成 117/120 请求；失败来自 64 线和 256 行参考容量。全部失败仍在分母内。', '',
             '## 覆盖与比较口径', '',
             '固定 40 首留出歌曲（mania / Phira 各 20 首），每首从零、填空、参考三种请求。两检查点使用相同请求、种子、修复后的生成代码和贪心解码；每请求推理上限 90 秒。', '',
             '| 检查点 | 已尝试 / 计划 | 完成 | 失败 |', '|---|---:|---:|---:|']
    for ck, c in summary['coverage'].items():
        lines.append(f"| {ck} | {c['attempted']}/{c['expected']} | {c['completed']} | {c['attempted']-c['completed']} |")
    lines += ['', '另有 36 次输入诊断：4 次新增历史通路移除、16 次参考交换尝试、16 次音频错配。参考交换中原面板的 4 首 Phira 没有同音频替代谱，对应 8 次记录为不可用，不当作模型失败或成功。', '',
              '各均值只计完成请求；空输出保留。失败单列，未以补零或删除失败来隐瞒。配对变化只计双方完成的相同歌曲与请求。置信区间按歌曲重采样 5000 次，不能把同歌多个音符当成独立样本；多项指标没有做多重检验校正，属于探索性复盘。', '',
              '本轮没有“原配方继续训练 1000 步”的匹配训练对照，所以不能将检查点变化归因于某个新增模块。', '',
              '## 整曲产物', '',
              '音符 F1 要求时间 ±20 ms 且类型相符，mania 还要求键道相符；Phira 不计空间位置。下面的密度比是生成数 / 原谱目标数，原谱不是唯一合理答案，偏离 1 本身不构成失败。', '',
              '| 模式 / 请求 | 完成数 3500 / 4500 | 音符 F1 3500 → 4500 | 起音 F1 3500 → 4500 | 密度比 3500 → 4500 | 空输出 3500 / 4500 |',
              '|---|---:|---:|---:|---:|---:|']
    for mode in labels:
        for kind in kinds:
            a, b = [summary['groups'][f'{ck}/baseline/{mode}/{kind}'] for ck in ('3500', '4500')]
            pair = lambda f, scale=1: f'{num(point(a,f),scale)} → {num(point(b,f),scale)}'
            lines.append(f"| {labels[mode]} / {kinds[kind]} | {a['completed']} / {b['completed']} | {pair('note_f1_20ms')} | {pair('onset_f1_20ms')} | {pair('density_ratio')} | {a['empty_completed']} / {b['empty_completed']} |")
    lines += ['', '| 模式 / 请求 | 配对音符 F1 变化（4500 − 3500） |', '|---|---|']
    for mode in labels:
        for kind in kinds:
            p = summary['paired'][f'4500_minus_3500/{mode}/{kind}']
            lines.append(f"| {labels[mode]} / {kinds[kind]} | {change(p,'note_f1_20ms')} |")
    empty_targets = [r for r in rows if r['condition']=='baseline' and r.get('complete') and not r.get('truth_events')]
    if empty_targets:
        lines += ['', f"空目标口径问题：{len(empty_targets)} 次完成请求的真值为空，其中 {sum(not r['generated_events'] for r in empty_targets)} 次模型也保持空白。现有原始 F1 对双方为空返回 0；这个数不应解释为质量失败。保留原始评分，同时在 summary 的 empty_target_silent 与 nonempty_target_note_f1_20ms 中单列静默成功和非空目标均值。", '']
    lines += ['', '| 模式 / 请求 | 音符 precision 3500 → 4500 | 音符 recall 3500 → 4500 |', '|---|---:|---:|']
    for mode in labels:
        for kind in kinds:
            a,b = [summary['groups'][f'{ck}/baseline/{mode}/{kind}'] for ck in ('3500','4500')]
            pair = lambda f: f'{num(point(a,f))} → {num(point(b,f))}'
            lines.append(f"| {labels[mode]} / {kinds[kind]} | {pair('note_precision_20ms')} | {pair('note_recall_20ms')} |")
    lines += ['', 'precision 对非空目标而空输出计 0；双方为空未定义，recall 在空目标时未定义。分母数见 summary。结合二者与密度比，判断 F1 变化是否主要来自更积极地放音符。']
    lines += ['', '### 音乐锚定与模式诊断', '',
              '音频锚定使用固定 mel 谱通量峰；它偏向可检测瞬态，不代表所有音乐事件，也会遗漏持续音乐下的合理 drag。平移超额是相对于同一输出在可编辑区内平移后的机会基线。强度相关、模式分布与人谱区间完整保存在 JSON 记分卡中，尚未校准成及格线。', '',
              '| 模式 / 请求 | 锚定率 3500 → 4500 | 同一人谱参照 | 平移超额 3500 → 4500 | 密度-谱通量相关 3500 → 4500 |',
              '|---|---:|---:|---:|---:|']
    for mode in labels:
        for kind in kinds:
            a,b = [summary['groups'][f'{ck}/baseline/{mode}/{kind}'] for ck in ('3500','4500')]
            human = [r['benchmark']['human']['anchor_20ms_precision'] for r in rows
                     if r['checkpoint']=='3500' and r['condition']=='baseline' and r['mode']==mode and r['request_kind']==kind and r.get('complete')]
            human = [v for v in human if v is not None]
            pair = lambda f: f'{num(point(a,f))} → {num(point(b,f))}'
            lines.append(f"| {labels[mode]} / {kinds[kind]} | {pair('audio_anchor_20ms')} | {num(float(np.mean(human)) if human else None)} (n={len(human)}) | {pair('audio_anchor_shift_excess')} | {pair('density_flux_spearman')} |")
    lines += ['', '| mania 请求 | 小于 100 ms 的同键间隔占比 | 键道 bigram 熵 |', '|---|---:|---:|']
    for kind in kinds:
        a,b = [summary['groups'][f'{ck}/baseline/mania_4k/{kind}'] for ck in ('3500','4500')]
        pair = lambda f: f'{num(point(a,f))} → {num(point(b,f))}'
        lines.append(f"| {kinds[kind]} | {pair('rapid_same_lane_fraction')} | {pair('lane_bigram_entropy')} |")
    lines += ['', '100 ms 是诊断阈值，不能直接宣布 jack 不可玩；熵也没有越高越好或越低越好的单向结论。人评、难度和实际指法仍需补充。']
    lines += ['', '### 时间误差不能脱离覆盖率解释', '',
              '以下误差仅在 ±50 ms 内成功一对一匹配的起音上计算。较低误差可能来自只保留容易匹配的音符，因此同时列覆盖率；这也不足以单独验收 σ=1.5 ms 软目标。', '',
              '| 模式 / 请求 | 匹配误差中位数 ms | P90 ms | 匹配覆盖率 |', '|---|---:|---:|---:|']
    for mode in labels:
        for kind in kinds:
            a, b = [summary['groups'][f'{ck}/baseline/{mode}/{kind}'] for ck in ('3500', '4500')]
            pair = lambda f: f'{num(point(a,f))} → {num(point(b,f))}'
            lines.append(f"| {labels[mode]} / {kinds[kind]} | {pair('matched_error_median_ms')} | {pair('matched_error_p90_ms')} | {pair('match_coverage_50ms')} |")
    lines += ['', '## 输入诊断：模型使用了什么', '',
              '### 音频错配', '',
              '对原面板每模式 4 首曲，将各层音频特征、mel 和全曲摘要全部换成另一首曲，按时长重采样，评测仍针对原音频。没有用目标谱选择错配内容。供体只匹配时长，没有匹配节拍；这是一项输入依赖压力测试，不是自然音频变换，也不能证明乐句层面的理解。', '',
              '| 检查点 / 模式 | 完成配对 | 输出改变 | 音符 F1 变化 | 音频锚定率变化 |', '|---|---:|---:|---|---|']
    for ck in ('3500', '4500'):
        for mode in labels:
            p = summary['paired'][f'{ck}/audio_mismatch/{mode}']
            lines.append(f"| {ck} / {labels[mode]} | {p['both_complete']}/{p['pairs']} | {p['output_changed']} | {change(p,'note_f1_20ms')} | {change(p,'audio_anchor_20ms')} |")
    lines += ['', '锚定率明显下降支持模型在使用音频。仍需区分“对局部起音敏感”和“懂得乐句、选音与动作表达”；谱通量峰本身不是完整音乐标注。', '',
              '### 新增近期键道历史移除', '']
    p = summary['paired']['4500/history_removed/mania_4k']
    lines += [f"4500 检查点、mania 从零生成：{p['both_complete']}/{p['pairs']} 配对完成，{p['output_changed']} 首输出改变。只将新增 Y / 内层 GRU 的 K=8 组适配器归零；原有自回归前文与合法性掩码保留。", '',
              f"音符 F1：{change(p,'note_f1_20ms')}。起音 F1：{change(p,'onset_f1_20ms')}。", '',
              '这项结果检验当前权重对通路的依赖，不等价于从训练开始移除通路的因果对照。4 首歌的区间较宽，不足以宣布历史模块有效或有害。', '',
              '### 同音频参考交换', '',
              '保持音频、目标、控制与已观测上下文不变，只交换同音频的另一张人谱参考，并重新编码模型实际读取的 evidence。原有已知上下文可能压过参考影响，因此弱响应不能直接解释成“完全不读参考”。风格改变可能合理降低对原真值的 F1，不能拿该 F1 差作为风格迁移成败。', '']
    panels = [('main_mania', [r for r in rows if r['mode'] == 'mania_4k' and r['request_kind'] == 'reference'])]
    addon_path = ROOT / 'reports/phira_reference_review_raw/results.json'
    if addon_path.exists():
        addon = json.loads(addon_path.read_text())
        panels.append(('additional_phira', addon['records']))
        lines += [f"Phira 补充面板：从留出集 {addon['candidate_audio_groups']} 个同音频多谱组中按固定顺序选择 {len(addon['selected'])} 首，已保存 {len(addon['records'])}/16 次请求。对照与交换使用相同的线索引映射，预先检查 64 线容量。核查后这 4 首均不在主 40 首中。", '',
                  '**发现参考交换的抽样混淆。** 其中 phira:5822/9756、48098/48099、35939/36329 的选中参考音符在时间、类型、时长和位置上相同，变化仅来自重抽的负证据与模型不读的事件身份字段；两检查点在这 3 对上均改变了输出。这是负证据抽样敏感性，不能作为风格迁移成功，也不自动说明使用负证据本身错误。原始 reference_changed 包含元数据变化，不能当成有效风格对比计数。唯一有真正音符内容差异的是 12626/4758。', '']
    else:
        lines += ['Phira 补充参考面板尚无结果。', '']
    positive_path = ROOT/'reports/reference_positive_review_raw/results.json'
    if positive_path.exists():
        positive = json.loads(positive_path.read_text())
        for mode in ('mania_4k','phira'):
            panels.append((f'positive_only_{mode}',[r for r in positive['records'] if r['mode']==mode]))
        lines += [f"为隔离这一混淆，补做 {len(positive['records'])}/20 次正证据对照：保留 4 对真实有差异的 mania 参考，以及唯一有效的 Phira 对；A/B 两侧都删除随机 unselected 证据行，其他条件相同。这是依据输入内容差异追加的探索诊断，不依据模型哪次成功挑样本；它不代表原有完整参考请求的分布。", '']
    reference_result = {}
    lines += ['| 面板 / 检查点 | 双方完成 | 输出改变 | 参考方向一致次数 / 有变化的特征次数 |', '|---|---:|---:|---|']
    for panel, records in panels:
        for ck in ('3500', '4500'):
            a = [r for r in records if r['checkpoint'] == ck and r['condition'] == 'baseline']
            b = [r for r in records if r['checkpoint'] == ck and r['condition'] == 'reference_swap']
            p = paired(a, b)
            responses = []
            for r in b:
                base = next((x for x in a if x['song'] == r['song']), None)
                if not base or not base.get('complete') or not r.get('complete'):
                    continue
                styles = [style([e for ref in r[k] for e in ref['selected']]) for k in ('reference_a', 'reference_b')]
                styles += [style(base['generated_events']), style(r['generated_events'])]
                if not all(s is not None for s in styles):
                    continue
                for key in styles[0]:
                    if any(s[key] is None for s in styles):
                        continue
                    dr, dg = styles[1][key]-styles[0][key], styles[3][key]-styles[2][key]
                    responses.append({'song':r['song'],'feature':key,'reference_delta':dr,'generated_delta':dg,
                                      'direction_agrees':dr*dg>0 if abs(dr)>1e-8 else None})
            terms = []
            for key in ('hold_fraction', 'drag_fraction', 'chord_fraction', 'median_ioi_ms'):
                eligible = [s for s in responses if s['feature'] == key and s['direction_agrees'] is not None]
                terms.append(f"{key} {sum(s['direction_agrees'] for s in eligible)}/{len(eligible)}")
            lines.append(f"| {panel} / {ck} | {p['both_complete']}/{p['pairs']} | {p['output_changed']} | {'；'.join(terms)} |")
            reference_result[f'{panel}/{ck}'] = {'paired':p,'feature_responses':responses}
    (ROOT/'reports/capability_reference_review.json').write_text(json.dumps(reference_result,ensure_ascii=False,indent=2,allow_nan=False))
    lines += ['', '受控正证据对照中，两版均只有 2/4 个 mania 对改变输出，唯一有效的 Phira 对也改变输出。这支持存在参考内容依赖，但方向响应不稳定：4500 的 hold 比例在 4 个 mania 对和 1 个 Phira 对上都没有朝参考变化方向移动，和弦比例也只有少数同向。样本太少，且原已知上下文保持不变，不能宣布整体风格理解或确定退化；当前没有证据支持 4500 的参考控制能力更强。', '',
              '这些特征次数相互相关，不是独立成功率或因果效应量。完整特征位移和配对数据见 capability_reference_review.json；未在小样本上拟合总分权重。', '',
              '## Drag：逐音符分数与连续动作应分开', '',
              '一串 drag 可以共同表达持续音乐或连续操作；每颗未必对应独立起音。它们可以是真实可判音符，不能以“不贴起音”为由删除。原子事件与重数继续保留，以下分项仅用于解释当前 F1，不是新的语义真值。', '',
              '| Phira 请求 | 全部音符 F1 | 非 drag F1 | drag F1 | 生成 drag 占比 |', '|---|---:|---:|---:|---:|']
    for kind in kinds:
        a,b = [drag['groups'][f'{ck}/{kind}'] for ck in ('3500','4500')]
        pair = lambda label: f"{num(a['note_f1_20ms'][label]['mean'])} → {num(b['note_f1_20ms'][label]['mean'])}"
        fraction = f"{num(a['generated_drag_fraction']['mean'])} → {num(b['generated_drag_fraction']['mean'])}"
        lines.append(f"| {kinds[kind]} | {pair('all')} | {pair('non_drag')} | {pair('drag')} | {fraction} |")
    lines += ['', 'drag/非 drag 双方都为空时该分项未定义，不用 0 或 1 代替；各分项样本数与配对区间见 capability_drag_review.json。', '',
              '另按人谱 drag 的相邻间隔 80 / 160 / 320 ms 分别构造至少含三个不同起音时刻的密集段，比较段内外 F1。这些是阈值敏感性诊断，不能当作连续手势标签：同刻叠键不自动算连续动作，时间接近也不能证明共用同一操作路径。空间、触摸状态与人评仍缺失。', '',
              '### 与操作空间集成提案的关系', '',
              'STATIC/BIND、请求级引用池、Hold 时长先于引用检查，以及显式未知反馈，能补上目前缺失的空间表达。它们仍以单个音符为输出单位，不自动建立连续 drag 与乐句的对应关系。接入前建议补四项：', '',
              '1. 为连续接触片段建立有证据的评测案例：段起止、区域变化、持续性与音乐段落关系。保留原始音符数量，避免把假定的手势等价当成原谱完全等价。',
              '2. context 阶段分别比较旧输入、仅移除旧 line/x、替换为合法操作空间，拆开分布迁移和新增信息的作用。',
              '3. “不显著退化”改为预先约定可接受退化幅度并检查歌曲级区间；检不出差异不等于等价。',
              '4. 联合检查时间与空间：1 ms 重求标签不能自动消除 σ=1.5 ms 软目标跨越动画跳变的问题。对跳变两侧做时间扰动和物化检查，决定是否需要局部调整时间目标。', '',
              '现有 65.65% 动态 BIND 覆盖来自特定授权条件；动态样本约 99% 来自 Solace，是表示与权限覆盖，不是模型准确率或曲库总体上限。本次未修改该设计或接入模型。', '',
              '## 清洗、失败与表示限制', '']
    failures = [r for r in rows if r['condition']=='baseline' and not r['complete']]
    for r in failures:
        lines.append(f"- {r['checkpoint']} / {r['chart_id']} / {r['request_kind']}：{r.get('failure')}。")
    lines += ['', '已下载并核查 phira:12870：691 个真实事件，16 个显式 fake 单独存储；真实事件使用 140 条判定线，events 中 fake 标记为真者为 0。原始编号 282 的线有 2 个真实音符，编号不是线总数。模型只按真实事件构建线映射，纯装饰线不占槽；64 线容量使部分已知/参考输入无法编码。', '',
              '另一张 phira:56053 已在本地用完全相同请求复现：两段参考分别有 53+53、88+88 行正/负证据，合计 282 行，超过 MAX_REF=256。它是请求构造与模型容量没有对齐，不能解释为脏谱或模型没学会。审计同时核对了当前 Corpus 默认启用的 codec 常量及 −2 ms 时间平移，避免用不同请求误复现。', '',
              '不能为通过评测删除这些真实音符。原生导入会分离显式 isFake；目前未独立检测未标 fake 的表演音符。canon 读取处还会移除 fake 字段而未再次校验，应另补边界检查，但现有样本没有发现 fake 混入。', '',
              '这次保留相同失败口径，不在两个检查点中途更换表示。Phira 没有输出真实摆位，时间/类型合法不能视为完整可玩性通过。', '',
              '另一个旧表示限制：本次沿用训练路径的全谱 line 索引，已观测音符的编号映射可能受隐藏侧使用的线影响；补充参考面板使用固定的双谱 union 映射。没有把隐藏未来音符序列送入生成，但这种映射仍不满足新提案“池成员和排序只由授权条件决定”的严格隔离要求。本次只能比较两检查点在同一旧表示下的行为；空间模式需要重建输入并重新冻结 benchmark。', '',
              '## 下一轮的裁决原则', '',
              '继续暂停训练；4500 不因 loss 降低或某个格子的 F1 提升而自动晋升。保留数值正确性修复、数据与费用审计、原守卫和生成评测。新增历史、软目标与反事实目标分别保留实现或方案，是否纳入训练由匹配实验裁决。', '',
              '下一步先收敛 Phira 的输出与动作评测定义，再从同一检查点做单变量、相同优化器与采样日程的对照。不要再次同时更改请求配比、控制 dropout、历史、软目标和前文回放后试图归因。每 1000 步是固定复盘节点，不要求每次都增加模块。', '',
              '控制量响应、完整可玩性、动作语义和人评没有被本次三个输入诊断替代。人谱区间与音频锚定只是参照，当前没有经过校准的统一“及格总分”。', '',
              '## 费用与复现', '']
    upper = budget['prior_cost_upper_usd']+budget['other_allowance_usd']+sum(s['reserved_usd'] for s in budget['stages'].values())
    lines += [f"历史已确认费用 ${budget['prior_confirmed_usd']:.4f}；含检查额度、失败批次全额预留与本次评测的保守承诺上界 ${upper:.4f}，低于原 $10 上限。预留不是实付，最终状态与计费核查另见 capability_review_closeout.json。", '',
              '首批因元数据 tuple/list 比较失败，第二批因本地连接丢失取消，第三批因落盘 protocol 的 tuple/list 比较失败。均保留费用。恢复只修正封装层的规范化比较；已核对源码差异，原输出没有重算或替换。protocol.original.json、protocol.json、runner_resume_fix.json 保留前后指纹与原记录 ID。', '',
              '数据文件：', '',
              '- `capability_review_raw/evaluation-review-v1/`：检查点哈希、冻结协议、逐请求原始输出、失败与原始指标。',
              '- `capability_review_summary.json`：完整分母、分组分数、歌曲级配对区间与参考特征响应。',
              '- `capability_drag_review.json`：类型分项和密集 drag 段敏感性。',
              '- `capability_reference_review.json`：参考 A/B 的方向、幅度与输出变化。',
              '- `phira_12870_cleaning_audit.json`：显式 fake 和实际使用判定线审计。',
              '- `capability_review_budget.json`：全部派发、失败及保守费用预留。', '',
              '本地验证包括输入干预的真值/控制/权限不变与历史恢复测试、monitor 分母与暂停状态测试，以及 drag 分组边界检查；无优化器更新。']
    path = ROOT/'reports/capability_review_2026-10-24.md'
    path.write_text('\n'.join(lines)+'\n')
    print(path)


if __name__=='__main__':
    main()
