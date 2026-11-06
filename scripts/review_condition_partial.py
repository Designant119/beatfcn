"""Review committed outputs and reject failed directions without any new inference."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from beatfcn.events.canonical import Event
from beatfcn.model.condition_probe import condition_inventory, screen_control
from beatfcn.model.evaluation import score_events

LABELS = ('3500', 'r02-density-supervised', 'r02-density-contrastive')
KINDS = ('from_scratch', 'middle_holes', 'reference')


def main():
    directory = ROOT / 'reports/condition_v1_benchmark'
    files = sorted((directory / 'records').glob('*.json'))
    records = [json.loads(p.read_text()) for p in files]
    summary = json.loads((ROOT / 'reports/condition_v1_benchmark_summary.json').read_text())
    result = {'scope': 'partial development panel, user stopped; no new GPU or training',
              'received': len(records), 'expected': summary['expected'],
              'not_recorded': summary['not_recorded'], 'failed_recorded': summary['failed_recorded'],
              'protocol_identity': summary['protocol_identity'],
              'record_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
              'cells': summary['cells'], 'paired_changes': summary['paired_changes'],
              'type_breakdown': {}, 'control_screens': {},
              'all_condition_status': condition_inventory()['numeric_fields'],
              'decision': 'Do not promote either candidate; cancel the remaining old-protocol evaluation.'}
    for label in LABELS:
        for mode in ('mania_4k', 'phira'):
            for kind in KINDS:
                rows = [r for r in records if (r['checkpoint'], r['mode'], r['request_kind']) == (label, mode, kind)
                        and r['complete'] and not r['legality']]
                types = {}
                for typ in ('tap', 'hold', 'flick', 'drag'):
                    counts, true, pred = Counter(), 0, 0
                    for r in rows:
                        p = [Event.from_json(e) for e in r['generated_events'] if e['type'] == typ]
                        t = [Event.from_json(e) for e in r['truth_events'] if e['type'] == typ]
                        score = score_events(p, t, mode)
                        pred += len(p)
                        true += len(t)
                        counts['head_matches'] += score['note_20ms']['tp']
                        counts['full_matches'] += score['hold_joint_20ms']['tp'] if typ == 'hold' else score['note_20ms']['tp']
                    types[typ] = dict(counts, pred=pred, true=true,
                                      head_recall=counts['head_matches'] / true if true else None,
                                      full_recall=counts['full_matches'] / true if true else None,
                                      full_f1=2 * counts['full_matches'] / (pred + true) if pred + true else None)
                result['type_breakdown']['/'.join((label, mode, kind))] = {'songs': len(rows), 'types': types}
            for scope, prefix in (('window', 'density_'), ('whole_song', 'density_song_')):
                kinds = {prefix + x for x in ('low', 'mid', 'high')}
                rows = [r for r in records if r['checkpoint'] == label and r['mode'] == mode and r['request_kind'] in kinds]
                songs = sorted({r['song'] for r in rows})
                probes = [{**r, 'feasible': r.get('minimum_from_observed', float('inf')) < r['requested'][1]}
                          for r in rows if 'requested' in r]
                result['control_screens']['/'.join((label, mode, scope))] = screen_control(
                    'density_notes_per_second', mode, probes, songs)
    for field, entry in result['all_condition_status'].items():
        entry['capability_status'] = 'failed_current_density_trial' if field == 'density_notes_per_second' else 'not_controlled_in_this_panel'
        entry['next_gate'] = 'small fixed panel; zero directional response stops expansion; positive response still requires quality checks'
    path = ROOT / 'reports/condition_partial_review_2026-10-25.json'
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    lines = ['# 部分评测复盘：不晋升、不补跑旧协议', '',
             f"已取回 {len(records)}/{summary['expected']} 条记录，已记录失败 {summary['failed_recorded']}；"
             f"其余 {summary['not_recorded']} 条未形成记录，不能算模型失败。覆盖 5 首 mania、4 首 Phira。", '',
             '两组新模型均未达到控制与完整音符要求。取消剩余旧协议请求，不为补齐表格继续花费。'
             '这是对当前方案的工程否决，不是对所有歌曲或所有反事实方法的统计结论。', '',
             '## 质量变化', '',
             '每首歌曲等权；时间点与完整音符的编辑节省是相对人谱的插入/删除代理，不是人工工时。', '',
             '| 模型 / 模式 / 请求 | 歌曲 | note F1 | 数量比 | 时间点编辑节省 | 完整音符编辑节省 |',
             '|---|---:|---:|---:|---:|---:|']
    for key, cell in summary['cells'].items():
        lines.append(f"| {key} | {cell['complete_legal']} | {cell['note_f1_20ms']['mean']:.3f} | "
                     f"{cell['count_ratio']['mean']:.3f} | {cell['20ms']['timing_point']['mean']:.3f} | {cell['20ms']['full_note']['mean']:.3f} |")
    lines += ['', '## 被平均值掩盖的类型问题', '',
              '下表汇总已完成的从零生成歌曲。计数是这些歌曲的总数，不能当作控制响应，也不要求逐个复制人谱；'
              '几乎不输出整类音符足以说明能力不完整。分类型匹配与召回在同名 JSON。', '',
              '| 模式 / 模型 | tap 生成/人谱 | hold 生成/人谱 | flick 生成/人谱 | drag 生成/人谱 |',
              '|---|---:|---:|---:|---:|']
    for mode in ('mania_4k', 'phira'):
        for label in LABELS:
            types = result['type_breakdown'][f'{label}/{mode}/from_scratch']['types']
            lines.append('| ' + mode + ' / ' + label + ' | ' + ' | '.join(f"{v['pred']}/{v['true']}" for v in types.values()) + ' |')
    lines += ['', '## 控制与提前止损', '',
              '精确窗口中，两组新模型在 5 首 mania 和 4 首 Phira 上均没有低→中→高的严格单调响应；'
              '已经触发小面板止损，不扩大旧方案测试。整曲小面板的响应同样不足，详见 JSON。', '',
              '全部 8 项数值条件都属于验收范围：密度、峰值密度、长按头比例、长按占用、和弦比例、'
              'mania 同键连打比例、Phira flick 比例、Phira drag 比例。其他 7 项尚未经过独立干预，不能由当前数据宣布有效或无效。', '',
              '每项先用固定小面板、三个预先确定的档位，只改该条件；锁定音符和类型白名单导致的不可行请求单列。'
              '至少 4 首完整可行歌曲仍无严格正向响应，就停止扩测。响应存在也只进入质量复查，不自动晋升。', '',
              '## 已确认的推理错误与修复边界', '',
              '生成时起音指针每次只查询一个 Y 行，旧代码因此总取 prev_onsets 的第 0 行；'
              '训练则逐行使用正确历史。已改成按实际生成行索引读取。'
              '使用非零间隔核权重的真实生成回归测试在修复前失败（200/217 个 logits 不同），修复后通过。'
              '已有测试的间隔核初始化为零，因此此前未覆盖这个故障。', '',
              '这能证明训练/生成不一致已修复，不能证明它解释全部平台期，也不能宣布检查点质量已恢复。'
              '234 条旧结果保留原始代码身份；后续修复后的输出必须使用新协议身份，不能混并为同一次 benchmark。', '',
              '## 下一轮取舍', '',
              '- 保留：数据权限修复、纯节奏输出、同数据对照、原守卫、每 1000 次有效更新复盘。',
              '- 暂停晋升两组 4500；当前 density-only 的 time+cont 排序损失不作为已成功干预继续叠加。',
              '- 先验证历史修复对旧检查点的影响，之后才判断剩余问题属于训练目标、数据分布或解码；当前不启动云端验证。',
              '- 下一训练准备覆盖所有条件的真实窗口标签与样本分布。密度/峰值对应时间与数量决策，hold/flick/drag 对应类型与尾点，和弦对应同组数量，jack 对应键道转移；不能把 time+cont 排序损失直接当成全部条件的学习目标。',
              '- 先检查并平衡完整窗口中的稀少类型、低密度及各条件档位，保留人谱音符和 drag 序列，不靠删谱或硬凑数量实现响应。',
              '- 增加分类型召回/完整音符 F1、过量/缺失分布、条件响应作为独立退化检查。4000→4500 两组固定验证 loss 都反弹；继续降总训练 loss 不构成续训理由。',
              '- 其他 7 项、参考/锚点能力仍须独立证明；不得宣称代码接了输入就代表可控制。', '',
              '## 评测优化与预算', '',
              '已将每个生成步骤重建全部训练 metadata 改为复用固定输入、追加新组和历史行；'
              '对比检查两种模式、桶边界、长按/和弦、K8、随机与贪心生成及空组重试。'
              '本地计时见 evaluation_metadata_timing_2026-10-25.json；只证明本地测试条件下的开销变化，不代表 L40S 加速。', '',
              '评测 app 已停止、容器数为 0。此次分析与修改没有新启动 GPU 或训练。预算仍为累计 $15；'
              '账本保留旧任务预留，未将预留当作实际支出，也未擅自释放未核对费用。', '']
    report = path.with_suffix('.md')
    report.write_text('\n'.join(lines))
    print(json.dumps({'review': str(report), 'records': len(records), 'decision': result['decision']}))


if __name__ == '__main__':
    main()
