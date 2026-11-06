"""Recompute D08 sensitivity and pointer margins from saved decision distributions."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
from statistics import median

from beatfcn.model.information_diagnostics import response

ROOT=Path(__file__).resolve().parents[1]


def analyze(payload):
    if not payload['complete'] or payload['optimizer_updates'] or payload['checkpoint_saved']:
        raise ValueError('requires a complete frozen-weight diagnostic')
    rows=payload['records'];base={(r['case_id'],r['prefix_groups']):r for r in rows if r['variant']=='base'}
    if len(rows)!=256 or len(base)!=16 or len(payload['parity'])!=16 or len(payload['replays'])!=8:
        raise ValueError('unexpected evidence count')
    pairs=defaultdict(dict)
    for row in rows:
        if row['variant']=='control':
            key=(row['mode'],row['field'],row['case_id'],row['prefix_groups'])
            if row['level'] in pairs[key]: raise ValueError('duplicate control arm')
            pairs[key][row['level']]=row
    grouped=defaultdict(list);contrasts=[];threshold=payload['protocol']['response_tv_threshold']
    for (mode,field,case,prefix),pair in pairs.items():
        if set(pair)!={0,2}: raise ValueError('missing low/high request')
        delta=response(pair[0],pair[2]);contrasts.append(dict(mode=mode,field=field,case_id=case,prefix_groups=prefix,heads=delta))
        for head,value in delta.items():grouped[(mode,field,prefix,head)].append(value)
    def aggregate(values):
        return dict(n=len(values),median_tv=median(x['total_variation'] for x in values),
            maximum_tv=max(x['total_variation'] for x in values),
            above_threshold=sum(x['total_variation']>=threshold for x in values),
            argmax_changes=sum(x['argmax_changed'] for x in values))
    controls=[dict(mode=k[0],field=k[1],prefix_groups=k[2],head=k[3],**aggregate(v)) for k,v in grouped.items()]
    if len(pairs)!=104 or any(x['n']!=4 for x in controls): raise ValueError('condition panel coverage differs')
    grouped=defaultdict(list)
    for row in rows:
        if row['variant'].endswith('_mismatch'):
            delta=response(base[(row['case_id'],row['prefix_groups'])],row)
            for head,value in delta.items():grouped[(row['mode'],row['variant'],row['prefix_groups'],head)].append(value)
    audio=[dict(mode=k[0],variant=k[1],prefix_groups=k[2],head=k[3],**aggregate(v)) for k,v in grouped.items()]
    margins=[]
    for row in base.values():
        selected,target=row['time_argmax'],row['true_cell_index']
        margin={key:values[target-1]-values[selected-1] for key,values in row['pointer_components'].items()} if selected else None
        margins.append(dict(case_id=row['case_id'],prefix_groups=row['prefix_groups'],
            selected_cell=selected,true_cell=target,true_cell_rank=row['true_cell_rank'],
            end_probability=row['time_probability'][0],true_minus_selected_components=margin,
            type_argmax=row['type_argmax'],true_first_type=row['true_first_type']))
    return dict(complete=True,diagnostic_only=True,product_promotion=False,records=len(rows),
        optimizer_updates=0,function_wall_s=payload['function_wall_s'],
        parity=dict(count=16,passed=all(x['passed'] for x in payload['parity']),
            hidden_max_absolute=max(x['hidden_max_absolute'] for x in payload['parity']),
            probability_max_absolute=max(d['maximum_absolute_difference'] for x in payload['parity']
                for d in x['probability_response'].values())),
        free_replays_equal=all(x['equal'] for x in payload['replays']),
        control_threshold_tv=threshold,control_contrasts=contrasts,controls=controls,audio=audio,
        base_margins=margins,limitations=[
            'Only one existing non-promoted density-control checkpoint and eight held-out windows.',
            'Type probes use the true next onset; continuation/lane use the true first note/type.',
            'TV measures sensitivity, not direction, correct attribute value, or musical understanding.',
            'Hold duration and reference conditioning were not probed; occupancy conclusions cannot rely only on these heads.',
            'Audio mismatch resamples donor feature lengths and is not a natural time-stretched recording.',
            'FP32 parity supports path equivalence; production bf16 replay is checked separately.'])


def main():
    path=ROOT/'reports/information-path-diagnostic-v1.json';payload=json.loads(path.read_text());r=analyze(payload)
    r['raw_sha256']=hashlib.sha256(path.read_bytes()).hexdigest()
    out=ROOT/'reports/d08_information_path_review.json'
    out.write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    lines=['# D08：条件与音频通路诊断','',
        '现有权重只读诊断：256 条决策分布，16 组训练/缓存生成路径对照，8 次原 bf16 自由生成重放，8 份实际张量体检。零优化、未保存权重、没有产品晋升。',
        '',f"16/16 路径对照通过；hidden 最大绝对差 {r['parity']['hidden_max_absolute']:.9f}，概率最大差 {r['parity']['probability_max_absolute']:.9f}；8/8 自由生成事件一致。函数 {r['function_wall_s']:.3f}s。",'',
        '## 条件低→高的概率响应','',
        '表中为四窗 TV 中位数（无历史 / 四个真实先前组），.01 是预注册敏感性线，不是质量或控制成功线。其他条件均缺失。',
        '', '| 模式 | 字段 | time | type | continuation | lane |','|---|---|---|---|---|---|']
    lookup={(x['mode'],x['field'],x['prefix_groups'],x['head']):x for x in r['controls']}
    fields=list(dict.fromkeys((x['mode'],x['field']) for x in r['controls']))
    for mode,field in fields:
        cells=[]
        for head in ('time','type','continuation','lane'):
            vals=[lookup.get((mode,field,prefix,head)) for prefix in (0,4)]
            cells.append(' / '.join(f"{v['median_tv']:.5f}" for v in vals) if all(vals) else '不适用')
        lines.append('| '+' | '.join([mode,field,*cells])+' |')
    lines+=['','## 换音频后的响应','','| 模式 | 换音频范围 | 前缀组数 | 决策头 | TV 中位数 | 改变最高概率选择 |','|---|---|---:|---|---:|---:|']
    for x in r['audio']:
        lines.append(f"| {x['mode']} | {x['variant']} | {x['prefix_groups']} | {x['head']} | {x['median_tv']:.5f} | {x['argmax_changes']}/{x['n']} |")
    lines+=['','## 首点错误的分量','','| 片段 | 选中 cell（0=END） | 真首点 cell | 真点排名 | 音频分数真点减选中 | 时间距离先验真点减选中 |','|---|---:|---:|---:|---:|---:|']
    for x in r['base_margins']:
        if x['prefix_groups']:continue
        d=x['true_minus_selected_components'];a=f"{d['audio']:.4f}" if d else 'END';b=f"{d['relative_time']:.4f}" if d else 'END'
        lines.append(f"| {x['case_id']} | {x['selected_cell']} | {x['true_cell']} | {x['true_cell_rank']} | {a} | {b} |")
    lines+=['','## 裁决和限制','',
        '本面板不支持当前训练/缓存生成路径错位解释。换音频的决策响应大，说明音频确实影响模型，不能据此认定它理解音乐。所有动作头的低/高条件响应中位数均低于 .01，四组真实历史后多数更弱；优先检验直接条件注入这一架构假设，但还不能归因于特定 attention kernel 或证明所有注意力机制有缺陷。',
        '', '六个未选 END 的首步中，音频指针都把后面的候选排在真首点之前；时间距离先验反而偏向真首点。因此继续加强“偏爱近点”的 loss/先验没有当前证据支持。另两窗直接选 END。',
        '', '只测一个未晋升检查点；参考通路及 hold 时长未测。动作分布使用真起音/真第一音符隔离混杂，不能当成自由生成结果。音频错配包含特征长度重采样。下一轮仍须真实 batch 50–100 步探针和独立生成 benchmark；D08 本身没有模型改善。']
    out.with_suffix('.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({k:r[k] for k in ('records','function_wall_s','parity','free_replays_equal','product_promotion')}))


if __name__=='__main__':main()
