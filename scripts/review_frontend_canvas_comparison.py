"""Matched small-frontend evidence for bounded H100 investment, not release."""
from collections import Counter,defaultdict
import json
import shutil

import numpy as np

from scripts.run_local_bundle import ROOT,write,sha
from scripts.run_frontend_canvas_quality_v2 import PLAN,REPORT,ART,save,journal


def metrics(rows):
    types=Counter();truth=Counter();by=defaultdict(list);per_type=defaultdict(lambda:Counter())
    for row in rows:
        types.update(e['type'] for e in row['prediction']['events']);truth.update(e['type'] for e in row['target'])
        by[row['component']].append(row)
        for k,d in row['scores']['per_type'].items():
            per_type[k].update({n:d[n] for n in ('tp','pred','true')})
    fields=dict(full_note_f1=lambda r:r['scores']['full_note_f1'],
        onset10=lambda r:r['scores']['onset_10ms']['f1'] or 0,
        onset50=lambda r:r['scores']['onset_f1'],shift_null10=lambda r:np.mean(r['shifted_onset10']))
    components={k:{field:float(np.mean([fun(r) for r in values])) for field,fun in fields.items()} for k,values in by.items()}
    out={field:float(np.mean([v[field] for v in components.values()])) for field in fields}
    out.update(attempted=len(rows),complete=sum(r['prediction']['complete'] for r in rows),
        nonempty=sum(bool(r['prediction']['events']) for r in rows),predicted_types=dict(types),target_types=dict(truth),
        density_ratio=sum(types.values())/max(sum(truth.values()),1),components=components,
        per_type={k:dict(d,f1=2*d['tp']/max(d['pred']+d['true'],1)) for k,d in per_type.items()})
    return out


def main():
    p=json.loads(PLAN.read_text());r=json.loads(REPORT.read_text())
    if r['status']!='evaluated':raise ValueError('Complete paired evaluation required')
    criteria_path=ROOT/'bench/rhythm-v2/frontend-investment-screen-v1.json';criteria=json.loads(criteria_path.read_text())
    source={**p['source_sha256'],**p['mert_cache_sha256']}
    for path,expected in source.items():
        if sha(ROOT/path)!=expected:raise ValueError('Frozen source/cache changed '+path)
    tasks=('old_draft','new_draft','completion','density_low','density_high','reference_masked','empty')
    raw={a:{t:json.loads((ART/a/f'{t}-final.json').read_text()) for t in tasks} for a in ('pupu','mert')}
    result={a:{t:metrics(rows) for t,rows in values.items()} for a,values in raw.items()}
    for arm in result:result[arm]['combined_draft']=metrics(raw[arm]['old_draft']+raw[arm]['new_draft'])
    for task in tasks:
        for a,b in zip(raw['pupu'][task],raw['mert'][task]):
            if a['index']!=b['index'] or a['target']!=b['target'] or a['request']!=b['request']:
                raise ValueError('Paired request/target mismatch')
    b=result['pupu']['combined_draft'];m=result['mert']['combined_draft'];c=criteria['preference']
    improved={field:sum(m['components'][k][field]>v[field] for k,v in b['components'].items()) for field in ('full_note_f1','onset10')}
    full=(m['full_note_f1']-b['full_note_f1']>=c['mert_full_event_gain'] and improved['full_note_f1']>=c['minimum_components_improved']
          and m['onset10']>=b['onset10']-c['noninferiority_other_metric_drop'])
    timing=(m['onset10']-b['onset10']>=c['mert_onset10_gain'] and improved['onset10']>=c['minimum_components_improved']
            and m['full_note_f1']>=b['full_note_f1']-c['noninferiority_other_metric_drop'])
    signals={}
    for arm in result:
        q=result[arm]['combined_draft'];g=criteria['candidate_signal']
        positive=sum(v['onset10']>v['shift_null10'] for v in q['components'].values())
        signals[arm]=dict(positive_components=positive,onset10_null_gain=q['onset10']-q['shift_null10'],
            signal_for_bounded_h100_trial=bool(q['complete']==q['attempted'] and positive>=g['minimum_components_positive']
                and q['onset10']-q['shift_null10']>=g['onset10_minus_fixed_shift_null']))
    r.update(status='completed',complete=True,review=result,mert_preference_signal=bool(full or timing),
        improved_components=improved,h100_investment_signals=signals,product_release_ready=False,
        original_v1_health_gate_unchanged='health_failed',cloud_compute_calls=0,
        source_cache_identity_count=len(source),screen_criteria_sha256=sha(criteria_path),
        interpretation='Small-model native frontend route comparison; only signal for bounded H100/0.25B testing, not a proof of large-model ranking or full product quality.')
    save(r)
    manifest={};archive=ART/'source'
    for path in source:
        if not path.endswith(('.py','.md')):continue
        dest=archive/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest);manifest[path]=sha(dest)
    for path in ['scripts/review_frontend_canvas_comparison.py','bench/rhythm-v2/frontend-investment-screen-v1.json']:
        dest=archive/path;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dest);manifest[path]=sha(dest)
    shutil.copy2(PLAN,archive/'frozen-plan.json');manifest['frozen-plan.json']=sha(archive/'frozen-plan.json')
    write(archive/'manifest.json',manifest)
    write(ART/'verification.json',dict(source_cache_sha256=source,plan_sha256=sha(PLAN),
        prediction_sha256={str(path.relative_to(ROOT)):sha(path) for arm in ('pupu','mert') for path in (ART/arm).glob('*-final.json')},
        weights_sha256={arm:sha(ART/arm/'quality-0900.pt') for arm in ('pupu','mert')}))
    lines=['# MERT2-FullSong / Pupu本地路线比较','',
        '投资目标：Modal H100上约0.25B可训模型的一次有界试训。本地不是产品出场验收，也不按小模型胜负否定模型族。',
        f"两臂各900更新，参数{r['parameters']}，共同参数新初始化精确相同、逐批监督相同，健康权重丢弃。训练{r['wall_s']['training']:.1f}s、评价{r['wall_s']['evaluation']:.1f}s，新增云调用0。",'',
        '| 16验证歌曲组件/32窗 | Pupu | MERT2-FullSong |','|---|---:|---:|']
    for label,key in [('完整音符F1@50ms','full_note_f1'),('起点F1@10ms','onset10'),('起点F1@50ms','onset50'),
        ('错位起点F1@10ms','shift_null10'),('输出/真值数量比','density_ratio'),('非空窗口','nonempty')]:
        lines.append(f'| {label} | {b[key]:.4f} | {m[key]:.4f} |')
    lines+=['',f'MERT完整事件改善{improved["full_note_f1"]}/16组件，10ms定位改善{improved["onset10"]}/16组件；预注册前端优先信号={bool(full or timing)}。',
        '有界H100试训定位信号：','```json',json.dumps(signals,ensure_ascii=False,indent=2),'```','',
        '独立面板分列：','```json',json.dumps({a:{t:{k:v[k] for k in ['full_note_f1','onset10','onset50','density_ratio']} for t,v in data.items() if t in ('old_draft','new_draft')} for a,data in result.items()},ensure_ascii=False,indent=2),'```','',
        '类型与长按正确匹配：','```json',json.dumps({a:result[a]['combined_draft']['per_type'] for a in result},ensure_ascii=False,indent=2),'```','',
        '输入路线差异：Pupu三层八频带/两个查询，对MERT两层原生向量/一个查询；精度、表示和记忆token数不同，是完整前端路线比较。40音频逐文件SHA检查，39单次整曲+1长曲分窗拼接；共有10msmel逐值相等。',
        '相对尾端以毫秒对齐的单元起点为基准，保留微秒COPY头、跨窗hold和全部多重集；不做NMS、最小间隔、类型配额或按用户密度裁数。',
        'v1稀有类型健康门失败保留；v2在见过该结果后显式调整工程入场证书，原生离散字段100%/全局F1>=.95/空窗0/数值一致，不能声称原门事前通过。',
        '当前16验证组件均已在本地研发使用，不能称全项目盲测；每组件两窗宏平均。尚未试听、真人编辑或Phira原生试玩。条件、参考和各类型风险在机器报告完整保留。',
        '本地MPS耗时不外推H100速度；后续0.25B目前只meta-device核算约252M参数，CUDA数值/显存/吞吐和整段费用必须实测并封顶。']
    (ROOT/'reports/local_frontend_comparison_review_v2.md').write_text('\n'.join(lines)+'\n')
    journal(f'最终路线比较：Pupu完整F1={b["full_note_f1"]:.6f}/onset10={b["onset10"]:.6f}，MERT={m["full_note_f1"]:.6f}/{m["onset10"]:.6f}，优先信号{bool(full or timing)}，H100定位信号{signals}；未执行云调用。')
    print(json.dumps(dict(mert_preference_signal=bool(full or timing),signals=signals,comparison={a:result[a]['combined_draft'] for a in result}),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
