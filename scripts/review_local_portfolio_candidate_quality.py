"""Audit completed mel/package curves from immutable saved predictions."""
import argparse
from collections import Counter
import json
from pathlib import Path

from scripts.run_local_portfolio_mamba_health import sha,write


ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate',choices=('mel','package'),required=True)
    candidate=parser.parse_args().candidate
    stem=f'local_portfolio_{candidate}_quality'
    output=ROOT/f'reports/{stem}_review_v1.json'
    markdown=ROOT/f'reports/{stem}_review_v1.md'
    if output.exists() or markdown.exists():raise FileExistsError('Preserve previous review')
    report_path=ROOT/f'reports/{stem}_v1.json';report=json.loads(report_path.read_text())
    plan_path=ROOT/f'bench/rhythm-v2/local-portfolio-{candidate}-quality-v1.json'
    plan=json.loads(plan_path.read_text());art=ROOT/f'artifacts/local-portfolio-v1/{candidate}-quality'
    if report['status']!='completed' or report['updates']!=2000:
        raise ValueError('Require completed fixed2000 endpoint')
    if report['plan_sha256']!=sha(plan_path) or report['health_weights_used']:
        raise ValueError('Plan or initialization provenance failed')
    for name,value in plan['source_sha256'].items():
        if sha(ROOT/name)!=value:raise ValueError('Frozen source drift: '+name)
    if report['final_weight_sha256']!=sha(art/'quality-2000.pt'):raise ValueError('Weight identity drift')
    control=json.loads((ROOT/'reports/local_portfolio_data_scale_v1.json').read_text())
    mamba=json.loads((ROOT/'reports/local_portfolio_mamba_quality_resume_v1.json').read_text())
    earlier,final=(report['evaluations'][str(s)] for s in (1000,2000))
    baseline=control['evaluations']['2000']['small64']['draft']
    mamba_final=mamba['evaluations']['2000']['mamba']['draft']
    components=final['draft']['components']
    deltas={name:{key:final['draft'][key]-other[key] for key in ('full_note_f1','onset10','onset50')}
        for name,other in [('small64',baseline),('mamba2000',mamba_final)]}
    comparison=dict(recipe_deltas=deltas,
        relative_to_small64_components={key:sum(v[key]>baseline['components'][c][key] for c,v in components.items()) for key in ('full_note_f1','onset10')},
        relative_to_mamba_components={key:sum(v[key]>mamba_final['components'][c][key] for c,v in components.items()) for key in ('full_note_f1','onset10')},
        caveat='Whole routes differ in width/depth/seed/audio context; no isolated frontend or backbone causality claimed.')
    raw={task:json.loads((art/f'quality-2000-{task}.json').read_text()) for task in final}
    failed=[dict(task=t,index=r['index'],failure=r['prediction'].get('failure'))
        for t,rows in raw.items() for r in rows if not r['prediction']['complete']]
    nonempty_types=Counter(tuple(sorted({e['type'] for e in r['prediction']['events']})) for r in raw['draft'])
    empty_counts=[len(r['prediction']['events']) for r in raw['empty']]
    temporal_gain=final['draft']['onset10']-final['draft']['shift_null10']
    positive=sum(v['onset10']>v['shift_null10'] for v in components.values())
    worst=sorted(raw['draft'],key=lambda r:(r['scores']['full_note_f1'],r['index']))[:6]
    summary=dict(candidate=candidate,status='reviewed',optimizer_updates=0,new_inference_calls=0,
        actual_training_updates=2000,health_weights_used=False,confirmation_scored=False,cloud_compute_calls=0,
        snapshots={str(s):report['evaluations'][str(s)] for s in (1000,2000)},comparison=comparison,
        endpoint_change={k:final['draft'][k]-earlier['draft'][k] for k in ('full_note_f1','onset10','onset50')},
        temporal_evidence=dict(onset10_minus_fixed_shift=temporal_gain,positive_components=positive),
        density_control=[dict(index=l['index'],low=len(l['prediction']['events']),high=len(h['prediction']['events']),
            low_nps=len(l['prediction']['events'])/(l['request']['end_us']-l['request']['start_us'])*1e6,
            high_nps=len(h['prediction']['events'])/(h['request']['end_us']-h['request']['start_us'])*1e6)
            for l,h in zip(raw['density_low'],raw['density_high'])],
        reference_full_gain=final['completion']['full_note_f1']-final['reference_masked']['full_note_f1'],
        empty_counts=empty_counts,failed_requests=failed,window_type_patterns={'+'.join(k):v for k,v in nonempty_types.items()},
        worst_native_windows=worst,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in [plan_path,report_path,
            ROOT/'reports/local_portfolio_data_scale_v1.json',ROOT/'reports/local_portfolio_mamba_quality_resume_v1.json',
            ROOT/'scripts/review_local_portfolio_candidate_quality.py']},
        prediction_sha256={str(p.relative_to(ROOT)):sha(p) for p in art.glob('quality-*-*.json')},
        weight_sha256=report['final_weight_sha256'])
    write(output,summary)
    lines=[f'# {candidate} train64 质量审查','',
        f'固定2000更新已完成，参数{report["parameters"]:,}；fresh初始化与健康初态身份相同，健康权重未加载。'
        f'训练含评价{report["wall_s"]["training"]:.1f}秒，确认集未评分，新增云调用0。','',
        '| 开发16组件/32窗 | 1000更新 | 2000更新 | small64 2000 | Mamba2000 |','|---|---:|---:|---:|---:|']
    for title,key in [('完整音符F1','full_note_f1'),('onset10','onset10'),('onset50','onset50'),('预测/真值数量比','density_ratio')]:
        lines.append(f'| {title} | {earlier["draft"][key]:.6f} | {final["draft"][key]:.6f} | {baseline[key]:.6f} | {mamba_final[key]:.6f} |')
    lines+=['',f'同一候选1000→2000变化：完整F1 {summary["endpoint_change"]["full_note_f1"]:+.6f}，'
        f'onset10 {summary["endpoint_change"]["onset10"]:+.6f}。2000是冻结终点，不事后改用1000。','',
        f'真实起点相对固定移位零模型增益{temporal_gain:.6f}，正向组件{positive}/16。'
        '这是时间定位证据，不证明高层情绪／段落理解。','',
        f'两个空窗各输出{empty_counts}；失败请求{len(failed)}。参考完整F1增益{summary["reference_full_gain"]:+.6f}。', '',
        '类型正确匹配：','', '| 类型 | TP | 预测 | 真值 | F1 |','|---|---:|---:|---:|---:|']
    for kind,v in final['draft']['per_type'].items():lines.append(f'| {kind} | {v["tp"]} | {v["pred"]} | {v["true"]} | {v["f1"]:.6f} |')
    lines+=['','候选间数据流和评分一致，但参数规模、种子、计算与上下文不同；以上是完整路线比较，不能单独归因于是否使用预训练表征。',
        '没有试听、真人编辑计时、情绪标签或Phira原生试玩。机械完成、健康拟合和总体F1不能代替这些证据。',
        f'逐组件、逐条件、最差6窗口原始事件及来源SHA见 `{output.name}`。','']
    markdown.write_text('\n'.join(lines))
    print(json.dumps({k:summary[k] for k in ('candidate','endpoint_change','comparison','temporal_evidence','empty_counts','failed_requests','reference_full_gain')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
