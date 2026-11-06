"""Screen a small local candidate for bounded additional compute, not release."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import shutil

import numpy as np

from scripts.run_local_typed_canvas import ROOT, ART, PLAN, REPORT, save, journal
from scripts.run_local_bundle import sha, write


def metrics(rows):
    active=[r for r in rows if r['attempted']];pred=Counter();target=Counter();per=defaultdict(list)
    song=defaultdict(list);onset_song=defaultdict(list)
    for r in active:
        pred.update(r['types']);target.update(r['target_types']);song[r['chart_id']].append(r['scores']['full_note_f1'])
        onset_song[r['chart_id']].append(r['scores']['onset_f1'])
        for k,v in r['scores']['per_type'].items():per[k].append(v)
    types={}
    for k,vs in per.items():
        tp=sum(v['tp'] for v in vs);npred=sum(v['pred'] for v in vs);nt=sum(v['true'] for v in vs)
        types[k]=dict(tp=tp,pred=npred,true=nt,f1=2*tp/max(npred+nt,1))
    return dict(attempted=len(active),complete=sum(r['prediction']['complete'] and not r['legality'] for r in active),
        legal_nonempty=sum(bool(r['prediction']['events']) and not r['legality'] for r in active),
        full_note_f1=float(np.mean([r['scores']['full_note_f1'] for r in active])) if active else 0.,
        onset_f1=float(np.mean([r['scores']['onset_f1'] for r in active])) if active else 0.,
        single_type_windows=sum(len(r['types'])==1 for r in active),all_tap_windows=sum(set(r['types'])=={'tap'} for r in active),
        predicted_types=dict(pred),target_types=dict(target),per_type=types,
        density_ratio=sum(pred.values())/max(sum(target.values()),1),
        positive_gaps_under10ms=sum(r['positive_gaps_under_10ms'] for r in active)/max(sum(r['positive_gaps'] for r in active),1),
        per_song_f1={k:float(np.mean(v)) for k,v in song.items()},
        per_song_onset_f1={k:float(np.mean(v)) for k,v in onset_song.items()})


def main():
    plan=json.loads(PLAN.read_text());report=json.loads(REPORT.read_text())
    if report['status']!='evaluated':raise ValueError('Final evaluation not completed')
    verified={}
    for path,expected in plan['source_sha256'].items():
        actual=sha(ROOT/path)
        if actual!=expected:raise ValueError('Frozen input/source changed: '+path)
        verified[path]=actual
    tasks=('draft','completion','density_low','density_high','reference_masked')
    rows={t:json.loads((ART/f'{t}-final.json').read_text()) for t in tasks}
    baseline={t:json.loads((ROOT/f'artifacts/local-bundle-v1/bundle/{t}-final.json').read_text()) for t in tasks if t!='reference_masked'}
    old_verification=json.loads((ROOT/'artifacts/local-bundle-v1/verification.json').read_text())
    for path,expected in old_verification['prediction_sha256'].items():
        if '/bundle/' in path and sha(ROOT/path)!=expected:raise ValueError('Frozen comparator prediction changed')
    for task in baseline:
        for new,old in zip(rows[task],baseline[task]):
            if new['index']!=old['index'] or new['request']!=old['request'] or new['target']!=old['target']:
                raise ValueError('Frozen baseline comparison target/request mismatch')
    result={t:metrics(v) for t,v in rows.items()};base={t:metrics(v) for t,v in baseline.items()}
    draft=result['draft'];prior=base['draft'];g=plan['gate']
    better_full=sum(draft['per_song_f1'].get(s,0)>v for s,v in prior['per_song_f1'].items())
    better_time=sum(draft['per_song_onset_f1'].get(s,0)>v for s,v in prior['per_song_onset_f1'].items())
    guard=(all(x['attempted'] and x['prediction']['complete'] and not x['legality'] for v in rows.values() for x in v)
           and draft['attempted']==16 and draft['legal_nonempty']>=g['minimum_legal_nonempty_drafts']
           and g['density_ratio'][0]<=draft['density_ratio']<=g['density_ratio'][1])
    full_route=draft['full_note_f1']-prior['full_note_f1']>=g['full_gain'] and better_full>=g['minimum_improved_songs']
    time_route=(draft['onset_f1']-prior['onset_f1']>=g['onset_gain'] and better_time>=g['minimum_improved_songs']
                and draft['full_note_f1']>=prior['full_note_f1']-g['maximum_full_drop_if_timing_route'])
    promising=bool(guard and (full_route or time_route))
    density=[]
    for a,b in zip(rows['density_low'],rows['density_high']):
        width=(a['request']['end_us']-a['request']['start_us'])/1e6
        lo=len(a['prediction']['events'])/width;hi=len(b['prediction']['events'])/width
        density.append(dict(index=a['index'],low=lo,high=hi,direction=hi>lo,
                            low_in_band=2<=lo<3,high_in_band=6<=hi<8))
    reference_gain=result['completion']['full_note_f1']-result['reference_masked']['full_note_f1']
    report.update(status='screen_closed_positive' if promising else 'screen_closed_no_progress',complete=True,
        promising_for_bounded_scale_pilot=promising,release_ready=False,review=result,baseline_review=base,
        screening_gates=dict(correctness_resource_guard=bool(guard),full_event_progress=bool(full_route),time_progress=bool(time_route)),
        improved_songs=dict(full=better_full,timing=better_time),density_pairs=density,reference_full_f1_gain=reference_gain,
        source_verification_count=len(verified),source_verification_passed=True,
        interpretation='Only an engineering resource screen; type/condition/reference shortcomings are measured risks, not a conjunctive product-release gate.',
        inference_budget_note='Old bundle uses a 1024 autoregressive-generation token cap; canvas uses bounded addresses plus request max_events=4096 and native serialization max_tokens=32768. Different output computation contracts are intentional; completion is not a like-for-like token-budget quality result.',
        new_cloud_spend_usd=0,human_playtest=False,attribution='Complete recipe, not matched parameter/update/compute ablation; old single-dimension counters unchanged')
    save(report)
    archive=ART/'source';manifest={}
    for path in sorted(plan['source_sha256']):
        if not path.endswith(('.py','.md')):continue
        dst=archive/path;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(ROOT/path,dst);manifest[path]=sha(dst)
    shutil.copy2(PLAN,archive/'frozen-plan.json');manifest['frozen-plan.json']=sha(archive/'frozen-plan.json')
    write(archive/'manifest.json',manifest)
    write(ART/'verification.json',dict(verified_source_sha256=verified,plan_sha256=sha(PLAN),
        prediction_sha256={str((ART/f'{t}-final.json').relative_to(ROOT)):sha(ART/f'{t}-final.json') for t in tasks}))
    lines=['# 本地分类型画布：资源筛选结果','',
        f'判定：{"值得一次有界增加资源的试验" if promising else "本配方尚未显示足够扩算收益"}。这不是产品出场判定。',
        f"实际{report['parameters']:,}参数；100健康后丢弃权重、新鲜初始化{report['updates']}正式更新；云调用0。",'',
        '| 草稿指标 | 旧组合1000步 | 新画布900步 |','|---|---:|---:|']
    for label,key in [('完整音符F1@50ms','full_note_f1'),('起音F1@50ms','onset_f1'),('完整窗口','complete'),
                      ('单类型窗口','single_type_windows'),('全tap窗口','all_tap_windows'),('输出/真值数量比','density_ratio'),('正间隔<10ms比例','positive_gaps_under10ms')]:
        lines.append(f'| {label} | {prior[key]:.4f} | {draft[key]:.4f} |')
    lines+=['',f'完整音符改善{better_full}/8歌曲，时间改善{better_time}/8。',
        f"补全F1={result['completion']['full_note_f1']:.4f}，相对参考屏蔽收益={reference_gain:.4f}。",'',
        '各类型（聚合精确率/召回的F1；hold要求双端点50ms）：','```json',json.dumps(draft['per_type'],ensure_ascii=False,indent=2),'```',
        '密度条件实际音符/秒：','```json',json.dumps(density,ensure_ascii=False,indent=2),'```','',
        '筛选门：','```json',json.dumps(report['screening_gates'],ensure_ascii=False,indent=2),'```','',
        '健康拟合不是泛化，历史控制与新候选不是等参数/等训练/等计算对照；分数仅为现有验证集上的局部工程信号。',
        '固定地址自然结束不等于音乐质量；无最小间隔、NMS、删重复、网格吸附、类型配额或密度裁剪。',
        '旧组合有1024自回归生成token限制；画布使用有界地址与原请求4096事件/32768原生序列化token上限，推理预算合同不同。完成率不作等token预算质量收益。',
        '尚未试听、真人编辑或原生试玩；条件/参考/类型短板作为下一阶段风险完整保留。',
        '原始40请求、曲线、权重、原生序列化tokens与源码在artifacts/local-typed-canvas-v1/，机器结果reports/local_typed_canvas_v1.json。']
    (ROOT/'reports/local_typed_canvas_review_v1.md').write_text('\n'.join(lines)+'\n')
    journal(f'资源筛选终态{report["status"]}，fullF1={draft["full_note_f1"]:.6f}、onsetF1={draft["onset_f1"]:.6f}，完整音符改善{better_full}/8歌；门{report["screening_gates"]}。这是扩算筛选，非出场门。新云0。')
    print(json.dumps(dict(status=report['status'],gates=report['screening_gates'],review=result),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
