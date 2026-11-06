"""Join authored judgments with raw receipts, not an automatic causal judge."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from scripts.mechanism_notes_v2 import notes
from scripts.validate_mechanism_reviews import validate_review


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/local-portfolio-v1/mechanism-reviews-v2'


def read(path):return json.loads((ROOT/path).read_text())
def sha(path):return hashlib.sha256((ROOT/path).read_bytes()).hexdigest()
def write(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')


def evaluations(c,source):
    ident=c['id'];arm=c['arm']
    if 'arms' in source:
        r=source['arms'][arm];e=r['evaluations'];return r,e.get('1000',e)
    if ident=='ssd-reference1000':return source,source['evaluations']['1000']['mamba']
    if ident.startswith('small'):return source,source['evaluations']['2000'][arm]
    if ident in ('mel-2000','package-2000'):return source,source['evaluations']['2000']
    return source,source['evaluations']


def health_receipt(c,current):
    ident=c['id'];path=c['report'];h=current
    if ident.startswith('small'):
        path='reports/local_portfolio_scale_v1_width_probe.json';h=read(path)['small24']
        return dict(status='shared_small_architecture_health_cited_not_parent_capacity_failure',source=path,
                    updates=100,initial_loss_mean=float(np.mean([v['loss'] for v in h['before']])),
                    final_loss_mean=float(np.mean([v['loss'] for v in h['after']])),
                    caveat='small24/small64 quality resets share exact initialization; parent capacity64 health failure is separate.')
    mapping={'ssd-reference1000':'reports/local_portfolio_mamba_health_v1.json',
             'mel-2000':'reports/local_portfolio_mel_health_v1.json',
             'package-2000':'reports/local_portfolio_package_health_v1.json',
             'local-canonical-package-quality-v2':'reports/local_canonical_package_v1.json',
             'local-group-diffusion-quality-v1':'reports/local_group_diffusion_health_v3.json'}
    if ident in mapping:path=mapping[ident];h=read(path)
    if ident.startswith('access-'):path='reports/local_access_health_v1.json';h=read(path)['arms'][c['arm']]
    result=dict(status=h.get('status'),source=path,updates=h.get('health_updates'),
                gates=h.get('health_gates',h.get('gates')),parity=h.get('parity',h.get('zero_update_parity')),
                admission_after_observation=current.get('admission_after_observation',False),
                health_weights_used=current.get('health_weights_used',False))
    for a,b in [('initial_diagnostic','final_diagnostic'),('diagnostic_initial','diagnostic_final')]:
        if a in h and b in h:
            for key,target in [(a,'initial_fixed_loss_mean'),(b,'final_fixed_loss_mean')]:
                rows=h[key].get('rows',[]) if isinstance(h[key],dict) else h[key]
                values=[v.get('loss',v.get('velocity_mse')) for v in rows]
                if values and all(v is not None for v in values):result[target]=float(np.mean(values))
    if ident in ('local-canonical-package-quality-v2','local-group-diffusion-quality-v1'):
        result['positive_generated_count_ratios']=h.get('positive_ratios',h.get('positive_count_ratios'))
        empty=next(v for v in h['final_diagnostic'] if v['index']==6074)
        result['empty_index']=6074
        result['empty_expected_events']=empty.get('expected',empty.get('expected_events'))
        prediction_path=('artifacts/local-canonical-package-v1/health-predictions.json'
                         if ident=='local-canonical-package-quality-v2'
                         else 'artifacts/local-group-diffusion-v3/health-predictions.json')
        prediction=next(v for v in read(prediction_path) if v['index']==6074)
        result['empty_generated_events']=len(prediction['prediction']['events'])
        result['empty_generated_receipt']=prediction_path
    return result


def scalar_metrics(value):
    return {k:value[k] for k in ('full_note_f1','onset10','onset50','shift_null10','density_ratio','complete','attempted','nonempty') if k in value}


def micro_types(metric):
    rows=metric.get('samples',[metric]);result={}
    for kind in ('tap','hold','flick','drag'):
        if all(kind in r.get('per_type',{}) for r in rows):
            totals={k:sum(r['per_type'][kind][k] for r in rows) for k in ('tp','pred','true')}
            totals['f1']=2*totals['tp']/max(1,totals['pred']+totals['true']);result[kind]=totals
    return result


def controls(ev,metrics):
    values=[dict(name='count_preserving_output_shift',scope='Moves output onsets, preserves count; not input-audio causality',
                 onset10=metrics['onset10'],null_onset10=metrics['shift_null10'],
                 positive_components=sum(v['onset10']>v['shift_null10'] for v in metrics['components'].values()))]
    for task,value in ev.items():
        if task=='draft' or not isinstance(value,dict):continue
        if 'full_note_f1' in value:
            item=dict(name=task,metrics=scalar_metrics(value))
            if task=='empty':
                rows=value.get('samples',[value]);item['events_each_draw']=[sum(r.get('predicted_types',{}).values()) for r in rows]
            values.append(item)
    return values


def main():
    catalog=read('reports/local_portfolio_terminal_catalog_v1.json')['candidates'];authored=notes()
    index=read('reports/local-portfolio-v1/mechanism-reviews-v2/index.json')
    selection=read('reports/local_portfolio_finalist_selection_v1.json');selected={r['id'] for r in selection['finalist_identities']}
    if set(authored)!={r['id'] for r in catalog if r['id']!='local-whole-song-arrangement-v1'}:
        raise ValueError('Every remaining candidate must have an authored judgment')
    for c in catalog:
        ident=c['id']
        if ident=='local-whole-song-arrangement-v1':continue
        jp=FOLDER/(ident+'.json');md=FOLDER/(ident+'.md')
        if jp.exists() or md.exists():
            existing=next(r for r in index['candidates'] if r['id']==ident)
            if existing['analysis_status']=='complete' and existing.get('review_sha256')==sha(jp.relative_to(ROOT)):
                if validate_review(read(jp.relative_to(ROOT))):raise ValueError('Existing review no longer validates')
                continue
            raise FileExistsError('Unregistered review requires explicit diagnosis: '+ident)
        source=read(c['report']);current,ev=evaluations(c,source);a=authored[ident];m=c['metrics']
        health=health_receipt(c,current);ct=controls(ev,m)
        role='replicated_and_confirmed_local_candidate' if ident in selected else 'eligible_fixed_trial_not_selected_under_frozen_top2' if c['eligible'] else 'failed_complete_recipe'
        evidence=list(dict.fromkeys([c['report'],c['plan'],c['checkpoint'],health['source'],*a['evidence']]))
        row=dict(candidate=ident,analysis_status='complete',written_at=datetime.now(timezone.utc).isoformat(),
            identity={p:sha(p) for p in evidence},
            layers=dict(health=health,quality=dict(status=role,metrics=scalar_metrics(m),original_screen=c['eligibility'],
                        parameters=c['parameters'],quality_updates=c['steps'],per_type=micro_types(m)),
                        structure_conditions=dict(status='measured_capabilities_and_unmeasured_controls_separated',
                            reference_full_gain=ev['completion']['full_note_f1']-ev['reference_masked']['full_note_f1']
                                if 'completion' in ev and 'reference_masked' in ev else None,
                            controls=ct)),
            mechanism_hypotheses=a['hypotheses'],controls=ct,missing_controls=a['missing'],lessons=a['lessons'],
            successor_contract=dict(required_changes=a['changes'],forbidden_repeats=a['forbidden']),
            closure_reason=a['story'],source_record=str(md.relative_to(ROOT)),
            interpretation='Original metric/health failures unchanged; this authored postmortem replaces gate-only rationale. Selection completion is not method-family rejection; missing causal comparisons are explicit.',
            optimizer_updates_in_review=0,cloud_compute_calls=0)
        def link(path):return os.path.relpath(ROOT/path,FOLDER)
        lines=[f'# {ident}：终态机制复盘 v2','',a['story'],'',
               '## 分层现象','',f'健康/工程来源：[{health["source"]}]({link(health["source"])}).',
               '```json',json.dumps(health,ensure_ascii=False,indent=2),'```','',
               '| 正式主结果 | 数值 |','|---|---:|']
        for k,v in scalar_metrics(m).items():lines.append(f'| {k} | {v:.6f} |' if isinstance(v,float) else f'| {k} | {v} |')
        lines+=['',f'实际模型参数 {c["parameters"]:,}；固定质量更新 {c["steps"]}；本批处置 `{role}`。',
                '原屏只决定原试验的数值结果；以下机制、对照和后继合同决定为何停止或保留。','',
                '| 类型（所有固定draw合并） | TP | 预测 | 真值 | F1 |','|---|---:|---:|---:|---:|']
        for kind,v in micro_types(m).items():lines.append(f'| {kind} | {v["tp"]} | {v["pred"]} | {v["true"]} | {v["f1"]:.6f} |')
        lines+=['','## 可证伪机制与判断','']
        for h in a['hypotheses']:
            lines += [f'### {h["id"]}','',f'观察：{h["observation"]}',f'机制：{h["mechanism"]}',
                      f'预测：{h["prediction"]}',f'判断：{h["judgment"]}','']
        lines+=['## 已有对照与边界','','| 对照 | 完整F1 | onset10 | 数量比/备注 |','|---|---:|---:|---|']
        for v in ct:
            if 'metrics' in v:
                x=v['metrics'];remark=str(x.get('density_ratio','—'))
                if 'events_each_draw' in v:remark='真值为零，数量比不适用；两空窗各draw总事件 '+str(v['events_each_draw'])
                lines.append(f'| {v["name"]} | {x.get("full_note_f1",0):.6f} | {x.get("onset10",0):.6f} | {remark} |')
            else:lines.append(f'| 输出保数量错位 | — | null {v["null_onset10"]:.6f} | 正向组件{v["positive_components"]}/16；不是输入因果 |')
        lines+=['','额外冻结权重/源帧/概率/真实输入干预见下面证据。若本候选没有某条输入干预，不能借用其它模型的结果充当完成。空窗误发与条件方向不等于同一指标，真实参考与遮罩对照也不证明人类编辑收益。','',
                '## 缺失的判别对照','','| 未分开的解释 | 具体对照 | 结果改变哪一步动作 |','|---|---|---|']
        for v in a['missing']:lines.append(f'| {v["question"]} | {v["comparison"]} | {v["decision_if"]} |')
        lines+=['','## 经验与后继合同','']+[f'- {v}' for v in a['lessons']]
        lines+=['','后继必须改或明确检验：','']+[f'- {v}' for v in a['changes']]
        lines+=['','禁止重复：','']+[f'- {v}' for v in a['forbidden']]
        lines+=['','本复盘不启动新训练、不改原主读出或门、不追加seed，不授权云任务。下一实验必须先通过机制复盘与同根失败处理检查。','',
                '## 证据','']+[f'- [{p}]({link(p)})' for p in evidence if not p.endswith('.pt')]
        md.write_text('\n'.join(lines)+'\n');write(jp,row)
        errors=validate_review(row)
        if errors:raise ValueError((ident,errors))
        old=next(r for r in index['candidates'] if r['id']==ident)
        old.update(analysis_status='complete',review=str(jp.relative_to(ROOT)),review_sha256=sha(jp.relative_to(ROOT)),markdown=str(md.relative_to(ROOT)))
        write(FOLDER/'index.json',index)
    index.update(status='all_candidate_mechanism_reviews_completed',new_training_allowed=False,
                 completed_candidates=len(index['candidates']),new_training_note='Future training needs a new successor contract and actual pretraining receipts; finishing reviews does not auto-launch.')
    write(FOLDER/'index.json',index)
    lines=['# 完整候选机制复盘 v2','',
           'Owner2026-10-29补正：23个候选均已逐项写出分层结果、可证伪机制、已有/缺失对照和后继合同。'
           '原数字、健康失败和旧裁决保留；这不是追加质量训练。B优先的深度复盘和新冻结干预单列。','',
           '| 候选 | 机制复盘 | 原始质量筛选 |','|---|---|---|']
    for r in index['candidates']:
        lines.append(f'| {r["id"]} | [完整分析]({os.path.relpath(ROOT/r["markdown"],FOLDER)}) | {r["original_quality_eligible"]} |')
    lines+=['','字段完整性与身份校验不等于自动证明因果。所有分析明确区分直接干预、受控配对、候选解释和未测边界；没有用“不同参数”作为终止分析的理由。','']
    (FOLDER/'README.md').write_text('\n'.join(lines)+'\n')
    print('23 candidate mechanism reviews complete; no training launched.')


if __name__=='__main__':main()
