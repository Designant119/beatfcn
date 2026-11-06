"""Present measured learning trends without turning partial metrics into passes."""
from collections import defaultdict
import datetime as dt
import html
import json
import math
from pathlib import Path
import time
import statistics
import unicodedata


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def fmt(value, suffix=''):
    return f'{value:.4g}{suffix}' if finite(value) else '—'


def trend(values, direction, comparable=True, tolerance=.005):
    values = [v for v in values if finite(v)]
    if not comparable:
        return dict(status='不可判', tone='muted', delta=None, reason='生成超时或条件不可比')
    if len(values) < 2:
        return dict(status='待积累', tone='muted', delta=None, reason='需要第二个同口径点')
    change = values[-1]-values[-2]
    if abs(change) <= max(1e-6, abs(values[-2])*tolerance):
        return dict(status='持平', tone='neutral', delta=change, reason=f'变化不超过{100*tolerance:g}%')
    good = change*direction > 0
    return dict(status='改善' if good else '变差', tone='good' if good else 'bad', delta=change, reason='与前一同口径点比较')


def metric(key, label, direction, value, values=(), *, comparable=True, unit='', note='', target=None, tolerance=.005):
    result = dict(key=key,label=label,direction=direction,value=value,unit=unit,note=note,
                  points=list(values),target=target,**trend(values,direction,comparable,tolerance))
    if target is not None and finite(value):
        passed = value >= target if direction > 0 else value <= target
        result.update(status='达标' if passed else '未达标',tone='good' if passed else 'bad')
    return result


# Benchmark v2 categories (design/BENCHMARK_GUIDE_v2.md). Each row: key, label,
# direction (+1 higher better, -1 lower better), source, path, unit, note.
BENCHMARK_V2=(
    ('accuracy','准确度 · 对照人谱','与人类谱逐事件匹配；截断输出按完整真值计（偏低），前缀F1与覆盖率另列',(
        ('note_f1_50ms','音符F1@50ms',1,'gen','accuracy.note_f1_50ms','','类型+mania键位+时间；主读出(T=1采样)K种子均值'),
        ('note_minus_null','F1 超出平移基线',1,'gen','accuracy.note_f1_50ms_minus_null','','减去同输出平移1–3秒的F1；≈0 表示没对齐音乐'),
        ('hold_f1','Hold 头尾联合F1',1,'gen','accuracy.hold_joint_f1_50ms','','跨窗合并计数，单侧有hold的窗也计入'),
        ('complete','完整输出比例',1,'gen','accuracy.complete_fraction','ratio','token预算内自然结束；watchdog/预算截断记失败'),
        ('prefix_f1','已覆盖前缀F1',1,'gen','accuracy.prefix_note_f1_50ms','','只看已生成时间段；与覆盖率一起读'),
        ('coverage','平均时间覆盖率',1,'gen','accuracy.mean_coverage_fraction','ratio',''))),
    ('human_likeness','人类相似度 · 节奏统计','与同窗人谱的统计分布比较；不是试玩评价',(
        ('ioi_js','间隔分布 JS',-1,'gen','human_likeness.ioi_js_nats','','含<30ms间隔；0=分布一致，0.69=完全不重叠'),
        ('short_gap_excess','<40ms间隔比例偏差',-1,'gen','human_likeness.short_gap_excess','','|模型−人类|；人类约1%'),
        ('type_js','类型构成 JS',-1,'gen','human_likeness.type_mix_js_nats',''),
        ('density_r','密度轮廓相关',1,'gen','human_likeness.density_profile_pearson','','0.5秒格；平坦窗不定义并计数'))),
    ('condition_response','条件响应 · 密度请求','同音频改密度请求：低0.5×/高2×/时间反转',(
        ('monotonic','低<自由<高 单调比例',1,'gen','condition_response.monotonic_fraction','ratio','三者都完整才计'),
        ('high_low','高/低 产出比',1,'gen','condition_response.mean_high_over_low','','请求为4×；≈1 表示不听条件'),
        ('request_mae','请求−产出 MAE',-1,'gen','condition_response.mean_request_mae_events_per_s','',''))),
    ('hacks','投机/退化检测','越低越好；任一项高说明数字可能是假象',(
        ('empty','正例空输出率',-1,'gen','hacks.empty_positive_rate','ratio','空输出=失败，不是“完整”'),
        ('template','模板化超出人类',-1,'gen','hacks.template_excess_mean','','不同歌输出的相似度−人谱同口径'),
        ('silence_rate','空窗事件/秒',-1,'gen','hacks.silence_events_per_s','','未完成时为下界'),
        ('foreign_same','换音频输出不变比例',-1,'gen','hacks.foreign_identical_fraction','ratio',''))),
    ('audio_grounding','音频依赖 · 是否在听','teacher-forced CE差（替换/清零/平移音频−真实），>0 表示用到音频',(
        ('gap_foreign_time','换歌 CE差·头时间',1,'teacher','foreign.head_time','','约200窗；时间字段'),
        ('gap_zero_time','清零 CE差·头时间',1,'teacher','zero_local_mel.head_time','','local+mel清零'),
        ('gap_roll_time','平移 CE差·头时间',1,'teacher','roll.head_time','','local+mel 平移±1–3秒'),
        ('gap_foreign_kind','换歌 CE差·类型',1,'teacher','foreign.kind','','Phira kind'),
        ('gap_foreign_mania','换歌 CE差·键位',1,'teacher','foreign.mania','','Mania 键位组合'),
        ('f1_real_minus_foreign','真实−换歌 音符F1',1,'gen','audio_grounding.real_minus_foreign_note_f1_50ms','','生成侧'),
        ('onset_head','起音头 F1（真实音频）',1,'onset','real','','仅配置有 onset_head 时'))))


def _get(value,path):
    for key in path.split('.'):
        value=value.get(key)if isinstance(value,dict)else None
    return value


def benchmark_values(checkpoint):
    """Primary-readout scalars of one v2 checkpoint; None for old schemas."""
    bench=checkpoint.get('benchmark')
    if not isinstance(bench,dict)or not bench.get('readouts'):return None
    readouts=bench['readouts']
    name=next((k for k,v in readouts.items()if v.get('primary')),next(iter(readouts)))
    summary=readouts[name]
    seeds=list((summary.get('per_seed')or{}).values())
    def gen(path):
        means=summary.get('mean_sd_over_seeds')or{}
        if path in means:return means[path]['mean']
        if path=='human_likeness.short_gap_excess':
            values=[abs(s['human_likeness']['short_gap_fraction_model']-s['human_likeness']['short_gap_fraction_human'])
                    for s in seeds if finite(_get(s,'human_likeness.short_gap_fraction_model'))
                    and finite(_get(s,'human_likeness.short_gap_fraction_human'))]
            return statistics.mean(values)if values else None
        values=[_get(s,path)for s in seeds];values=[v for v in values if finite(v)]
        return statistics.mean(values)if values else None
    teacher=((checkpoint.get('teacher_screen')or{}).get('summary')or{})
    out={}
    for category,_,_,rows in BENCHMARK_V2:
        for key,_,_,source,path,*_ in rows:
            if source=='gen':value=gen(path)
            elif source=='teacher':value=_get(teacher.get('headline')or{},path)
            else:value=_get(teacher.get('onset_head_f1_50ms')or{},path+'.f1')
            out[(category,key)]=value
    complete=gen('accuracy.complete_fraction')
    return dict(values=out,readout=name,complete=complete,
        teacher_complete=bool(teacher.get('complete')),teacher_windows=teacher.get('measured_windows'),
        spread={k:v.get('sd')for k,v in(summary.get('mean_sd_over_seeds')or{}).items()},
        oracle=(summary.get('oracle')or{}).get('mean_note_f1_50ms'),
        failures=_get(seeds[0],'accuracy.failures')if seeds else None)


def benchmark_view(checkpoints):
    """Five categories with direction arrows; trends only across same-readout points."""
    points=[(c,benchmark_values(c))for c in checkpoints]
    points=[(c,v)for c,v in points if v is not None]
    if not points:return []
    readout=points[-1][1]['readout']
    points=[(c,v)for c,v in points if v['readout']==readout]
    view=[]
    for category,label,meaning,rows in BENCHMARK_V2:
        metrics=[]
        for key,name,direction,source,path,unit,*note in rows:
            series=[v['values'][(category,key)]for _,v in points]
            if source=='gen' and category!='accuracy':
                comparable=all(v['complete']==1 for _,v in points[-2:])
            elif source!='gen':
                comparable=all(v['teacher_complete']for _,v in points[-2:])
            else:comparable=True
            if not finite(series[-1])and source=='onset':continue
            metrics.append(metric(key,name,direction,series[-1],series,comparable=comparable,
                unit=unit,note=(note[0]if note else '')))
        view.append(dict(key=category,label=label,meaning=meaning,metrics=metrics))
    last=points[-1][1]
    view.append(dict(key='_identity',readout=readout,opt=points[-1][0].get('optimizer_updates'),
        oracle=last['oracle'],failures=last['failures'],teacher_windows=last['teacher_windows']))
    return view


def checkpoint_metrics(checkpoint):
    records = checkpoint.get('records', [])
    positive = [r for r in records if r['case_role']=='positive']
    silence = [r for r in records if r['case_role']=='natural_silence']
    mean = lambda values: statistics.mean([v for v in values if finite(v)]) if any(finite(v) for v in values) else None
    summary = checkpoint.get('summary') or {}
    return dict(opt=checkpoint['optimizer_updates'],sha=checkpoint.get('sha256',''),
        teacher_positive=mean([r.get('teacher_joint_ce')for r in positive]),
        teacher_silence=mean([r.get('teacher_joint_ce')for r in silence]),
        useful_complete=mean([int(r['prediction']['complete'] and bool(r['prediction']['events']))for r in positive]),
        positive_complete=sum(r['prediction']['complete']for r in positive),positive_count=len(positive),
        all_positive_complete=bool(positive)and all(r['prediction']['complete']for r in positive),
        all_silence_complete=bool(silence)and all(r['prediction']['complete']for r in silence),
        hold_f1=summary.get('hold_joint_f1_50ms'),
        silence_false=summary.get('silence_false_events'),
        per_type=summary.get('per_type',{}),
        teacher_complete=len(records)==summary.get('expected'))


def snapshot(state, contract, history=()):
    rows = sorted(state.records,key=lambda r:r.get('step',0))
    current=state.latest()
    threshold=contract.get('operator_override',{}).get('minimum_util')
    if not finite(threshold):threshold=85.
    checkpoints={}
    protocol=contract.get('quality_eval_protocol',{}).get('panel_sha256')
    for evaluation in [*history,state.evaluation]:
        if evaluation and evaluation.get('protocol_sha256')==protocol:
            for c in evaluation.get('checkpoints',[]):checkpoints[c.get('sha256',str(c['optimizer_updates']))]=c
    ordered=sorted(checkpoints.values(),key=lambda c:c['optimizer_updates'])
    evaluations=[checkpoint_metrics(c)for c in ordered]
    try:benchmark=benchmark_view(ordered)
    except (KeyError,TypeError,ValueError,AttributeError,statistics.StatisticsError):benchmark=[]
    points=[]
    for row in rows:
        lengths=[n for part in row.get('composition',[])for n in part['valid_lengths']]
        modes={mode for part in row.get('composition',[])for mode in part['modes']}
        points.append(dict(opt=(row.get('step',0)+15)//16,loss=row.get('tok_ce'),
            mode=next(iter(modes))if len(modes)==1 else 'mixed',
            length=statistics.mean(lengths)if lengths else None,
            zero=any(0. in(e.get('rates')or[])for e in row.get('source_exposures',[])),
            gpu=row.get('gpu_util_rolling'),grad=row.get('grad'),memory=row.get('gpu_mem_mib'),elapsed=row.get('elapsed')))
    # One point per actual completed optimizer update for resource trends.
    resource=[r for r in rows if r.get('step',0)%16==0]
    loss=[p['loss']for p in points if finite(p['loss'])]
    chunks=[statistics.mean(loss[i:i+80])for i in range(0,len(loss),80)]
    train=metric('loss','训练复合目标',-1,statistics.mean(loss[-80:])if loss else None,chunks,
                 comparable=False,note='最近约10次更新均值；样本组成变化，下降不等于能力提升')
    quality=[]
    if evaluations:
        def q(key,label,direction,**kwargs):
            quality.append(metric(key,label,direction,evaluations[-1][key],
                [c[key]for c in evaluations],**kwargs))
        full_teacher=evaluations[-1]['teacher_complete']
        q('teacher_positive','正例留出 CE',-1,comparable=full_teacher,note='同一固定独立歌；越低越好，仍非自由能力')
        q('teacher_silence','空窗 END CE',-1,comparable=full_teacher,note='越低越好；衡量遇到空窗是否愿意停止')
        q('useful_complete','完整且非空正窗',1,unit='ratio',note='限时完成并有事件；仅工程完成率，不代表谱面正确')
        complete=all(c['all_positive_complete']for c in evaluations[-2:])
        q('hold_f1','Hold 联合 F1',1,comparable=complete,note='头尾联合匹配；未完成输出不判改善')
        q('silence_false','空窗误生成数',-1,comparable=all(c['all_silence_complete']for c in evaluations[-2:]),note='输出截断时计数不可比')
    rolling=current.get('gpu_util_rolling')
    memory=current.get('gpu_mem_mib')
    total=(contract.get('memory_admission',{}).get('results')or[{}])[0].get('total_device_bytes')
    memory_fraction=(memory*2**20/total*100)if finite(memory)and total else None
    resources=[metric('gpu','GPU 最近8区间',1,rolling,[r.get('gpu_util_rolling')for r in resource],unit='%',target=threshold,
                      note=f'原480秒后守卫，低于{threshold:g}%停止；不是越接近100越有产品价值'),
               metric('memory','设备显存占用',-1,memory_fraction,unit='%',target=80,note='安全范围≤80%；缓存占用下降不等于模型改善')]
    speeds=[]
    for a,b in zip(resource,resource[8:]):
        elapsed=(b.get('elapsed')or 0)-(a.get('elapsed')or 0)
        if elapsed>0:speeds.append(60*(b['updates']-a['updates'])/elapsed)
    speed_windows=([statistics.median(speeds[-16:-8]),statistics.median(speeds[-8:])]
                   if len(speeds)>=16 else [])
    resources.append(metric('speed','实际更新 / 分钟',1,speeds[-1]if speeds else None,speed_windows,
        tolerance=.05,note='近期窗口中位数比较，5%内持平；含供数与checkpoint间隙，只衡量效率'))
    active=not state.failed and not state.app_state.endswith('; tasks=0')and state.app_state not in('idle','stopped','not found','0')
    phase='训练中'if active and rows else'准备 / 评测中'if active else'已停止'
    age=time.time()-state.last_record_wall if state.last_record_wall else None
    if active and rows and age is not None and age>90:phase='日志停滞'
    raw_runtime=getattr(state,'runtime',None)
    runtime=({key:raw_runtime.get(key)for key in ('optimizer_updates','checkpoint','checkpoint_sha256','resource_guard')}
             if raw_runtime else None)
    # In-run benchmark v2 headlines (training report benchmark_v2_evaluations), when a run has no
    # separate learning evaluation; same frozen val panel and readout at every point.
    in_run=[e for e in((raw_runtime or{}).get('benchmark_v2_evaluations')or[])
            if e.get('status')=='done'and isinstance(e.get('headline'),dict)]
    readouts={e['headline'].get('readout')for e in in_run}
    bench_headline=[]
    if in_run and len(readouts)==1:
        for key,label,direction,unit in(('accuracy.note_f1_50ms','音符F1@50ms',1,''),
                ('accuracy.note_f1_50ms_minus_null','F1 超出平移基线',1,''),
                ('accuracy.complete_fraction','完整输出比例',1,'ratio'),
                ('hacks.empty_positive_rate','正例空输出率',-1,'ratio'),
                ('condition_response.monotonic_fraction','密度请求单调比例',1,'ratio'),
                ('teacher_gap_foreign_head_time','换歌 CE差·头时间',1,''),
                ('teacher_gap_zero_head_time','清零 CE差·头时间',1,'')):
            series=[e['headline'].get(key)for e in in_run]
            bench_headline.append(metric(key,label,direction,series[-1],series,unit=unit))
    bench_identity=(dict(opt=[e.get('optimizer_updates')for e in in_run],readout=next(iter(readouts)))
                    if bench_headline else None)
    hard_cap=contract.get('runtime_accounting',{}).get('next_process_hard_cap_s')
    remaining=max(0,hard_cap-current['elapsed'])if finite(hard_cap)and finite(current.get('elapsed'))else None
    return dict(updated=dt.datetime.now().astimezone().strftime('%H:%M:%S %z'),call=state.call_id,app=state.app_id,app_state=state.app_state,
        phase=phase,active=active,failed=state.failed,error=state.logged_error,raw_phase=state.phase,
        updates=current.get('updates',state.restored_updates),step=current.get('step'),threshold=threshold,lr=current.get('lr'),grad=current.get('grad'),
        train=train,quality=quality,resources=resources,points=points,evaluations=evaluations,benchmark=benchmark,
        bench_headline=bench_headline,bench_identity=bench_identity,
        checkpoints=state.checkpoints,notes=[('整曲结构、试玩/编辑仍未测' if benchmark or bench_headline else '音频反事实、密度响应、整曲结构、试玩/编辑仍未测'),
        '绿=同口径改善/当前达标；红=变差/未达标；灰=未测、截断或不可比'],
        runtime=runtime,remaining_s=remaining,source_optimizer_updates=contract.get('memory_admission',{}).get('source_optimizer_updates'))


def _display(value,width):
    result='';n=0
    for ch in str(value):
        step=2 if unicodedata.east_asian_width(ch)in'WF' else 1
        if n+step>width:break
        result+=ch;n+=step
    return result+' '*(width-n)


def _spark(values):
    values=[v for v in values if finite(v)][-12:]
    if len(values)<2:return '—'
    lo,hi=min(values),max(values)
    return ''.join('▁▂▃▄▅▆▇█'[min(7,int((v-lo)/max(hi-lo,1e-12)*7.99))]for v in values)


def render_text(view):
    lines=['╭─ BeatFCN · 学习趋势 ─────────────────────────────────────────────────────────╮',
           f'│ {view["phase"]}  ·  真实参数更新 {view["updates"] or "—"}  ·  {view["updated"]}',
           f'│ call {view["call"] or "—"}',f'│ app {view["app"] or "—"} ({view["app_state"]})',
           '╰───────────────────────────────────────────────────────────────────────────╯']
    if not view['active']:lines.append('■ 当前无训练 / nothing is training；保留最后实测快照')
    if view['failed']:lines.append('✕ training reported failure：'+(view['error']or'等待退出原因'))
    elif view['error']:lines.append('! '+view['error'])
    if not view['points']:lines.append('waiting / 准备阶段：'+view['raw_phase'])
    if finite(view['remaining_s']):lines.append(f'本call硬时限剩余约 {view["remaining_s"]/60:.0f} 分钟；原累计训练帽保持')
    def table(title,metrics):
        lines.extend(['',title,'  '+_display('指标',20)+_display('方向',12)+_display('当前',10)+_display('变化',12)+_display('走势',13)+'判定'])
        for m in metrics:
            value=f'{100*m["value"]:.1f}%'if m['unit']=='ratio'and finite(m['value'])else fmt(m['value'],m['unit'])
            delta=m['delta'];change=('↗ 'if delta>0 else'↘ 'if delta<0 else'→ ')+fmt(abs(delta))if finite(delta)else'—'
            direction=('↑ 越高越好'if m['direction']>0 else'↓ 越低越好'if m['direction']<0 else'≈ 参考')if m['target']is None else('≥ 'if m['direction']>0 else'≤ ')+fmt(m['target'],m['unit'])
            lines.append('  '+_display(m['label'],20)+_display(direction,12)+_display(value,10)+_display(change,12)+_display(_spark(m['points']),13)+m['status'])
    table('训练 · 复合目标需按模式/长度比较',[view['train']])
    if view['quality']:
        table('留出能力 · 固定独立歌 / checkpoint 指标',view['quality'])
        last=view['evaluations'][-1]
        lines.append(f'  评测身份 opt {last["opt"]} · SHA {last["sha"][:12]} · 正例完成 {last["positive_complete"]}/{last["positive_count"]}')
        lines.append('  逐类型匹配 F1 ↑：'+ '  '.join(k+'='+fmt(v.get('f1'))for k,v in last['per_type'].items()))
        if not last['all_positive_complete']:lines.append('  ! 自由输出有截断；类型/hold/空窗计数不判改善')
    elif not view.get('bench_headline'):lines+=['','留出能力 · 未测 / 等待固定checkpoint评测']
    if view.get('bench_headline'):
        ident=view['bench_identity']
        table(f'周期评测 · 训练内 benchmark v2（EMA，固定 val 面板，读出 {ident["readout"]}；已测 opt '
              +', '.join(str(o)for o in ident['opt'])+'）',view['bench_headline'])
    categories=[c for c in view.get('benchmark',[])if c['key']!='_identity']
    identity=next((c for c in view.get('benchmark',[])if c['key']=='_identity'),None)
    if categories:
        lines+=['',f'Benchmark v2 · 读出 {identity["readout"]} · opt {identity["opt"]}（↑越高越好 ↓越低越好；每类单独看）']
        for category in categories:
            table(f'{category["label"]} — {category["meaning"]}',category['metrics'])
        if finite(identity.get('oracle')):
            lines.append(f'  ORACLE best-of-K 音符F1 {fmt(identity["oracle"])}（用答案挑选，不是能力）')
        if identity.get('failures'):
            lines.append('  失败计数（首个种子）：'+', '.join(f'{k}={v}'for k,v in identity['failures'].items()))
    table('资源与效率 · 达标范围，不等同于模型能力',view['resources'])
    if view['runtime']:
        stable=(view['runtime'].get('resource_guard')or{}).get('stable_mean')
        lines.append(f'  稳定期总均值 {fmt(stable,"%")} · 来自opt {view["runtime"]["optimizer_updates"]}落盘')
    lines.append(f'  preclip grad {fmt(view["grad"])}（有限/尖峰检查，无单调好坏） · LR {fmt(view["lr"])}（按计划）')
    lines+=['','未测：'+view['notes'][0],
            '网页趋势图：reports/600m-monitor.html（每轮刷新；含分层筛选）',
            'checkpoints: '+(', '.join(view['checkpoints'])or'本call尚未落盘')]
    return '\n'.join(lines)+'\n'


def render_html(view):
    data=json.dumps(view,ensure_ascii=False,allow_nan=False).replace('<','\\u003c')
    template=Path(__file__).with_suffix('.html').read_text()
    return template.replace('__DATA__',data)
