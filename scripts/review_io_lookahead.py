"""Apply frozen I02 CPU runtime gates; never promote a model from an I/O test."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def decide(plan,current,ahead):
    reasons=[]
    if not current.get('complete') or not ahead.get('complete'):reasons.append('incomplete replay')
    if current.get('identities')!=ahead.get('identities'):reasons.append('feature identities differ')
    if not all(r.get('same_sample_order') and r.get('consumed_batches')==plan['batches'] for r in (current,ahead)):
        reasons.append('sample order or count differs')
    before=current.get('pool_stats',{}).get('wait_s',0);after=ahead.get('pool_stats',{}).get('wait_s',0)
    wait_ratio=after/before if before>0 else None
    wall_ratio=ahead.get('loop_wall_s',0)/current['loop_wall_s'] if current.get('loop_wall_s',0)>0 else None
    if wait_ratio is None or wait_ratio>plan['gate']['maximum_wait_time_ratio']:reasons.append('Future wait did not halve')
    if wall_ratio is None or wall_ratio>plan['gate']['maximum_total_wall_ratio']:reasons.append('loop gain below twenty percent')
    if ahead.get('conservative_feature_gib',float('inf'))>plan['gate']['maximum_conservative_feature_gib']:
        reasons.append('feature memory planning reserve exceeded')
    return dict(passed=not reasons,reasons=reasons,future_wait_ratio=wait_ratio,loop_wall_ratio=wall_ratio,
        same_feature_payloads=current.get('identities')==ahead.get('identities'),
        model_quality_verdict=None,gpu_performance_verified=False)


def main():
    p=ROOT/'bench/rhythm-v2/i02-io-lookahead-v1.json';plan=json.loads(p.read_text())
    reg=json.loads((ROOT/'reports/i02_registration.json').read_text())
    if hashlib.sha256(p.read_bytes()).hexdigest()!=reg['plan_sha256']:raise ValueError('frozen plan changed')
    rows={arm:json.loads((ROOT/f'reports/i02_{arm}.json').read_text()) for arm in plan['arms']}
    for arm,r in rows.items():
        if (r['arm']!=arm or r['source_sha256']!=reg['source_sha256'] or r['plan_sha256']!=reg['plan_sha256']
            or r['sequence_sha256']!=reg['sequence_sha256']):raise ValueError('paired identity differs')
    decision=decide(plan,rows['current'],rows['ahead'])
    report=dict(schema='beatfcn.io_lookahead_review.v1',decision=decision,
        plan_sha256=reg['plan_sha256'],source_files_sha256={k:hashlib.sha256((ROOT/f'reports/i02_{k}.json').read_bytes()).hexdigest() for k in rows},
        summaries={k:{field:r.get(field) for field in ['complete','prefetch','consumed_batches','loop_wall_s','function_wall_s','pool_stats','conservative_feature_gib','distinct_audio']} for k,r in rows.items()},
        limitations=plan['limitations'])
    out=ROOT/'reports/i02_review.json';out.write_text(json.dumps(report,indent=2)+'\n')
    text=['# I02 预取前视窗口的实际文件回放','','同一实际序列、无 GPU、无模型更新；CPU sleep 仅模拟 .13s/批的计算间隔。','',
          '| 前视批数 | 完成批次 | 循环耗时 s | Future 等待 s | 特征内存规划 GiB |','|---|---:|---:|---:|---:|']
    for r in rows.values():
        text.append(f"| {r['prefetch']} | {r['consumed_batches']} | {r['loop_wall_s']:.3f} | {r['pool_stats']['wait_s']:.3f} | {r['conservative_feature_gib']:.3f} |")
    text+=['','CPU 保留关：'+('通过，仍须实际 GPU 对照。' if decision['passed'] else '未通过，不直接应用到训练。')]
    text+=['- '+x for x in decision['reasons']]+['']+['- '+x for x in plan['limitations']]
    out.with_suffix('.md').write_text('\n'.join(text)+'\n');print(json.dumps(decision))


if __name__=='__main__':main()
