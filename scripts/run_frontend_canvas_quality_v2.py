"""Explicit gate-policy revision, reusing archived zero-update health evidence.

The failed v1 gate remains untouched. Only process paths/bookkeeping and the
engineering admission certificate change; v1's model/train/eval equations run
unchanged. No health weights or optimizer state enter quality training.
"""
import argparse
import json
import signal

from scripts import run_frontend_canvas_comparison as engine
from scripts.run_local_bundle import ROOT,write,sha,configure


ID='local-frontend-comparison-v2'
PLAN=ROOT/'bench/rhythm-v2/local-frontend-comparison-v2.json'
REPORT=ROOT/'reports/local_frontend_comparison_v2.json'
ART=ROOT/'artifacts/local-frontend-comparison-v2'
V1_REPORT=ROOT/'reports/local_frontend_comparison_v1.json'
AUDIT=ROOT/'artifacts/local-frontend-comparison-v1/health-decision-audit.json'


def journal(text):
    with (ROOT/'EXPERIMENT_LOG.md').open('a') as f:f.write('\n\n'+ID+' '+text+'\n')


def save(report):
    write(REPORT,report);p=ROOT/'reports/rhythm_v2_budget.json';b=json.loads(p.read_text())
    b['local_cpu_diagnostics'][ID]={'status':report['status'],'updates':report['updates'],
        'new_health_updates':report['health_updates'],'borrowed_health_source':'reports/local_frontend_comparison_v1.json',
        'wall_s':report.get('wall_s',{}),'new_cloud_spend_usd':0,'new_cloud_compute_calls':0,
        'report':str(REPORT.relative_to(ROOT)),'resource_target':'Modal H100 ~0.25B trainable event model'}
    write(p,b)


def install_paths():
    # Process-local dispatch of the frozen engine; no source monkey-edit or
    # transformation of model equations, data, loss, or readout occurs here.
    engine.PLAN=PLAN;engine.REPORT=REPORT;engine.ART=ART;engine.journal=journal;engine.save=save


def certificate():
    old=json.loads(V1_REPORT.read_text());audit=json.loads(AUDIT.read_text())
    if old['status']!='health_failed' or any(old['updates'].values()):raise ValueError('Expected archived untrained v1 failure')
    if max(old['cpu_mps_parity'].values())>1e-4:raise ValueError('Device parity')
    for item in audit:
        if item['count_correct']!=item['count_total']:raise ValueError('Count field not fitted')
        for name in ('offset','tail_block','tail_cell','tail_ms'):
            if item[name]['correct']!=item[name]['total']:raise ValueError('Native event fields not fitted')
    raw=json.loads((ROOT/'artifacts/local-frontend-comparison-v1/health-predictions.json').read_text())
    for rows in raw.values():
        for row in rows:
            if not row['prediction']['complete']:raise ValueError('Incomplete health')
            if row['target']:
                if row['scores']['full_note_f1']<.95 or (row['scores']['onset_10ms']['f1'] or 0)<.95:
                    raise ValueError('Global free-readout fit too weak')
            elif row['prediction']['events']:raise ValueError('Health empty false positives')
    return dict(passed=True,criterion='All native discrete count/offset/tail fields exact; actual global event and onset10 F1>=.95; empty0; finite/parity checks. Per-small-type .9 retained as unmet v1 quality risk, not engineering veto.',
        policy_revision_after_observation=True,original_v1_gate_still_failed=True,
        old_report_sha256=sha(V1_REPORT),decision_audit_sha256=sha(AUDIT),health_weights_reused=False)


def prepare():
    cert=certificate()
    if PLAN.exists() or REPORT.exists():raise FileExistsError('Preserve v2')
    engine.prepare()  # Same deterministic data/config/cache/initialization audit.
    p=json.loads(PLAN.read_text());p['id']=ID;p['health_gate_revision']=cert
    p['source_sha256'].update({str(V1_REPORT.relative_to(ROOT)):sha(V1_REPORT),
        str(AUDIT.relative_to(ROOT)):sha(AUDIT),'scripts/run_frontend_canvas_quality_v2.py':sha(ROOT/'scripts/run_frontend_canvas_quality_v2.py')})
    write(PLAN,p)
    r=json.loads(REPORT.read_text());r.update(id=ID,status='health_passed',health_gate_revision=cert,
        borrowed_health_updates={'pupu':100,'mert':100},health_updates={'pupu':0,'mert':0})
    save(r);journal('原v1每稀有类型.9门未过原样保留；0更新核验原生计数/偏移/尾字段全精确、实际整体F1>=.95/空窗0，登记工程门修订。复用证据，不复用健康权重或重复计费。接下来按原数学合同fresh两臂900更新。')
    print(json.dumps(dict(prepared=True,health_gate_revision=cert)),flush=True)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','train','evaluate']);args=ap.parse_args();install_paths()
    if args.stage=='prepare':prepare();return
    p=json.loads(PLAN.read_text());r=json.loads(REPORT.read_text());engine.verify(p);configure()
    if certificate()!=p['health_gate_revision']:raise ValueError('Archived certificate changed')
    def stop(sig,_):raise InterruptedError(f'Signal {sig}')
    signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
    try:(engine.train if args.stage=='train' else engine.evaluate)(p,r)
    except BaseException as exc:
        if engine.LIVE:
            for arm,model in engine.LIVE['models'].items():
                engine.checkpoint(model,engine.LIVE['opts'][arm],arm,'quality-stopped',r['updates'][arm])
        r['status']=args.stage+'_stopped';r['failures'].append(f'{type(exc).__name__}: {exc}');save(r);raise


if __name__=='__main__':main()
