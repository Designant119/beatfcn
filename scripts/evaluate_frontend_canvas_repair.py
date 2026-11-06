"""Zero-update repair of the missing scoring-helper import; weights unchanged."""
import json
import shutil

from beatfcn.model.sample import match_f1
from scripts import run_frontend_canvas_quality_v2 as run
from scripts.run_local_bundle import ROOT,write,sha


def main():
    run.install_paths();p=json.loads(run.PLAN.read_text());r=json.loads(run.REPORT.read_text())
    if r['status']!='evaluate_stopped' or not any('match_f1' in e for e in r['failures']):
        raise ValueError('Expected specific missing-import evaluation failure')
    if list(run.ART.glob('*/*-final.json')):raise FileExistsError('Do not overwrite partial final predictions')
    archive=run.ART/'evaluation-attempt-01';archive.mkdir()
    shutil.copy2(run.REPORT,archive/'report.json')
    shutil.copy2(ROOT/'artifacts/local-frontend-comparison-v2-evaluate.log',archive/'evaluation.log')
    weights={a:sha(run.ART/a/'quality-0900.pt') for a in ('pupu','mert')}
    run.engine.verify(p)
    if run.certificate()!=p['health_gate_revision']:raise ValueError('Health certificate changed')
    # The frozen engine already calls this exact existing helper. Installing
    # its missing name changes no weights, predictions, cases or scoring math.
    run.engine.match_f1=match_f1
    r['evaluation_repair']=dict(reason='Missing match_f1 import before any final case was saved',optimizer_updates=0,
        original_attempt_status='evaluate_stopped',failed_attempt_wall_s='not recorded',
        accounting_allowance_for_failed_attempt_s=300,remaining_evaluation_wall_limit_s=300,
        weight_sha256_before=weights,source_sha256={name:sha(ROOT/name) for name in (
            'scripts/evaluate_frontend_canvas_repair.py','beatfcn/model/sample.py','beatfcn/model/evaluation.py')})
    p['evaluation_wall_s']=300  # reduce remaining wall budget; leave frozen plan untouched
    r['status']='trained';run.save(r)
    run.engine.evaluate(p,r)
    if any(sha(run.ART/a/'quality-0900.pt')!=s for a,s in weights.items()):raise ValueError('Evaluation changed weights')
    r['evaluation_repair']['weight_files_unchanged']=True;run.save(r)
    run.journal('首评价因缺少match_f1名称在首窗打分中止，原日志/report归档。仅补既有指标函数依赖，0更新、相同案例/分数数学/权重。失败时间未记录，保守划300秒额度后剩余评价上限300秒；修复后的全评价和前后权重SHA均保存。')


if __name__=='__main__':main()
