"""Append the owner's completed mechanism judgments; never alter experiment results."""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json

from scripts.archive_council import ROOT, append_event, events
from scripts.validate_mechanism_reviews import validate_review


FOLDER = 'reports/local-portfolio-v1/mechanism-reviews-v2'
GROUPS = {
    'local-whole-song-arrangement-v1': ['C01-09','C01-16','C01-17','C04-S06','C05-S02'],
    'local-grouped-chart-v2': ['C00-01','C00-03','C01-18','C04-S05','C05-G02'],
    'local-lattice-complete-v1': ['C01-07','C01-08','C01-14','C01-15'],
    'local-noisy-proposal-refiner-v2': ['C01-10'],
    'local-event-code-flow-v2': ['C04-A05'],
    'local-canonical-package-quality-v2': ['C02-04','C02-05'],
    'local-ssm-event-package-v1': ['C04-S04'],
    'local-group-diffusion-quality-v1': ['C04-S03','C05-G01','C04-A03','C04-G03'],
    'package-2000': ['C04-S02'],
    'mel-2000': ['C04-G04'],
    'direct-song-full': ['C03-19'],
    'local-attention-control-quality-v1': ['C04-G01','C04-S01'],
    'ssd-reference1000': ['C04-A02','C04-G02','C05-G03'],
}
SPECIAL = {
    'C00-01': (
        '纠正旧“候选物理音频完整实现”说法：本次 AR 实际采用 frame 音频项加 phase 项，'
        '固定 query 时格内候选的局部音频项在 log-odds 中抵消，24格自动微分均为零。'
        '原建议的参数匹配单开关研究未执行，不能由本次 AR 冒充。该历史单开关合同在本池明确不再执行：'
        '它继续保留已有时间/类型历史反馈，而现已同时发现候选内声学交互盲点与贪心边界吸引子，'
        '只移动粗音频地址不足以检验这两个机制。真正 frame×offset 声学交互列为新合同的先行实现要求；'
        '其效果尚未测，原建议不是被实测整体证伪。'),
    'C00-03': '历史内容 dropout 已实施；scheduled sampling 未实施且原设计否决仍保留。当前完整 AR 的失败不能归因于 dropout 单组件。',
    'C01-16': '本条 tested_failed 只指完整 spectral/flux 隐状态 B 配方；其它从零 mel 访问混合路线已复验支持，不能否定可训练前端。',
    'C04-S06': '本条只关闭完整 mel/flux＋状态事件包合同；不能把另一个无 flux 的 mel-only 或成功的 access-mixed 当同一实验。',
    'C05-G03': '同权重比较本身已完成，采纳状态不变；它没有支持事后改主读出。数量 DP 的新增纯先验反例针对 A2/SSM 包，不能移植到旧 SSD 独立槽头。',
}


def main():
    registry=json.loads((ROOT/FOLDER/'index.json').read_text())
    reviews={}
    for row in registry['candidates']:
        data=(ROOT/row['review']).read_bytes()
        if hashlib.sha256(data).hexdigest()!=row['review_sha256']:
            raise ValueError('Review hash mismatch: '+row['id'])
        review=json.loads(data)
        if validate_review(review):raise ValueError('Incomplete mechanism review: '+row['id'])
        reviews[row['id']]=(row,review)
    all_events=events(ROOT);latest={r['id']:r for r in all_events}
    counts=defaultdict(int)
    for row in all_events:counts[row['id']]+=1
    now=datetime.now(timezone.utc).isoformat();appended=0
    for candidate,ids in GROUPS.items():
        index,review=reviews[candidate]
        for ident in ids:
            old=latest[ident]
            if old.get('mechanism_review_version')=='owner-v2':continue
            row=dict(old);counts[ident]+=1
            row.update(event_id=f'{ident}.{counts[ident]}',recorded_at=now,
                analysis_status='complete',mechanism_review_version='owner-v2',
                mechanism_review=index['review'],
                rationale=review['closure_reason']+' '+SPECIAL.get(ident,'')+
                    ' 三层数字、已有/缺失判别对照、后继改动与禁止重复已写入完整复盘；原数值不改。',
                evidence=list(dict.fromkeys([index['markdown'],index['review'],
                    'reports/local-portfolio-v1/successor-contracts-v2.md',*old['evidence']])),
                recording_basis='Owner-required mechanism re-review with frozen-weight controls; zero optimizer updates, no new council.',
                context='All 23 fixed candidate postmortems complete; no active training or cloud task.',
                revisit='Only a new contract addressing the shared failure and completed pretraining probes can launch; no automatic extra steps/seed/quotas.')
            if old['status']=='deferred':row['status']='tested_failed'
            if ident=='C00-01':row.update(status='rejected',implemented_alternative_result='tested_failed',original_single_switch_study='not_executed_explicit_design_rejection')
            row['result_status']=row['status']
            append_event(ROOT,row);appended+=1
    research_path=ROOT/'reports/local-portfolio-v1/research-suggestions.jsonl'
    research=[json.loads(v) for v in research_path.read_text().splitlines()]
    research_latest={r['id']:r for r in research}
    research_pairs={'R-LFM3-01':['access-global','access-mixed'],
                    'R-LFM3-02':['music-depth-wide8','music-depth-deep16'],
                    'R-LFM3-03':['music-ssl-supervised','music-ssl-matching']}
    for ident,pair in research_pairs.items():
        old=research_latest[ident]
        if old.get('mechanism_review_version')=='owner-v2':continue
        row=dict(old);n=sum(r['id']==ident for r in research)+1
        row.update(event_id=f'{ident}.{n}',recorded_at=now,analysis_status='complete',
            mechanism_review_version='owner-v2',
            rationale=' '.join(reviews[k][1]['closure_reason'] for k in pair)+
                ' 对照与后继合同见两臂独立复盘，原配对假设失败不改。',
            evidence=[reviews[k][0]['markdown'] for k in pair]+old['evidence'],
            recording_basis='Owner-required mechanism analysis, actual frozen-weight field/retrieval/ablation controls, no additional training.')
        with research_path.open('a') as stream:stream.write(json.dumps(row,ensure_ascii=False)+'\n')
        appended+=1
    print(f'Appended {appended} mechanism decisions; raw quality files unchanged.')


if __name__=='__main__':main()
