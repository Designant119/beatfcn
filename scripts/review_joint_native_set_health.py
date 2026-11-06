"""Close two engineering/health terminals without inventing held-out quality."""
from collections import Counter
import json
from pathlib import Path
import statistics

from scripts.run_local_portfolio_scale import sha, write


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/local-integration-v1'


def main():
    raw='reports/local-integration-v1/joint-native-set.json'
    first=json.loads((ROOT/raw).read_text())
    fixed='reports/local-integration-v1/joint-native-set-v1a.json'
    second=json.loads((ROOT/fixed).read_text())
    predictions='artifacts/joint-native-set-v1a/health-predictions.json'
    rows=json.loads((ROOT/predictions).read_text())
    summary=[]
    for r in rows:
        p=Counter(e['onset_us'] for e in r['prediction']['events']);t=Counter(e['onset_us'] for e in r['target'])
        summary.append(dict(index=r['index'],target_events=len(r['target']),predicted_events=len(r['prediction']['events']),
            target_unique_heads=len(t),predicted_unique_heads=len(p),target_max_same_time=max(t.values(),default=0),
            predicted_max_same_time=max(p.values(),default=0),full50=r['scores']['full_note_f1'],onset10=r['scores']['onset_10ms']['f1']))
    positive=[r for r in summary if r['target_events']]
    stats=dict(per_request=summary,mean_full50=statistics.mean(r['full50'] for r in positive),
        mean_onset10=statistics.mean(r['onset10'] for r in positive),predicted=sum(r['predicted_events'] for r in summary),
        target=sum(r['target_events'] for r in summary),head_nll_start=[r['head'] for r in second['health_curve'][0]['parts']],
        head_nll_end=[r['head'] for r in second['health_curve'][-1]['parts']],
        tail_nll_start=[r['tail'] for r in second['health_curve'][0]['parts']],tail_nll_end=[r['tail'] for r in second['health_curve'][-1]['parts']])
    write(FOLDER/'joint-native-set-health-analysis.json',stats)
    narrative='''# 原生集合联合候选：工程失败与真实健康失败分开结案

## 分层现象

v1完成4000训练请求无损审计，148932事件曝光，最多227事件/窗、12事件/40ms，19575额外完全重复事件与1513跨core hold曝光保留。实际10,675,610参数。CPU forward/backward已走到损失，MPS在`aten::index_copy.out`停止；健康与质量优化均0。此版本未检验可拟合性、独立歌质量或条件，不给它填F1=0。

v1a仅修后端不支持的稀疏插入为incidence矩阵乘法；独立CPU index_copy参照对实际MPS前向/梯度回归通过。完整CPU/MPS kind/head/query最大差分别1.32e-6/4.30e-6/2.87e-6，loss差0，梯度最大差3.36e-7。100步真实5窗健康278.31秒，driver约2.69GB、无非有限数或资源故障；五窗自由输出均合法完整、空窗0事件。质量更新0，无独立歌生成，不能把健康F1当新基线。

健康loss从5.8593降至1.4984，残余约25.57%，未过15%合同。四个正窗full50宏平均约0.3715、onset10约0.3595，未过预注册拟合线。329目标发出383事件，不能称全空或全部类型消失。

## 主失败形态：数量接近，头的多重性分布失真

3817的87目标/73独立头，发出105/20独立头，预测最大同刻17而目标最大2；1283为91/80对95/26，最大16对4；7457为88/23对79/15，最大13对12；3851为63/45对104/22，最大15对2。7457的真实高多重性较容易拟合（full50=.6467），其余约.226–.311。多个查询反复选择少数峰，集中在真的某些头上，导致一对一匹配召回及精度都差。最近真头距离为0不能充当多重集精度。

训练匹配头NLL降至5.124/6.049/2.078/4.205，尚有明显残差。相对尾端NLL在有hold的三窗降到.193/.076/.028，说明相对尾端条件目标在匹配监督下可拟合，不能据自由完整hold分低一并否掉。自由生成使用自身头；错误头上的尾端尚未隔离。空窗0保留为有限拟合发现，不外推独立歌静默。

## 可检验解释与对照边界

最受支持的解释是本实现的查询竞争/头定位学习没有在100步内分离足够的事件，而不是旧count尾秩压空：旧count与tail-rank已彻底移除，硬输出数量没有消失，错在头上的集中。匹配保证目标分配到不同query，并不保证这些query自由argmax落到不同或正确的头；同刻允许重复，因此不能事后去重修分。

这不唯一证明查询对称性是根因。匹配每步改变、类型均衡损失与自然频率的差异、全窗约8000地址的粗细学习难度、lr=.002健康动力学都仍可能参与。只有一套联合配方，不能分别否定SSD、attention、直接song/reference或集合方法族。容量不足没有配对证据。真实原生时钟/支持及CPU/MPS等价已排除这一批可见的表示溢出、label删改与后端数值错误；未测声学错位/异歌和参考对照，因为没有正式训练。

最小后继对照：在同五窗/同100更新预算，让查询带可学习的物理时间寻址初值（所有query仍可访问全部原生地址，无硬分箱/配额），对原完整模型重训；记录固定初始随机数、匹配切换率、query间头分布相似度、预测多重性和匹配头NLL。若头分布分离且自由拟合大幅改善，支持可寻址性/分配问题；若仍失败，在相同预算检验S-D1规范格内联合law，不直接加宽。对尾端的独立归因需固定已预测头/同一权重，比较条件尾端与屏蔽头信息，并逐匹配hold报告，不用oracle补主结果。

## 经验、保留与禁止重复

关闭这两个版本的直接晋升：v1因明确MPS算子不可运行，v1a因真实局部可拟合性不足且已出现可测的query头集中；正式独立歌质量仍未检验。修复后数值等价、原生无损、多重集和相对尾端条件拟合予以保留。禁止启用MPS CPU回退、去重复、抬slot数、偷降健康门、续训原健康权重或直接加宽来掩盖当前失败。

后继先由C11审查可检验的查询寻址/匹配解释及完整竞争law，优先修合同而不是绑定所有建议。下次健康失败若需要冻结权重干预，预算内持久化一份失败模型：本次健康权重按原磁盘合同丢弃，故不能冒充已经完成冻结查询消融；这条证据保留不足必须修复。

原raw报告、预测、源码快照和失败状态均保留，未改门。此结案是实现/预算内健康边界，不是独立歌曲质量结论，亦不授权scaling。云调用及费用均0。
'''
    md=FOLDER/'joint-native-set-mechanism-review.md';md.write_text(narrative)
    for ident,report,health_layer in [('joint-native-set-v1',raw,'MPS unsupported index_copy;health0/quality0'),
                                      ('joint-native-set-v1a',fixed,'CPU/MPS pass;100health fitting failed;quality0')]:
        code=f'artifacts/{ident}/source/beatfcn/train/joint_native_set.py'
        review=dict(candidate=ident,analysis_status='complete',identity={p:sha(ROOT/p) for p in (report,code,str(md.relative_to(ROOT)))},
            layers=dict(health=health_layer,quality='not_run;no independent-song quality evidence',structure_conditions='Known-empty and duplicates representation pass;no learned condition/reference causal evaluation'),
            mechanism_hypotheses=[
                dict(id='backend_operator',observation=first['failures'][0]['error'],mechanism='unsupported MPS index_copy, not a tensor-value divergence',
                    prediction='Equivalent incidence multiplication executes with matching gradients',evidence=[raw,fixed,'tests/test_joint_native_set.py'],judgment='Confirmed implementation blocker;replacement actual parity passed'),
                dict(id='query_head_crowding',observation='329targets/383predictions;head multiplicity17vs2 and16vs4;positive head NLL remains2.08–6.05',
                    mechanism='Matching of distinct queries does not ensure distinguishable free head modes within100updates',prediction='Physical query addressing under matched health budget should separate query distributions if this dominates',
                    evidence=[fixed,predictions,'reports/local-integration-v1/joint-native-set-health-analysis.json'],judgment='Supported observed failure shape;query symmetry versus assignment/optimization not uniquely isolated')],
            controls=['Native representation4000requests','CPU/MPS actual forward/backward','Exact sparse-insertion independent regression','Empty health request0events','Per-field losses and emitted same-time multiplicity'],
            missing_controls=[dict(question='Query competition versus insufficient optimization/clock addressing',comparison='Matched100step query-addressed versus original five-window fit,save matching churn and query distributions',decision_if='Separation with same supervision supports addressing;otherwise next complete canonical law,not automatic widening'),
                dict(question='Independent acoustic or reference utility',comparison='After future admitted fresh quality model,actual input roll/foreign/reference removal with requests fixed',decision_if='Only claim learned utility if counterfactual direction and per-type scores support it;current quality remains unmeasured')],
            lessons=['Native multiset and relative head-conditioned tail retained','No fallback/dedup/quota','Save one failed-health weight if frozen interventions required'],
            successor_contract=dict(required_changes=['Supported MPS sparse insertion','Address observed query concentration or change complete generating law before quality','Preserve one necessary failure checkpoint within1GiB'],
                forbidden_repeats=['No silent CPU fallback','No deduplication','No healthy/quality promotion from numerical parity','No continuation of failed health weights']),
            closure_reason='Close exact v1/v1a promotion only: the first cannot execute a required MPS operator, and its corrected implementation passes numerical regression but fails local free fitting with concentrated head modes after100 updates. This has a concrete observed mechanism and a matched addressing control to discriminate it; no independent-song quality or architecture-family limit was tested.',
            source_record=str(md.relative_to(ROOT)))
        path=FOLDER/f'{ident}-mechanism-review.json';write(path,review)
        registry_path=ROOT/'reports/local-evolution-v1/mechanism-reviews/index.json';registry=json.loads(registry_path.read_text())
        if not any(r['id']==ident for r in registry['candidates']):
            registry['candidates'].append(dict(id=ident,analysis_status='complete',review=str(path.relative_to(ROOT)),review_sha256=sha(path),
                markdown=str(md.relative_to(ROOT)),narrative_sha256=sha(md)))
            write(registry_path,registry)
    print(json.dumps(stats,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
