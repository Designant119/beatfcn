"""Close resource-limited terminals while preserving their positive health facts."""
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT=Path(__file__).resolve().parents[1]
FOLDER=ROOT/'reports/local-integration-v1'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    raw='reports/local-integration-v1/native-cell-chain.json';r=json.loads((ROOT/raw).read_text())
    h=json.loads((ROOT/'artifacts/c11-native-cell-chain-v1/health-diagnostics.json').read_text());positive=[x for x in h if x['target']]
    probe='artifacts/c12-context-shape-probe-v1/summary.json';p=json.loads((ROOT/probe).read_text())
    n=r['quality_updates']*4;data=json.loads((ROOT/'artifacts/c11-native-cell-chain-v1/data-identity.json').read_text())
    facts=dict(health_updates=100,quality_counter=r['quality_updates'],health_full_macro=statistics.mean(x['scores']['full_note_f1'] for x in positive),
        health_onset10_macro=statistics.mean(x['scores']['onset_10ms']['f1'] for x in positive),
        recorded_applied_prefix_windows=n,recorded_applied_prefix_events=sum(x['events'] for x in data['checks'][:n]),
        seen_windows_including_unapplied=r['exposure']['windows'],seen_events_including_unapplied=r['exposure']['events'],
        quality_wall_s=r['quality_wall_s'],process_wall_s=r['process_wall_s'],last_resources=r['quality_curve'][-1],
        process_sample_physical_footprint_raw='33.9G',sample_global_swap_used_mib=15048.69,
        stop='Parent SIGTERM safety stop before registered6000squalitylimit;actualGPUdriver/diskguards had not tripped',
        checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],quality_predictions='not_run',runtime_probe=p)
    write(FOLDER/'native-cell-runtime-facts.json',facts)
    text='''# 原生格内链与固定context零更新对照：资源层结案

## 工程、健康、质量与条件分层

完整L1实际10,240,592参数，93份源码闭包和588输入payload冻结，4000原目标SHA一致。8项支持/匹配/概率回归通过；真实CPU/MPS loss差0、梯度最大差7.16e-7，无missing gradient。100健康438.47秒全部门通过：四正窗full50均值约.99097、onset10同值；3817的87真/90预测full.97175，1283和7457完全拟合，3851为63/64、full.99213，空窗0。真count/真前缀四窗字段模式F1为1。保留局部可拟合性，不当独立歌证据。

质量fresh重置，健康权重没有复用。原1000×4流在计数50更新时由父级SIGTERM安全停止，保存quality-interrupted.pt；不是在看到独立歌分数后挑checkpoint。质量墙639.15秒，进程1190.17秒。原6000秒尚未耗尽，GPUdriver与磁盘硬线也未触发，不能改写为“原预注册守卫自动通过/失败”。

触发人工提前停止的是实际总内存与吞吐：sample报告本进程physical footprint33.9G；全系统swap used约15048.69MiB。最近50步记录MPSdriver4,587,634,688 bytes、current tensor280,204,544，说明只看driver/current不足以保护该机器。50步626.32秒，前25与后25均约每步12秒，1000步线性外推超过原6000秒；外推不是已经测到1000步。sample调用栈同时出现SDPA math与MPSGraph创建/编译，不能据此量化是哪一层拥有多少总内存，也不能把libtorch_cpu库名误称CPU fallback。

目前没有32开发窗预测、未见空窗/参考/条件/音频置换质量结果。不能填F1=0或判定模型学不到非tap。quality_updates=50对应计划前200请求/7075事件；seen账203请求/7274事件包括中断批尚未应用的3请求，不能全算优化曝光。信号可能打断一个步骤，state_dict没有optimizer，故不保证原子optimizer边界，不用于自动精确续训。

## 实际区分对照

1. 新增own-process TASK_VM_INFO physical-footprint读数，依据本机SDK结构；与编译SDK的同进程C参考逐字节数值差0。该指标不同于RSS和MPSdriver。
2. 新增严格mask的post-SSD cross-attention padding：query200/key2048；只在最后pointwise cross-attention/FFN中补行，SSD不推进padding。CPU两组变形状前向/输入梯度/参数梯度回归通过，dropout固定0，溢出显式失败。
3. 两臂各在独立冷进程、相同完整初始权重、同前32请求计划、无optimizer更新，physical cap10GiB/arm300s/磁盘1GiB及15GiB保底。两臂均仅完成15请求，在第16请求资源停止。动态臂触发时footprint10,828,898,464 bytes，固定context臂10,805,780,304；二者driver约2.6GB。共同请求countlogits最大差1.91e-6、loss9.54e-7、首请求全部梯度4.77e-7。完整对照86.93秒、云0。

这排除了“只固定新增context层的形状就足够解决总内存增长”，没有证明新增层完全无开销，也没有排除其他MPSGraph路径。当前最有根据的范围是随真实请求形状/端点/索引变化增长的执行/框架资源，而非权重参数或已验证无损域。

## 竞争解释与下一判别

候选解释包括：其余SSM/song/reference的动态尺寸图缓存；成员/端点循环中按Python标量选择和多形状算子形成大量图；框架分配/压缩内存保留；活跃计算图未释放或其他主机分配。仅31个样本范围的padding对照不能将其中某项定为唯一原因。

最小下一步是相同数据/权重/预算的串行零更新阶段剥离：encode-only、完整frame/count forward-backward、加成员、加endpoint逐层记录physical-footprint增量，保持每条源输入和梯度范围声明；再针对增长路径做tensor-index/gather、固定完整branch形状或独立后端的一个改动，并验证输出/梯度1e-4。不能在未定位时同时改生成law、换前端、加宽或延长。若仅MPS后端异常，<=6美元的独立CUDA机制探针可另行注册及记账，不等于0.25B授权。

## 执行缺项与保留

P2的三种计数/前缀诊断及字段概率已执行，但原建议的“固定机器state/head只改字段音乐”与健康audio-zero/shift控制未接全，健康权重又已丢弃，不能补造旧时点结果。记为本次明确实施缺项；它不抹去自由拟合数值，也不提供音乐依赖证据。后继在代码/保存策略中提前接齐，需冻结干预的健康权重不能先丢。质量提前资源停止，G2/G3本来未执行，不能用缺控制给其质量判负。

关闭当前具体本机运行的直接晋升：生成law可局部拟合，但当前执行路径在原预算下没有完成独立歌检验。context-only padding也关闭为“不充分的资源修复”，正确mask/参数等价可回收。主生成路线/SSD/GRU/精确COPY均不作方法族否决。下一候选优先修执行并扩展总physical-footprint守卫，不靠提高上限、禁用watermark或静默CPU fallback继续跑。C12重新独立审查，Argon缺席，云仍0/6USD。
'''
    md=FOLDER/'native-cell-runtime-review.md';md.write_text(text)
    for ident,paths,health in [
        ('c11-native-cell-chain-v1',[raw,'artifacts/c11-native-cell-chain-v1/quality-interrupted.pt'],'100health passed;50quality counter then parent resource safety stop'),
        ('c12-context-shape-probe-v1',[probe,'artifacts/c12-context-shape-probe-v1/dynamic.json','artifacts/c12-context-shape-probe-v1/static_context.json'],'Two cold processes;zero optimizer;both stopped at own-process10GiB cap')]:
        review=dict(candidate=ident,analysis_status='complete',identity={p:sha(ROOT/p) for p in paths+[str(md.relative_to(ROOT))]},
            layers=dict(health=health,quality='Independent-song quality not measured;do not fill zeroF1',structure_conditions='Actual multiset/COPY/numeric support andhealth fit retained;reference/audio/music causal utility unmeasured'),
            mechanism_hypotheses=[dict(id='total_process_memory',observation='33.9G process footprint while MPSdriver4.59GB;global swap15GB',mechanism='Framework/host memory outside tensor/driver-only guard grows under varied requests',prediction='Own-process physical guard catches danger before oldguard',evidence=['reports/local-integration-v1/c11-quality-runtime-sample.txt','reports/local-integration-v1/process-memory-validation.json',probe],judgment='Observed resource problem;not uniquely assigned to layer or cache'),
                dict(id='context_shape_only',observation='Both cold arms stop on request16;common count/loss/gradient parity1e-4',mechanism='Newcontextlayer shape alone might explain runaway',prediction='Paddingonlycontext would allow moreboundedrequestsif sufficient',evidence=[probe,'tests/test_static_context_cell_chain.py'],judgment='Sufficiency rejected within32request probe;other dynamic paths/allocator retained')],
            controls=['Exactdata/initialweights','CPU/MPSnewfieldparity','Healthtrue/machineprefix','CPUtaskvmSDKreference','Pairedcoldprocesscontextpadding'],
            missing_controls=[dict(question='Which phase owns growth?',comparison='Same-seed/data encode-only,count,member,endpoint phases with own-process accounting',decision_if='Repair the growing path only andvalidategradients beforefreshquality'),
                dict(question='MPS-specific or common algorithm execution?',comparison='Afterlocalmechanismgate,registered<=6USD CUDApairednumeric/runtimeprobe',decision_if='Backend-specificbenefit preserveslaw;not a 0.25B scalingauthorization'),
                dict(question='Field music dependence/independentquality?',comparison='Nexthealthyfinalweight fixed-head/state field interventions plustrueinput/roll/foreign/empty/reference',decision_if='No musicalability claims until actual proper-score/free-event controls')],
            lessons=['Driver/RSS are not fullprocess footprint','Fixed-five health does not test varied-shape memory','Seenrequests are not allappliedexposures','Save neededstate before discarding health','Keep native law positivefindings'],
            successor_contract=dict(required_changes=['Own-processphysicalcap10GiB beforequality and perrequest','RealvariedrequestFWDBWD ingressprobe inisolatedprocess','Fix observedexecutionpath with output/gradient parity','Complete fieldintervention records beforehealthweightrelease'],
                forbidden_repeats=['No raising memory/time caps toforce completion','No watermark disabling orCPUfallback','No architecturefamily verdictfrom resource failure','No pretending interruptedstate is exactresume','No contextpadding-only fullquality without further evidence']),
            closure_reason='The complete generator passed real local free fitting and numerical/native-domain checks but its varied-request quality execution was manually stopped for measured severe total-process memory and insufficient throughput, before independent-song evaluation. The parameter-identical context-padding control preserved numerics yet also hit the physical cap, rejecting that limited fix as sufficient. Preserve the generating law and identify the remaining runtime path rather than inventing a quality failure or widening.',source_record=str(md.relative_to(ROOT)))
        path=FOLDER/(ident+'-mechanism-review.json');write(path,review)
        registry_path=ROOT/'reports/local-evolution-v1/mechanism-reviews/index.json';registry=json.loads(registry_path.read_text())
        if not any(x['id']==ident for x in registry['candidates']):registry['candidates'].append(dict(id=ident,analysis_status='complete',review=str(path.relative_to(ROOT)),review_sha256=sha(path),markdown=str(md.relative_to(ROOT)),narrative_sha256=sha(md)));write(registry_path,registry)
    print(json.dumps(facts,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
