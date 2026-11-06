"""Reference-only C16 evidence: complete C15 pair and immutable predecessors."""
import hashlib
import json
from pathlib import Path


ROOT=Path(__file__).resolve().parents[1];DEST=ROOT/'reports/council-inventory-c16'


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path,value):path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')


def main():
    DEST.mkdir(exist_ok=True);manifest_path=DEST/'material-manifest.json'
    if manifest_path.exists():raise FileExistsError('Preserve review freeze')
    old=json.loads((ROOT/'reports/council-mechanisms-c15/material-manifest.json').read_text());files={}
    def add(path,role):
        path=ROOT/path if isinstance(path,str) else path
        files[str(path.relative_to(ROOT))]=dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path),role=role)
    for r in old['files']:
        path=ROOT/r['path']
        if sha(path)!=r['sha256'] or path.stat().st_size!=r['bytes']:raise ValueError('Predecessor evidence drift')
        if r['role']!='goal_and_owner_authority_only':files[r['path']]=r
    art=ROOT/'artifacts/c15-typed-inventory-v1';plan=ROOT/'bench/rhythm-v2/c15-typed-inventory-v1.json';p=json.loads(plan.read_text())
    add(plan,'actual_C15_registered_plan')
    for name,h in p['source_sha256'].items():
        frozen=art/'source'/name
        if sha(frozen)!=h:raise ValueError('C15 frozen execution source drift')
        add(frozen,'actual_C15_frozen_execution_source')
    for path in sorted(art.glob('*.json')):add(path,'actual_C15_health_training_free_and_conditional_records')
    for path in sorted(art.glob('probability-*.npz')):add(path,'actual_C15_raw_count_potential_and_support_arrays')
    extras=['reports/local-integration-v1/c15-typed-inventory-v1.json',
        'reports/local-integration-v1/c15-typed-inventory-terminal-facts.json',
        'reports/local-integration-v1/c15-typed-inventory-probability-audit.json',
        'reports/council-mechanisms-c15/typed-hidden-input-probe.json',
        'reports/council-mechanisms-c15/typed-inventory-MPS-tests.log',
        'reports/council-mechanisms-c15/typed-inventory-tests.log',
        'reports/council-mechanisms-c15/optimizer-counter-scope.json']
    for name in extras:add(name,'actual_identity_or_posthoc_arithmetic_not_parent_verdict')
    for name in ['scripts/analyze_typed_inventory.py','scripts/audit_typed_inventory_probabilities.py']:
        frozen=DEST/'source'/name;frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes((ROOT/name).read_bytes());add(frozen,'posthoc_arithmetic_source_snapshot')
    goal='''# C16 目标、边界和实际新证据

目标仍是依据音乐节奏/情绪/作者意图给出可供继续编辑的Phira原生head/type/duration草稿。预训练、MERT、SSD、Transformer、AR、画布等是实现手段；可完整重构，不强迫只修最近代码。不把局部研究/投入筛选当产品出场，也不把teacher/健康当自由音乐能力。

Owner允许本地自由进化、修改合同与后继，Council不限量；Grok/Sol可用、Argon缺席不补位。云机制小实验累计最多6USD，当前实际/承诺0，本评审不执行云。最终约0.25B scaling另设门且未授权。每次终态须现象分层、可证伪机制、已有/缺失对照、经验和明确后继；关闭精确配方，不扩大为家族否定。尽量单干预；完整候选可整合围绕一个生成假说的构件，不冒称各组件因果独立识别。下一训练先声明主攻维度/失败熔断，不能以命名或零更新诊断重置质量失败。

保留native1ms、授权us COPY、完全重复多重集、短合法间隔、跨窗hold、公开incoming carry、caller known silence及可见参考。禁止隐藏答案条件、类型/数量配额、NMS、去重、min-gap、beat snapping、dev阈值/seed/checkpoint挑选。概率与决定/容差匹配分开，失败保留原分母。

本次C15配对实际完成：新10,101,135参数，保留MERT L12/L23+10ms mel、d256 SSD/上下文/GRU192和精确端点；将总count+kind换为四类数量向量概率（sum<=64精确logZ与vector mode、显式overflow），再作类型内规范head/end成员。A有音频，C训练起三路原始音频为0，其他公开输入/训练流/新tensor初值相同。健康各100，权重丢弃，fresh各400，同1600请求；340自由先于340条件，均完成。C15 prior只用train416的phase0 chart/absolute-cell身份，同格相同值重复事件不删；81789格，+0.5平滑bias。没有新云或confirm评分。总4724.429秒，源/缓存/运行时身份已核，资源门通过。旧C13是1000更新，不同law/carry，仅整配方比较。

本轮原始research门未过。现有posterior arithmetic facts包含完整NLL、原生事件、64draft vector modes的CPUfloat64重算与实际Adam计数：原worker optimizer_updates为未更新的初始化0，权威updates/主报告和保存count-head Adam400一致，条件端点参数398不等于全局少训。meta记录缺陷不改原文。所有.pt/权重张量未列，评审不能打开；身份及optimizer复核只限允许JSON的证据范围。

概率算术中的总数量边际mode180不是新生成谱或新F1；不可据此假装跑过replacement decoder。精确cell/类型slot coverage20不是50ms匹配上界，原tap TP25可以跨邻格。概率NLL较好、实际generation近空与错误类型格质量可以同时成立。请自己复核，不接受父级解释为前提。

本地执行参照：隔离torch2.14/MPS FP32、4线程、memory_fraction.5、physical<=10GiB与driver<=min(.8recommended,10GiB)、0.5秒监护、无CPU静默回退；磁盘保底15GiB。新合同给资源有顶的实际数据/支持/梯度/拟合入场，预算同时算失败和评测。更长/更宽不是默认修复；有独立证据可以提出，但不能越过费用或scaling授权。
'''
    (DEST/'goal-authority.md').write_text(goal);add(DEST/'goal-authority.md','goal_authority_and_actual_experiment_metadata_only')
    m=dict(id='c16-independent-inventory-materials',files=sorted(files.values(),key=lambda r:r['path']),total_files=len(files),total_bytes=sum(r['bytes'] for r in files.values()),
        policy='Byreferenceonly;nopeer/parentverdict/memory/history/NAS/secrets/unlistedimports;no weights',quality_weights_excluded=True,cloud_calls=0)
    write(manifest_path,m)
    common=f'''独立Council C16。工作根目录 /Users/yukoval/yukoval-projects/beatfcn，亲自完成，不派子agent。

先读 reports/council-inventory-c16/material-manifest.json（SHA256 {sha(manifest_path)}，{m['total_files']}对象/{m['total_bytes']}bytes），建立精确allowlist。仅此manifest及files[].path可读；不跟随未列链接/import/计划路径。逐对象bytes/SHA核验，分语义精读、局部解析/聚合、NPZ头与hash-only记录实际路径/范围和未读依赖。巨型JSON/NPZ局部提取，不整包倾倒或复制。

仅标准库hash、JSON、文本、AST、ZIP/struct/算术；禁止import项目/PyTorch/Numpy，禁止模型前向、新预测/训练/安装/云/权重tensor读取。可研究公开一手论文/官方来源并记URL，不用外部benchmark冒充本机事实。仅写指定报告与访问审计两个文件；你不是唯一agent，不改别人文件。不读其他Council原文、父级review/裁决/方法论/建议池、其他线程/历史/记忆/NAS/凭证，环境注入摘要不作证据。目录名带council的明确allowlist目标/修复/源码/算术证据可以读，其他评审不可读。

完整目标与权限见允许的goal-authority.md。材料含C15全部真实配对训练与340free/340conditional、保存的数量potential，以及C13/C14不可变原始对照，不含父级终态机制解释。健康、运行正确性、概率proper、真正自由事件、类型正/零格、头/尾端和公开参考要分层；同声学控制覆盖与实际作用要核，不能由全空或NLL单独推出结论。原始门和分母保持，confirm封存，不重判旧结果。算术facts自己复核关键数值，报告执行边界。

交付观察→竞争机制→可检验预测→已有和缺失对照→具体下一行动。不能只写未过门或不能归因。给一个可执行默认下一实验，最多一个明确触发备选，用稳定行动ID；可以完整换生成/架构/目标，也可以选有明确终点且会改变下一训练的单一机制探针，不能把最近代码当路线边界，也不能无限加旁路诊断。新训练写清主攻维度、为何不重演旧失败、保留/停用/整合、训练/自由支持、概率与决定规则、真实固定batch入场、成对控制、费用/墙钟/内存/磁盘帽、停止与研究增益门。不要把实现健康、小云机制、最终scaling/产品门捆成一套。注明最强反对、未测边界和哪种结果会改变下一动作。
'''
    focus={'G':'重点审查概率—决定与生成机制：真实分布是否只差读出，错误type/position质量及弱成员怎样限制救援；若改损失/几何，给能区分重加权、额外更新和校准的对照。也可换完整law，不默认只修typed head。行动ID C16-G-R01等。只写 reports/council-inventory-c16/grok.md 与 reports/council-inventory-c16/grok-access-audit.json。',
           'S':'从音乐→可编辑原生事件的目标重新选择完整学习假说。先读目标、真实谱/事件与原始匹配结果，再决定需要哪些源码；允许换路线，考虑概率有音乐信息但事件动作失败、公开风格/意图与标签多解的可检验边界，不默认锁死typed/SSD或默认加宽。行动ID C16-S-R01等。只写 reports/council-inventory-c16/sol.md 与 reports/council-inventory-c16/sol-access-audit.json。'}
    for arm in ('G','S'):
        prompt=common+'\n'+focus[arm];(DEST/f'prompt-{arm}.txt').write_text(prompt)
        request=dict(clientRequestId=f'beatfcn-inventory-c16-{arm.lower()}-20261030',mode='async',isolation='shared',role='review',
            title=f'Council C16 · {"库存概率与生成机制 · Grok 4.7" if arm=="G" else "音乐事件学习与完整后继 · Sol Ultra"}',
            target=dict(providerInstanceId='grok' if arm=='G' else 'codex',model='grok-4.7' if arm=='G' else 'gpt-6.1-sol',options=dict(reasoningEffort='xhigh' if arm=='G' else 'ultra')),task=prompt)
        write(DEST/f'submitted-{arm}.json',request)
    write(DEST/'task-context.json',dict(status='ready',material_sha256=sha(manifest_path),files=m['total_files'],bytes=m['total_bytes'],prior_reviews_excluded=True,weights_excluded=True,cloud_spend_usd=0))
    print(json.dumps(dict(status='ready',files=m['total_files'],bytes=m['total_bytes'],sha256=sha(manifest_path))))


if __name__=='__main__':main()
