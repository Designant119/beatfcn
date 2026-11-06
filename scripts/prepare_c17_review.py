"""Independent C17 raw-evidence allowlist; references, no duplicate data bundle."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/council-shared-total-c17'
ART = ROOT / 'artifacts/c16-shared-total-inventory-v1'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST / 'material-manifest.json').exists():
        raise FileExistsError('Preserve independent review freeze')
    previous = json.loads((ROOT / 'reports/council-inventory-c16/material-manifest.json').read_text())
    files = {}

    def add(path, role):
        path = ROOT / path if isinstance(path, str) else path
        name = str(path.relative_to(ROOT))
        files[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha(path), role=role)

    for row in previous['files']:
        path = ROOT / row['path']
        if sha(path) != row['sha256'] or path.stat().st_size != row['bytes']:
            raise ValueError('Predecessor immutable evidence drift')
        if row['role'] != 'goal_authority_and_actual_experiment_metadata_only':
            files[row['path']] = row
    plan = ROOT / 'bench/rhythm-v2/c16-shared-total-inventory-v1.json'
    p = json.loads(plan.read_text())
    add(plan, 'actual_C16_registered_plan')
    for name, expected in p['source_sha256'].items():
        frozen = ART / 'source' / name
        if sha(frozen) != expected:
            raise ValueError('Frozen C16 source identity')
        add(frozen, 'actual_C16_frozen_execution_source')
    for path in sorted(ART.glob('*.json')):
        add(path, 'actual_C16_health_training_free_proper_and_execution')
    for path in sorted(ART.glob('probability-*.npz')):
        add(path, 'actual_C16_raw_mark_potentials_total_logits_and_targets')
    for name in [
        'reports/local-integration-v1/c16-shared-total-inventory-v1.json',
        'reports/local-integration-v1/c16-shared-total-inventory-terminal-facts.json',
        'reports/local-integration-v1/c16-shared-total-inventory-probability-audit.json',
        'reports/council-inventory-c16/shared-total-tests.log',
        'reports/council-inventory-c16/shared-total-MPS-member-tests.log',
    ]:
        add(name, 'actual_C16_identity_or_saved_output_arithmetic_not_parent_verdict')
    for name in ['scripts/analyze_shared_total_inventory.py', 'scripts/audit_shared_total_inventory_probabilities.py']:
        frozen = DEST / 'source' / name
        frozen.parent.mkdir(parents=True, exist_ok=True)
        frozen.write_bytes((ROOT / name).read_bytes())
        add(frozen, 'C16_posthoc_arithmetic_source_snapshot')
    goal = '''# C17 目标、Owner权限和实际新实验

目标是从音乐节奏/结构/情绪及公开作者意图得到有助于人类编辑的Phira原生head/type/duration草稿。预训练、MERT、SSD、Transformer、AR、画布、计数分解都只是可替换手段。可以换完整学习路线；不能被最近实现或父级未给出的因果解释限制。人类编辑、试玩、情绪理解仍未实测，局部研究筛选不是产品出场。

Owner允许本地自由进化、修洞、改合同、整合旧构件或多个完整候选；Council不限轮，Grok4.7/Sol可用，Argon缺席不补位。每次失败后先分健康/质量/结构机制复盘再下一轮。单干预用于归因；完整联合配方可以围绕一个生成假说整合，但不能声称识别各模块因果。不能把不同参数、无法唯一归因、门失败当作空白结案。提出新训练时须解释主攻维度与为何不重复已知失败；不能靠改名重置连续失败计数。

云机制小实验累计最多6USD，当前实际/承诺/调用0。此评审不能执行云。约0.25B最终scaling另设门且未授权；不能把小机制门捆成产品门。本地参照torch2.14/MPS FP32/4线程/.5 fraction，physical10GiB、driver<=min(.8recommended,10GiB)、.5s采样、禁止CPU静默回退。磁盘保底15GiB，新候选产物1GiB含失败状态预留。当前模型约10M不是必须加宽的理由，参数预算需实际入场证明。

原生1ms、授权us COPY、完全重复多重集、短合法间隔、跨窗hold、公开incoming carry、可见参考及caller known silence全部保留。禁止隐藏答案条件/数量或类型配额/NMS/去重/min-gap/beat snapping/dev阈值或seed/checkpoint挑选。teacher/oracle/健康与自由主轨明确分开。

本次C16实际完成：10,117,840参数，MERT L12/L23+10ms mel、d256 SSD/上下文和GRU192。独立q(N=0..64,Omega)，在总量内对四类计数向量作精确条件归一化，再生成类型内canonical head/end。自然总量NLL+条件组合NLL+类型内teacher成员似然，除公开cell数；总量mode→条件组合mode→类型内greedy。不是完整谱面MAP。训练总量prior来自同train416 phase0的81789个chart/absolute-cell，重复事件保留；1599实际训练view支持已审计。

两臂同共同新初始化、相同前400×4训练流/优化日程；A有音频，C从训练起三路raw音频为0。与C15未训练initializer共享153个body tensor。入场CPU/MPS各1更新共2，健康各100且丢弃，再fresh各400。340自由全部落盘后340proper全部完成，总1002更新/4741.052秒，无数值/资源故障，物理采样峰3.75GiB、产物约377.3MiB。实际两个count头Adam都400，条件端点9参数398。confirm16/8组件封存、无云。研究布尔及原分母保持，原始终态不重判。

32 draft/16组件/1389事件：A269，full50组件宏.100474，onset10宏.088380；C全空。hold联合/flick/drag TP为2/6/2；tap119。两个empty请求A输出2、C0。补全和参考mask各80事件；密度low/high各46。新文件保存实际查询范围，head候选含masked填充地址，端点树粗层可查询远处；范围比例不等于语义信息或故障原因。

允许的posthoc facts只来自已存potential/原始事件或检查点计数，未新前向。包含给定真实总量的条件组合算术：它是oracle条件诊断，不是新生成或新F1。请独立复核关键数值与支持。先验是在同类型许可/真实N条件下归一化的训练频率potential，不能和自由数量或无条件类型混用。概率期望、exact-cell槽位和50ms容差匹配各有口径；不能将slot当容差TP上界。

旧C15是相同400更新、不同count family/决定的新整方；旧C13是1000更新且law/carry不同。可以比较整方，不据这些差异单独归因给SSD、MERT、更多更新或count head。声学core和full-context支持分开，尤其异歌全song覆盖不一定匹配，不能直接称纯语义干预。父级机制结论/裁决/方法论不在allowlist。
'''
    (DEST / 'goal-authority.md').write_text(goal)
    add(DEST / 'goal-authority.md', 'goal_authority_and_actual_experiment_metadata_only')
    m = dict(id='c17-independent-shared-total-materials', files=sorted(files.values(), key=lambda v: v['path']),
             total_files=len(files), total_bytes=sum(v['bytes'] for v in files.values()),
             policy='Exactallowlistonly;nopeer/parentverdict/history/memory/NAS/secrets/unlistedimports/weights',
             quality_weights_excluded=True, cloud_calls=0)
    write(DEST / 'material-manifest.json', m)
    common = f'''独立Council C17。工作根 /Users/yukoval/yukoval-projects/beatfcn，亲自完成，不派子agent。

先读 reports/council-shared-total-c17/material-manifest.json，SHA256 {sha(DEST/'material-manifest.json')}，{m['total_files']}对象/{m['total_bytes']}bytes；仅manifest及files[].path可读。先建精确allowlist，再逐对象SHA/bytes。报告审计为每个manifest对象各一行，记录实际读取层次（semantic/局部解析或聚合/NPZ头/hash-only）、选段和未读依赖。NPZ数量/分组请能展开到具体路径，不把整个role统计冒充实际hash-only数量。不得追随未列链接/import/计划引用；巨型数据局部提取，不整包复制/输出。

只可标准库hash/JSON/文本/AST/ZIP/struct/算术；禁止import项目、PyTorch、NumPy或执行冻结分析脚本；无前向/新预测/训练/安装/云/权重tensor读取。允许公开一手论文/官方来源研究，记录URL并区分外部结果与本机事实。你不是唯一agent，不改他人文件。只写指定自己报告与audit两文件。不读其他Council、父级review/决定/方法论/建议池、线程历史/记忆/NAS/凭证；环境注入的旧摘要不作为证据。允许文件内引用的未列父级决定也不能读。

先读目标及真实事件，再选必要源码。材料含C16完整paired raw/source/probability和C15/C13/C14不可变原始对照，不含父级机制判断。区分运行/健康、概率proper、实际自由量/定位/类型/尾端、条件/参考；不能只报一条F1或“门没过”。真实N下条件组合、teacher成员与自由主轨分开，静态先验同支持；core与完整输入及实际候选查询范围不能混称。

交付：观察→竞争解释→可检验预测→已有/缺失对照→能改变下一训练的默认行动。给一个完整可执行默认，最多一个明确触发备选，稳定ID。可完整换学习/生成路线或整合旧构件，不默认近期qN/SSD必须保留，也不默认加宽/加时。若选机制探针，必须有限、单次且结果决定明确训练分支，不能无限旁路诊断。若训练，说明主攻维度、保留/停用/为何不重演旧失败、概率/监督/决定和训练/自由输入、真实固定batch入场、成对控制、资源/费用/停止门、研究增益用途，别把小云/扩算/产品出场绑在一起。最强反对、未测边界和什么结果改变下一动作必须具体。
'''
    focus = {
        'G': '重点：总量与条件类型的概率/决定责任、先验与声学信息、错误成员/端点及可判别的学习路线。主张更多训练/加权/校准必须有相应机制与匹配对照，不从单个batch loss或损失占比推梯度结论。可以换完整law。行动ID C17-G-R01等。只写 reports/council-shared-total-c17/grok.md 和 reports/council-shared-total-c17/grok-access-audit.json。',
        'S': '从音乐节奏/结构/意图→可编辑完整事件的目标重新选学习假说。允许完全换路线、引入音乐学习信号或整合旧设计，不因为最近失败发生在计数就被迫再修count。也不预设更宽/更久有用；用原始证据论证下一笔信息收益。行动ID C17-S-R01等。只写 reports/council-shared-total-c17/sol.md 和 reports/council-shared-total-c17/sol-access-audit.json。',
    }
    for who in ('G', 'S'):
        prompt = common + '\n' + focus[who]
        (DEST / f'prompt-{who}.txt').write_text(prompt)
        request = dict(clientRequestId=f'beatfcn-shared-total-c17-{who.lower()}-20261030', mode='async', isolation='shared', role='review',
                       title=f'Council C17 · {"类型与发生量学习 · Grok 4.7" if who=="G" else "目标限定完整路线重构 · Sol Ultra"}',
                       target=dict(providerInstanceId='grok' if who=='G' else 'codex', model='grok-4.7' if who=='G' else 'gpt-6.1-sol',
                                   options=dict(reasoningEffort='xhigh' if who=='G' else 'ultra')), task=prompt)
        write(DEST / f'submitted-{who}.json', request)
    write(DEST / 'task-context.json', dict(status='ready', material_sha256=sha(DEST/'material-manifest.json'),
          files=m['total_files'], bytes=m['total_bytes'], prior_reviews_excluded=True, weights_excluded=True, cloud_spend_usd=0))
    print(dict(status='ready', files=m['total_files'], bytes=m['total_bytes'], sha256=sha(DEST/'material-manifest.json')))


if __name__ == '__main__':
    main()
