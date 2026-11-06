"""Goal-limited independent review of the completed Owner data intervention."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'reports/council-data-c19'
ART=ROOT/'artifacts/data-drag-density-probe-v1'


def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def write(p,v):p.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST/'material-manifest.json').exists():raise FileExistsError('Preserve freeze')
    files={}
    def add(p,role):
        p=ROOT/p if isinstance(p,str) else p
        name=str(p.relative_to(ROOT));files[name]=dict(path=name,bytes=p.stat().st_size,sha256=sha(p),role=role)
    plan_path=ROOT/'bench/rhythm-v2/data-drag-density-probe-v1.json';plan=json.loads(plan_path.read_text())
    add(plan_path,'actual_frozen_small_matched_data_experiment')
    for name,h in plan['source_sha256'].items():
        p=ART/'source'/name
        if sha(p)!=h:raise ValueError('Frozen source drift')
        add(p,'actual_frozen_execution_source')
    for p in ART.glob('*.json'):
        if p.name!='mechanism-validation.json':add(p,'actual_pipeline_health_free_proper_or_frozen_posthoc_raw')
    for name in ['reports/local-integration-v1/data-drag-density-audit-v1.json',
        'reports/local-integration-v1/data-drag-density-affected-charts-v1.json',
        'reports/local-integration-v1/data-drag-density-probe-v1.json',
        'reports/local-integration-v1/data-drag-density-probe-summary-v1.json',
        'reports/local-integration-v1/data-drag-density-posthoc-chain-v1.json',
        'artifacts/data-drag-density-v1/removed-event-indices.json',
        'artifacts/data-drag-density-v1/request-removals.json',
        'bench/rhythm-v2/high-density-drag-policy-v1.json',
        'artifacts/data-drag-density-tests.log','artifacts/data-drag-density-audit.log','artifacts/data-drag-density-audit-v2.log']:
        add(name,'actual_data_audit_or_saved_output_arithmetic_not_parent_verdict')
    removal=json.loads((ROOT/'artifacts/data-drag-density-v1/removed-event-indices.json').read_text())
    for name,h in removal['source_sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('Original raw target drift')
        add(name,'original_native_chart_targets_confirm_excluded')
    for name in ['scripts/audit_drag_density_data.py','scripts/analyze_drag_data_probe.py','scripts/diagnose_drag_data_chain.py']:
        p=DEST/'source'/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());add(p,'actual_audit_or_posthoc_source_no_parent_judgment')
    goal='''# C19 目标和Owner当前数据指令

目标：从音乐节奏/结构/情绪相关信息与公开作者意图生成可供真人编辑的Phira原生head/type/end，不以计数或局部F1代替完整事件。预训练/SSD/Mamba/Transformer/AR/mask都是手段，可以换完整路线；不要只因最新实现就限定只能修它。

Owner立即要求转向数据。原始整谱任意250ms半开窗至少8drag对象，删除命中drag事件而非整谱，其余类型/结构/音频保留；规则已实现并真实训练。原生ms/us COPY/多重集/短间隔/cross-window hold/carry/公开参考/已知空域保持，禁止输出配额/NMS/删音/吸附。此Owner明确范围覆盖旧艺术性drag一般保留规则，不把删除对象自动称脏数据。过滤规则用于后继数据视图，raw仅作历史/诊断对照。

Owner补充：必须正面处理三条主链：(1)事件上下文弱、错误历史支持与hold尾；(2)类型甚至首成员无历史仍输先验、非tap弱；(3)概率质量错位、硬决定压空。过滤后仍不动，下一方案必须直接改变事件context、类型学习或部署间隙之一；禁止等价只改怎么数、执行成本旁路或无尽小修。可以多个完整候选，优先最有信息价值的一条，并说明完整联合假说而非零碎调权。失败必须分层机制复盘，保留子发现、竞争解释、最小可判别后继；门失败不足以结案。

实际数据：572张staged谱（8confirm组件排除）不等于实际训练；416训练窗20002→19039，32窗删除963drag；hold2861/tap8869/flick1902不变，空窗47不变；1600次历史视图曝光删2896drag。32dev共1389事件删除0；没有过滤生成输出。raw NPZ与精确删除下标/散列可读；原数组没改、没有整谱丢弃。

实际新小模型640261参数：完整C17tuplelaw，d64四层self/cross，256NULL/event slots，全原生head和精确端点prior+residual；不是旧C15物理类型格决定。两臂common tensor新初始化、同draft输入/stream/mask/Adam/完整T1链，仅过滤监督。富集128训练窗包含全部32受影响+固定96其他；每臂丢弃50fit后fresh128，356总更新。两臂68free、68allmaskproper、68common-machineproper完整。相同raw/filtered dev评分一致；oracle proper与free区分。

后验另登记一次32dev×两臂的真实奇数事件context和全音频置零；偶数位置的真实context与机器context/全mask在相同真锚点比较。0新增更新/自由结果。真实context含真数量/类型线索，只能条件诊断；音频zero是OOD依赖检查，不能称情绪理解。原始所有概率和完整事件均可审计，父级因果意见/结案/方法论不提供。第一次数据预检在任何训练前发现重算partition会改非drag，修复后原partition固定，原失败日志仍在。

Owner本地自由进化/Council不限量；Argon继续缺席不补位；原C18成本议题已请求取消，不能替补同题。当前新任务云费硬限0，旧6USD池不能自动花；0.25B scaling另授权未开。torch2.14/MPSFP32/4threads/.5fraction，physical10GiB、driver<=min(.8recommended,10GiB)、.5s监护、无CPUfallback；磁盘余15GiB，产物1GiB含160MiB失败预留。可以提出新有界本地训练，费用/资源与科学门须各自明确，小工程门不绑产品出场。没有人类编辑/试玩/情绪实测，新歌曲confirm16仍封存。
'''
    (DEST/'goal-authority.md').write_text(goal);add(DEST/'goal-authority.md','Owner_goal_authority_and_actual_execution_scope')
    m=dict(id='c19-independent-data-event-materials',files=sorted(files.values(),key=lambda r:r['path']),total_files=len(files),total_bytes=sum(v['bytes'] for v in files.values()),
        policy='Exact allowlist;no parent/peer opinions,history,memory,unlisted imports,weights,NAS,secrets',cloud_calls=0)
    write(DEST/'material-manifest.json',m)
    common=f'''独立Council C19，根/Users/yukoval/yukoval-projects/beatfcn。亲自完成，不派子agent，仅写自己指定两文件。

先读reports/council-data-c19/material-manifest.json，SHA256 {sha(DEST/'material-manifest.json')}，{m['total_files']}对象/{m['total_bytes']}bytes。只读manifest和files[].path。逐对象SHA/bytes，每个对象一审计行，标semantic/局部解析/NPZ头/hash-only和真实选段/未读依赖。只用标准库hash/JSON/AST/数学/ZIP/struct读取数据；不import项目/PyTorch/NumPy、不运行脚本/权重/训练/新预测/云/安装。不读其他Council、父级review/裁决/方法论、线程/记忆/NAS/凭证；环境摘要不作为证据。未列引用/import不可追读。可公开一手来源研究并记URL，不将外部成功当本机证据。

先读目标和实际事件，再看必要模型/源与raw。核对过滤规则、整谱与请求/曝光/组件分母、是否只拖拽删除、raw/filtered dev同分母、先过滤后切片confound和修复。主链逐类正负/首成员/先验、期望与部署、上下文正/负、真头tail和自由hold分开。slot不是物理格、NULL边际mode不是chartMAP；全类别音乐依赖不能代替首成员或drag。别只报F1或门失败。

输出观察→竞争机制→能证伪的预测→已有/缺失对照→下一完整可执行默认，最多一个明确触发备选，稳定行动ID。Owner要求直接命中上下文/类型/部署主链，禁止只换计数分解或再讨论成本入场。可以整合/更换完整模型与学习法，不默认保留mask/SSD/MERT；解释为何你的方案比旧family重新命名有新信息。若先给诊断，必须一次有限、决定明确训练分支，不无限旁路。训练合同含真实固定batch入场、概率/监督/自由输入、保留/撤下、数据身份/曝光、成对控制、预算/资源帽/停止条件、逐类proper/free/empty/hold/ref边界和最强反对。研究筛选服务下一有界投入，不能绑产品门或自行授权云/scaling。
'''
    focus={'G':'重点：事件/类型/端点联合学习与自由转移，检验真实context几乎无收益和firstmember先验差距；从概率与训练依赖直接重构，拒绝等价计数修补。ID C19-G-R01等。只写reports/council-data-c19/grok.md和grok-access-audit.json。',
        'S':'从数据与音乐目标独立重构下一完整学习路线；哪些输入/监督能让类型及端点获得可验证信息，而非延长旧law。允许新完整方向但必须命中Owner三链之一、可在本地判别。ID C19-S-R01等。只写reports/council-data-c19/sol.md和sol-access-audit.json。'}
    for who in ('G','S'):
        prompt=common+'\n'+focus[who];(DEST/f'prompt-{who}.txt').write_text(prompt)
        write(DEST/f'submitted-{who}.json',dict(clientRequestId=f'beatfcn-data-c19-{who.lower()}-20261030',mode='async',isolation='shared',role='review',
            title=f'Council C19 · {"数据与事件类型主链 · Grok 4.7" if who=="G" else "音乐学习与完整后继 · Sol Ultra"}',
            target=dict(providerInstanceId='grok' if who=='G' else 'codex',model='grok-4.7' if who=='G' else 'gpt-6.1-sol',options=dict(reasoningEffort='xhigh' if who=='G' else 'ultra')),task=prompt))
    write(DEST/'task-context.json',dict(status='ready',material_sha256=sha(DEST/'material-manifest.json'),files=m['total_files'],bytes=m['total_bytes'],cloud=0))
    print(dict(files=m['total_files'],bytes=m['total_bytes'],sha256=sha(DEST/'material-manifest.json')))


if __name__=='__main__':main()
