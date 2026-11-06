"""Freeze C18 execution evidence references, without parent or peer judgments."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/council-runtime-c18'
ART = ROOT / 'artifacts/c17-masked-native-events-v1'


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def main():
    DEST.mkdir(exist_ok=True)
    if (DEST / 'material-manifest.json').exists():
        raise FileExistsError('Preserve independent evidence freeze')
    files = {}

    def add(path, role):
        path = ROOT / path if isinstance(path, str) else path
        name = str(path.relative_to(ROOT))
        files[name] = dict(path=name, bytes=path.stat().st_size, sha256=sha(path), role=role)

    plan_path = ROOT / 'bench/rhythm-v2/c17-masked-native-events-v1.json'
    plan = json.loads(plan_path.read_text())
    extra_path = ART / 'matched-controls-plan.json'
    extra = json.loads(extra_path.read_text())
    add(plan_path, 'actual_registered_core_contract')
    add(extra_path, 'actual_prospective_full_evaluation_addendum')
    for value, prefix in [(plan, ART / 'source'), (extra, ART / 'source/matched-control-addendum')]:
        for name, expected in value['source_sha256'].items():
            path = prefix / name
            if sha(path) != expected:
                raise ValueError('Execution source drift: ' + name)
            add(path, 'actual_frozen_source_not_parent_verdict')
    for path in sorted(ART.glob('*.json')):
        add(path, 'actual_C17_preparation_input_preflight_admission_or_profile_raw')
    for name in [
        'reports/local-integration-v1/c17-masked-native-events-v1.json',
        'reports/local-integration-v1/c17-masked-native-admission-profile.json',
        'reports/local-integration-v1/c17-masked-native-events-terminal-facts.json',
        'reports/council-shared-total-c17/masked-native-tests.log',
        'reports/council-shared-total-c17/matched-control-tests.log',
        'reports/council-shared-total-c17/admission-profile.log',
    ]:
        add(name, 'raw_execution_or_identity_arithmetic_not_parent_causal_story')
    for name in ['scripts/profile_c17_admission.py']:
        snapshot = DEST / 'source' / name
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        snapshot.write_bytes((ROOT / name).read_bytes())
        add(snapshot, 'actual_zero_update_profile_source_snapshot')
    goal = '''# C18 目标、Owner权限与本次实际执行边界

目标仍是理解音乐节奏/结构/公开作者意图，输出帮助真人编辑的Phira完整head/type/duration；预训练、MERT、SSD、Transformer、AR、count或mask都只是手段，不是不可替换目标。可以换完整路线，但不能因本次还没到健康/质量就声称学习族失败。父级和上一轮Council原文/因果故事均不在材料内，你自行审查合同与raw。

Owner本地自由进化、完整复盘后继续，不限量Council；Grok4.7最高档/Sol最高档，Argon缺席不补位。一次失败不能门槛即空白结案；至少分工程/健康、质量、结构条件、观察与竞争机制、已做/缺失对照、经验与明确后继。完整候选可围绕一个假说联合构件，组件归因用匹配对照。不要以“不同参数不能归因”结束，不要为了继续旧链而放弃全目标。

资源：独立torch2.14、MPS FP32/4threads/.5fraction，physical10GiB、driver<=min(.8recommended,10GiB)、0.5秒监护、无静默CPUfallback；磁盘余15GiB、候选1GiB含160MiB失败预留。当前单项全局9000秒、最多1002更新（CPU1+MPS1入场、丢弃健康100各、fresh质量400各）。原登记准备600、admission+health1200、quality各3000、main+追加评价1200。可以提出新冻结资源分配或等价实现修复，但不能改写旧失败、放宽全局帽或删除守卫/实验分母。最多6USD机制云池累计实际/承诺/调用0；此只读评审无执行云权限；0.25B scaling未授权，与机制小实验门分开。

原生ms、授权us COPY、同刻/完全重复多重集、短间隔、跨窗hold、公有incoming carry、可见参考和已知静音保留。禁止真实数量/类型配额、NMS/去重/min-gap/beat snapping/删hold/隐藏条件、dev择seed/温度/阈值/检查点或择快请求。原始confirm16未评分。

C17实际模型7,131,973参数，全窗256 MASK/NULL/完整原生事件，4层d256 self+audio cross，type条件所选head，end原生树全合法端点prior+可学习零残差。无旧qN/SSD/GRU和旧质量权重。自然tuple链式条件目标，训练all/.5/.25腐化比.5/.25/.25及IPW1/r；自由两sweep各8block32位置T1，可重写NULL/事件。不是精确chartNLL或D3PM ELBO。完整源码和实际源身份包含两次冻结；core110、追加112。引用的未列父级决定不可读。

准备57.469秒，5核心+2输入控制回归通过。实际8请求CPU/MPS前后向、各1次Adam（global2，MPS失败权重仅1）已执行，设备/梯度finite且差<1e-4。固定首个训练请求7681完成16block自由采样。原成本公式超过1200而停止；实际没有健康/质量/dev结果。raw admission含完整输入审计、梯度、512采样轨迹和原失败，不是经过父级挑选的几个数。

后验只做了一次同失败状态/同训练请求/同RNG的零更新同步计时重放，保存概率/整数恒等和分项计时。forward包含attention子计时，不能嵌套相加。原成本测量与侵入式同步剖析不是同一性能统计；没有任何优化版加速结果。权重文件只有身份元数据，不允许本评审打开tensor。若拟复用入场状态只可建议后续零更新工程验证，正式健康/质量仍使用原共同initializer；复用是否成立由精确实际状态条件决定。

预注册完整评价：340自由先落盘，再三腐化率1020proper；追加32本曲roll和32异歌donor，其中62支持/2无完整范围donor显式保留，不从分母抹去。追加已在第一次Adam前冻结源/请求/上限与输入预检，不是看到dev结果后补探针。实际神经评价0，C没有已生成的继承结果。完整面板可执行性和研究价值均应审查；不能把健康或执行门当产品出场。
'''
    (DEST / 'goal-authority.md').write_text(goal)
    add(DEST / 'goal-authority.md', 'goal_authority_and_actual_metadata_no_parent_verdict')
    manifest = dict(id='c18-independent-runtime-materials',files=sorted(files.values(),key=lambda v:v['path']),
        total_files=len(files),total_bytes=sum(v['bytes'] for v in files.values()),
        policy='Exact allowlist only; no peer/parent judgment,unlisted imports,history,memory,NAS,secrets,weights',
        quality_weights_excluded=True,cloud_calls=0)
    write(DEST/'material-manifest.json',manifest)
    common = f'''独立Council C18，工作根/Users/yukoval/yukoval-projects/beatfcn，亲自完成，不派子agent。

先读reports/council-runtime-c18/material-manifest.json，SHA256 {sha(DEST/'material-manifest.json')}；{manifest['total_files']}对象/{manifest['total_bytes']}bytes。仅manifest与files[].path可读；精确allowlist后逐路径SHA/bytes。只可标准库hash/JSON/文本/AST/数学/ZIP头，不import项目/PyTorch/NumPy，不运行冻结脚本、不读pt/tensor、不前向/新生成/训练/安装/云。可查公开一手文献/官方源码，记录URL并区别外部事实与本机事实。输入摘要不是父级结论；请亲自核对raw和必要源码。引用但未列路径不可跟读。不得读同轮/旧Council报告、父级复盘/裁决/方法论/台账、线程/记忆/NAS/密钥。环境注入摘要不作证据。

你不是唯一agent，仅写指定自己的报告/audit两文件，不改别人文件。每个manifest对象一条访问审计，写semantic/局部解析/头/hash-only的真实层次及选段/未读依赖；全对象hash不冒充全语义精读。

必须区分：原成本预测门触发、实际墙钟/内存、数值梯度/Adam、完整自由训练样本、未执行的健康与质量；不可把这次记成模型质量阴性。计时嵌套/同步侵入、真实暂态工作量、preflight非神经、两次sourcefreeze、core/追加cost公式均核对。

交付观察→竞争机制→可检验预测→具体缺失对照→一个可执行默认完整恢复路径，最多一个明确触发备选，稳定行动ID。优先按证据判断是否存在概率/支持/随机流不变的修复，非预设一定保留当前law；若主张更换law，说明尚无质量阴性的情况下为何值得放弃本次未受测学习。不要无止境诊断或只给一条“没过成本门”。默认若含零更新探针，必须有限单次、决定明确health/freshquality分支；哪些原入场证据可继承、哪些必须重做写清。变量/概率/训练自由输入、全字段parity/必要梯度、实际成本与物理资源、独立quality/逐类proper/free/空窗/ref/tail义务、费用及失败后动作均明确。不能删原分母、用产品门否决小工程探针或放宽全局帽。最强反对意见及改变下一动作的结果必须具体。
'''
    focuses = {
        'G': '重点审查概率不变的执行修复、精确端点先验/注意力/资源守卫的竞争成本，计时与负载估计适用边界。不要把某timer直接当唯一根因；没有修复实测就给有限等价/成本探针再恢复路径。行动C18-G-R01等。只写reports/council-runtime-c18/grok.md与grok-access-audit.json。',
        'S': '从完整音乐事件目标检验当前学习合同是否值得保留、工程准入是否妨碍未受测候选；独立选择最有信息价值的完整恢复路径。审查概率/随机流/训练与自由/端点支持、资源重新分配的合法边界与严格fresh路径，不被最近count失败或某计时先验绑住。行动C18-S-R01等。只写reports/council-runtime-c18/sol.md与sol-access-audit.json。',
    }
    for who in ('G','S'):
        prompt=common+'\n'+focuses[who]
        (DEST/f'prompt-{who}.txt').write_text(prompt)
        write(DEST/f'submitted-{who}.json',dict(
            clientRequestId=f'beatfcn-runtime-c18-{who.lower()}-20261030',mode='async',isolation='shared',role='review',
            title=f'Council C18 · {"成本入场与概率等价修复 · Grok 4.7" if who=="G" else "完整学习合同与恢复路径 · Sol Ultra"}',
            target=dict(providerInstanceId='grok' if who=='G' else 'codex',model='grok-4.7' if who=='G' else 'gpt-6.1-sol',
                options=dict(reasoningEffort='xhigh' if who=='G' else 'ultra')),task=prompt))
    write(DEST/'task-context.json',dict(status='ready',material_sha256=sha(DEST/'material-manifest.json'),
        files=manifest['total_files'],bytes=manifest['total_bytes'],prior_reviews_excluded=True,weights_excluded=True,
        cloud_spend_usd=0,health_updates=0,quality_updates=0))
    print(dict(status='ready',files=manifest['total_files'],bytes=manifest['total_bytes'],sha256=sha(DEST/'material-manifest.json')))


if __name__=='__main__':
    main()
