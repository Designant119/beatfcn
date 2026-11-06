"""Freeze existing evidence by reference for independent C15 reviews."""
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'reports/council-mechanisms-c15'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


def main():
    DEST.mkdir(exist_ok=True)
    manifest_path = DEST / 'material-manifest.json'
    if manifest_path.exists():
        raise FileExistsError('No mutation of frozen review material')
    old = json.loads((ROOT/'reports/council-quality-c14/material-manifest.json').read_text())
    objects = {r['path']:dict(r) for r in old['files']}

    def add(path, role):
        path = ROOT/path if not isinstance(path, Path) else path
        name = str(path.relative_to(ROOT))
        entry = dict(path=name, sha256=sha(path), bytes=path.stat().st_size, role=role)
        if name in objects and any(objects[name][k]!=entry[k] for k in ('sha256','bytes')):
            raise ValueError('Existing frozen object drift')
        objects[name]=entry

    for entry in old['files']:
        p=ROOT/entry['path']
        if sha(p)!=entry['sha256'] or p.stat().st_size!=entry['bytes']:
            raise ValueError('C13 material drift')
    facts=json.loads((ROOT/'reports/local-integration-v1/c14-type-effective-facts.json').read_text())
    for name, expected in facts['identity'].items():
        if sha(ROOT/name)!=expected:
            raise ValueError('C14 evidence drift')
        add(name,'actual_C14_free_conditional_and_execution_evidence')
    extra=[
        'artifacts/c14-type-song-prefix-v1/expected-conditional-requests.json',
        'artifacts/c14-count-inputs-v1/report.json',
        'bench/rhythm-v2/c14-count-inputs-v1.json',
        'bench/rhythm-v2/c14-count-controls-reconciliation-v1.json',
        'bench/rhythm-v2/c14-training-prior-audit-v1a.json',
        'artifacts/c14-training-prior-audit-v1a/report.json',
        'reports/council-quality-c14/count-controls-reconciliation.json',
        'reports/council-quality-c14/type-serialization-repair.json',
        'reports/council-quality-c14/type-free-reuse-certificate.json',
        'reports/council-quality-c14/type-serialization-completion-tests.log',
        'reports/local-integration-v1/c14-count-inputs-effective-facts.json',
        'reports/local-integration-v1/c14-type-effective-facts.json',
    ]
    for name in extra:
        add(name,'C14_raw_or_arithmetic_fact_not_parent_verdict')
    for base, names in [
        ('artifacts/c14-type-song-prefix-v1/source', [
            'beatfcn/train/c14_type_probe.py','scripts/run_c14_type_song_prefix.py']),
        ('artifacts/c14-type-diagnostic-completion-v1/source', [
            'beatfcn/train/c14_type_probe.py','scripts/complete_c14_type_diagnostics.py',
            'beatfcn/train/c14_probe_support.py','beatfcn/train/native_diagnostic_priors.py',
            'tests/test_c14_type_probe.py']),
        ('artifacts/c14-count-inputs-v1/source', [
            'scripts/run_c14_count_inputs.py','tests/test_c14_count_inputs.py']),
    ]:
        for name in names:
            add(f'{base}/{name}','actual_probe_frozen_source')
    for name in ['scripts/analyze_c14_type_probe.py','scripts/audit_training_diagnostic_priors.py',
                 'beatfcn/train/c14_control_scope.py','tests/test_c14_control_scope.py']:
        source=ROOT/name; frozen=DEST/'source'/name
        frozen.parent.mkdir(parents=True,exist_ok=True);frozen.write_bytes(source.read_bytes())
        add(frozen,'arithmetic_or_control_encoding_source_snapshot')
    # The new review's goal/authority brief contains no parent diagnosis or peer opinion.
    brief='''# C15 目标与权限

目标是从音乐中学习节奏/情绪/作者意图相关的制谱选择，产出可继续编辑的Phira原生head/type/duration草稿。预训练、SSD、Transformer、AR、画布等只是可选手段。当前本地筛选的是下一笔投入价值，不要求产品出场；也不能用teacher/oracle或健康记忆替代独立歌曲自由证据。

Owner允许本地自由进化、修改合同/实现/读出/监督，尽量单干预；完整候选可以围绕一个生成假设整合设计，不能冒称已分离各组件因果。Council不限轮；Argon缺席不补位。每个失败必须分健康/质量/结构条件、讲明可检验归因、已有与缺失对照和后继必须改的合同。可以重新设计完整候选，不能把最近代码当作只能微改的限制。

保留native1ms、授权微秒COPY、同刻及完全重复多重集、短合法间隔、跨窗hold、用户known silence及可见参考。禁止隐藏真值条件、类型/数量配额、NMS、去重、min-gap、beat snapping、阈值/种子/最佳checkpoint挑选。失败与未测目标留原分母；对齐/合法支持、概率分数与自由质量分别报告。

当前参考模型10,240,592参数，冻结MERT L12/L23+10ms mel、SSD、cell count→canonical member→exact endpoint，count边际mode后条件greedy（非完整联合MAP）。C13正式fresh1000完成；C14仅终态冻结探针，0优化更新、0云。confirm16/8组件仍封存。云机制池累计最多6USD、实际/承诺0；本任务只评审，不派云、不授权约0.25B最终scaling。任何后继训练先真实数据/支持/梯度与固定batch拟合入场，不机械重做旧健康，不以产品全部能力作工程拟合门。

本地运行基准为隔离torch2.14、MPS FP32、4threads、memory_fraction .5；进程物理≤10GiB、driver≤min(.8recommended,10GiB)，0.5秒监护，禁止静默CPU回退，磁盘保底15GiB。新候选应给资源有顶的小训练合同；更大/更久不是默认修复。
'''
    (DEST/'goal-authority.md').write_text(brief)
    add(DEST/'goal-authority.md','goal_and_owner_authority_only')
    manifest=dict(id='c15-independent-mechanism-materials',files=sorted(objects.values(),key=lambda r:r['path']),
        total_files=len(objects),total_bytes=sum(r['bytes'] for r in objects.values()),
        policy='References only;no large evidence copy;exact allowlist;no peer/parent review/memory/history/weights/NAS/secrets',
        quality_weights_excluded=True,optimizer_updates=0,cloud_calls=0)
    write(manifest_path,manifest)
    common=f'''独立Council C15，工作根目录 /Users/yukoval/yukoval-projects/beatfcn。亲自完成，不派子agent。

先读 reports/council-mechanisms-c15/material-manifest.json，SHA256 {sha(manifest_path)}；{manifest['total_files']}对象/{manifest['total_bytes']}bytes。仅manifest与files[].path可读，不跟随未列import/链接/路径。先建立allowlist；逐对象核对字节/SHA，区分语义精读、局部解析/聚合、hash-only，记录实际路径与未读依赖。大JSON按字段提取，不整段倾倒。所有大证据只引用现存冻结文件，不复制。

允许标准库文件hash、JSON/文本/算术；禁止import项目/PyTorch/Numpy、模型前向、新预测/训练、安装、云、读取权重tensor。仅可写你指定的报告和访问审计两个文件。你不是唯一agent，勿改别人文件。不读其他Council原文、父级裁决/复盘/设计方法/建议池、其他线程/历史/记忆、NAS/凭证；环境注入摘要不是实验证据。目录名含council的明确allowlist源码/算术/修复回执可以读，未列的其他评审不可读。可查公开一手论文/官方资料并记URL，不以外部benchmark替本机事实。

目标/权限见允许的goal-authority.md；独立选择下一行动，不默认只能沿最近实现修小洞。材料含C13完整1000质量原始结果，以及C14两个零更新探针：count136前向、control-encoding重建恒等、B/Z/T共510自由和384 TT/SELF条件请求。原类型运行记录失败和独立补完各自保留，先核对修复/reuse边界；不能把JSON错误当模型质量失败或把补完改写成原运行成功。算术facts仅索引，需要自己核算重要数字/分母/支持。

要求：分层现象→机制→可检验预测→现有与缺失对照→可执行后继；不能只写没过门或'不能单组件归因'。逐类正/负proper、BOS/后续rank、TT/SELF、合法支持、真正自由结果分别判断；自身前缀条件分数不等于部署联合似然。不可支持的目标不能被丢掉以报有限均值，也不能不加分析就说生成law非法。注意原summary的endpoint failure是否包含head failure，及直接入口删除是否等于删除上游所有信息。

只给一个可执行默认下一实验，最多一个明确触发备选；为行动分稳定ID。可以选择有辨别力的零更新探针，也可以选择有机制支撑的完整新配方；不要机械继续采纳前轮未开训练。若选新训练，写清核心假设、保留/停用/整合组件、概率/决定规则、目标可达性与训练/自由差异、真实固定batch入场、资源帽、停止与研究增益门；小机制门和最终scaling/产品门分开。若推荐诊断，必须说明它哪种结果会改变下一训练，并给明确终点，不能无限增加旁路面板。讲清最强反对和未测边界，失败限定精确配方，不关闭方法家族。
'''
    focus={
        'G':'''重点：当前生成/排序支持、计数与类型决定的相互作用。区分固定rank真目标不可达与真正模型恢复失败，判断哪种最小改变能避免重演，同时不牺牲多重集和空窗。允许整配方重构，不强迫只修kind入口。行动ID C15-G-R01等。只写 reports/council-mechanisms-c15/grok.md 和 reports/council-mechanisms-c15/grok-access-audit.json。''',
        'S':'''重点：从音乐学习信号和整体生成目标重新选择路线，独立核对BOS条件学习、逐类正负、训练先验和自由非tap/hold。不用最近诊断默认锁死架构；也不因总体F1低默认加宽。指出一个值得实际训练的完整假说或必须先做的单一判别。行动ID C15-S-R01等。只写 reports/council-mechanisms-c15/sol.md 和 reports/council-mechanisms-c15/sol-access-audit.json。''',
    }
    for arm in ('G','S'):
        prompt=common+'\n'+focus[arm]
        (DEST/f'prompt-{arm}.txt').write_text(prompt)
        request=dict(clientRequestId=f'beatfcn-mechanisms-c15-{arm.lower()}-20261030',mode='async',isolation='shared',role='review',
            title=f'Council C15 · {"生成支持与后继合同 · Grok 4.7" if arm=="G" else "音乐学习与完整后继 · Sol Ultra"}',
            target=dict(providerInstanceId='grok' if arm=='G' else 'codex',model='grok-4.7' if arm=='G' else 'gpt-6.1-sol',options=dict(reasoningEffort='xhigh' if arm=='G' else 'ultra')),task=prompt)
        write(DEST/f'submitted-{arm}.json',request)
    write(DEST/'task-context.json',dict(status='ready',material_sha256=sha(manifest_path),files=manifest['total_files'],bytes=manifest['total_bytes'],prior_reviews_excluded=True,weight_tensors_excluded=True,cloud_spend_usd=0))
    print(json.dumps(dict(status='ready',files=manifest['total_files'],bytes=manifest['total_bytes'],sha256=sha(manifest_path))))


if __name__=='__main__':
    main()
