"""Authored mechanism judgments. Numbers come from identified raw receipts."""


def hypothesis(ident,observation,mechanism,prediction,judgment,evidence):
    return dict(id=ident,observation=observation,mechanism=mechanism,prediction=prediction,judgment=judgment,evidence=evidence)


def missing(question,comparison,decision):
    return dict(question=question,comparison=comparison,decision_if=decision)


def spec(story,hypotheses,controls,lessons,changes,forbidden,evidence):
    return dict(story=story,hypotheses=hypotheses,missing=controls,lessons=lessons,changes=changes,forbidden=forbidden,evidence=evidence)


def notes():
    out={}
    data=['reports/local_portfolio_data_scale_review_v1.json','reports/local_portfolio_data_scale_v1.json']
    for name in ('small24-2000','small64-2000'):
        out[name]=spec(
            '本项是数据覆盖对照，不是把旧容量健康失败算回小模型。两臂同1.824M、同初态、相同4000请求预算，24→64组件后完整F1 .219454→.261957、onset10 .212632→.223358。支持扩大训练池这个完整干预；变化也含样本组成与稀有类型曝光，不能把全部收益说成纯歌曲数量效应。两者都保留局部定位，类型与hold弱；该终态可作基线，本批未因未入围第二seed而判模型失败。',
            [hypothesis('data_coverage','同初态同更新扩池，完整F1增加.042503，onset10增加.010727。','覆盖改变比单纯重复24组件提供更可迁移的声音/谱面组合。','若覆盖有用，应在独立组件改善而非只降低训练CE。','支持整个扩池干预；纯数量/类型构成的归因仍未分离。',data),
             hypothesis('remaining_semantics','两臂总体通过筛选，但逐类型/空窗/参考仍需与下面原始分项一起看。','单参考事件监督加数量/槽读出可先学常见tap与时点，作者手势选择仍欠监督。','更大数据若只扩大tap，非tap/hold联合和参考收益不会同步改善。','可检验解释；不能由此认定容量不足或预训练必需。',data)],
            [missing('组件数量还是训练组成','固定类型/难度/请求曝光分布后做24/64匹配池，并保留相同总更新。','若收益消失，路由到组成/采样；若仍在，支持覆盖而非重复次数。'),missing('数据与容量能否叠加','同一64组件、优化日程与读出做小/大配置，或对新组件独立复验。','只有同条件改善才能把更大容量列为下一收益来源。')],
            ['保留同初态和实际曝光账，窗口不算独立歌曲。','扩池成功不能替代类型、尾端和条件诊断。','后续锁箱已打开，继续开发须另冻新组件。'],
            ['明确下一项改的是数据组成还是模型容量，记录类型支持和独立组件。','固定生成读出并做数量保持错位与逐类型验证。'],
            ['把容量64的旧健康失败当数据两臂失败。','靠增加相同窗重复数宣称新增歌曲。'],data)
    ssd=['reports/local_portfolio_mamba_quality_review_v1.json','reports/local_ssd_acoustic_routes_v1.json','reports/local_mamba_readout_v1.json','reports/local_portfolio_confirmation_v1.json']
    out['ssd-reference1000']=spec(
        '保留此路线的依据是独立歌声音定位、同权重读出对照和双seed复验，不是参数名叫Mamba。冻结1000步F1 .319804/onset10 .314420；去全部音频onset10 .017705、换歌.066143，而仅去整曲摘要32条事件流不变，说明此权重主要靠局部MERT/mel。相同权重改逐格mode数量比掉到.259、F1 .214309；基数DP .290550，四次T1约.222，原读出确有作用。2000终点下降到.286165，不应机械加步。确认两seed支持当前定位基础，但hold/flick/drag和参考收益仍弱，整体不能称产品成熟。',
        [hypothesis('local_audio_and_decision','正确onset10.314420，换歌.066143；同权重mode.190587、DP.281561，原读出.314420。','物理寻址局部声学提供时点，数量/槽决策影响能否把该信息读成稳定事件。','移除/错配声音或改为稀疏mode会掉定位；改读出不等于重训主干。','冻结权重支持路径和决策作用；不是SSD相对attention的独立因果。',ssd),
         hypothesis('semantic_and_global_gap','去整曲支路32/32事件相同；确认hold联合F1.0460、flick.0143、drag.0036，参考增益-.0011。','可用局部时点先学会，整曲融合和作者手势/参考信号没有产生相称收益。','加参数若只强化局部tap，原来的类型和参考问题会保留。','当前弱项直接成立；融合衰减、目标不对齐和条件数据不足需分开。',ssd)],
        [missing('SSD机制还是整个配方','共享输入/输出边界初值、数据/时钟/目标/日程的attention与SSD配对，分别报告相同更新及时间。','相同条件仍有优势才归给主干；当前跨配方排序不作主干结论。'),missing('全局无效是融合还是任务监督','参数相同的受控全局读取/门控与原路由；量和类型与相位分开。DirectSong固定强度试验已有负面完整F1。','后续路由必须比原完整事件更好或有明确任务收益，不能只使输入变化。'),missing('扩大容量能否修属性','把246M视为同配方容量对照，记录逐类型/尾端/参考及空窗曲线；新增确认组件。','只涨tap定位就只支持定位容量，不称解决制谱理解。')],
        ['保留证实有用的局部声音与原生寻址。','旧1000与2000身份都保留，不能用后验最好步数讲收敛。','下一项必须明确处理/测量非tap、空窗、参考，而非只报告总F1。'],
        ['原读出与mode/DP比较随新权重重做，避免把数量功能错配带上去。','整曲路线改动先测实际增益；容量试验只回答容量，不替其他弱项交卷。'],
        ['把Mamba名称当因果或CUDA吞吐承诺。','把本地定位通过解释成全局/情绪/可玩通过。'],ssd)
    mel=['reports/local_portfolio_mel_quality_review_v1.json','reports/local_access_quality_review_v1.json']
    out['mel-2000']=spec(
        '此9.644M从零mel终态不是空化：数量比.939、完整F1 .178637，但onset10仅.103184，且hold/flick没有正确匹配。继续到2000未解决精细位置和属性。不能据此说mel信息不够，因为另两条从零mel的access路线达到约.30并复验。否决的是这一编码/混合/目标/读出组合；优先检查细时间条件读出和损失分工，而不是把前端换成预训练当作唯一修复。',
        [hypothesis('coarse_time_and_type','全数量未塌，严格10ms定位弱且hold/flick匹配0；原review含1000/2000曲线和空窗17。','模型学到了大致事件强度，细时刻/稀有属性分布没有被读成准确事件。','放宽到50ms或只看数量会显得较好，类型与严格时间仍差。','支持症状分解，未做该模型固定数量head-only对照，不能断言唯一码头。',mel),hypothesis('frontend_necessity','同为从零mel的access模型F1 .30级且确认通过。','可训练mel信息与计算组织/目标的配合影响结果，不是没预训练就无法定位。','若mel不可用，其它mel路线也不会在独立歌有稳定定位。','否定预训练必要性的强断言；不等于两模型算力匹配。',mel)],
        [missing('读出还是特征','同权重固定数量，比较原偏移与受控head-only决策；测真实数量条件下headCE和合法均匀基准。','teacher好而自由差改决策；两者都弱改声学/时间头。'),missing('混合结构还是规模/日程','同参数日程种子比较原temporal和access骨架。','匹配后仍改善才归因计算组织，不能用跨大小胜负定前端。')],
        ['保留从零路线研究资格。','严格时间与类型拆开，不用正常总量掩盖偏移。','记录1000到2000实际变化，不自动再延长。'],
        ['先对细时间读出作固定数量诊断。','后继不只换前端，明确损失与混合结构改变。'],['称mel-only家族失败。','以总体密度正常代替定位。'],mel)
    package=['reports/local_portfolio_package_quality_review_v1.json','reports/local_package_readout_review_v1.json']
    out['package-2000']=spec(
        '该归一化hurdle+GRU的原主读出在2000仅15/1389个tap，F1 .008073；教师/概率可以非空不代表逐格众数有事件。冻结T1次要读出F1 .146507/onset10 .126541，对帧置换.114113/.066330仍有声学定位差，但两个空窗每draw误发24/35/19/34。主失败是分散概率与逐格众数的功能不一致；改成采样只从空化换成嘈杂稀疏事件，仍未达到可靠原生属性。',
        [hypothesis('sparse_mode_mismatch','主众数15个，T1恢复大量事件而时间超帧置换。','独立位置上的零为众数，同时事件概率质量分散在全窗；mode不保留总概率质量。','同权重采样会恢复数量，不必重新加宽。','同权重干预支持；不是GRU完全学不动。',package),hypothesis('diffuse_distribution','T1有定位差但空窗持续误发，且严格时间/完整F1仍弱。','非空质量中存在真实声学线索与错误概率质量的混合，单换采样不能修校准与属性。','保数量零模型差保留，但空窗和联合尾仍失败。','支持整体分布问题；未隔离数量与成员采样各自份额。',package)],
        [missing('数量和成员谁主导','固定原采样数量、只改成员；固定成员条件、比较数量法，禁止用真数量当正式输出。','若只改数量解决空化却不解决定位，则改时间/属性合同。'),missing('联合训练概率改法','匹配自然计数先验/尾概率与原hurdle，保持声学和预算。A2已证明简单全基数DP可能打包未见16人组。','先做纯先验DP反例与未见count条件审计再训练。')],
        ['mode空化不是无信息的同义词。','T1次要结果不替换原主终点。','概率质量与整窗空谱概率都要报。'],['计数与成员决策分开，训练前审计未见乘数的概率尾部。','正概率不能靠人工数量/类型配额变成通过。'],['把采样数量恢复当机制成功。','重复逐格argmax而不审计P(N=0)。'],package)
    access=['reports/local_access_quality_review_v1.json','reports/local_portfolio_confirmation_v1.json']
    for arm in ('global','mixed'):
        out['access-'+arm]=spec(
            '两臂16.696M与所有初始张量完全相同，只改变两层attention访问mask；上游音频/address本来就能全局访问，故这里不是“模型看没看整曲”的实验。mixed完整F1比global高.011145，但onset10低.037378，drag TP10→20、hold TP均8、flick均0，是较粗事件/类型与严格时间的取舍，不是全面改善。两条独立模型都过本地门，mixed按既定排序入围并复验；不因联合假说失败撤销可用基线，也不把没有MERT视为缺陷。',
            [hypothesis('access_tradeoff','full .300751→.311896，onset10 .300427→.263049；仅两层mask改变。','中程窗口化改变局部模式的整合，使该预算下粗匹配与严格定位的取舍不同。','若全面改善，full和严格时间应同向且组件广泛受益；实际不是。','受控mask的总效应成立；哪层/哪类模式引发取舍未隔离。',access),hypothesis('scratch_audio_sufficiency','输入仅10msmel及其0.5秒池化，无MERT/Pupu；mixed第二seed与确认均过。','从零训练也能学到足够的局部音频相关定位。','输入错配应削弱定位，复验不能只靠窗口密度。','支持此任务/预算的局部定位路线，不证明情绪、整曲编排或预训练无价值。',access)],
            [missing('访问mask具体损伤点','同权重对两层窗口mask逐层恢复，分1/5/10/50ms、类型与同刻组对照。','只一层影响fine定位就单修那层，避免全部全局化。'),missing('类型与参考监督','同配方对齐训练中的多谱风格/参考差异并保留空窗，而非只增加深度。','真实参考的同权重收益和非tap匹配改善才支持作者选择学习。')],
            ['访问分工试验失败不等于两条候选都失败。','把预训练当实现方式，不当能力定义。','层的实际感受野与输入整曲长度分开报告。'],['保持严格时间/类型联合账，不只看full增加。','后继同根任务增加可验证作者选择信号或作mask因子对照。'],['把8秒canvas全局attention称整曲理解。','用更多层自动宣称更多推理。'],access)
    lattice=['reports/local_portfolio_contract_diagnostics_v2.json','reports/local_lattice_complete_v1.json']
    out['local-lattice-complete-v1']=spec(
        '全空不是程序未结束或原生表示装不下：真实健康通过，2000请求目标往返无损，正式所有草稿/补全/高低密度仍零事件。冻结权重的四个不同开发组件，presence仅.0539–.0565附近，AUROC .420/.455/.379/.510；期望数量58.7–59.7几乎恒定，目标16–98。两个空窗也期望51.0/53.1。因而不只是0.5阈值太高：概率排序本身接近或低于随机，改阈值会把密度先验摊成假音符。',
        [hypothesis('presence_prior_attractor','四窗p约.055，0个过.5，AP大多不优于占用率；空窗期望同样高。','Focal、全局期望铰链及属性训练的组合保留了近常量占用先验，未形成音频起音辨别。','仅调发射阈值无法选择真实位置，应看到AUROC/AP弱且不同密度窗期望相近。','冻结概率支持；不是把所有失败归于阈值。',lattice),hypothesis('objective_competition','训练有1016个反事实请求，但硬输出全部空，软数量可非零。','铰链优化期望presence×人数，未直接约束真实起音排序；共同骨干上的属性损失可能压过presence学习。','presence-only与原联合损失的同架构对照可区分竞争和特征不可分。','可检验的目标解释；各项因果贡献尚未由单项训练隔离。',lattice)],
        [missing('presence学习信号还是音频特征','同一输入/骨干/人数头，先匹配presence-only自然BCE/Focal与联合目标，记录共享梯度与heldout PR。','若单任务有排序，路由到目标竞争；若仍随机，查特征/地址对齐。'),missing('反事实铰链的替代自由度','记录p与条件人数分别变化，固定人数的零更新计数诊断；再做只改铰链训练的配对。','若只改人数满足期望而presence不动，重写条件合同，不能靠quota。')],
        ['原生1ms/COPY容量验证不等于学会发声。','软期望、排名和硬决策同时验。','概率近随机时禁止阈值搜索来制造非空。'],['新合同先让presence学到独立歌排序，列共同梯度分工。','任何条件项必须同时核对位置与人数贡献。'],['再缩网格或加类型配额后声称定位修复。','把所有Focal/10ms模型一起否掉。'],lattice)
    flow=['reports/local_portfolio_contract_diagnostics_v2.json','reports/local_flow_within_cell_oracle_null_v2.json','reports/local_event_code_flow_v2.json']
    out['local-event-code-flow-v2']=spec(
        'v1的MPS index_copy不支持是0更新实现停止，v2已独立修复。v2质量失败是连续数量码残余噪声，经最近整数投影放大成过密：数量码相邻差.125、半边界.0625，零计数组残差std .300–.330，40%–45%跨过第一计数边界；即使空窗也生成762/819个。16→64Euler未改善，不能只补求解步数。81维码里数量只占1/81，绝大多数成员槽为背景；这份均匀MSE没有把稀疏离散计数读稳。',
        [hypothesis('continuous_count_noise','四开发窗数量码MSE .094–.112，空窗.090/.099；残差远大于.0625，原数量比18.803。','连续码的残余方差穿过密集计数边界，被逐格投影累积成大量假事件。','若计数噪声主导，空窗同样过密；更细Euler若不是主要根因，不会清掉方差。','实测空窗与16/64配对支持，属于已训练向量场/决策合同问题。',flow),hypothesis('mse_allocation_and_fine_info','count仅占总速度误差约1.2%–1.5%；活跃槽仅0.1%–0.8%。真cell/type计数oracle会令50msF1近1，但它已泄露格/类型，细时间并不稳定优于同格均匀随机。','稀疏离散对象被背景与属性欧氏重建主导，主观码距离不等于事件决策质量。','单纯把oracle高F1当条件发射成功会被同格随机对照揭穿。','统计和oracle-null支持该风险；需匹配目标分解训练才能归因MSE权重。',flow)],
        [missing('损失加权还是连续表示','同音频/训练预算，分离归一化的count目标与条件属性流，或真正离散count变量；原生成员支持不变。','只有计数校准与细时间一起好转才支持修复，不以输出总量正常就通过。'),missing('条件向量场是否用声音','相同终态码比较真实/置换音频的count与active-member速度误差；再匹配训练排除条件失用。','声学不影响有效方向时修条件注入，不能只加Euler步数。')],
        ['连续可逆编码只保证能表示，不保证噪声尺度适合离散决策。','计数oracle泄露40ms时间和类型，50ms高分不算新证据。','全空/全密要看码坐标残差与决策边界。'],['重写count与active-member目标分配，保留完整原生支持。','增加计数噪声尺度、正确音频收益和同格随机对照。'],['用更多Euler步数或删音符充当改模型。','把学习VAE未实施路线说成已实测否定。'],flow)
    canonical=['reports/local_canonical_group_phase_diagnostics_v2.json','reports/local_canonical_count_packing_mechanism_v2.json']
    for name in ('local-canonical-package-quality-v2','local-ssm-event-package-v1'):
        evidence=canonical+[f'reports/{name.replace("-","_")}.json']
        out[name]=spec(
            '原实现合法和质量不足是两层。数量DP得到正常总量，却将很多事件打包为未见的16人组：A2有45组、SSM包53组；对应720/848个事件，占总1071/1192的大部。训练count16支持为0，count embedding的Adam一二阶动量均为0。只把半计数平滑的训练先验送DP，无音频/无模型就把48个事件配成3组×16。因此它不是先缺容量：条件数量MAP会偏爱少付presence代价的高乘数，并让GRU在未训练的count条件下推理。格内非递减贪心随后集中到38ms，放大错误。',
            [hypothesis('count_map_packing','A2/SSM主总量比.771/.858；16人组45/53，训练从未有16，纯先验DP也3×16。','全局基数选择后只最大化count-law，平滑的未见高计数尾部按每事件代价更便宜；分配忽略成员条件概率。','不需要骨干或音频就会出现打包；未训练count16嵌入动量为0。','解析反例、真实分配和优化器状态共同支持；DP忠实执行目标，未发现回溯算术bug。',evidence),
             hypothesis('untrained_member_condition','A2 783/1071、SSM907/1192起点在38ms；大乘数占大部，逐格众数全空，T1次要比主greedy更好。','GRU被请求未见count16，首个错误offset后非递减合法域限制后续，产生重复的格尾模板。','相位集中应主要在多成员组；只换主干会保留相同故障。','多成员相位和跨主干复现支持；需固定数量单改head进一步分离。',evidence)],
            [missing('概率尾与MAP目标','同权重固定模型总基数，比较自然衰减的软计数尾先验/成员概率入分配与原目标；保留count16合法支持。','不再无依据选择未见16并改善位置，才支持尾/决策修复；禁止硬截count。'),missing('数量与GRU分别贡献','同一分配固定后比较head-only/tail-only或非AR成员头；另以原成员固定改分配。','若正确分配仍格尾锁定，单修成员；若分配解释大部则先修计数。')],
            ['正常总量可隐藏极端局部打包。','未见count条件必须审计，平滑不能等于给所有高乘数同样尾质量。','A2原健康计数门失败保留，后来的fresh质量不改判旧健康。'],
            ['训练前运行无音频先验DP反例及所有乘数曝光/嵌入状态审计。','后继必须改变尾概率/分配目标或成员决策合同，不能只换SSD/attention。'],
            ['再把同一DP+GRU移到更宽骨干。','硬删合法16人组或用总数量门掩盖聚集。'],evidence)
    grouped=['reports/local_grouped_chart_review_v2.json','reports/local_process_frame_sources_v1.json','reports/local_grouped_address_alias_diagnostic_v2.json']
    out['local-grouped-chart-v2']=spec(
        'v2数值相位修复后健康通过，T1主生成不是1ms塌缩：正间隔中位160ms、<10ms约9.75%；greedy次要才是中位1ms、<10ms96.91%、数量比21.05。源帧置换令onset10 .142481→.018082，证明跨格用音频。新增源码/梯度检查却发现同一40ms格的所有候选共用address_key(x_frame)，它在格内两毫秒地址的logodds中抵消；固定全局query时24格局部音频梯度全为0。原生ms可输出，但候选局部的格内音频对比没有实现。不能把这条实现限制说成AR家族或容量失败。',
        [hypothesis('within_cell_audio_alias','4窗24格，固定query的格内音频logodds梯度0；native submsphase恒0；直接换格音频项只改共同偏置。','音频项与clock项加性可分，格内由共享时钟模板而非各地址局部声音评分。','改x_frame能改变跨格分数，却不能在固定query下改变该格内相对毫秒偏好。','代数、自动微分和替换对照支持；全局pooled/query仍能依赖声音，不说所有音频无效。',grouped),
         hypothesis('greedy_boundary_feedback','greedy1ms占96.91%，T1中位160ms；参考增益-.00463，T1完整F1.118998。','单调合法边界与近似平滑地址分数形成贪心推进吸引子；采样逃离它但细时间/类型仍散。','状态/音频掩蔽与固定正确历史的自由生成应揭示反馈份额；不能由teacher CE代替。','生成对照支持该吸引子，未独立证明哪段历史反馈为主。',grouped)],
        [missing('局部候选音频交互','同骨干/计数/STOP，把keys改为显式frame-audio×native-offset或候选mel读出非线性交互，先验证格内audio logodds可变化。','若alias消除并提升严格位置，才认定物理音频读出修复；只加时间embedding不够。'),missing('历史还是地址造成微步','同权重保留previoushead合法域，分别固定正确历史内容与生成历史；只改头部地址交互单独训练。','若反馈恢复也不改变格内偏好，先修地址合同；若先修地址后仍微步再修序列目标。')],
        ['区分T1主失败与greedy次要崩溃。','候选能表示ms不等于候选能用局部ms音频。','STOP起始合法且主输出非空，不能再归咎共享END塌缩。'],
        ['新AR合同先验证候选局部音频与格内地址真正交互。','保留合法短间隔，不用最小gap挡住吸引子。'],['只更换Mamba/Transformer而保留别名地址打分。','把teacher历史剥夺当自由生成因果结论。'],grouped)
    diffusion=['reports/local_group_diffusion_quality_v1.json','reports/local_process_frame_sources_v1.json','reports/local_group_diffusion_quality_review_v1.json']
    out['local-group-diffusion-quality-v1']=spec(
        '原100步计数/空窗健康门失败真实保留，后来是显式的新工程入场，不是旧门翻案。新鲜1000的32步四抽样全部完成，数量比.850、完整F1 .147497/onset10 .125016；不像旧slot扩散的纯tap模式，也不像连续flow的18倍密度。类型有flick/drag命中，但hold10联合四抽样均0，空窗平均26.75。源帧置换sample0将onset10 .123963→.074176，仅.049787、10/16正向；存在声学收益但不够广且精度低。',
        [hypothesis('diffuse_event_distribution','四T1均可完成并恢复类型/数量，但严格时间弱，空窗持续响，hold10联合0。','完整包去噪避免单token循环，却仍学习了较散的count与条件成员分布；更多反向步骤本身不保证稀疏位置正确。','若只是旧语法循环，非AR固定32步应解除；实际解除结束问题但未解除定位/校准。','支持生成终止与质量分离；不足以唯一定位denoiser还是canonical成员。',diffusion),
         hypothesis('partial_audio_grounding','同sample0源帧置换onset10下降.049787，CPU32/32精确重放；count .853→.816。','编码声音包含部分位置信号，去噪/读出尚不能可靠使用所有组件。','更纯声学干预应保有优势；该置换还移动编码位置并改变count，不能当保数量因果。','有条件支持输入依赖；不证明所有概率质量有效或hold结构学会。',diffusion)],
        [missing('去噪count还是成员读出','冻结终态count包，分别做count已知/未知的条件成员评估并加同格随机null；只改变去噪步数的零更新诊断单列。','count-oracle泄露格/类型，若只50ms高分不接受；严格时间/尾端才指向成员能力。'),missing('训练噪声目标/稀疏概率','固定音频/参数比较噪声时刻覆盖与校准的count目标，报各t的count和active-member NLL。','高噪声count先验失控先修目标；低噪声很好而自由链漂移再修反向过程。')],
        ['它是真吸收腐化/噪声时刻/反向链，不拿反复argmax冒充。','原健康失败与后续工程准入分账。','空窗与hold联合不能由合理数量和类型存在替代。'],['后继先分解噪声时刻、count、成员和联合尾端。','避免沿用A/SSM未见高count条件而不审计。'],['单因.20未过就否定扩散家族。','挑最佳draw、加配额或改温度救终态。'],diffusion)
    noisy=['reports/local_portfolio_contract_diagnostics_v2.json','reports/local_noisy_proposal_refiner_review_v2.json']
    out['local-noisy-proposal-refiner-v2']=spec(
        'v1在合法空提议请求267步处停止是调用遗漏，v2修复后完整训练，二者分账。质量拒绝的实质是精修毁掉强提议：同源1545个提议变991个，完整F1 .319804→.152324，onset10 .314420→.281279；仅1/16组件完整F1胜出。它将509个事件判为hold，但联合正确仅17。冻结模型的类型盲匹配混淆表里真实tap517个仅176保留tap，171变hold、163被拒；未匹配提议也有170变hold。类别平衡的行动logits没有自然频率校准。',
        [hypothesis('classification_prior_miscalibration','128个真实训练增强请求的自然标签比例约tap.265/hold.110/flick.072/drag.168/reject.384；原模型大量hold。固定train频率偏置令full .2210但onset.2403，所有非tap几乎消失。','逐请求逐类等权CE改变决策先验，属性证据弱时给稀有类过高优势；简单校正又退成tap/reject。','若先验失配重要，固定train-only偏置会大幅改变hold误判；若它是唯一问题，应能恢复原源F1及非tap，但没有。','干预支持先验敏感；否定仅加logprior即可修复，非精确Bayes校准证明。',noisy),
         hypothesis('proposal_and_label_identifiability','原source强而精修丢tap/加hold；训练混用真值jitter与实际源，推理只有实际源，重复同刻提议的typeblind配对可不唯一。','训练提议分布与作者类型标记可能不匹配，且本模型不能插入源中没有的任意起点。','实际源-only训练/固定头属性-only对照会分离曝光差与头尾修改的伤害。','有结构依据和混淆证据，增强分布/配对歧义的因果份额未单独测。',noisy)],
        [missing('平衡CE还是表示不够','同body/提议/增强/日程，真实频率CE与逐类均衡CE配对，校准只用训练留出的数据。','恢复非tap且不毁正确tap才支持分类修复，整体F1靠全tap上涨不算。'),missing('增强迁移还是精修位置伤害','实际SSD提议only vs原jitter混合训练；分别固定head做属性与固定属性只修head。','源-only改善则修提议训练合同；固定头改善则取消无依据的位置改写。')],
        ['先有强baseline，精修必须报每类保留、误转和拒绝代价。','不能把平衡训练logits当自然后验。','空提议修复是实现问题，与后续质量失败分别解释。'],['后继合同先验证类别校准、真实提议分布与重复事件标记。','保留原源草稿，比较新增收益和被破坏的正确音符。'],['以少数非tapTP增加掩盖大规模tap损失。','把先验偏置全tap高分当语义修复。'],noisy)
    context=['reports/local_mert_context_pair_review_v1.json','reports/local_mert_context_cache_audit_v1.json']
    for arm in ('full','short32'):
        out['direct-song-'+arm]=spec(
            '同一MERT权重/层/提取runtime、相同时间戳、下游参数和实际4000监督/共同mel均精确配对。full完整F1 .282290、short32 .286833；full onset10高.014421，但完整事件低.004544、只7/16组件胜出，两秒数量误差仅降6.27%。两者后续都池化完整歌曲，所以此试验只检验提取上下文，并未移除下游整曲输入。附加远端信息未带来预期的联合收益，不能推断缓存没信息，也不能说短上下文在所有容量最优。',
            [hypothesis('extra_extraction_context','只有冻结编码器的提取上下文改变，full在完整事件/组件覆盖不胜，严格时间略好。','更长预训练上下文的信息在当前池化/小下游与事件目标中未转化为额外完整事件。','若额外上下文有实用收益，完整F1、组件广度与结构误差应共同改善。','这组受控效果不支持预期增益；不等于完全没有上下文信息。',context),hypothesis('access_vs_usefulness','DirectSong路由两臂相同；完整全局数据已输入，full的去整曲/换整曲干预也改变输出。','不是完全没接上；相关性、池化保留、作者选择监督可能限制信息用途。','若瓶颈在池化，固定编码器的分辨率/局部保留对照会改善；若目标不需要则不会。','候选解释，缺少池化/目标独立对照。',context)],
            [missing('提取上下文还是池化瓶颈','同前端输出比较当前0.5s摘要与保持局部细分/跨段检索，投影预算匹配。','长上下文只在保留细信息后获益才指向池化。'),missing('容量还是编排目标','固定full/short配对，在同预算新下游或带任务相关段落监督复验。','若只扩大tap，仍不能称额外音乐理解。')],
            ['完整音频覆盖和编码器注意力上下文是两件事。','两臂所有策略差异明确，不能以换MERT掩盖优化器/读出变更。','27秒控制音频两套特征逐值相同，长曲变化不等于质量收益。'],['后继声明改提取、池化、路由还是监督，只改一项来归因。','保持同一编码器/数据/实际监督配对。'],['用维度1024→1152保证语义保全。','把full负面小模型结果当MERT族无效。'],context)
    route=['reports/local_direct_song_routing_review_v1.json','reports/local_mert_context_pair_v1.json','reports/local_concat_control_quality_v1.json']
    out['local-concat-control-quality-v1']=spec(
        '这是原串接路由在新batch4合同的真实对照，不是旧SSD1000换名字。它与DirectSong/full同初态、参数和4000实际监督；串接fullF1 .310364，直读.282290，直读起音10却高.025157。直读只5/16组件full胜出，tap TP377→351、hold9→5、drag54→29，数量比.952→.844。读到了整曲不等于这些额外特征有益；此处保留串接作为配对中更好的完整事件控制。',
        [hypothesis('routing_tradeoff','参数/初值/数据/目标完全匹配；直读严格时间好但完整事件和类型匹配差。','固定强度的新路由改变局部特征、数量与属性的平衡，可能干扰已有有效路径。','若全面有效，不能靠少发事件或只改头精度换掉大量正确类型/尾端。','受控路由总效应成立；幅度/门控/语义各自责任未隔离。',route),hypothesis('connected_not_necessarily_useful','移除/换整曲能改变DirectSong输出，但它仍输完整事件。','输入依赖和任务收益是不同条件。','提高敏感性而不提高正确上下文相对控制的质量，不能叫融合修复。','现有干预直接支持这一区别。',route)],
        [missing('额外特征幅度还是内容','同初态固定预算，测试规范化/门控的单一路由改动；冻结权重时作同范数错内容对照。','幅度控制修复而内容错配无益才指向注入尺度。'),missing('数量变化还是类型变化','固定合法预测总量的跨模型诊断，分别比较head、类型和tail，不用隐藏数量输入。','同量仍退化修属性融合；仅数量差解释则修计数校准。')],
        ['新路由必须和原同初态比较。','不以全局输入敏感度当整曲理解收益。','旧batch2与新batch4不可当纯batch因果。'],['有全局依赖目标时先解决注入平衡和监督，不盲目固定强度相加。','逐类和数量副作用纳入路由复盘。'],['用onset单项上涨掩盖完整事件退步。','把相同参数量当相同语义利用。'],route)
    attention=['reports/local_attention_control_quality_v1.json','reports/local_portfolio_scale_v1_width_probe.json']
    out['local-attention-control-quality-v1']=spec(
        '旧10M在100步稀有类F1健康门失败，只证明早期读出门没过，不是容量极限。后来明确改为拟合/数值/合法性入场，丢弃健康权重，1000步真实独立歌F1 .251031/onset10 .259413，说明原早拒绝过于粗。当前模型的类型表仍是tap主导，hold8、flick3、drag40个TP；合法非AR仍有属性问题，所以去掉token历史反馈不是充分条件。此候选完成但未进冻结前两名，不将未复验写成实测失败。',
        [hypothesis('early_health_not_capacity','原稀有类型健康门未过，新的fresh质量通过。','短拟合解码和长程跨歌优化回答不同问题，尤其稀有类小支持数F1敏感。','若原100步足以裁决，后续同表示不应产生有效独立歌定位；实测反例。','否定原健康失败即可判容量极限。',attention),hypothesis('nonar_not_sufficient','独立帧输出可结束且有定位，但罕见类型/hold联合仍弱。','共同的作者类型/尾端信号与场读出还限制表现，不只AR历史。','换同类非AR主干不一定修属性，需相同输入/头的机制对照。','症状成立；尚无单独证明某个共享损失造成。',attention)],
        [missing('增加宽度还是策略差异','同种子/数据流/有效批/日程/边界初值的小大attention配对。','匹配后才谈宽度收益，不能拿small64不同日程作因果。'),missing('类型与尾端损失','同骨干分解真实head条件tail和真实数量条件kind，另对作者可见风格/参考监督作配对。','若条件诊断仍差先修监督/读出；仅自由差再修联立决策。')],
        ['健康准入与质量判别分层。','10M大小不是原因本身。','不把没有进入前两名叫架构淘汰。'],['下一容量/主干试验匹配训练策略和边界。','保留逐类与hold联合，不只看完整F1。'],['重演100步稀有F1即关大模型。','凭一次跨种子排名宣布attention不如SSD。'],attention)
    depth=['reports/local_music_depth_pair_v1.json','reports/local_research_pair_field_diagnostics_v2.json']
    for arm in ('wide8','deep16'):
        out['music-depth-'+arm]=spec(
            '本次检验参数预算如何分配，不是数学上纯层数因果。浅宽5.096M、深窄5.223M，共同输入/输出初值一致；深窄full少.012512、onset10少.021008，只7组件胜出，训练535.6s对424.4s。固定面板深窄训练复合损失略好6.3003<6.3842，开发反而5.6731>5.6270，主要offset/数量项较差；这比“更深没用”更像泛化/归纳偏置取舍。去掉已学temporal后两臂定位都大跌，故深层并非完全未使用。',
            [hypothesis('allocation_generalization_tradeoff','同面板深窄训练offset3.5575<浅宽3.6144，开发3.6443>3.6002；count开发也差。','增加串行组合并缩窄，可能更适合拟合已有样本而不适合当前fine-time迁移；不是数值不收敛。','若只是没有学会训练，训练项应同样更差；实际训练稍好/开发稍差。','支持这组分配的泛化取舍；不是已证明过拟合唯一原因。',depth),hypothesis('temporal_is_used','16首窗trunk关闭，浅宽full .32425→.09804/onset.27888→.06544；深窄.30163→.07257/.25146→.04894。','已学时间混合对两臂均有用，失败是分配的相对收益，不是整个trunk无效。','移除trunk会损伤，两臂都如此。','冻结干预支持；OOS删除不等于重新训练无trunk基线。',depth)],
            [missing('泛化还是优化预算','相同预算分配下独立种子及固定面板曲线，单独增加正则或只改日程，分别冻同时间。','深窄训练更好开发更差持续出现，才支持泛化问题；多步改善则预算不足仍在。'),missing('深度还是变窄的子效应','在同接口设计浅窄/深宽对照，或固定参数的其他分配，各自独立预注册。','2×2交互支持后才归某个单组件，不能靠“参数近似”取消变窄变量。')],
            ['所有层数和跨宽边界投影计入参数。','原D136/head4违反四分RoPE，改144在训练前完成并留档。','同时间包络与同更新结果分开；本次两臂1000均在600s内。'],['容量分配后继先查fixed-train/dev分项和偏移泛化。','真正只问深度时匹配宽度/预算或明确联合分配假说。'],['宣称更多层等于音乐推理。','训练拟合更好就当开发质量更好。'],depth)
    ssl=['reports/local_music_ssl_pair_v1.json','reports/local_research_pair_field_diagnostics_v2.json','reports/local_ssl_retrieval_diagnostic_v2.json']
    for arm in ('supervised','matching'):
        out['music-ssl-'+arm]=spec(
            '相同1.602M/初态/音频视图与额外头，唯一训练差异是局部—上下文InfoNCE权重0/.1。匹配确实学起来：8训练/16开发检索正确率从.25/.1875到.75/.625，开发auxCE1.4036→.7896；但完整F1仅+.005119、9/16胜出，正确粗上下文相对错配的收益反而少.007276。固定面板监督loss几乎相同5.70218/5.70048。因此失败不是“辅助任务根本没学”，而是预文本对照关联没有充分转成所需编排选择。',
            [hypothesis('pretext_transfer_mismatch','检索开发.625明显高于.1875控制，但监督字段与完整事件改善小，coarsebenefit增益为负。','同一录音的局部与周围匹配可由音色/录制/连续纹理线索解决；不必学习作者的手势/留白选择。','辅助检索好而编排/正确上下文优势不相称。','症状与目标不对齐解释受支持；音色捷径本身尚未用匹配音色对照证实。',ssl),hypothesis('timing_partial_gain','onset10+.016276，full+.005119，fixeddev offsetCE仅改善.00389；tail略变差。','辅助信号帮助一部分声音关联/时点，却没有让各字段共同提升。','不应只报告auxloss或onset单项；尾与参考/上下文可能维持弱。','观察成立，不能据此说所有音乐SSL无效。',ssl)],
            [missing('录音身份捷径还是任务对齐','保持当前同等音量/节奏proxy，加入经标注的同曲不同段/不同编排对照或匹配音色的异曲；保留真实重复动机为正/不确定。','只有去掉身份捷径后任务与正确上下文收益同涨才支持音乐结构学习。'),missing('辅助权重还是目标本身','先在同架构上单独对任务相关的正负例定义作配对，权重固定；若再测权重另冻一轮。','正确例定义有效后再调权重，禁止仅换.1→.2当作已吸收失败。')],
            ['预文本学习成功与下游任务学习成功分开。','2秒中心与12秒环绕无重叠、视图时钟归零已核对。','不把检索准确率或更低auxCE称情绪理解。'],['下一SSL合同改变任务相关监督/正负例判别并预先规定正确上下文的收益。','继续保留同参数/同额外音频计算的监督控制。'],['重复辅助loss下降就是成功。','把同曲所有片段都当负例损坏合法重复。'],ssl)
    return out
