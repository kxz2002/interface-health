# TT 注入关联标签基础设施 Spec

日期：2026-10-20。用户已认可前序Spec内容，本文正式整理该设计。
配套实施计划见 [Plan](labeling-infrastructure-plan.md)。Task 1–5 已实施打标与生产验收，未训练；实际计数及证据更正见 [实现报告](labeling-infrastructure-implementation.md)。

## 目标与语义

第一版实现方案 A：应用 Service×15秒窗口的注入关联评价标签。
标签不等于完整健康状态真值。保留 Normal-only 拟合、独立 Normal 校准、
Fault 仅测试和完整 Case 隔离。方案 B 的执行证据准入要求后续单独细化。

## 已确认的标签规则

- 正类：直接注入目标明确映射到25个应用服务候选之一，且完整窗口满足
  `actual_start_ms <= window_start_ms < window_end_ms <= actual_end_ms`。
- 负类：独立 Normal Case 及 Fault Case 首次实际干预请求之前的完整窗口，
  通过预定运行质量准入且无已知异常或前次故障残留记录时，称为协议正常。
  故障前窗口要求 `window_end_ms <= first_intervention_requested_ms`，不能用更晚的
  生效确认时刻替代；请求时间缺失或存在更早干预疑点时保留 Unknown。
  质量准入及例外规则见下文。
- 其余保留 Unknown，包括部分重叠窗、干预开始后的非目标服务及恢复段。
- 数据库注入不向应用调用方展开；不依据 Case 名推断实际症状。
- 使用实际执行记录的开始与撤销时刻，不用计划时刻替代。
  缺失、无效或瞬时操作不能强行构造持续区间。
- Unknown 不属于第三个预测类别，不转为正常，不进入相应分类指标。

## 已确认的主划分

6个 Normal Case 按4个拟合、1个独立阈值校准、1个独立正常测试分配。
11个 Fault Case 全部仅测试，其故障前负类不得用于拟合、校准或模型选择。
完整 Case 不跨划分；所有模型与基线共享划分。具体 Case ID 在实施前固定，
不得根据故障测试效果选择分配。阈值规则尚未确定，不默认99%分位数。
该4/1/1决策替代既有准备产物的3/1/2划分；原产物保持不变，
后续生成独立版本的划分侧表，并将标签、划分及评价结果绑定。

采用时间顺序分配，避免依据结果挑选 Case。以下清单作为本协议固定划分，
不是对所有窗口正常性的认证：

|用途|Case ID|
|---|---|
|fit|Normal_20261017T020706Z_95064d21e8204018|
|fit|Normal_20261017T110632Z_4c14d1c71d664ca9|
|fit|Normal_20261017T115148Z_cc9a6da5e99a4263|
|fit|Normal_20261018T030819Z_9adec05d73684fb6|
|calibration|Normal_20261018T035358Z_4496c85fde8e4f6e|
|test|Normal_20261018T153211Z_e448a74ed7104ebd|

## 运行质量准入与已确认例外

仅加载 selected17。来源hash、Case/attempt/plan身份、主键或时间范围不一致
属于资产完整性错误，生成程序应报错退出，不发布部分标签包。
运行准入要求 collection_status=complete、coverage_status=passed、
processing_status=passed、顶层errors为空，load_status无采集错误且
attempt_health.healthy=true；无故障注入的身份须由metadata与plan相互确认。
不得用 CPU 平稳、日志无 ERROR 或 HTTP 全200作为负类准入标准。
Normal 的恢复/清理状态与已知运行例外应登记；缺失或失败需显式例外复核，
不得自动变为正常。Fault 的末端恢复失败不自动否定故障前区间，恢复段仍Unknown。
缺失模态或单个特征无效仅影响相应模型资格，不自动改变参考标签。

2026-10-20 核查 selected17 的 collection_quality、collection_events、
case_metadata、load_status，以及六次 Normal 的 load/api_attempts。
六次 Normal 均 collection complete、coverage/cleanup/processing passed、
errors 为空、recovery terminal passed，负载 attempt_health healthy。
这些状态证明采集流程通过其检查，不证明每个服务健康。

最早 Normal 的负载有54条 ReadTimeout；其请求时间区间与52个15秒时间窗
重叠（按全25服务暂避会涉及1300行）。不能把该 Case 全窗无条件认定健康。
最后 Normal 的 load_status failures=2，但请求完成记录均为 HTTP200；
没有业务响应体证据，原因尚未闭合，HTTP200不证明业务成功。
不按 failures 计数直接构造服务异常正类。

用户已确认的简单保守例外规则：在原本候选负类中，对明确记录的请求
传输失败／超时区间 `[request_start_ms, request_end_ms)`，将有重叠的完整
时间窗内所有应用服务暂置 Unknown；不能由 Gateway 请求唯一归因服务。
这是客户端证据辅助的负类排除，不是独立服务异常正类，也会产生选择偏差，
必须报告排除数量与原因。故障前只使用在首次干预边界前已结束的失败请求，
不能把跨注入后才失败的请求倒推为注入前服务异常。
仅有失败汇总而无可定位证据的记录单列质量说明，不能自动造区间。
采集脚本快照显示 failures 既包含预订返回false，也包含业务循环异常，
不等于HTTP失败次数。最后Normal的2次失败原因仍未闭合；本协议不据此
宣称服务异常或健康，仅按无注入运行提供协议正常参考，保留噪声限制。
例外规则须同时应用于fit、calibration和test候选负类。被排除行仍留在标签表，
label=null，且不得进入拟合、阈值校准或二分类指标。

探查还发现多数故障前有 probe_write_pause_verified，早于 apply_requested_ms。
负类上界取实际干预请求和记录的probe暂停边界的最早值，
不使用更晚的暂停确认事件。暂停边界来源为 control/start_signal.json：
`start_ms + floor((probe_pause_at_mono_ns - start_mono_ns)/1e6)`，
并与probe_pause_verified的pause_at_mono_ns和Case绑定交叉核对。
该值是同一运行时钟锚点映射出的暂停边界，不声称获得真实接收时间；
若锚点缺失、身份冲突或时钟转换不一致，相关负类区间Unknown并列原因。
修正后11个 Fault 共10600行故障前候选负类；旧11575计数由更晚的确认时刻
产生，已被替代。在此边界前未发现已结束的load传输失败；不等于无业务异常。
Gateway KILL 的3条 ConnectionError、NETLOSS 的1条 ReadTimeout 虽在干预前
发起，却均在干预后结束，不能作为独立故障前异常证据。

可重跑探查脚本：`/tmp/tt_label_protocol_probe.py`，汇总与来源 SHA256：
`/tmp/tt_label_protocol_probe.json`。均为设计探查临时资产，不是标签发布。

## 设计探查计数参考（已由生产复核更正）

第一Normal的54次超时排除1300行后，六个Normal合计16550行候选负类；
按4/1/1分配分别为fit 10600、calibration 2975、Normal test 2975。
Fault的10600行故障前负类只用于test。加上194行候选正类，
全50575行参考分布为0类27150、1类194、Unknown23231；
test已知负类13575。以上均未加模型特征有效性掩码，实施验收须重新复算。
这些是暂定协议下的资格，不是已有产物的新标签；例外清单变更会改变数量。

## 数据流与实现边界

读取 selected17 清单、已有窗口稳定主键、原始注入和运行质量证据，
生成独立标签侧表。不得重建特征、改写原始数据或覆盖既有准备产物。
标签侧字段不能进入模型预测特征。

标签表至少包含稳定样本键、label、label_source、适用范围、规则原因与
证据引用。版本清单绑定协议配置、证据内容标识、窗口资产标识和生成代码版本。
输出拒绝覆盖旧版本；支持按稳定键比较版本变化及原因。
同一输入和配置应生成确定的标签内容。

标签包接口：`protocol.json`（规则及版本）、`evidence.json`（来源、hash及
字段定位）、`exceptions.json`（范围与证据）、`labels.parquet`、`splits.parquet`、
`coverage.json`、`manifest.json`。标签和划分覆盖全50575行，不删除 Unknown。
标签内容版本与划分版本分别标识，manifest绑定两者和输入资产；不靠目录名识别。
协议重标输出到新目录，比较命令生成稳定键上的旧值、新值及原因汇总。
label采用nullable整数0/1/null，Unknown用null表示；label_source区分
injection_association、normal_run_reference、pre_intervention_reference和unknown。
exceptions同时保留证据事件清单与合并后的区间，不以行号关联。
splits独立覆盖全网格，使用fit/calibration/test；划分不会因Unknown而改写。
coverage按Case、服务、split和原因报告0/1/Unknown及模态有效性覆盖。
后续评分接口须拒绝重复键、缺失必需评分和标签/划分版本不匹配；
仅单类的AUROC/AP为null。当前基础设施不包含模型、阈值选择或训练器。
边界或例外修改仅影响测试标签时可复用固定模型分数重评估；影响训练资格、
校准资格或划分时必须重拟合相应模型／阈值。不能以测试效果选择标签版本。

状态标签与模态有效性分离。缺少资源特征不自动改变标签；评分资格由
单独有效性掩码决定。不同模型使用同一预定评价集合，评分失败不得静默删行。
评估结果绑定标签版本；Unknown 告警不计 TP/FP，另报数量。

既有 preparation 验收要求全 Unknown，应保留；新标签版本另设验收入口，
不静默放宽旧准备协议。

## 当前覆盖及结论边界

2026-10-20 只读可行性计数：HTTPABORT、NETLOSS、CPU、DISKIO、DNSFAIL
分别有39、39、39、38、39个完整关联窗口，共194个，全部属于 preserve。
其中171个资源特征有效；该计数不是已发布标签或端到端验收结果。
实现后须用可重跑检查重新确认，不能将临时计数当永久实验产物。

三个 KILL 为一次 UID 约束删除，现有开始／结束记录不能作为服务异常持续区间；
三个数据库目标不在应用候选中。六个事件保留在覆盖清单中，注明未纳入原因，
不产生第一版窗口正类，不作故障已检出或未检出的推断；其中合格的故障前
窗口仍可作为测试负类，不能因此删除整个 Case。

第一版不支持跨服务检测性能结论；194窗不等于194次独立实验。
具备已知正负标签的 Fault Case 可报告逐 Case 二分类指标；只有负类的 Case
其 AUROC/AP 返回 null 并注明原因。单列留出 Normal 的协议告警率，
报告每个 Case 的正负及 Unknown 覆盖，避免将总体指标解释为全网格性能。

## 后续服务覆盖扩展

用户希望纳入 gateway、order 等服务，扩展应保留可插拔事件区间来源和
版本化规则，不在第一版加入无依据的固定延长时间。

- KILL：核对目标删除、服务可用副本、替代实例就绪等已有生命周期记录，
  再协商能支持的区间及语义；Pod 消失不自动等于整个服务不可用。
- 数据库：保留独立事件记录；应用侧纳入需要明确的对象归属和影响证据，
  不能仅凭依赖关系将所有调用方标正。
- 若现有证据不足，如实记录缺口；不能承诺无需补采即可覆盖其他服务。

## 验收要求与执行边界

实施计划需覆盖主键唯一性、全网格行守恒、区间边界、缺失及冲突证据、
瞬时 KILL、候选外数据库、Unknown 排除、版本可重放和差异报告。
未执行这些检查前不宣称通过。

验收需用生产输入重算本Spec计数，同时校验任意配置下的规则与行守恒，
不能只锁定某个计数；例外覆盖优先于候选负类，冲突不静默选择。
来源完整性测试、跨干预请求的失败请求测试和probe暂停边界测试必须覆盖。
协议配置不得包含待模型训练后决定的边界；不读取模型分数生成标签。

配套Plan须覆盖基础设施测试和真实标签包生成验收步骤；
执行前完成计划审阅与执行方式选择。
当前不启动模型训练。Spec没有授权重采、修改原始资产或扩展B规则。

## 判定顺序与最小接口

1. 校验输入资产和稳定主键，校验失败则不发布标签包。
2. 按固定Case清单赋予split，为每个窗口初始化Unknown。
3. 按准入规则确定Normal与故障前候选负类；应用有证据的例外排除。
4. 对非瞬时操作的应用服务直接目标，按完整包含规则生成注入关联正类。
5. 校验正负候选互斥；冲突报错，不靠后写覆盖前写。
6. 汇总覆盖与排除原因，验证行守恒后发布独立版本。

主键复用现有 `dataset_id/run_id/canonical_service_id/window_start_ms`，
并保留 `sample_id`、`window_end_ms`、`case_id`；sample_id与原准备资产一致。
label侧表增加 `label`、`label_source`、`label_scope`、`label_reason`、`evidence_ids`。
split侧表包含上述主键与sample_id、`split`、`split_version`。
`label_scope=service_window`表示映射粒度，不表示独立服务健康证明。
多个Unknown原因保存为排序列表，主原因采用固定顺序：运行准入失败、负类例外、
瞬时操作、目标不在候选集、部分边界、其余证据不足；顺序仅用于报告。

无记录的probe暂停与缺失本应存在的暂停锚点必须区分：由本Case的
probe_write_policy/事件/信号共同判定是否要求暂停，不按文件不存在一律当无暂停。
传输例外只消费load来源、event=end、transport_started=true、outcome=exception
且具有有效起止时间的请求；suppressed不属于已发出的失败请求。
未知状态不静默解释为健康，应登记质量警示；有证据表明记录完整性受损时报错。

输出位于新 `outputs/service/<run_id>/labels/`，run目录已存在则拒绝执行，
无force覆盖选项。先写临时目录，校验通过后发布；失败不得留下成功manifest。
manifest保存文件hash、输入hash、协议/划分内容hash、代码commit及dirty、依赖版本；
运行时间与路径不进入语义版本hash。重放要求排序后的标签/划分值及语义hash相同，
不要求Parquet字节完全一致。

## 生产复核更正（2026-10-20）

正式实现保留TRANSACTION的unknown_pause_policy，不推断无暂停。其原参考875个故障前负类撤销；实际故障前9,725、总负26,275、正194、Unknown24,106、test负12,700。fit10,600/cal2,975不变，资源有效正171已复核。此为证据不足的生产纠正，不是为满足计数而变更规则；详见实现报告。
