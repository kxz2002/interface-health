# Service训练前数据准备 Spec

状态（2026-10-20）：输入准备、labels/splits及独立复验已通过，最新验收入口为 `outputs/service/preparation-exploratory-20261020-task4-acceptance-final/quality_report.json`。TT限输入准备，SN限用户批准假设下的exploratory输入准备；模型mask、训练及真实状态性能尚未验证。保留逐项brainstorming共识与Task2b历史范围。

## 1. 目标与范围

为TT selected17和Eadro SocialNetwork准备共同模型架构可消费的三模态数据。允许两边资源和执行字段、维度不同；统一样本身份、模态组织和质量语义，不强制取特征交集。

本任务包含来源核验、解析、窗口聚合、标签/划分和质量验收，不包含BERT编码、模板挖掘、归一化拟合、模型实现、基线训练或GPU实验。后续归一化仅用Normal拟合池估计。

## 2. 数据与时间窗

- TT仅加载 `data/tt_service_20261017_004546/batch_control/final_acceptance/selected_cases.json` 的17项，核验迁移绑定和路径映射；不混failed27、旧四Case或new_merge。
- SN为Eadro版本 `10.5281/zenodo.7615394` 的SN包，MD5为 `4c523faf64770b1e9a709cdd610e07c4`；不重复计算嵌套/外层重复封装。
- 15秒、不重叠、仅完整窗口。TT沿用已验收边界；SN三模态采用同组边界。两数据集无需绝对起点相同。
- 保留来源清单确定的完整Service网格，不因日志/Trace缺失删行。严格profile时区、单位或实例身份不明时保留待定，不猜偏移或跨身份/运行拼接；显式SN exploratory例外按末节用户批准假设执行。

## 3. 五张交付表

主键：`(dataset_id, run_id, canonical_service_id, window_start_ms)`；保留 `window_end_ms`。每个数据集分别输出：

|表|内容|
|---|---|
|`resource_windows`|自身资源字段、有效性和质量状态|
|`execution_windows`|入口/Span统计、统计对象及覆盖|
|`log_windows`|消息列表、频次、原文引用和采集状态|
|`labels`|正常/异常标签与依据|
|`splits`|运行角色、划分及正常训练资格依据|

schema记录各自字段顺序、单位和来源。各表按主键关联，不以inner join删除缺模态窗口；暴露索引与来源manifest单独保存，不进入模型。

## 4. 三模态规则

### 资源

TT保留现有CPU核数、内存比例、进程数和质量状态，不重新计算。SN严格profile先确认单位及瞬时/累计语义再定物理聚合；exploratory仅保存raw序列和纯数学均值（见末节）。SN缺少进程数不导致TT删除该字段；不同物理量不得冒称同一语义。

### 执行

优先统计服务端入口，避免内部Span重复计请求。保留入口观测数、耗时均值/P95/最大值、错误标记已知数和错误数，附观测与错误覆盖状态。

TT沿用Entry语义；SN先验证入口识别，无法识别则称Span统计，不冒称请求数。耗时和错误针对窗口内已结束的入口Span，明确记录归窗规则；按离线闭窗解释，不宣称在线提前量。没有观测不直接填零请求或零错误率。

### 日志

每个Service窗保存消息及频次，多行堆栈合并为事件，保留原文引用。时间、容器身份及采集包装用于对齐溯源；故障名/注入脚本内容不进入模型消息。

不做模板挖掘、BERT编码、硬截断或激进清洗。相同消息可压缩保存但保留次数。区分观察到消息、未观察到消息、已知采集缺失；无消息不等于健康，ERROR不直接生成异常标签。

## 5. 简单标签与划分

始终二分类，不增加第三预测类。labels除主键外仅要求：

- `label`：0有依据的正常，1有依据的异常，空值暂不评价。
- `label_reason`：标签依据或空缺原因，可引用证据文件以保留来源和适用范围，不强制复杂多列标签体系。

注入对象和时间仅为暴露线索，不把所有注入窗、非目标或恢复窗批量赋状态标签。Case、phase、target、fault type、注入时间及probe不进入预测特征。

splits独立保存角色与依据。TT保留前三Normal fit、第四cal、第五六holdout、故障全测试。SN核实运行/重复/正常资格后固定候选划分；不凭no-fault目录认定健康。常态业务错误须解释，不自动删除运行或认定全健康。

Normal-only及整Case隔离不变；Fault不用于拟合、校准或超参数选择。评价标签空缺与正常训练资格分别核验，不给所有训练行制造金标；资格未确认的行不进训练池。

## 6. 交付与验收

raw只读；派生输出写新的 `outputs/service/<run_id>/`，拒绝覆盖。保存数据hash、配置、Git状态、环境、schema、服务映射和稳定行ID；原始包及大Parquet不随源码提交。

验收标准：

1. 两套数据各有三模态表、labels和splits，待定项明确。
2. 主键唯一、同运行三模态边界一致、不跨运行、不因缺模态删网格。
3. 单位、来源、有效/缺失状态清楚，未确定项列出。
4. 质量报告包括每运行窗数、模态覆盖率、标签计数、排除原因和训练资格范围。
5. 少量样本人工回查原始记录，并执行相应自动校验；未执行不写通过。

不强求所有窗口有标签。关键时间、字段或划分未解决，只能称结构性准备完成，不能称该数据集训练就绪。

## 7. 模型阶段衔接

准备验收通过后可进入模型实现。再确认编码器、融合机制、SVDD目标、mask和训练配置，完成Loader/模型、前向梯度检查及小规模试跑，随后在专用设备同步推进两数据集各自Normal-only训练、独立校准和评价。

主验证为同架构分别训练，不要求冻结一套权重跨系统。模型实现和实验计划另立，不因本Spec完成自动启动训练。查新仍有未完成项，不宣称新颖性已通过或预先保证增益。


## 2026-10-20 用户批准的 SN exploratory profile（Task2b）

严格 profile 的未知语义阻断保留，历史核验结果不改写。用户明确采用三模态探索方案：仅日志与Span采用固定 -28800 秒对齐 metadata/资源数值坐标；这是公开 alignment assumption，不是已证明时区，不按测试效果选偏移。日志 naive 时间显式按 UTC 数值化后平移；Span 按结束时间归15秒完整半开窗，统计对象为 Span（不是请求），原生键去重，冲突报错，未知 error 保留。

资源七字段保留 raw 1Hz 序列、实际时间与覆盖；unit/measurement_semantics=unknown。允许纯数学均值，不作 rate/diff 或物理换算。日志逐条检查解析、边界与回跳，合并多行并保留全文、频次、来源引用，不模板化、不截断。只读固定 MD5 包，精确选一个故障封装和三个 Normal 封装，不重复外层 data，全12Service网格保留。规则和来源小统计保存在 sn-exploratory-evidence.json 及 configs/service/sn_exploratory_v1.json。

Task2b仅实现 adapter、小fixture和角色/证据接口；生产全量执行与全日志质量验收留Task4。Task3独立生成labels/splits：三个无声明注入运行按20日fit、21日上午calibration、21日下午holdout，4Fault仅test；资格为无声明注入常态参考分布，常态ERROR不删运行，labels仍Unknown，不冒充全健康。Task2b不训练、不拟合归一化、不编码。
