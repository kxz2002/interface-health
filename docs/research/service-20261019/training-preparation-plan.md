# Service训练前数据准备 Plan

状态（2026-10-20）：Task1—4的输入准备与独立复验已完成，最新验收入口为 `outputs/service/preparation-exploratory-20261020-task4-acceptance-final/quality_report.json`。SN只在用户批准的exploratory假设下通过；模型mask、训练及真实状态性能尚未验证。以下保留brainstorming共识、任务顺序及Task2b历史修订。

## 一、来源核验

工作：核对TT selected17、迁移绑定、资源网格；核对SN包hash、重复封装、七个运行和服务集合。确认SN日志时区、资源单位/类型，以及Normal共享Trace ID裁窗后的关系。不凭no-fault认定健康。

交付：来源manifest、服务/运行映射、时间/字段说明、未决问题清单。

验收：来源身份固定，关键语义有证据，未知项阻断依赖步骤。复用已有证据，只补缺口，不为重复确认重扫TT大日志/Trace。

## 二、两套适配

工作：TT复用资源原值和质量；SN核实语义后聚合自身字段，不强制交集、不归一化。执行分支确定TT Entry与SN入口/Span对象、裁窗去重并统计数量、耗时及错误覆盖。日志按Service窗保留消息、频次、引用，合并多行事件，不做编码/模板挖掘/激进清洗。

交付：两套resource_windows、execution_windows、log_windows及各自schema，统一主键、15秒完整窗、全Service网格和质量状态接口。

验收：不跨身份/运行；缺失不补健康零值；内部Span不冒称请求；未知error tag不填false；时区不明不猜偏移。

## 三、标签与划分

工作：生成labels(label,label_reason)和独立splits。TT保留既有划分；SN核实运行和训练资格后固定分配。注入暴露单独记录，仅作证据线索。

交付：两套labels/splits、暴露索引、正常训练资格及评价范围说明。

验收：标签仅0/1/空值，Unknown不填负类，非目标/recover不批量正常；整运行隔离，Fault不进fit/cal。评价标签空缺和训练资格分别说明。

## 四、质量验收

工作：核对五表身份、窗口、网格和质量状态；每套数据抽少量正常/故障、边界和缺模态样本回查原始记录，执行新增解析与相关旧接口自动校验。

交付：质量报告、人工核对记录、自动校验日志、配置和产物manifest，明确每数据集就绪状态。

验收：满足Spec第6节，数量可重算；候选窗不冒充有效训练样本。关键问题未解决不能宣布训练就绪。不重训或重建旧管线。

## 实施约束

各数据集自己的adapter保留字段差异；共享主键、网格、状态/schema校验和输出管理。复用纯函数，不套旧四Case配置，不改Endpoint默认接口。互不依赖的适配可并行，语义核验先于聚合。

规划时的未验收代码：src/service/preparation.py、sn_preparation.py、window_preparation.py、scripts/prepare_service_data.py及相关测试。该历史状态已由Task1/2及Task2b逐项更新，详见对应报告；历史49项测试不等于本版整体验收。

## 后续衔接（本计划不执行）

数据准备验收 → 确认模型/训练细节 → 模型和Loader实现 → 前向、梯度及小规模试跑 → 专用设备两数据集分别Normal-only训练、独立校准和评价。

同架构允许输入字段不同，不等于相同权重直接迁移。模型贡献靠各数据集内相同输入基线与消融，不用跨数据集绝对分数差直接证明创新。本计划完成不代表模型已实现、训练已启动或查新已通过。


## 2026-10-20 用户批准的 SN exploratory profile（Task2b）

严格 profile 的未知语义阻断保留，历史核验结果不改写。用户明确采用三模态探索方案：仅日志与Span采用固定 -28800 秒对齐 metadata/资源数值坐标；这是公开 alignment assumption，不是已证明时区，不按测试效果选偏移。日志 naive 时间显式按 UTC 数值化后平移；Span 按结束时间归15秒完整半开窗，统计对象为 Span（不是请求），原生键去重，冲突报错，未知 error 保留。

资源七字段保留 raw 1Hz 序列、实际时间与覆盖；unit/measurement_semantics=unknown。允许纯数学均值，不作 rate/diff 或物理换算。日志逐条检查解析、边界与回跳，合并多行并保留全文、频次、来源引用，不模板化、不截断。只读固定 MD5 包，精确选一个故障封装和三个 Normal 封装，不重复外层 data，全12Service网格保留。规则和来源小统计保存在 sn-exploratory-evidence.json 及 configs/service/sn_exploratory_v1.json。

Task2b仅实现 adapter、小fixture和角色/证据接口；生产全量执行与全日志质量验收留Task4。Task3独立生成labels/splits：三个无声明注入运行按20日fit、21日上午calibration、21日下午holdout，4Fault仅test；资格为无声明注入常态参考分布，常态ERROR不删运行，labels仍Unknown，不冒充全健康。Task2b不训练、不拟合归一化、不编码。
