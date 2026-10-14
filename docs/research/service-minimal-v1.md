# Service-Minimal v1：任务定义与冻结证据（初稿，T0）

> 状态：T0 协议冻结（2026-10-14）。本文档与 `configs/service/minimal_v1.yaml`、
> `configs/service/evidence_minimal_v1.json`、`src/service/protocol.py` 同步；
> 证据条目全部 `pending_human_review`，数字结论待人工核验后才能进入论文。

## 1. 任务定义

在现有 per-endpoint contract 之外，建立一条 **service 粒度、资源指标最小集** 的
单类异常检测支线，目的不是再刷一个 AUROC，而是回答一个更便宜、更可审计的问题：

- 只用 Normal 训练（one-class），在 service × 15s 窗粒度上检测资源层异常；
- 模型主输入冻结为 R3 = `[cpu_cores, memory_usage_ratio, process_count]`，
  另设 CPU-only 单变量臂；身份、阶段、注入事件、客户端字段一律不进模型；
- 候选实体、case、切分、方法、信息预算全部在数据处理前冻结
  （`configs/service/minimal_v1.yaml`），错误实体或非法输出路径在处理前失败。

### 1.1 为什么需要这条支线

现有 endpoint 融合链路的最强结论目前被平凡基线占据（rel_pos 0.9656，见 CLAUDE.md
研究状态），根因是采集模板的时间可预测性。service-minimal 支线主动收窄问题：

1. **标签语义更硬**：Killing 事件、CPU stress 是可独立核验的干预事实，
   不依赖故障注入时刻的机械对齐；
2. **信息预算更小、更可审查**：6 个资源列 + 状态列，不存在四模态拼接里
   "哪一模态泄漏了标签"的归因困境；
3. **与 P-B（per-endpoint baseline）协议互补**：它检验"资源层是否真的发生了
   可观测变化"（S1），而不是直接声称"服务不可用"（S2）。

### 1.2 冻结范围

| 项 | 冻结值 |
|---|---|
| case | Normal_20260728T161044Z；Lv_P_CPU_preserve_…164711Z_em；Lv_S_KILLPOD_order_…203705Z_em；Lv_S_KILLPOD_preserve_…200124Z_em |
| 候选服务 | 25 个 `ts-*-service`（字典序；数据库/基础设施排除），不随标签/分数/覆盖改变 |
| 窗口 | 15000ms，ceil 对齐，只保留完整窗；guard 30000ms |
| Normal 切分 | 时间窗 60/20/20（normal_fit 72 窗 / calibration 24 / tail 24），同窗 25 服务不拆散 |
| Fault case | 全部窗为 fault_evaluation，无任何训练窗 |
| 方法 | L2、max、OCSVM、IF；特征组 R3 与 CPU-only；seed 42 |
| 阈值参数 | min_fit_rows=30、min_calibration_rows=10、max_counter_gap_s=30、transition_ms=30000、event_min_coverage=.8、event_top_fraction=.2 |
| 禁止项 | Fault 专属适配、超参数搜索、score-dependent 选择 |

### 1.3 三个任务的边界定义（S1/S2/S3）

本支线包含三个语义强度不同的任务，v1 只开发 S1 与受限版 S2，S3 明确 NO-GO。
三者共用同一冻结协议（候选/case/信息预算），但允许声明的结论严格不同，
禁止把弱任务的结果叙述成强任务。

**S1 — 资源层单类异常检测（本版主任务，开发）**

- 输入：仅 R3（CPU-only 臂仅 `cpu_cores`）+ 实体键与观测有效性；
- 训练：只用 normal_fit/normal_calibration 段，故障 case 零训练窗；
- 允许结论：某 service × 15s 窗的资源形态偏离了正常分布；
  对 CPU case 可额外报告与受控暴露弱对照（proxy_label）的一致性；
- 禁止结论：把"资源偏离"直接叙述为"服务不可用"或"该服务是根因"。

**S2 — 事故级注入目标 Service 检索（受限开发，T4 固定事件检索）**

- 任务设定：对每个事故 case（CPU preserve、KILL order、KILL preserve 三个事件），
  在冻结的 25 个候选 Service 内做排序检索，排序依据是协议冻结的事件检索规则
  （event_min_coverage=.8、event_top_fraction=.2），不允许逐 case 调参或按分数挑规则；
- 指标：Hit@1、Hit@3、MRR，真值为该事故 CONTROLLED_EVENT 记录的实际干预对象
  （ts-preserve-service / ts-order-service / ts-preserve-service）；
- 允许结论：**在受控事件范围内**，系统能否把实际干预对象排在候选前列
  （事故级 injection-target retrieval）；
- 禁止结论：
  1. 不得叙述为"完整根因定位"——干预目标是实验者外部施加的对象，
     不等于自然事故中因果链上的责任服务；
  2. 不得叙述为"全部实际异常 Service 集合"——下游受波及服务、共享资源受害者
     （如 CPU contention 的邻居）不在该真值内，检索结果不提供集合覆盖语义；
  3. 证据不足（KILL 两 case 的服务可用性当前为 UNKNOWN）时报告 Unknown，
     不用排序结果反推状态金标。

**S3 — 完整根因 Service 定位（本版 NO-GO，不开发）**

- 显式声明：service_minimal_v1 不开发、不评估完整根因定位任务，
  论文/报告中不出现 "root cause localization" 口径的 S3 数字；
- NO-GO 理由：① 数据只有受控注入的干预标签，没有自然事故的因果链真值，
  干预对象 ≠ 根因；② 资源指标无法区分"施害者"与"受害者"
  （CPU stress 的目标与其资源受挤压的邻居在 R3 上可能同向异常）；
  ③ 缺独立于模型输入的服务依赖/状态证据，无法闭合因果判断；
- 硬约束：禁止以注入标签（target_service / inject 时段）替代根因真值来"造" S3 评估；
  若未来重启 S3，必须先补独立因果证据与评估协议，走新协议版本，不得在 v1 上扩展。

## 2. 三个主事件的选入理由与证据定级

证据全部落在 `configs/service/evidence_minimal_v1.json`（9 条），
source_locator 只记录稳定标识（事件时间/Pod/指标选择器/source 文件 sha256），
不引用会漂移的数组行号。

### 2.1 CPU stress（Lv_P_CPU_preserve，目标 ts-preserve-service）

- **CONTROLLED_EVENT**：StressChaos YAML 证明干预设计（app=ts-preserve-service，
  cpu.load=60/workers=1），计划区间来自 case_metadata；但现存 YAML 不是采集当时
  冻结的 spec/status，只证明"计划做了什么"。
- **OBSERVATION_DERIVED**：metric 输入证实资源确实变化——baseline CPU rate
  最大 0.0597 cores（n=80），注入后首个采样间隔（inject+10.4s，ts=1785258453s）
  0.2826（4.7×），随后样本 0.37~0.60（约 7.5~10×）。因为 metric 同时是模型输入，
  这条变化**不能**当独立状态金标（S1），只能作受控暴露弱对照（proxy_label）。
- **UNKNOWN（恢复时序）**：configured inject_end 后第一个采样间隔仍为 0.996 cores，
  约 35s 后才回落到 0.008；卸载滞后机制无独立证据，恢复边界保留 Unknown，
  不把 configured end 当实际恢复时刻。

### 2.2 KILL order（Lv_S_KILLPOD_order，目标 ts-order-service）

- **CONTROLLED_EVENT**：Schedule(PodChaos pod-kill, mode=one, @every 3s) 的计划。
- **INSTANCE_TERMINATION（INDEPENDENT_STATE）**：K8s Killing 事件 201 条、
  201 个不同 Pod，首条 1785272238000（计划起点 +2.3s）、末条 1785272838000。
  Killing 独立于模型输入，能证明"实例被终止"，但 Killing/Created/Started ≠ Ready。
- **SERVICE_UNAVAILABLE 不自动成立**：mode=one 每次只杀一个副本，其余副本可能继续
  服务；没有 readiness/endpoint 状态或独立于模型输入的服务端成功请求率，整窗
  服务可用性标 UNKNOWN（`unknown::…::service_availability`），等待 T4 固定检索。
- 该 case inject_end 比 window_end 大 90ms，网格截到 window_end，自然无 recover 窗。

### 2.3 KILL preserve（Lv_S_KILLPOD_preserve，目标 ts-preserve-service）

- 同 2.2 的三层定级；Killing 201 条/201 Pod，首条 1785270099000、末条 1785270700000。
- 审计划记中保留"目标运行观测歧义"（前序可行性审计提出，待 T4 核实），
  可用性 UNKNOWN 的 review_note 已显式记录该歧义。

### 2.4 定级规则（已由 validate_evidence 锁死）

- `depends_on_input_sources` 非空 ⇒ 禁止 INDEPENDENT_STATE
  （人工阅读同一 Log/Trace 只增强语义，模型也消费该来源时仍是 OBSERVATION_DERIVED）；
- Killing/实例终止类 fact_type ⇒ 禁止自动派生 SERVICE_UNAVAILABLE + independent_label=1；
- 未确认时段一律 state_class=UNKNOWN，不允许用计划区间回填实际状态。

## 3. S1（独立状态证据）为什么不足

当前三个事件里，唯一满足 INDEPENDENT_STATE 的事实只有 **实例终止**：

- CPU 变化的唯一观测通道是 metric——而 metric 正是模型主输入，循环论证；
- 服务不可用性没有独立观测：集群侧缺 readiness/endpoint Ready 历史导出，
  应用侧的错误信号（trace 5xx、log、客户端错误率）全部属于模型可消费来源，
  即便人工逐条读过也只能 OBSERVATION_DERIVED；
- 因此 v1 的评估叙述必须分层：
  - S1 可声明的：干预发生（CONTROLLED_EVENT）、实例终止（INDEPENDENT_STATE）；
  - 只能弱对照的：目标资源变化（proxy exposure，baseline 窗=0 / inject_core 窗=1）；
  - 暂不声明的：故障窗 = 服务不可用金标。
- 模型输出的解读相应限定为"资源层偏离正常"，下游若要主张检测到了服务故障，
  必须经过 T4 的 S2 固定事件检索，不允许在分数侧做阈值相关的事后选择。

## 4. PPT 第一页草案（一句话 + 三张证据卡）

**标题**：Service-Level One-Class Anomaly Detection with a Frozen, Auditable Information Budget

**一句话**：只用正常资源指标训练单类模型，在 25 个服务 × 15s 窗上检测资源层偏离；
评估标签按证据独立性分级，实例终止可独立核验，服务不可用性不做无证据假设。

- 卡 1（设计）：4 case / 25 service / 15s 窗全部冻结；模型只见
  cpu_cores、memory_usage_ratio、process_count（CPU-only 臂仅 1 列）；
  phase、target、Killing 事件在物理上不进入特征表（schema 校验拒绝）。
- 卡 2（证据）：CPU 注入后首个采样间隔 4.7×baseline 峰值（metric 派生，弱对照）；
  KILL 两 case 各 201 次独立 Killing 事件覆盖 201 个 Pod（实例终止金标）。
- 卡 3（诚实边界）：我们不声称故障窗等于服务不可用——mode=one 杀单副本，
  且没有独立于模型输入的可用性观测；未知时段显式标 UNKNOWN，
  检测结论以"资源偏离 + 独立事件"的合取呈现。

## 5. T0 交付物与后续边界

- 交付：`configs/service/minimal_v1.yaml`、`evidence_minimal_v1.json`、
  `src/service/protocol.py`（12 个公共签名）、`tests/test_service_protocol.py`（4 测试）。
- 明确不做（留给后续任务）：流读 metric CSV 与实体消解（T1）、prepare 入口/Hydra CLI（T2）、
  基线与检测器（T3）、S2 固定事件检索（T4）。
- 任何签名或 schema 与冻结 spec 不符时，停下来改 spec，不在实现里静默变通。
