# Interface-Health：Codex 工作指南

适用于本仓库。根据现有 CLAUDE.md 整理，补充截至 2026-10-14 的用户研究决策。具体任务以用户最新指令为准；旧实验协议、研究设想和目录中的计划文件不构成自动训练或重建授权。

## 1. 当前研究目标与工作状态

硕士论文暂定方向：**面向微服务系统的 Service–Endpoint 双层异常感知与业务 API 异常识别**。

- Service 层研究内部资源、执行与可用性状态；Endpoint 层研究客户端 RED、Response 与接口表现。
- 当前优先两层独立输出，后续按真实 Service/Operation 依赖关联。Service 报警不是 Endpoint 检测的前置条件。
- 数据集与既有小论文是研究基础，不计为两个新方法创新。两套不同输入的检测器不自动构成两个创新点。
- 当前先建立可信 Service 数据和简单强基线；新方法必须由真实基线失败模式与可验证能力支持。CPU 单列或固定错误指标已足够时，不为论文叙事强行开发复杂网络。
- 中期日期曾计划为 2026-11-04，预算与最新日期需依当前任务核实；不能把历史 40–60 小时当已确认预算。

**截至本指南创建时**：Service 代码调研、Brainstorming、Spec、Plan 已完成；Service 实施已在 feature/service-minimal-v1 分支推进中。下一轮用户要求开发时，先核对代码现状与设计文件，再按授权任务执行。不要把“文档已完成”写成“Service 模型已完成”。

### 新会话阅读入口

1. 本文件、history/index.md；按影响域阅读相关 history/entries/。
2. 用户要实施 Service 方案时，优先读取：
   - /tmp/service-model-design-20261013/decision_summary.md
   - /tmp/service-model-design-20261013/code_research.md
   - /tmp/service-model-design-20261013/spec.md
   - /tmp/service-model-design-20261013/plan.md
   - 方案取舍需要时读同目录 brainstorming.md。
3. 读取与修改路径相关的 .claude/rules/ 文件，见第 8 节。Codex 不应假设这些 Claude 路径规则已自动加载。
4. 只读检查 Git 状态、环境、输入路径与现有产物；保留用户已有修改。

上述设计文件要求输出在 /tmp，因此可能被清理。文件不存在时如实说明；利用本指南与现有源码继续独立工作，不能猜测完整 Spec、宣称已读或自动重新运行全部审计。若缺失内容影响实施正确性，再获取必要信息。

## 2. 数据与外部依赖

### 主数据

- data/new_merge/：Train-Ticket 单批次采集，27 case（1 Normal、26 故障），2026-07-28/29；由 configs/data/new_merge.yaml 声明。原始数据及 _pipeline_out 都保持只读。
- 当前 8 个业务入口分别对应 8 个不同 Service；约 23,868 条 Endpoint×15s 行。不能以此验证同一 Service 下多个外部入口的差异。
- 原始 Metric 覆盖 25 个应用 Service 及部分数据库/基础设施；现有 Endpoint Contract 只保留 8 个入口所属 Service。
- Metric/Log 是服务侧观测；Trace/Client 可有 Endpoint/Operation 粒度。Service 数据不能由 Endpoint 行是否存在来决定完整时间轴。
- 唯一 Normal 不提供第二次独立健康实验；同事件大量窗口和 KILL 内反复终止不算大量独立故障重复。

### AnoMod 依赖

- 关联仓库：/home/kxz2002/Code/Repos/AnoMod。
- new_merge 同时支撑 AnoMod 数据集论文。其管线会重写 _pipeline_out；不要为了运行本仓库实验自动调用 AnoMod 的整体 pipeline。
- tt_fused_*.csv 是 downstream benchmark 输入；不是正式 Contract 的直接输入。两条链路特征、划分和评价不能混用。
- AnoMod 论文历史统计用 26 case，另排除 TRANSACTION；本仓库 Contract 用 27。Service 的最小实验另有明确 4case 范围，不能沿用旧排除列表。
- AnoMod/processed_data/2026_0729/ 的原始文件曾核对一致，但派生产物版本不同，不能当作 new_merge 的可直接恢复副本。
- 修改 data/new_merge 任意内容或重新生成 _pipeline_out 属于改变已有数据资产；只有用户明确授权该操作才执行。已授权的动作无需重复询问。

### 历史不可恢复资产

CLAUDE.md/history 记录 endpoint_raw2、lo2-sample、normal_v2、new_ep1、new_ep2 已丢失；merged_v1/merged_v2/new_ep1 等旧链路不能重建。旧指标只保留为历史记录，不与现行结果拼成统一消融表。

data/anomod_v1 与 data/anomod 为历史批次，不能未经任务授权与 new_merge 混合。docs/agent-docs/dataset-guide.md 部分正文描述旧 pipeline；现行含义须核对实际源码、配置与历史记录。

## 3. Service 最小实验的默认设计

以下是已讨论的设计边界，完整细节以可访问的 Spec/Plan 为依据；实现或运行以当前用户授权为准。

- 独立包 src/service/、配置 configs/service/、入口 scripts/service_experiment.py；已在 feature/service-minimal-v1 分支落地，现状以代码为准。
- 默认不修改既有 Endpoint Contract、Preprocessor、训练/融合管线，不覆盖历史 scores/metrics。
- 资源主键为 (case_id, canonical_service_id, window_start_ms)，15s 完整网格，稳定样本键；不能用数组行号作跨产物身份。
- 最小 case：Normal_20260728T161044Z、Lv_P_CPU_preserve_20260728T164711Z_em、Lv_S_KILLPOD_order_20260728T203705Z_em、Lv_S_KILLPOD_preserve_20260728T200124Z_em。
- 固定 25 个应用 Service 候选，包括 Gateway。不能只选择 order/preserve 或删不可评分候选提高排名。
- CPU-only 与 R3=[cpu_cores, memory_usage_ratio, process_count]，主对照共享 R3 有效样本 ID。
- 每 Service 仅 Normal 固定早段拟合；Normal 全窗 60/20/20 分为 fit/calibration/tail，各 Service 同窗同分段。Fault baseline 也不参与训练或参数拟合。
- 固定 L2、max、OCSVM（rbf、gamma=scale、nu=.1）、Isolation Forest（100 trees、seed42、n_jobs1），不按测试结果调参或翻分数方向。
- 默认资源观测采用保守稳定单实例语义。Counter 实际 dt、最大间隔30s；实例切换30s过渡；首点/reset/gap/身份歧义保留 Unknown，不补健康0。
- S2 按预定注入区间、30s边界保护聚合；有效覆盖至少80%，取最高20%校准分数均值；候选并列和缺评分作保守评价。目标不可评分保留该事件并报告检索失败、覆盖和原因。
- 生命周期事件默认留作评价/诊断，不输入资源模型。不能利用 Killing 清理实例后，又用同一 Killing 证明独立可用性检测。
- 先完成资源基线，再按失败模式决定资源—执行分离（方案B）或观测可靠性机制（方案C）。没有独立验证条件时，只能作诊断，不能宣称新方法成立。

### 任务与标签必须分开

- **S1：Service 运行状态检测。** 完整独立 Service×Window 金标尚不存在；Unknown 不当负类；独立 AUROC/AP 无足够正负证据时输出 null。
- **S2：注入目标 Service 检索。** 可以评价受控干预对象排序；不是全部异常服务集合，也不是完整根因定位。
- **S3：完整根因定位。** 当前不作为开发目标。

注入目标、实际运行异常 Service、传播 Service、实际受影响 Endpoint 是不同集合。HTTPChaos 的 Path 干预不自动成为整个 Service 内部异常正类。CPU/Log/Trace 输入同时派生的标签只能作观测规则/弱评价；人工读取同一来源也不自动产生独立真值。

## 4. 必须保留的观测语义

- MetricPreprocessor 旧实现丢失 container/id，混合父cgroup和应用容器，固定15s差分、首点/reset补0、内存比例裁剪；不能直接把旧5Metric字段当正确 Service 输入。
- 其 docstring 中“内存limit无定义、process全0”来自旧数据，与 new_merge 审计冲突；当前具名应用容器有有效 limit，CPU压力下进程数可变化。
- CPU 差分保持完整系列身份，不能跨Pod/container id重启；memory ratio仅positive limit有定义；网络错误计数不等于丢包率。
- KILL旧新系列同时存在不表示多个活实例；Prometheus query_range时间不是真实scrape/接收时间，旧Gauge、CPU零值不能证明健康/死亡。
- 原始日志含历史内容，先裁实验时段；多行堆栈不等于多条日志事件。Pod重启后旧日志流可能结束，缺日志不等于无错误。
- 原始业务 Service Entry 与 Gateway/Exit/Local 分开；不能用 Trace summary 补不存在的业务 Entry。
- Endpoint 聚合 Trace 的“有请求”可能来自 Gateway/传播侧。旧材料所称 KILLorder/preserve Trace 完好，不代表目标业务 Service Entry 存在。
- Entry 的耗时/错误只能结束后获知；按start回填不能声称在线检测提前量。当前缺真实接收时间，优先报告离线闭窗结果。
- Client 缺少逐请求Trace ID，超时/连接失败可能未记录响应；HTTP200/hash变化不直接证明业务成功/失败。baseline与非目标不自动健康，后期preserve存在持续500。

## 5. 现有工程与复用边界

|位置|实际职责/注意|
|---|---|
|src/preprocessors/|Endpoint Trace、Client与服务Metric/Log预处理；旧语义不直接迁入Service|
|scripts/build_contract.py|Endpoint inner join、Service left join、标签/划分/归一化、产物落盘|
|src/contracts/|契约、标签校验、时间划分；旧scores_v0不允许Unknown标签/分数|
|src/data/contract_dataloader.py|耦合endpoint_key/is_anomaly；不伪造Endpoint列用于Service加载|
|src/data/normalization.py|已有多种scope和回退；统计拟合边界要随具体协议审核|
|scripts/benchmark_downstream.py|可复用fit_normalizer、apply_normalizer、OCSVM/IF、safe_auc/AP；不要运行main替代Service入口|
|scripts/score_trivial_baselines.py|旧z-score按(case,endpoint)；缺Fault组可能0偏离，不能充当Normal-only Service实现|
|src/fusion/、src/models/deep_svdd.py|已有神经融合与DeepSVDD；存在代码不等于研究必要性|
|scripts/train_baseline_v0.py|Hydra训练；训练循环依赖SVDD接口，不能仅改model名切到sklearn|
|scripts/eval_baseline_v0.py|依赖is_endpoint_anomaly，不是Service金标Evaluator|
|src/training/、src/evaluation/|历史为空stub；以实际代码核实，不假设通用Trainer已有|
|src/utils/seed.py|复用随机管理；不要另建冲突seed逻辑|
|src/utils/logger.py|MLflow封装；历史未接入，不为最小实验自动启动服务|

实现时优先复用纯函数；新增粒度/Unknown适配，不复制完整训练框架。没有必要时不改共享接口；确需改动要保证旧签名与默认行为，并验证相关兼容性。

## 6. 历史Endpoint实验：复现规则与结论边界

- 旧v1/v2 expanded池包含Normal、Fault baseline及inject/recover非目标行；它依赖注入关联标签，**不等于当前Service Normal-only训练**。
- is_anomaly=(phase==inject)；is_endpoint_anomaly按target_endpoint优先、再target_service、最后case级回退。label_granularity由target字段存在性决定，不是anomaly_level。
- 这些正类是注入关联标签，不是完整实际业务影响金标。旧协议把其他行当负，不表示本研究已确认这些接口/服务健康。
- 合同主链历史指标以artifacts/contract_v2_acceptance/main_table.md为准。时间位置基线很强；不能把高AUROC称严格跨故障泛化。
- downstream benchmark的18/21维、Contract特征、后续9/12维Client实验是不同协议。样本、标签、划分、参考、评分方向及macro口径一致才可横比。
- 纯Client RED实际9维，加3个Response字段为12维；18维是Trace9+RED9，21维再加Response3。不要延续RED10/13维笔误。
- 已完成公平实验显示Response增量集中于PATCH，非PATCH多种基线退步；不能预设更多模态一定更强。
- trace_5xx_rate的非空值曾恒为1，trace_error_rate受稀疏HTTP状态tag影响；新Service执行统计使用原始Entry.is_error，不把旧列解释成完整服务错误率。
- AUROC/AP单类case返回null并说明跳过；Unknown不转0；逐事件/机制与有效样本覆盖并列报告。相关窗口不作独立重复或置信证据。

## 7. 环境、命令与输出安全

使用conda环境interface；系统/base的python可能缺依赖。已核查环境为Python3.12、numpy2.1.2、pandas2.2.3、pyarrow17、sklearn1.5.2；下一会话核对实际版本，不自动升级。

```bash
# 只读查看主链DVC状态；不是授权重跑
conda run -n interface dvc status dvc_new_merge/dvc.yaml

# 用户要求测试/验证，或明确执行含这些检查的Plan时，按影响范围运行
conda run -n interface python -m pytest tests/test_dataset_config.py -q

# 已有Hydra配置预览：不训练
conda run -n interface python scripts/train_baseline_v0.py --cfg job
```

- make setup/sync、pip安装、torch安装只在任务确有环境需求时使用；不要为文档/只读任务改变环境。
- **禁止在根目录运行无target的dvc repro。** 根dvc.yaml及历史dvc_reliability_gate/dvc_deviation_weighted/dvc_new_ep1是死链，可能先删不可恢复outs再失败；全项目DVC状态也有历史噪音。
- dvc_new_merge/dvc.yaml是旧Endpoint主链。历史确认它可复现不表示本轮有重训授权；Service开发不应触发它。
- v2 Contract没有注册build stage，改上游源码不自动触发重建；不能靠DVC“全绿”证明新语义已体现在旧产物。
- DVC整目录dep容易被_pipeline_out漂移作废；发现new_merge变化先定位来源，不自动repro/checkout/restore。
- 新Service输出拟用outputs/service/<run_id>，独立data/run/evaluation/report子目录；已存在拒绝覆盖。不要把新数据写入raw目录或旧baseline目录。
- 读取Metric/Trace/Log要选case/列/服务、流式处理；不复制原始大文件，不为最小任务全解析约96GB日志。
- 每次实验保存resolved config、git commit/dirty状态、环境、data标识、seed、特征序、稳定train/cal/eval ID、模型参数、逐样本分数、事件排序和质量报告。单靠汇总指标不足以复现。
- Hydra旧入口用key=value；学习率为training.optimizer.lr。Service薄入口拟用argparse，不修改configs/base.yaml改变旧行为。

## 8. 按需读取的项目规则

这些文件沿用为共享知识来源，无需复制成Codex专用版本：

- .claude/rules/contract-and-labels.md：build_contract、src/contracts、Contract配置、旧评价；标签闸门/训练池/行集合。
- .claude/rules/normalization.md：Normalizer、退化组、回退链、拟合范围。
- .claude/rules/fusion-and-models.md：融合、SVDD、optimizer、旧Loader/训练；退役RG与相关限制。
- .claude/rules/preprocessors-and-raw-data.md：原始观测、Drain截断、日志/Trace/Metric语义。

读取后区分旧数据/Endpoint聚合事实与当前Service事实；冲突时核对源码和数据并记录依据，不照抄过时注释。CLAUDE.md中的历史研究设想不覆盖用户当前任务。

相关历史重点：025标签与丢失资产、027时间捷径、028下游benchmark、029平凡基线、030公平共同样本、031v2/RG退役、032数据漂移、033文档/粒度风险。索引以history/index.md为准。

## 9. 开发、验证与协作约定

- 用中文说明发现、决策、改动和结果；区分本轮验证、历史引用、设计假设、尚未验证。
- 搜索优先rg；独立只读检索可以批量；依赖步骤、编辑和验收顺序明确。
- 先检查Git状态，保留已有dirty/untracked文件；不清理、reset或覆盖用户修改。小范围实现，不顺手重构旧管线。
- 原始数据只读；用户只读审查/设计任务只写指定/tmp。用户明确允许开发时按批准范围改文件，不重复索取已给出的授权。
- 测试在用户要求验证、修复复现或明确执行带测试步骤的Plan范围内运行；不将全面重训、全量重建或大型测试当默认附带动作。
- 新数据/契约正确性必须有可检查依据；计划中列出的测试未执行时不能称通过。完成时报告实际改动、执行检查及未完成部分。
- 格式沿用black/isort、type hints、logging；超参数写配置。sklearn适配无需继承torch.nn.Module；神经模块才沿用项目nn.Module约定。
- feature/用于基础设施，baseline/用于经典基线，exp/用于实验迭代。提交仅相关文件，不自动push/merge或提交大parquet/raw。
- 提交格式沿用CLAUDE： [Feature]、[Bugfix]、[Experiment]、[Data]、[Model]、[Refactor]、[Docs]、[Test]、[Chore]，格式为“[Type]: description”；不加Co-Authored-By。
- 合并PR前遵循项目history记录要求；读取skills-local/project-history/SKILL.md，新增entry并更新index。单纯只读调研/创建本指南不自行发PR或补造实验历史。
- GitHub任务流程需要时阅读docs/agents/issue-tracker.md、triage-labels.md、domain.md及CONTEXT.md/docs/adr。不要无授权对外发送消息或发布结果。
- 应用当前会话实际提供的技能和工具；不要为了匹配Claude流程自动安装技能、修改配置或启动多代理。ARIS链接是外部skill源，不修改其symlink目标。

## 10. 已有审计与设计产物索引

|目录|用途|
|---|---|
|/tmp/service-model-design-20261013|本次Service代码调研、三方案比较、Spec、实施Plan与摘要|
|/tmp/service-endpoint-feasibility-20261013|Service资产、27case标签可行性、跨层证据与限制|
|/tmp/endpoint-red-response-minimal-20261013|已实际完成的9/12维公平Client RED/Response实验|
|/tmp/experiment-audit-20261013|历史实验资产及协议可比性审计|
|/tmp/api-impact-review-20261013|原始调用关系、影响标签证据与传播实例|
|/tmp/b-admission-20261012|此前候选路线准入审查，仅作历史参考|

这些临时资产有存在性与版本限制，先检查再引用。**核心开发原则：用正确的样本、明确的任务和可复现证据回答问题；数据工程、严格评估和现成算法组合不自动等于方法创新。**
