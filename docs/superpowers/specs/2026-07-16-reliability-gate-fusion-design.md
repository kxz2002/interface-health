# Deviation-Conditioned Reliability Gate Fusion — Design Spec

**Date:** 2026-07-16
**Branch:** `feature/reliability-gate-fusion`
**Status:** Design approved, pending spec review before implementation-plan.

## 1. 背景与动机

### 1.1 研究定位

本项目的融合创新点已经历两轮外部审稿（llm-chat / gpt-5.6-sol）打磨，前两个方案均被否决：

- **方案 A（entity-conditioned residual decomposition，mixed-effects 框架）**：被判定为"本质是 per-entity z-score 标准化喂给不变的 L2 门控"，是预处理层面改动而非融合架构创新；mixed-effects 术语不准确应删除；endpoint↔service 严格 1:1 映射使得"层次/广播"故事不成立。
- **方案 B（session-distribution-aware fusion）**：被判定为标准的 closed-set batch/domain correction，跨 session 分布偏移只在 2/8 endpoint 出现、仅 3 个 session，强度不足以撑方法论创新（只能作数据有效性修正/消融）；且校正对象（service 特征）与证据特征（`trace_request_count` 属 endpoint_red）错配。

第二轮审稿明确提出的第三方向被采纳：**reliability gate** —— 让融合门控依据"每个模态分支相对自己 normal 参照的偏离程度"去重新分配"该信任哪个模态"的权重，而不是校正数值本身。

### 1.2 数据信号验证（go/no-go 前置检查，已完成）

对 `artifacts/contract_v1/eval_all.parquet` 中 3121 个精确标注异常样本（`is_endpoint_anomaly==True`）做了模态偏离去相关性分析，结论支持该机制：

- 两分支偏离量整体 Pearson r=0.54（中等，非 lockstep）；约 40% 异常样本有某一模态明显占优（>2×），RED 占优与 SVC 占优近似各半。
- **按故障类型分解出结构性规律**（非随机噪声）：
  - HTTP 层 endpoint 注入故障（`Lv_E_HTTPABORT/REPLACE_*`）几乎纯 RED 占优（比例 5~12×，~100% 样本 RED 占优）——请求/响应层故障，几乎不扰动 CPU/内存/日志。
  - 资源耗尽类故障（`Lv_P_CPU_preserve`/`Lv_P_DISKIO_preserve`/`Lv_S_DNSFAIL_*`）几乎纯 SVC 占优（比例 0.3~0.65，RED 占优~0%）——基础设施层故障，只微弱渗透到 endpoint RED。
  - `HTTPPATCH`/`HTTPDELAY` 及部分 node 级故障（`cachelimit`/`CONNECTION_POOL_exhaustion`）两模态均弱偏离，属低信噪比困难样本。

这正是 reliability gate 需要的信号：**该信哪个模态，系统性地随故障类型翻转**。简单的"总偏离幅度门控"捕捉不到这个翻转，必须能感知两分支各自偏离量并据此路由。

### 1.3 novelty 定位（诚实边界）

定位为 **reliability-aware 模态路由门控**，不夸大为通用融合框架。论文中需主动说明与 mixture-of-experts / attention / domain-adaptive fusion 的关系并对比。核心区别于 L2 之处：**gate 的输入活在"异常度量空间"（各分支偏离量），而非 L2 的"原始特征空间"**——学的是"该信哪个分支"而非"该不该多信这个内容"。

## 2. 核心机制：Deviation-Conditioned Reliability Gate

### 2.1 数据流（单样本）

```
raw row (18 维特征 + endpoint_id)
  → 拆分 raw_ep(10 维 endpoint_red) / raw_svc(8 维 service_metric+service_log)
  → dev_ep  = summarize(zscore(raw_ep,  baseline[eid].ep))    # 2~3 维偏离摘要
  → dev_svc = summarize(zscore(raw_svc, baseline[eid].svc))   # 2~3 维偏离摘要
  → [w_ep, w_svc] = softmax(MLP([dev_ep, dev_svc]))           # 竞争性归一化的信任权重
  → e_ep = enc_ep(raw_ep);  e_svc = enc_svc(raw_svc)          # 沿用 L1 的分支编码器
  → z = w_ep * value_ep(e_ep) + w_svc * value_svc(e_svc)
  → SVDD 编码器 → 距离分数
```

### 2.2 与 L2 GatedFusion 的关键区别

| 维度 | L2 (`GatedFusion`) | Reliability Gate |
|------|-------------------|------------------|
| 门控输入 | 原始特征拼接 `[e_ep; e_svc]` | 两分支偏离摘要 `[dev_ep; dev_svc]` |
| 门控语义 | "该不该多信这个 service 内容" | "该信哪个分支" |
| 归一化 | 逐维 sigmoid（独立开关） | softmax（竞争性二选一） |
| 融合形式 | `e_ep + gate ⊙ value(e_svc)`（非对称，endpoint 恒为主） | `w_ep·value_ep + w_svc·value_svc`（对称加权） |
| 决策依据空间 | 原始特征空间 | 异常度量空间 |

softmax 相较 sigmoid 的意义：数据信号显示 HTTP 类故障 RED 偏离、资源类故障 SVC 偏离，两者此消彼长，softmax 天然表达这种信任转移。

### 2.3 偏离量表示形式（低维摘要，已定）

每分支输出 2~3 维偏离摘要，候选度量：总 z-score 范数 + 最大单特征 z-score + 偏离特征占比。gate MLP 输入 4~6 维。

选择理由：
1. **机制层面约束**：把 gate 职责限定为"只根据两分支各有多反常做路由"，从设计上让 gate 活在异常度量空间，与 L2 的特征空间门控形成干净区分（novelty 故事载体）。
2. **防过拟合第一道防线**：低维输入本身降低在 838~7359 行（稀疏 endpoint 仅几十行）数据上记住训练集巧合的风险。

**正交补充**：gate 的 MLP 上叠加常规 dropout / weight decay 作为训练层面兜底。偏离量表示形式约束"给 gate 看什么"，dropout 约束"gate 怎么学"，两者正交并用。

## 3. 组件划分与接入

遵循现有可插拔 fusion 约定（Hydra `_target_` 实例化，与 L0/L1/L2 平级切换）。

### 3.1 组件 1：`EndpointBaselineStats`（新增）

- **位置**：`src/data/`（与 `Normalizer` 同域）或 `src/fusion/`，实现时定夺。
- **职责**：每个 endpoint 一份，分别持有 RED 分支与 SVC 分支特征的 normal-only mean/std。
- **拟合**：仅在 `train_fit` 上 fit（沿用 `Normalizer` 的防泄漏纪律，`build_contract.py:379-381` 约定），fit 一次后固定，不参与 backprop。
- **稀疏 endpoint 稳健性**：对样本量少的 endpoint（如 `POST:/api/v1/users/login` ~17 行），std 估计不稳——用 shrinkage-to-global（向全局 std 收缩）兜底。作为可开关消融维度。
- **全 NaN group 兜底**：复用 `Normalizer` 已知坑教训（CLAUDE.md），某 endpoint 某列在 fit 集合全 NaN 时跳过标准化保留原值，不把 `[nan]` 统计量传播。
- **统计量产出位置（已定）**：在 contract build 阶段算好，存成 **sidecar 文件**（如 `artifacts/contract_v1/endpoint_baseline_stats.json/parquet`），fusion 模块运行时只读不算。**不 bake 进主 parquet 的行**——per-endpoint 分支统计量是 8×N 的小矩阵，逐行复制会把同一份统计量重复数千遍，违反数据规范化。理由：保持"所有统计量都在 train_fit 上算、防泄漏纪律集中一处"的一致性。

### 3.2 组件 2：`ReliabilityGatedFusion`（新增 `src/fusion/reliability_gate.py`）

- **职责**：接收模态字典 + `endpoint_id`，用组件 1 算出 `dev_ep`/`dev_svc`，softmax MLP 输出 `[w_ep, w_svc]`，加权融合 `value_ep(e_ep)`/`value_svc(e_svc)`。
- **接口**：继承 `FusionModule`（`src/fusion/base.py`），`forward()` 增加可选 `endpoint_id` 参数（向后兼容加法式改动，L0/L1/L2 忽略即可，已由基础设施调研确认）。
- **配置**：新增 `configs/fusion/reliability_gate.yaml`（`_target_` + `branch_dim` + dropout 率 + shrinkage 开关等超参）。

### 3.3 组件 3：管线接入（改动，非新增）

- **`scripts/build_contract.py`**：新增 `endpoint_id` int 列（8 个 endpoint 字符串→整数的确定性映射，`sorted(endpoint_to_service.keys())` 派生，baked 进 parquet）；`REQUIRED_ID_COLUMNS`（`src/contracts/contract_v0.py`）增加对应条目。同时产出组件 1 所需的 per-endpoint 分支统计量（sidecar 或列）。
- **`src/data/contract_dataloader.py`**：`_row_to_sample` 在 meta 中附带 `endpoint_id`。
- **`scripts/train_baseline_v0.py`**：`_collate`（40-45 行附近）把 `endpoint_id` 从 meta 拉进 batch dict，传给 `fusion.forward()`。

### 3.4 训练集扩容（配套基础设施改动）

为缓解数据稀缺（838 → ~7359 行），吸收故障 case 的 baseline 阶段行进 One-Class 训练池：

- **`build_contract.py`**：新增 `train_pool_mask = normal_mask | ((~normal_mask) & (phase == "baseline"))`。**只吸收 baseline，不吸收 recover**（recover 系统未稳定回正常态，分布未验证，保守排除）。
- **防泄漏（强制）**：`eval_all` 的构成从 `full[~normal_mask]` 收紧为 `full[~train_pool_mask]`——进入训练池的行必须同步从 eval 摘除，否则复现 v0 的 train⊆eval 泄漏 bug。
- **可审计标记**：新增 `source_phase` 列（`"normal_case"` / `"fault_baseline"`），标记新增行来源，可审计、未来可排除或降权。
- **已知风险留痕**：`inside_pay_service/inside_payment`、`preserveservice/preserve` 两 endpoint 的故障 case baseline 期流量比 normal_v2 高 1.8~1.9×（跨采集批次负载差异）——`source_phase` 列使其可事后排除/降权。
- **`split_v1.py` 无需改动**：其按 case+时间排序切分，不关心 case 是否 Normal，喂入混合池自动各 case 独立切分（但必须先按 phase 预过滤到 baseline，函数本身无 phase 感知）。

## 4. 实验方案

回应两轮审稿核心要求：归因清晰、区分"路由 vs 幅度门控"、区分"机制 vs 数据量"。

### 4.1 主对比

ReliabilityGate vs L0/L1/L2，多 seed（1/2/3/42），Contract v1 时序切分，报 AUROC/AUPRC ± std。

### 4.2 训练集扩容归因（2×2，审稿点名要求）

| 训练池 | L2 基线 | ReliabilityGate |
|--------|---------|-----------------|
| 原始 838 行 | ✓ | ✓ |
| 扩容 ~7359 行 | ✓ | ✓ |

证明提升不是纯数据量带来的。

### 4.3 关键消融（证明"路由"真有用）

- `dev` 输入换成"总偏离幅度标量"（去掉分支区分）→ 若性能掉，证明"该信哪个分支"确有用，非只靠总幅度。
- softmax 换成两个独立 sigmoid → 验证"竞争性归一化"是否必要。
- 固定 `[w_ep, w_svc] = [0.5, 0.5]`（禁用门控）→ 下界。

### 4.4 可解释性证据（论文卖点）

按故障类型分层展示 gate 学到的 `w_ep`/`w_svc`。预期 HTTP 类故障 `w_ep` 高、资源类故障 `w_svc` 高，直接对应 §1.2 已验证的数据信号，构成强定性证据。

### 4.5 稳健性

shrinkage-to-global 开/关对稀疏 endpoint（login 等）的影响。

## 5. 测试

遵循项目 `tests/` 约定 + CLAUDE.md「fusion 模块与契约层必须有单测」要求。

- **`EndpointBaselineStats`**：fit 仅用 train_fit；shrinkage 逻辑；全 NaN group 兜底（保留原值不传播 NaN）。
- **`ReliabilityGatedFusion`**：forward 形状；softmax 权重和为 1；`endpoint_id` 缺失时退化行为；与 `FusionModule` 接口/L0-L1-L2 兼容。
- **契约层**：`endpoint_id` 列存在性 + 8 个映射正确；扩容后 `test_build_contract_smoke.py:49` 断言更新（train 不再全部 `Normal` case_id）；`test_contract_v1_split.py` 泄漏回归测试扩一个"故障 baseline 行确实不出现在 eval_all"用例。

## 6. 已识别风险

1. **数据质量 bug（前置，独立修复）**：`endpoint_red__client_latency_p95` 与 `endpoint_red__latency_divergence` 约 25% 行含 1e9~1e13 异常值（疑似时间戳泄漏，根因未确认）。会污染偏离量计算，须在正式实现前定位修复。**作为独立前置 bugfix，不混进本设计实现分支**（用户将单独派 subagent 处理）。详见 memory `todo_latency_p95_corrupted_values`。
2. **困难样本**：`HTTPPATCH`/`HTTPDELAY` 及部分 node 级故障两模态均弱偏离，gate 在这类样本上大概率无优势——实验如实报告，不掩盖。
3. **扩容副作用**：session 流量偏移（`inside_payment`/`preserve`）、recover 阶段排除——`source_phase` 列可审计。
4. **novelty 边界**：论文诚实定位为 reliability-aware 模态路由 gate，主动对比 mixture-of-experts/attention。

## 7. 明确排除项（YAGNI）

- 不做 sibling cross-attention（endpoint↔service 1:1，无同伴信号）。
- 不做 endpoint 时序序列建模（数据量不足，稀疏 endpoint 单 case 个位数窗口）。
- 不做 session-conditional 数值校正作为独立创新（审稿判定强度不足）。
- 不把 mixed-effects/fixed-effect/random-effect 术语写进论文。
- 不做逐特征 z-score 全维门控输入（退化为特征门控，novelty 故事受损）。
- 目标耦合式融合（SVDD 超球体距离反馈门控）为 plan B，本 spec 不涵盖——仅当 reliability gate 实证效果不佳时再另立设计（且需约束式设计规避循环论证风险）。
