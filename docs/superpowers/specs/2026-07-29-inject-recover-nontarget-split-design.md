# Inject/Recover 非目标 Endpoint 行时序切分 — Design Spec

**Date:** 2026-07-29
**Branch:** 待定（`feature/eval-all-nontarget-split` 或类似）
**承接:** entry 019 遗留问题（`fault_baseline_train_fraction` 推到数值上限后，
eval_all 正负比仍封顶 12.24:87.76，根因是 inject 阶段非目标 endpoint 的
fan-out 残留 + recover 阶段行完全不参与任何训练池吸收逻辑）
**Status:** Design approved（经 brainstorming 逐问题访谈确认），pending spec
review before implementation-plan。

## 1. 背景与动机

issue #16 / PR #19（entry 016）修复了 `expand_train_pool=true` 时故障 case
`baseline` 阶段行的类别失衡问题，把"整段吸收进训练池"改为按时间窗时序切分。
但 entry 019 在 `new_ep1` 数据集上首次训练评估时发现，即使
`fault_baseline_train_fraction` 推到数值上限 1.0（baseline 阶段行全部吸收进
训练池），eval_all 正负比仍然封顶在约 **12.24:87.76**，达不到 1:1 目标。

分解 eval_all 负样本构成后确认三个固定不受 `fault_baseline_train_fraction`
影响的桶：

| 桶 | new_ep1 实测行数 | 现状 |
|---|---|---|
| inject 阶段、非目标 endpoint（fan-out 残留） | 2546 | 完全不参与任何吸收逻辑 |
| recover 阶段（不分目标/非目标） | 446 | 完全不参与任何吸收逻辑 |
| Normal holdout | 176 | 设计上不应参与（Normalizer 泄漏风险） |

正样本（目标 endpoint × inject 窗口）固定 442 行，不受任何 fraction 影响。

即使 baseline 全部吸收（fraction=1.0），负样本地板仍是
`2546 + 446 + 176 = 3168`，对比正样本 442，比例封顶 12.24:87.76——这是
**endpoint 级 fan-out 的结构性稀释**，`fault_baseline_train_fraction` 单参数
无法突破，entry 019 当轮决定不做定义层改动，留待本轮处理。

**范围边界**：数据集本身即将重采（entry 021：PATCH 类故障需要响应体模态才能
识别，现有 `merged_v2`/`new_ep1` 都会被替换），本轮实验数字不重要，**目标是
把切分逻辑做对、参数可调、有测试锁住行为，不是在当前数据集上真正逼近 1:1 的
具体数值**。

## 2. 核心方案

新增两路时序切分，分别处理 inject 阶段非目标 endpoint 行、recover 阶段非目标
endpoint 行，与已有的 baseline 阶段切分并列，三者共用同一套"按时间窗切分、
最早 fraction 比例进训练池"的切分哲学，但各自独立的 fraction 参数、独立的
判据字段。

### 2.1 判据语义：inject 与 recover 不能共用同一个字段

这是本轮设计中最容易踩错的地方，必须显式区分：

**inject 侧**用 `is_endpoint_anomaly == False` 作为"非目标行"判据。理由：
`is_endpoint_anomaly` 已经统一处理了两种 `label_granularity`——
`label_granularity=="endpoint"` 时精确等于 `is_target_endpoint & is_anomaly`；
`label_granularity=="case"` 时 fallback 等于 `is_anomaly`。在 inject 阶段
（`is_anomaly=True`），case 级标签的所有行 `is_endpoint_anomaly` 恒为
`True`，用该判据筛选时这批行自然一行都不会被选中——**不会误吸收正样本**，
不需要额外写 `label_granularity` 分支。

**recover 侧**不能用 `is_endpoint_anomaly`。原因：`is_anomaly` 的定义是
`phase == "inject"`，recover 阶段所有行的 `is_anomaly` 恒为 `False`，因此
`is_endpoint_anomaly` 在 recover 阶段对所有 endpoint（无论是否目标）恒为
`False`——用它筛选是空操作，无法区分目标/非目标。必须换成
`is_target_endpoint == False`。但这个字段只在 `label_granularity=="endpoint"`
的 case 里有精确含义；`label_granularity=="case"` 的 case（如 `anomod_v1`）
没有 `target_endpoint` 字段，`is_target_endpoint` 被 `trace_preprocessor.py`
统一填 `False`，对这批 case 该判据同样是无法区分目标/非目标的空判据——如果
不加约束，会把这些 case 里"实际上就是故障发生地"的 endpoint 的 recover 行也
当成"非目标"吸收进训练池，污染训练池对"正常"的定义。

**结论**：recover 侧切分**只对 `label_granularity == "endpoint"` 的 case
生效**，`label_granularity == "case"` 的 case 的 recover 行整段排除在新逻辑
外，原样保留在 `eval_all`。这不是"暂不支持"的临时限制，而是标签精度本身决定
的边界——沿用 CLAUDE.md 已有的"per-endpoint 标签精度取决于 target_endpoint
字段是否存在"这条原则。

### 2.2 切分范围：不动 recover 阶段目标 endpoint 行

目标 endpoint 自身的 recover 行维持现状，保留在 `eval_all`，不纳入吸收范围。
理由与 entry 014/016 对 baseline 扩容"不吸收 recover"的既有决策一致——系统
故障后未必立即稳定回正常态，目标 endpoint 的 recover 行分布未验证，保守排除。
这条保护本质针对的是目标 endpoint，非目标 endpoint 在 recover 阶段没有这个
顾虑（它们从未发生过故障），因此本轮把非目标 endpoint 的 recover 行纳入吸收
范围，同时继续保护目标 endpoint 的 recover 行不被碰。

### 2.3 切分粒度与方向：与 baseline 切分一致

沿用 `split_fault_baseline_temporal`（本次重命名，见 §3.1）已有的切分哲学：
按 `timestamp_window_ms` 整窗切分（不拆散同一窗内的多个 endpoint 行），每个
case 内最早 fraction 比例的窗口进训练池，其余留在 eval_all。三路切分（
baseline / inject 非目标 / recover 非目标）复用同一个通用函数，只是喂入不同
的行子集和不同的 fraction。

### 2.4 切分比例：两个独立可调参数，默认 0.0

新增两个配置字段：

```python
fault_inject_nontarget_train_fraction: float = 0.0
fault_recover_nontarget_train_fraction: float = 0.0
```

**inject 与 recover 各自独立的 fraction**，不共用一个参数——两个桶的规模、
对模型的影响机制不同（inject 非目标行仍是"故障发生时段"的正常流量，recover
非目标行是"故障已结束"的正常流量），需要能分别调节吸收强度。

**默认值 0.0**（向后兼容）：与 `fault_baseline_train_fraction` 当年引入时的
考虑一致——新参数默认不改变现有行为，`expand_train_pool=true` 的既有配置
（`v1_expanded_pool.yaml`、`v1_new_ep1.yaml`）不会因为字段新增而静默改变产物，
除非显式在 config 里打开。

**两个参数只在 `expand_train_pool=True` 时被消费**，复用现有总闸而不新开
独立开关——这两类吸收和 baseline 吸收语义上同属"训练池扩容"这个大伞，
`expand_train_pool=False` 时全部忽略（但仍参与加载期范围校验）。

### 2.5 数值可达性：本轮不追加动 recover 目标行或 holdout

用 `new_ep1` 的数字做一次参考计算（仅供设计参考，不代表未来重采数据集的真实
比例）：inject 非目标行 2546 行全部吸收后，负样本地板从 3168 降到
`recover 非目标部分 + holdout(176)`。recover 446 行里目标/非目标各占多少
当前未拆分统计，若大部分是目标 endpoint 自身产生的（会被继续保护、不参与
吸收），非目标可吸收部分可能远小于 446，实际可达比例需重采后用真实数据验证。
即便如此，与正样本 442 相比大概率仍达不到 1:1。本轮 `v1_expanded_pool.yaml` /
`v1_new_ep1.yaml` 里两个新 fraction 均设为 `1.0`（同步开启，尽量逼近本轮
逻辑能达到的上限），但**不再追加改动 recover 阶段目标 endpoint 行或 Normal
holdout**——这两者继续被各自的既有理由保护，若未来仍不够接近 1:1，留给下一轮
单独讨论。

## 3. 组件设计

### 3.1 函数重命名：`split_fault_baseline_temporal` → `split_fault_phase_temporal`

现有 `src/contracts/split_fault_baseline.py::split_fault_baseline_temporal`
本身是通用的"按 case 分组、按时间窗时序切分"函数，不关心调用方传入的是哪个
phase 的行——函数内部没有任何 baseline 专属逻辑。现有命名把函数功能和调用
场景绑死了，本轮要在三个场景复用它（baseline / inject 非目标 / recover 非
目标），继续叫 `split_fault_baseline_temporal` 会造成"函数叫 baseline 却处理
非 baseline 行"的名实不符。

**方案**：文件重命名为 `src/contracts/split_fault_phase.py`，函数重命名为
`split_fault_phase_temporal`，**签名不变**：

```python
def split_fault_phase_temporal(
    df: pd.DataFrame,
    fraction: float,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (train_part, eval_part)
```

同步改动点：
- `scripts/build_contract.py` 的 import 与调用处
- `dvc_new_ep1/dvc.yaml`、`dvc_reliability_gate/dvc.yaml` 里 deps 路径
  （`src/contracts/split_fault_baseline.py` → `split_fault_phase.py`）
- `tests/test_split_fault_baseline.py` 重命名为 `tests/test_split_fault_phase.py`
  （测试内容不变，只改 import 和函数名，函数行为本身不受本次改动影响）

### 3.2 配置字段

`ContractConfig`（`src/contracts/contract_config.py`）新增两个字段，照抄
`fault_baseline_train_fraction` 的既有模式：

```python
# v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 inject
# 阶段非目标 endpoint 行（is_endpoint_anomaly==False）按时间窗时序切分时，
# 最早 fraction 比例的窗口进训练池，其余留在 eval_all。默认 0.0（不吸收，向后
# 兼容），expand_train_pool=false 时被忽略（但仍参与 __post_init__ 范围校验）。
fault_inject_nontarget_train_fraction: float = 0.0
# v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 recover
# 阶段非目标 endpoint 行按时间窗时序切分时，最早 fraction 比例的窗口进训练池，
# 其余留在 eval_all。仅对 label_granularity=="endpoint" 的 case 生效——
# label_granularity=="case" 的 case 没有 target_endpoint 字段，is_target_endpoint
# 全为 False，无法区分目标/非目标，recover 行整段排除在外，原样留在 eval_all。
# 默认 0.0（不吸收，向后兼容）。
fault_recover_nontarget_train_fraction: float = 0.0
```

`__post_init__` 恒校验（不看 `expand_train_pool`）两个字段类型为数值且落在
`[0.0, 1.0]`，越界或类型错误抛 `ValueError`——理由与 `fault_baseline_train_fraction`
的既有校验一致：配置错误应在加载期暴露，不能等到 `split_fault_phase_temporal`
内部产出反直觉结果（负数触发 Python 负索引切片导致方向反转，大于 1 不报错但
吞掉全部窗口）才发现。

`load_contract_config()` 新增对应 `raw.get("fault_inject_nontarget_train_fraction", 0.0)`
和 `raw.get("fault_recover_nontarget_train_fraction", 0.0)`。

### 3.3 `_write_v1` 接线

在 `expand_train_pool=True` 分支里，`fault_baseline_train`/`fault_baseline_eval`
切分之后，追加两段结构相同的切分：

```python
# inject 非目标行：is_endpoint_anomaly==False 天然兼容两种 label_granularity
# （case 级 fallback 等于 is_anomaly，inject 阶段恒 True，该判据自动选不出任何
# 行，不会误吸收正样本），不需要额外按 label_granularity 分支。
fault_inject_nontarget_df = anomaly_df[
    (anomaly_df["phase"] == "inject") & (~anomaly_df["is_endpoint_anomaly"])
]
fault_inject_nontarget_train, fault_inject_nontarget_eval = split_fault_phase_temporal(
    fault_inject_nontarget_df, fraction=fault_inject_nontarget_train_fraction
)
fault_inject_nontarget_train["source_phase"] = "fault_inject_nontarget"

# recover 非目标行：is_target_endpoint==False 判据仅在 label_granularity=="endpoint"
# 时有精确含义，先过滤 granularity 再判据，case 级标签的 case 整段排除在外。
fault_recover_nontarget_df = anomaly_df[
    (anomaly_df["phase"] == "recover")
    & (anomaly_df["label_granularity"] == "endpoint")
    & (~anomaly_df["is_target_endpoint"])
]
fault_recover_nontarget_train, fault_recover_nontarget_eval = split_fault_phase_temporal(
    fault_recover_nontarget_df, fraction=fault_recover_nontarget_train_fraction
)
fault_recover_nontarget_train["source_phase"] = "fault_recover_nontarget"
```

`train_pool` 拼接新增 `fault_inject_nontarget_train`、
`fault_recover_nontarget_train`。

`eval_all` 的构成从"`phase != baseline` 全量 + `fault_baseline_eval` +
holdout"改为：

```python
eval_all = pd.concat(
    [
        # inject 阶段：目标行（正样本，从未被碰）+ 未被吸收的非目标行
        anomaly_df[
            (anomaly_df["phase"] == "inject") & anomaly_df["is_endpoint_anomaly"]
        ],
        fault_inject_nontarget_eval,
        # recover 阶段：case 级标签整段（从未被碰）+ endpoint 级标签里未被吸收的部分
        # （含目标行和未被吸收的非目标行）
        anomaly_df[
            (anomaly_df["phase"] == "recover") & (anomaly_df["label_granularity"] != "endpoint")
        ],
        anomaly_df[
            (anomaly_df["phase"] == "recover")
            & (anomaly_df["label_granularity"] == "endpoint")
            & anomaly_df["is_target_endpoint"]
        ],
        fault_recover_nontarget_eval,
        fault_baseline_eval,
        parts["eval_normal_holdout"],
    ],
    ignore_index=True,
)
```

`else` 分支（`expand_train_pool=False`）完全不动，两个新字段在该分支下不被
读取。`EndpointBaselineStats`/`Normalizer` 的 fit 范围不受影响，继续锁定
`parts["train_fit"]`（纯 Normal），本次只改 `train.parquet`/`eval_all.parquet`
的行构成。

### 3.4 日志

沿用现有 `expand_train_pool` 分支末尾的 `LOG.info` 汇总，追加
`fault_inject_nontarget_train/eval`、`fault_recover_nontarget_train/eval`
行数。沿用 `fault_baseline_train` 已有的"吸收 0 行"告警模式——若某个 fraction
`>0` 但吸收结果是 0 行，发 `LOG.warning` 提示检查 fraction 或候选行数是否
过少。

### 3.5 `is_train_eligible` 列：不动

`is_train_eligible = ~is_anomaly`（`_attach_label_columns` 里定义）保持现有
定义不变。该列在既有 docstring 里已明确是"逐行粗粒度可训练性标记，不是训练
池扩容逻辑的依据"——本轮新增的两类吸收逻辑同样不依据这一列路由，与
`fault_baseline` 吸收逻辑先例一致。实际路由结果统一看 `source_phase` 列
（本轮新增 `fault_inject_nontarget`/`fault_recover_nontarget` 两个取值）。

## 4. 组件依赖关系

```
configs/contract/v1_expanded_pool.yaml / v1_new_ep1.yaml
（声明 fault_inject_nontarget_train_fraction / fault_recover_nontarget_train_fraction: 1.0）
  └─ ContractConfig 新增两字段（§3.2，独立范围校验）
       └─ build_contract.py::_write_v1（expand 分支追加两段接线，§3.3）
            └─ split_fault_phase_temporal（重命名后的通用切分函数，§3.1）
                 └─ 产出 train.parquet / eval_all.parquet（sample_id 互斥不变量延续）
```

单向依赖链，实现顺序：§3.1 函数重命名 → §3.2 字段 → §3.3 接线 → §3.4 日志。

## 5. 测试策略

- **`split_fault_phase_temporal` 单测**（`tests/test_split_fault_phase.py`，
  由 `test_split_fault_baseline.py` 重命名）：保留原有全部用例，仅改 import
  和函数名——函数本身是纯粹的"按 case+时间窗切分"，与调用方喂入哪个 phase 的
  行无关，行为不受本次改动影响，不需要新增用例。
- **`ContractConfig` 字段测试**（`tests/test_contract_config.py`）：两个新
  字段各自默认值 0.0、覆写生效、越界（2.0 / -0.1）抛 `ValueError`，照抄
  `fault_baseline_train_fraction` 已有测试模式。
- **新 fixture**（`tests/fixtures/` 下新建，覆盖两种 `label_granularity`）：
  - 1 个 `label_granularity=="endpoint"` 的故障 case，含目标 endpoint + 至少
    1 个非目标 endpoint（fan-out），时间窗覆盖 baseline/inject/recover 三个
    阶段，非目标 endpoint 在 inject/recover 阶段均有足够窗口数验证非零分数
    切分（参照 `split_fraction_mini` 先例，`int(n*fraction)` 需要 n≥5 才能
    验证 0.2 这类小数）。
  - 1 个 `label_granularity=="case"` 的故障 case（无 `target_endpoint` 字段），
    验证 recover 切分对它是 no-op。
- **`_write_v1` 集成测试**（`tests/test_contract_v1_train_pool.py` 新增用例，
  基于上述新 fixture）：
  - inject 非目标行按 fraction 被部分吸收进 train，`source_phase=="fault_inject_nontarget"`。
  - inject 目标行（正样本）不管 fraction 多高都不会被吸收——这是对
    `is_endpoint_anomaly` 判据的核心保护，回归测试。
  - recover 非目标行（仅 endpoint 级 case）按 fraction 被部分吸收，
    `source_phase=="fault_recover_nontarget"`。
  - recover 目标行（endpoint 级 case）不管 fraction 多高都不会被吸收。
  - **case 级标签 case 的 recover 行，不管 fraction 多高，一行都不会被吸收**——
    对 §2.1 结论的直接回归测试，是本轮最容易被静默破坏的行为。
  - `train.sample_id ∩ eval_all.sample_id == ∅` 不变量在新增两路吸收后依然
    成立。
  - `expand_train_pool=False` 时两个新字段即使设为非默认值也是 no-op（照抄
    `test_fault_baseline_train_fraction_is_noop_when_expand_train_pool_false`
    模式）。
- **既有测试不回归**：`test_contract_v1_split.py`（未扩容路径）、
  `test_dvc_pipeline_v1.py`、`test_build_contract_v1_endpoint_id.py`
  （baseline_stats fit 范围锁定 train_fit，本次不碰 fit）、
  `test_e2e_reliability_gate_smoke.py`（走 expanded 路径，只断言
  `len(scores)==len(eval_all)`，行数会变但断言本身仍应通过）。
- **本轮不做端到端重跑验证指标数字**——entry 021 已决定 v0/v1/v1_expanded
  相关产物因 `MAX_CONTENT_CHARS` drift 暂不重跑，本次改动进一步加大 drift
  （eval_all 行构成变化），实验数字不重要，重点是切分逻辑正确性由单测/集成
  测试锁住。是否需要跑一次 `dvc repro` 仅用于验证流程本身不报错（不采信数字），
  留给实现阶段决定。

## 6. DVC 与文档同步

- **DVC 依赖**：`dvc_new_ep1/dvc.yaml`、`dvc_reliability_gate/dvc.yaml` 的
  `deps` 列表里 `src/contracts/split_fault_baseline.py` 改为
  `src/contracts/split_fault_phase.py`。
- **配置文件**：`v1_expanded_pool.yaml`、`v1_new_ep1.yaml` 均新增两个字段并
  设为 `1.0`（§2.5）。
- **CLAUDE.md**：在 "expand_train_pool 开关会改变 eval_all 行数" 与
  "fault_baseline_train_fraction 控制"那两条已知问题附近，补充本次两个新
  字段的语义、判据差异（inject 用 `is_endpoint_anomaly`，recover 用
  `is_target_endpoint` 且仅对 endpoint 级标签生效）、以及"即使三路吸收全部
  推满仍可能达不到 1:1"这一结论（具体数字待重采后用真实数据验证）。
- **history entry**：新建条目记录本次改动动机（承接 entry 019 遗留问题）、
  判据语义踩坑（§2.1，最值得记录的坑）、以及"本轮不重跑既有 baseline 指标"
  的决定（沿用 entry 021 的判断依据）。更新 `history/index.md` 列表与影响域
  索引。

## 7. Out of Scope

- **recover 阶段目标 endpoint 行的吸收**——继续沿用"系统未稳定回正常态，分布
  未验证"的保守排除理由，本轮不重新讨论。
- **Normal holdout 的压缩**——holdout 存在的理由是 Normalizer 防泄漏，与
  eval_all 类别平衡是两个独立问题，本轮不碰。
- **在真实重采数据集上验证是否真正逼近 1:1**——数据集本身待重采（entry 021），
  本轮只保证逻辑正确、参数可调，不追求当前数据集上的具体比例数字。
- **`is_train_eligible` 列语义调整**——维持"文档性粗粒度标记，非路由依据"的
  现状，不因本轮新增路由逻辑而重新定义。
- **case-aware / per-anomaly-type 的 fraction 分级机制**——entry 019 遗留的
  另一个独立 TODO（PATCH 类故障在 `fraction=1.0` 时的训练池污染副作用），
  与本轮问题不同，不在本次范围内一并解决。

## 8. 实现拆分（预计 6 个 commit）

1. `[Refactor]`：`split_fault_baseline.py`/`split_fault_baseline_temporal`
   重命名为 `split_fault_phase.py`/`split_fault_phase_temporal`，同步改
   `build_contract.py` import、DVC deps、测试文件重命名。纯重命名，不改行为，
   独立一个 commit 便于审查 diff。
2. `[Feature]`：`ContractConfig` 新增两个字段 + `__post_init__` 范围校验 +
   loader 装配 + 单测。
3. `[Test]`：新 fixture（含 endpoint 级 + case 级两种标签、fan-out 多
   endpoint 的故障 case）。
4. `[Feature]`：`_write_v1` 接线两路新切分（inject 非目标 + recover 非目标）
   + `eval_all` 构成同步改写 + 日志。
5. `[Test]`：`test_contract_v1_train_pool.py` 新增用例（新切分行为、
   `is_endpoint_anomaly`/`is_target_endpoint` 判据回归、case 级标签 recover
   no-op、sample_id 互斥不变量），全部跑通。
6. `[Docs]`：`v1_expanded_pool.yaml`/`v1_new_ep1.yaml` 新字段值 + CLAUDE.md
   已知问题更新 + history entry 新建 + index 更新。

> Commit 顺序 1→2→4 是依赖链，3 可与 1/2 并行；5 依赖 3+4；6 依赖 5。
