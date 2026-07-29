# Inject/Recover 非目标 Endpoint 行时序切分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `expand_train_pool=true` 时的训练池除 baseline 阶段行外，还能按两个独立可调 fraction 吸收故障 case 的 inject 阶段非目标 endpoint 行与 recover 阶段非目标 endpoint 行，压低 `eval_all` 的负样本地板。

**Architecture:** 复用现有"按 case 分组、按时间窗整窗时序切分"的通用函数（本次从 `split_fault_baseline_temporal` 重命名为 `split_fault_phase_temporal`），在 `scripts/build_contract.py::_write_v1` 的 `expand_train_pool=True` 分支里追加两路切分。判据字段两路不同：inject 侧用 `is_endpoint_anomaly == False`（天然兼容 case/endpoint 两种 `label_granularity`），recover 侧用 `is_target_endpoint == False` 且仅对 `label_granularity == "endpoint"` 的 case 生效。

**Tech Stack:** Python 3, pandas, PyYAML, pytest, DVC, conda 环境 `interface`

**Spec:** `docs/superpowers/specs/2026-07-29-inject-recover-nontarget-split-design.md`

**环境提醒：** 本仓库代码必须在 conda 环境 `interface` 里跑。所有 `pytest`/`python` 命令统一写成 `conda run -n interface <cmd>`。直接 `python` 走的是 base 环境，会 import 失败。

---

## File Structure

**重命名（Task 1）：**
- `src/contracts/split_fault_baseline.py` → `src/contracts/split_fault_phase.py`（函数 `split_fault_baseline_temporal` → `split_fault_phase_temporal`）
- `tests/test_split_fault_baseline.py` → `tests/test_split_fault_phase.py`

**修改：**
- `src/contracts/contract_config.py` — 新增两个 fraction 字段 + 校验 + loader 装配（Task 2）
- `scripts/build_contract.py` — import 改名（Task 1）、`_write_v1` 追加两路切分（Task 4）
- `configs/contract/v1_expanded_pool.yaml`、`configs/contract/v1_new_ep1.yaml` — 声明两个新字段 = 1.0（Task 6）
- `dvc_reliability_gate/dvc.yaml`、`dvc_new_ep1/dvc.yaml` — deps 路径改名（Task 1）
- `tests/test_contract_config.py` — 新字段单测（Task 2）
- `tests/test_contract_v1_train_pool.py` — 新集成测试（Task 5）
- `CLAUDE.md`、`history/entries/`、`history/index.md` — 文档（Task 6）

**新建：**
- `tests/fixtures/nontarget_split_mini/` — 新 fixture（含 endpoint 级 + case 级两种标签的故障 case，每个 case 含 2 个 endpoint 制造 fan-out）（Task 3）
- `tests/fixtures/nontarget_split_mini.yaml` — 该 fixture 的 dataset config（Task 3）

**不动：** `src/contracts/split_v1.py`（Normal 三路切分）、`src/data/endpoint_baseline_stats.py`、`src/data/normalization.py`、根 `dvc.yaml`。

---

## Task 1: 重命名 `split_fault_baseline` → `split_fault_phase`（纯重命名，不改行为）

现有函数内部没有任何 baseline 专属逻辑（不做 phase 过滤，调用方喂什么切什么），本轮要在 3 个场景复用它，继续叫 `baseline` 会名实不符。本 task 只做重命名，行为零变化，独立一个 commit 便于审查。

**Files:**
- Rename: `src/contracts/split_fault_baseline.py` → `src/contracts/split_fault_phase.py`
- Rename: `tests/test_split_fault_baseline.py` → `tests/test_split_fault_phase.py`
- Modify: `scripts/build_contract.py:27`（import）、`scripts/build_contract.py:542`（调用）
- Modify: `src/contracts/contract_config.py:61`（注释里提到的函数名）
- Modify: `dvc_reliability_gate/dvc.yaml:21`、`dvc_new_ep1/dvc.yaml:20`（deps 路径）
- Modify: `tests/test_contract_v1_train_pool.py:78`（docstring 里提到的函数名）

- [ ] **Step 1: 先跑一遍现有测试，确认起点是全绿**

```bash
conda run -n interface pytest tests/test_split_fault_baseline.py tests/test_contract_v1_train_pool.py -q
```

Expected: 全部 PASS（重命名前的基线，若这里就有 fail 先停下来搞清楚原因，不要继续）

- [ ] **Step 2: git mv 两个文件**

```bash
git mv src/contracts/split_fault_baseline.py src/contracts/split_fault_phase.py
git mv tests/test_split_fault_baseline.py tests/test_split_fault_phase.py
```

- [ ] **Step 3: 改 `src/contracts/split_fault_phase.py` 的模块 docstring 与函数名**

把文件顶部 docstring 第 1 行改为（保留其余段落不动）：

```python
"""Contract v1 训练池扩容专属：故障 case 指定阶段行的两路时序切分。
```

并在该 docstring 的"与 split_v1.split_normal_rows_temporal 的差异"段落之后，追加一段说明为什么函数名不带 phase 限定：

```python
本函数不假设输入是哪个 phase 的行——内部只按 case_col 分组、按 time_col 整窗切分，
不读 phase 列。调用方负责筛出目标行子集（baseline / inject 非目标 / recover 非目标
三个场景共用本函数，各自传不同的 fraction），因此函数名用 phase 而非 baseline。
"""
```

函数签名改名（签名参数完全不变）：

```python
def split_fault_phase_temporal(
    df: pd.DataFrame,
    fraction: float,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
```

同时把函数内 `KeyError` 消息里的函数名改掉（原第 42 行）：

```python
            raise KeyError(
                f"split_fault_phase_temporal: 列 {col!r} 不存在于输入 df，"
                f"可用列={list(df.columns)}"
            )
```

- [ ] **Step 4: 改 `scripts/build_contract.py` 的 import 与调用**

第 27 行：

```python
from src.contracts.split_fault_phase import split_fault_phase_temporal
```

第 542 行附近的调用：

```python
        fault_baseline_train, fault_baseline_eval = split_fault_phase_temporal(
            fault_baseline_df, fraction=fault_baseline_train_fraction
        )
```

- [ ] **Step 5: 改 `src/contracts/contract_config.py` 注释里的函数名**

第 61 行那句注释里的 `split_fault_baseline_temporal` 改为 `split_fault_phase_temporal`。

- [ ] **Step 6: 改两个 DVC pipeline 的 deps 路径**

`dvc_reliability_gate/dvc.yaml` 第 21 行、`dvc_new_ep1/dvc.yaml` 第 20 行，均把

```yaml
      - src/contracts/split_fault_baseline.py
```

改为

```yaml
      - src/contracts/split_fault_phase.py
```

- [ ] **Step 7: 改测试文件里的 import / 函数名 / docstring**

`tests/test_split_fault_phase.py`：第 1 行 docstring 与第 8 行 import 改名，文件内全部 15 处 `split_fault_baseline_temporal(` 调用改为 `split_fault_phase_temporal(`。可用 sed 批量替换后人工核对一遍：

```bash
sed -i 's/split_fault_baseline_temporal/split_fault_phase_temporal/g; s/from src.contracts.split_fault_baseline import/from src.contracts.split_fault_phase import/' tests/test_split_fault_phase.py
```

`tests/test_contract_v1_train_pool.py` 第 78 行 docstring 里的 `split_fault_baseline_temporal` 同样改名（这行只是注释文字，不是调用）。

- [ ] **Step 8: 确认仓库里再无旧名残留**

```bash
grep -rn "split_fault_baseline" --include="*.py" --include="*.yaml" --include="*.md" . | grep -v "^./docs/superpowers/" | grep -v "^./history/"
```

Expected: 无输出。（`docs/superpowers/` 下的 spec/plan 与 `history/` 下的历史记录保留旧名是正确的——它们是写作当时的事实记录，不该改。）

- [ ] **Step 9: 跑测试确认重命名没破坏任何行为**

```bash
conda run -n interface pytest tests/test_split_fault_phase.py tests/test_contract_v1_train_pool.py tests/test_dvc_pipeline_v1.py -q
```

Expected: 全部 PASS，测试数量与 Step 1 一致

- [ ] **Step 10: Commit**

```bash
git add -A src/contracts scripts/build_contract.py dvc_reliability_gate/dvc.yaml dvc_new_ep1/dvc.yaml tests/test_split_fault_phase.py tests/test_contract_v1_train_pool.py
git commit -m "[Refactor]: split_fault_baseline 重命名为 split_fault_phase

该函数内部不读 phase 列、不做 phase 过滤，是通用的按 case+时间窗整窗切分
逻辑，原命名把功能与调用场景（baseline）绑死了。后续要在 inject 非目标行、
recover 非目标行两个新场景复用它，先改名避免名实不符。纯重命名，行为零变化。"
```

---

## Task 2: `ContractConfig` 新增两个 fraction 字段

**Files:**
- Modify: `src/contracts/contract_config.py`（dataclass 字段 + `__post_init__` 校验 + `load_contract_config` 装配）
- Test: `tests/test_contract_config.py`

- [ ] **Step 1: 写失败的测试**

追加到 `tests/test_contract_config.py` 末尾（`_MINIMAL_MODALITIES` 这个辅助常量本文件里没有，所以每个测试自己拼 YAML，与该文件既有风格一致）：

```python
def test_nontarget_fractions_default_to_zero(tmp_path):
    """两个新字段默认 0.0：向后兼容——config 不写时行为与本次改动前完全一致
    （expand_train_pool=true 也只吸收 baseline，不吸收 inject/recover 非目标行）。"""
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_inject_nontarget_train_fraction == 0.0
    assert cfg.fault_recover_nontarget_train_fraction == 0.0


def test_nontarget_fractions_override(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fault_inject_nontarget_train_fraction: 0.75\n"
        "fault_recover_nontarget_train_fraction: 0.4\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_inject_nontarget_train_fraction == 0.75
    assert cfg.fault_recover_nontarget_train_fraction == 0.4


def test_nontarget_fractions_out_of_range_raises(tmp_path):
    """越界一律在加载期报错，不看 expand_train_pool——理由同
    fault_baseline_train_fraction：负数会触发 Python 负索引切片导致方向反转的
    错误切分，大于 1 会让 train 吞下全部窗口且不报错，两者都是"看起来合理但
    错误的数据"，必须在入口堵住。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        for bad in (1.5, -0.1):
            cfg_path = tmp_path / f"c_{field}_{bad}.yaml"
            cfg_path.write_text(
                "contract_version: v1\n"
                "window_size_s: 15\n"
                f"{field}: {bad}\n"
                "modalities:\n"
                "  endpoint_red:\n"
                "    preprocessor: TracePreprocessor\n"
                "    preprocessor_version: v0\n"
                "    features: [trace_request_count]\n"
                "    normalization: per_endpoint_min_max\n"
            )
            with pytest.raises(ValueError, match=field):
                load_contract_config(cfg_path)


def test_nontarget_fractions_boundary_values_valid(tmp_path):
    """0.0 与 1.0 是合法边界值，不能被范围校验误拒。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        for boundary in (0.0, 1.0):
            cfg_path = tmp_path / f"c_{field}_{boundary}.yaml"
            cfg_path.write_text(
                "contract_version: v1\n"
                "window_size_s: 15\n"
                f"{field}: {boundary}\n"
                "modalities:\n"
                "  endpoint_red:\n"
                "    preprocessor: TracePreprocessor\n"
                "    preprocessor_version: v0\n"
                "    features: [trace_request_count]\n"
                "    normalization: per_endpoint_min_max\n"
            )
            cfg = load_contract_config(cfg_path)
            assert getattr(cfg, field) == boundary


def test_nontarget_fractions_quoted_string_raises(tmp_path):
    """YAML 里误加引号（字符串类型）必须报出指名字段的错误，而不是在比较运算处
    抛与本意无关的 TypeError——类型检查须先于范围检查。"""
    for field in (
        "fault_inject_nontarget_train_fraction",
        "fault_recover_nontarget_train_fraction",
    ):
        cfg_path = tmp_path / f"c_{field}_str.yaml"
        cfg_path.write_text(
            "contract_version: v1\n"
            "window_size_s: 15\n"
            f'{field}: "0.5"\n'
            "modalities:\n"
            "  endpoint_red:\n"
            "    preprocessor: TracePreprocessor\n"
            "    preprocessor_version: v0\n"
            "    features: [trace_request_count]\n"
            "    normalization: per_endpoint_min_max\n"
        )
        with pytest.raises(ValueError, match=field):
            load_contract_config(cfg_path)
```

- [ ] **Step 2: 跑测试确认失败**

```bash
conda run -n interface pytest tests/test_contract_config.py -k nontarget -q
```

Expected: FAIL，报 `AttributeError: 'ContractConfig' object has no attribute 'fault_inject_nontarget_train_fraction'`（或 `TypeError: __init__() got an unexpected keyword argument`）

- [ ] **Step 3: 加两个 dataclass 字段**

`src/contracts/contract_config.py`，紧跟现有 `fault_baseline_train_fraction: float = 0.2` 之后插入：

```python
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 inject
    # 阶段"非目标 endpoint"行按时间窗时序切分，最早 fraction 比例的窗口进训练池，
    # 其余留在 eval_all。判据是 is_endpoint_anomaly == False，不是 is_target_endpoint
    # ——前者天然兼容两种 label_granularity：case 级标签下 is_endpoint_anomaly
    # fallback 等于 is_anomaly，inject 阶段恒 True，该判据自动选不出任何行，
    # 不会把正样本误吸收进训练池。默认 0.0（不吸收，向后兼容本字段引入前的行为）。
    fault_inject_nontarget_train_fraction: float = 0.0
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 recover
    # 阶段"非目标 endpoint"行按时间窗时序切分。判据必须是 is_target_endpoint == False
    # 而非 is_endpoint_anomaly——is_anomaly 定义为 phase == "inject"，recover 阶段
    # 恒 False，导致 is_endpoint_anomaly 对 recover 阶段所有 endpoint（含目标）恒为
    # False，拿它筛"非目标"是空操作。且该判据只对 label_granularity == "endpoint"
    # 的 case 有精确含义：case 级标签的 case 没有 target_endpoint 字段、
    # is_target_endpoint 全填 False，无法区分目标/非目标，其 recover 行整段排除在
    # 本切分外、原样留在 eval_all（见 _write_v1）。默认 0.0（不吸收，向后兼容）。
    fault_recover_nontarget_train_fraction: float = 0.0
```

- [ ] **Step 4: 把范围校验抽成循环，覆盖三个 fraction 字段**

`__post_init__` 现有实现只校验 `fault_baseline_train_fraction` 一个字段。三个字段的校验逻辑与报错语义完全相同，DRY 化为循环，避免复制三份（同时保留原有报错文案里的关键信息，`pytest.raises(match=...)` 靠字段名匹配）：

```python
    def __post_init__(self) -> None:
        # 恒校验（不看 expand_train_pool）：越界/非数值比例是配置错误，即便当前未被
        # 消费也应在加载期暴露，而不是等到 expand_train_pool 被打开时才在
        # split_fault_phase_temporal 内部产出反直觉结果——大于 1.0 的正数会让 train
        # 吞下全部窗口而不报错，负数触发 Python 负索引切片导致方向反转的错误切分（都
        # 不是"崩溃"而是"看起来合理但错误的数据"）；YAML 里若误加引号（字符串类型）
        # 则会在比较运算处抛出与本意无关的 TypeError，因此类型检查须先于范围检查。
        for field_name in (
            "fault_baseline_train_fraction",
            "fault_inject_nontarget_train_fraction",
            "fault_recover_nontarget_train_fraction",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, (int, float)):
                raise ValueError(
                    f"{field_name} 必须是数值类型，"
                    f"实际类型 {type(value).__name__}"
                    f"（值={value!r}，检查 YAML 中是否误加了引号）"
                )
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} 必须落在 [0.0, 1.0]，实际 {value}")
```

注意 `isinstance(True, int)` 是 `True`，所以 YAML 里写 `true` 会被当成 1.0 通过校验——这与现有 `fault_baseline_train_fraction` 的行为一致，本次不改（不在本轮范围内）。

- [ ] **Step 5: loader 装配两个新字段**

`load_contract_config()` 的 `return ContractConfig(...)` 里追加：

```python
        fault_inject_nontarget_train_fraction=raw.get(
            "fault_inject_nontarget_train_fraction", 0.0
        ),
        fault_recover_nontarget_train_fraction=raw.get(
            "fault_recover_nontarget_train_fraction", 0.0
        ),
```

- [ ] **Step 6: 跑测试确认通过**

```bash
conda run -n interface pytest tests/test_contract_config.py -q
```

Expected: 全部 PASS（新增 5 个 + 既有全部；既有 `fault_baseline_train_fraction` 的 4 个测试必须仍通过，验证校验循环化没改变原字段行为）

- [ ] **Step 7: Commit**

```bash
git add src/contracts/contract_config.py tests/test_contract_config.py
git commit -m "[Feature]: ContractConfig 新增 inject/recover 非目标行吸收比例字段

新增 fault_inject_nontarget_train_fraction / fault_recover_nontarget_train_fraction
两个字段，默认 0.0（向后兼容）。__post_init__ 的范围+类型校验循环化，三个
fraction 字段共用同一套恒校验逻辑。本 commit 只加字段与校验，_write_v1 尚未消费。"
```

---

## Task 3: 新建 `nontarget_split_mini` fixture

**为什么需要新 fixture：** 现有 `split_fraction_mini` 的故障 case 只有 1 个 endpoint 且它就是目标 endpoint，`is_endpoint_anomaly` 在 inject 阶段恒 True、`is_target_endpoint` 恒 True——新逻辑的两个候选池都是空集，无法验证任何新行为。现有 `nan_propagation_mini`/`mini_data_root` 的故障 case 都没有 `target_endpoint` 字段（全是 case 级标签），同样验证不了 endpoint 级判据。

**新 fixture 设计：**
- `Normal`（6 个时间窗 × 2 个 endpoint = 12 行）：Normal 必须覆盖故障 case 用到的全部 endpoint，否则 `Normalizer`（`per_endpoint_min_max`）在 fit 集合里见不到该 endpoint 会跳过归一化并发告警，`EndpointBaselineStats` 也只 fit `train_fit`。这是 `test_e2e_reliability_gate_smoke.py` 当年换 fixture 的同一个坑。
- `Lv_E_NTGT_travel`（`label_granularity == "endpoint"`，含 `target_endpoint`）：15 个时间窗 × 2 endpoint = 30 行，5 窗 baseline + 5 窗 inject + 5 窗 recover。目标 endpoint = `POST:/api/v1/travelservice/trips/left`，非目标（fan-out）= `POST:/api/v1/travel2service/trips/left`。5 窗 × fraction 0.2 = 1 窗，能验证非零分数切分。
- `Lv_D_CASELVL_travel`（`label_granularity == "case"`，**无** `target_endpoint` 字段）：同样 15 窗 × 2 endpoint = 30 行。用来验证 recover 切分对 case 级标签是 no-op、inject 切分同样选不出行（正样本不被误吸收）。`anomaly_level` 故意写成 `"endpoint"`——CLAUDE.md 明确写着判定 `label_granularity` 的依据是 `target_endpoint` 字段是否存在、**不是** `anomaly_level`，这个 fixture 主动把两者设成矛盾值，钉死实现不会退回用 `anomaly_level` 判断。

两个 endpoint 都已在 `configs/contract/endpoint_to_service.yaml` 白名单里（`travelservice` → `ts-travel-service`，`travel2service` → `ts-travel2-service`），不需要改映射表。fixture 只提供 trace + api 两个模态（无 `metric_data`/`log_data`），缺失模态由 `_fill_missing_feature_cols` 填 NaN，与 `mini_data_root/Lv_E_HTTPABORT_assurance_mini` 一致。

**Files:**
- Create: `tests/fixtures/nontarget_split_mini/_generate.py`（fixture 生成脚本，随 fixture 一起提交保证可复现）
- Create: `tests/fixtures/nontarget_split_mini.yaml`（dataset config）
- Create（由生成脚本产出）: `tests/fixtures/nontarget_split_mini/{Normal,Lv_E_NTGT_travel,Lv_D_CASELVL_travel}/case_metadata.json` + `_pipeline_out/{tt_traces_red_15s.csv,tt_endpoint_health_15s.csv}`

- [ ] **Step 1: 写 dataset config**

`tests/fixtures/nontarget_split_mini.yaml`：

```yaml
name: nontarget_split_mini
roots:
  - tests/fixtures/nontarget_split_mini
normal_source: tests/fixtures/nontarget_split_mini
fused_window: 15s
```

- [ ] **Step 2: 写 fixture 生成脚本**

`tests/fixtures/nontarget_split_mini/_generate.py`。下划线前缀让 pytest 不收集它；它不是测试，是可复现地重造 fixture 数据的工具。

```python
"""生成 nontarget_split_mini fixture 数据。

验证 inject/recover 非目标 endpoint 行时序切分（见 docs/superpowers/specs/
2026-07-29-inject-recover-nontarget-split-design.md）。手写 70 行 CSV 易错且难
复核，改由本脚本确定性生成；脚本随 fixture 一起提交，改 fixture 时重跑它而不是
手改 CSV。

用法（repo 根目录）：conda run -n interface python tests/fixtures/nontarget_split_mini/_generate.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

_HERE = Path(__file__).parent

TARGET_EP = "POST:/api/v1/travelservice/trips/left"
NONTARGET_EP = "POST:/api/v1/travel2service/trips/left"

_TRACE_COLUMNS = [
    "case_id",
    "anomaly_type",
    "timestamp_window",
    "window_str",
    "endpoint_key",
    "method",
    "normalized_path",
    "trace_request_count",
    "trace_latency_mean",
    "trace_latency_p50",
    "trace_latency_p95",
    "trace_latency_p99",
    "trace_error_rate",
    "trace_5xx_rate",
    "trace_4xx_rate",
    "trace_status_coverage",
    "is_target_endpoint",
    "weak_is_anomaly",
    "label_confidence",
    "latency_anomaly_signal",
    "error_anomaly_signal",
    "5xx_anomaly_signal",
    "phase",
    "injection_start_ms",
    "injection_end_ms",
    "target_service",
]

_HEALTH_COLUMNS = [
    "case_id",
    "anomaly_type",
    "timestamp_window",
    "endpoint_key",
    "request_count",
    "error_rate",
    "latency_mean",
    "latency_p50",
    "latency_p95",
    "latency_p99",
    "status_2xx_rate",
    "status_4xx_rate",
    "status_5xx_rate",
    "method",
    "normalized_path",
]


def _iso(ts_ms: int) -> str:
    return pd.to_datetime(ts_ms, unit="ms", utc=True).strftime("%Y-%m-%dT%H:%M:%SZ")


def _split_ep(endpoint_key: str) -> tuple[str, str]:
    method, path = endpoint_key.split(":", 1)
    return method, path
```

- [ ] **Step 3: 续写生成脚本的行构造逻辑**

追加到 `_generate.py`：

```python
def _make_rows(
    case_id: str,
    anomaly_type: str,
    base_ts: int,
    n_windows: int,
    inject_window_idx: set[int],
    recover_window_idx: set[int],
    target_endpoint: str | None,
    target_service: str | None,
    inject_start_ms: int | None,
    inject_end_ms: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """产出 (trace_df, health_df)。每个时间窗都含 TARGET_EP 与 NONTARGET_EP 两行，
    制造 endpoint 级 fan-out——这是本 fixture 存在的理由（现有 fixture 每个故障
    case 只有单 endpoint，非目标候选池恒为空，验证不了新切分）。

    特征值随窗序号 w 与 endpoint 变化，避免 per_endpoint_min_max 归一化在 fit
    集合上零方差退化（退化会触发跳过归一化 + 告警，干扰测试信噪比）。
    inject 窗口内只有目标 endpoint 的特征异常抬高，非目标 endpoint 维持正常量级
    ——非目标 endpoint 在 inject 阶段本来就没有故障，它就是我们要吸收进训练池的
    "正常"数据。
    """
    trace_rows, health_rows = [], []
    for w in range(n_windows):
        ts = base_ts + w * 15_000
        if w in inject_window_idx:
            phase = "inject"
        elif w in recover_window_idx:
            phase = "recover"
        elif inject_start_ms is None:
            phase = "normal"
        else:
            phase = "baseline"

        for ep in (TARGET_EP, NONTARGET_EP):
            method, path = _split_ep(ep)
            is_target = target_endpoint is not None and ep == target_endpoint
            faulty = phase == "inject" and is_target

            ep_offset = 0 if ep == TARGET_EP else 3
            req = 8 + w + ep_offset
            lat_p50 = 20_000.0 + w * 500 + ep_offset * 100
            lat_p95 = lat_p50 * 2
            err = 0.0
            if faulty:
                req += 4
                lat_p50 *= 2.6
                lat_p95 *= 2.1
                err = 0.4

            trace_rows.append(
                {
                    "case_id": case_id,
                    "anomaly_type": anomaly_type,
                    "timestamp_window": ts,
                    "window_str": _iso(ts),
                    "endpoint_key": ep,
                    "method": method,
                    "normalized_path": path,
                    "trace_request_count": req,
                    "trace_latency_mean": lat_p50 * 1.05,
                    "trace_latency_p50": lat_p50,
                    "trace_latency_p95": lat_p95,
                    "trace_latency_p99": lat_p95,
                    "trace_error_rate": err,
                    "trace_5xx_rate": err,
                    "trace_4xx_rate": 0.0,
                    "trace_status_coverage": 1.0,
                    "is_target_endpoint": is_target,
                    "weak_is_anomaly": 1 if faulty else 0,
                    "label_confidence": "strong" if faulty else "normal",
                    "latency_anomaly_signal": 1 if faulty else 0,
                    "error_anomaly_signal": 1 if faulty else 0,
                    "5xx_anomaly_signal": 1 if faulty else 0,
                    "phase": phase,
                    "injection_start_ms": inject_start_ms,
                    "injection_end_ms": inject_end_ms,
                    "target_service": target_service,
                }
            )
            # client 侧（health）比 trace 侧略高，让 latency_divergence 非零非常数
            health_rows.append(
                {
                    "case_id": case_id,
                    "anomaly_type": anomaly_type,
                    "timestamp_window": _iso(ts),
                    "endpoint_key": ep,
                    "request_count": req,
                    "error_rate": err,
                    "latency_mean": lat_p50 * 1.10,
                    "latency_p50": lat_p50 * 1.02,
                    "latency_p95": lat_p95 + 1_500 + w * 50,
                    "latency_p99": lat_p95 + 2_000 + w * 50,
                    "status_2xx_rate": 1.0 - err,
                    "status_4xx_rate": 0.0,
                    "status_5xx_rate": err,
                    "method": method,
                    "normalized_path": path,
                }
            )

    return (
        pd.DataFrame(trace_rows, columns=_TRACE_COLUMNS),
        pd.DataFrame(health_rows, columns=_HEALTH_COLUMNS),
    )
```

- [ ] **Step 4: 续写生成脚本的三个 case 定义与落盘**

追加到 `_generate.py`：

```python
def _write_case(case_dir: Path, meta: dict, trace_df: pd.DataFrame, health_df: pd.DataFrame) -> None:
    out = case_dir / "_pipeline_out"
    out.mkdir(parents=True, exist_ok=True)
    (case_dir / "case_metadata.json").write_text(json.dumps(meta, indent=2) + "\n")
    trace_df.to_csv(out / "tt_traces_red_15s.csv", index=False)
    health_df.to_csv(out / "tt_endpoint_health_15s.csv", index=False)


def main() -> None:
    # Normal：6 窗 × 2 endpoint = 12 行。split_normal_rows_temporal 默认
    # fit_frac=0.6 / val_frac=0.2 → int(6*0.6)=3 窗 train_fit（6 行）、
    # int(6*0.2)=1 窗 train_val（2 行）、剩 2 窗 eval_normal_holdout（4 行）。
    # 必须覆盖故障 case 用到的两个 endpoint，否则 Normalizer/EndpointBaselineStats
    # 的 fit 集合缺 endpoint。
    normal_trace, normal_health = _make_rows(
        case_id="Normal",
        anomaly_type="Normal",
        base_ts=1_796_000_000_000,
        n_windows=6,
        inject_window_idx=set(),
        recover_window_idx=set(),
        target_endpoint=None,
        target_service=None,
        inject_start_ms=None,
        inject_end_ms=None,
    )
    # Normal case 的 trace CSV 不应有 is_target_endpoint 列（真实 Normal 数据没有
    # 注入目标），与 split_fraction_mini/Normal 的既有形态一致
    normal_trace = normal_trace.drop(columns=["is_target_endpoint"])
    _write_case(
        _HERE / "Normal",
        {
            "case_id": "Normal",
            "anomaly_type": "Normal",
            "anomaly_level": "none",
            "target_service": None,
            "inject_start_ms": None,
            "inject_end_ms": None,
        },
        normal_trace,
        normal_health,
    )

    # Lv_E_NTGT_travel：endpoint 级精确标签（含 target_endpoint）。
    # 15 窗 = 窗 0-4 baseline / 窗 5-9 inject / 窗 10-14 recover。
    # 每阶段 5 窗，fraction=0.2 → int(5*0.2)=1 窗进 train，能验证非零分数切分。
    ntgt_base = 1_796_100_000_000
    ntgt_trace, ntgt_health = _make_rows(
        case_id="Lv_E_NTGT_travel",
        anomaly_type="Lv_E_NTGT_travel",
        base_ts=ntgt_base,
        n_windows=15,
        inject_window_idx=set(range(5, 10)),
        recover_window_idx=set(range(10, 15)),
        target_endpoint=TARGET_EP,
        target_service="ts-travel-service",
        inject_start_ms=ntgt_base + 5 * 15_000,
        inject_end_ms=ntgt_base + 9 * 15_000,
    )
    _write_case(
        _HERE / "Lv_E_NTGT_travel",
        {
            "case_id": "Lv_E_NTGT_travel",
            "anomaly_type": "Lv_E_NTGT_travel",
            "anomaly_level": "endpoint",
            "target_service": "ts-travel-service",
            "target_endpoint": TARGET_EP,
            "inject_start_ms": ntgt_base + 5 * 15_000,
            "inject_end_ms": ntgt_base + 9 * 15_000,
        },
        ntgt_trace,
        ntgt_health,
    )

    # Lv_D_CASELVL_travel：case 级近似标签（**无** target_endpoint 字段）。
    # anomaly_level 故意写成 "endpoint" 而与缺失的 target_endpoint 矛盾——
    # CLAUDE.md 明确 label_granularity 的判据是 target_endpoint 是否存在、不是
    # anomaly_level，这个 fixture 主动钉死实现不会退回用 anomaly_level 判断。
    case_base = 1_796_200_000_000
    case_trace, case_health = _make_rows(
        case_id="Lv_D_CASELVL_travel",
        anomaly_type="Lv_D_CASELVL_travel",
        base_ts=case_base,
        n_windows=15,
        inject_window_idx=set(range(5, 10)),
        recover_window_idx=set(range(10, 15)),
        target_endpoint=None,
        target_service="ts-travel-service",
        inject_start_ms=case_base + 5 * 15_000,
        inject_end_ms=case_base + 9 * 15_000,
    )
    # case 级标签的真实数据（如 anomod_v1）trace CSV 里没有 is_target_endpoint 列，
    # TracePreprocessor 会填 False。这里删掉该列以复刻真实形态。
    case_trace = case_trace.drop(columns=["is_target_endpoint"])
    _write_case(
        _HERE / "Lv_D_CASELVL_travel",
        {
            "case_id": "Lv_D_CASELVL_travel",
            "anomaly_type": "Lv_D_CASELVL_travel",
            "anomaly_level": "endpoint",
            "target_service": "ts-travel-service",
            "inject_start_ms": case_base + 5 * 15_000,
            "inject_end_ms": case_base + 9 * 15_000,
        },
        case_trace,
        case_health,
    )


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: 跑生成脚本**

```bash
conda run -n interface python tests/fixtures/nontarget_split_mini/_generate.py
```

Expected: 无输出、无报错

- [ ] **Step 6: 核对产出的文件与行数**

```bash
find tests/fixtures/nontarget_split_mini -type f | sort
conda run -n interface python -c "
import pandas as pd
for c,n in [('Normal',12),('Lv_E_NTGT_travel',30),('Lv_D_CASELVL_travel',30)]:
    t=pd.read_csv(f'tests/fixtures/nontarget_split_mini/{c}/_pipeline_out/tt_traces_red_15s.csv')
    h=pd.read_csv(f'tests/fixtures/nontarget_split_mini/{c}/_pipeline_out/tt_endpoint_health_15s.csv')
    assert len(t)==n and len(h)==n, (c,len(t),len(h),n)
    assert t['endpoint_key'].nunique()==2, (c,'endpoint 数应为 2')
    print(c,'trace',len(t),'health',len(h),'phases',sorted(t['phase'].unique()),'has_is_target', 'is_target_endpoint' in t.columns)
"
```

Expected：
```
Normal trace 12 health 12 phases ['normal'] has_is_target False
Lv_E_NTGT_travel trace 30 health 30 phases ['baseline', 'inject', 'recover'] has_is_target True
Lv_D_CASELVL_travel trace 30 health 30 phases ['baseline', 'inject', 'recover'] has_is_target False
```

- [ ] **Step 7: 跑一次 build_contract 确认 fixture 能被 pipeline 正常消费**

用当前 `v1_expanded_pool.yaml`（此时两个新 fraction 还未在 config 里声明，取默认 0.0，即新逻辑尚未生效；Task 4 才接线，这一步只验证 fixture 本身没坏）：

```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v1_expanded_pool.yaml \
  --dataset tests/fixtures/nontarget_split_mini.yaml \
  --out-dir /tmp/contract_ntgt_check --seed 42
conda run -n interface python -c "
import pandas as pd
ev=pd.read_parquet('/tmp/contract_ntgt_check/eval_all.parquet')
print('eval_all', len(ev))
print(ev.groupby(['case_id','phase','label_granularity']).size())
print('is_endpoint_anomaly 计数'); print(ev['is_endpoint_anomaly'].value_counts())
"
```

Expected: 命令成功退出；`Lv_E_NTGT_travel` 的 `label_granularity` 为 `endpoint`、`Lv_D_CASELVL_travel` 为 `case`；不出现 `ContractV0Error`（说明两个 endpoint 都在白名单映射里）。

- [ ] **Step 8: 清理临时产物**

```bash
rm -rf /tmp/contract_ntgt_check
```

- [ ] **Step 9: Commit**

```bash
git add tests/fixtures/nontarget_split_mini tests/fixtures/nontarget_split_mini.yaml
git commit -m "[Test]: 新增 nontarget_split_mini fixture

含 2 个 endpoint 制造 endpoint 级 fan-out 的故障 case，覆盖两种 label_granularity：
Lv_E_NTGT_travel（有 target_endpoint，endpoint 级精确标签）与 Lv_D_CASELVL_travel
（无 target_endpoint，case 级近似标签，且 anomaly_level 故意与之矛盾以钉死判据
不是 anomaly_level）。现有 fixture 的故障 case 均为单 endpoint 或纯 case 级标签，
非目标 endpoint 候选池恒为空，无法验证 inject/recover 非目标行切分。
CSV 由 _generate.py 确定性生成，随 fixture 提交保证可复现。"
```

---

## Task 4: `_write_v1` 接线两路新切分

**Files:**
- Modify: `scripts/build_contract.py:471-480`（调用处签名）、`scripts/build_contract.py:491-596`（`_write_v1` 本体）

**签名改动的理由（必须做，不是可选的清洁）：** `_write_v1` 现有签名已有 7 个位置参数，其中 `fault_baseline_train_fraction` 是 float。再追加两个同类型 float 参数后会出现**三个连续的 `float` 位置参数**，调用处任意两个写反都不报错、只是静默产出错误切分——正是本仓库反复警惕的"看起来合理但错误的数据"。因此把这三个 config 派生参数收拢为直接传 `cfg: ContractConfig`。全仓只有一处调用 `_write_v1`（`scripts/build_contract.py:472`），改动面很小。

- [ ] **Step 1: 改 `_write_v1` 签名与调用处**

调用处（原 471-480 行）改为：

```python
    if cfg.contract_version == "v1":
        _write_v1(out, full, normal_mask, args.seed, cfg)
```

`_write_v1` 签名改为：

```python
def _write_v1(
    out: Path,
    full: pd.DataFrame,
    normal_mask: pd.Series,
    seed: int,
    cfg: ContractConfig,
) -> None:
```

函数体内原先直接使用的裸参数名，改为从 `cfg` 取值。函数体开头加一行局部别名，避免正文里满屏 `cfg.`：

```python
    expand_train_pool = cfg.expand_train_pool
    fault_baseline_train_fraction = cfg.fault_baseline_train_fraction
    fault_inject_nontarget_train_fraction = cfg.fault_inject_nontarget_train_fraction
    fault_recover_nontarget_train_fraction = cfg.fault_recover_nontarget_train_fraction
    fit_endpoint_baseline_stats = cfg.fit_endpoint_baseline_stats
```

`ContractConfig` 已在文件顶部 import（`from src.contracts.contract_config import ContractConfig, load_contract_config` 一类），若没有则补上 type-hint 需要的那一个。

- [ ] **Step 2: 扩写 `_write_v1` 的 docstring**

在现有 docstring 里"（不吸收 recover——系统未稳定回正常态，分布未验证，保守排除）"这句之后，追加说明两路新吸收：

```
    除 baseline 阶段行外，`fault_inject_nontarget_train_fraction` /
    `fault_recover_nontarget_train_fraction` 两个独立比例分别控制"inject 阶段
    非目标 endpoint 行"与"recover 阶段非目标 endpoint 行"的吸收（默认均 0.0，
    即不吸收，与本字段引入前行为一致）。动机：eval_all 的负样本地板里，inject
    窗口内非目标 endpoint 的 fan-out 残留是最大一项，`fault_baseline_train_fraction`
    推到 1.0 也压不动它，正负比在 new_ep1 上封顶 12.24:87.76（见
    history/entries/019）。

    两路的"非目标"判据**不同**，不能互换：
    - inject 侧用 `is_endpoint_anomaly == False`。该列已统一处理两种
      label_granularity——case 级标签下它 fallback 等于 is_anomaly，inject 阶段恒
      True，故该判据在 case 级 case 上自动选不出任何行，不会把正样本误吸收进
      训练池，也因此不需要额外按 label_granularity 分支。
    - recover 侧必须用 `is_target_endpoint == False`，且只对
      `label_granularity == "endpoint"` 的 case 生效。原因：is_anomaly 定义为
      phase == "inject"，recover 阶段恒 False，导致 is_endpoint_anomaly 对 recover
      阶段所有 endpoint（含目标）恒为 False，拿它筛"非目标"是空操作；而
      is_target_endpoint 只在有 target_endpoint 字段的 case 上有精确含义，case 级
      标签的 case 该列全为 False、无法区分目标/非目标，其 recover 行整段排除在本
      切分外、原样留在 eval_all（否则会把"实际就是故障发生地"的 endpoint 的
      recover 行也吸收进训练池，污染训练池对"正常"的定义）。
    目标 endpoint 自身的 recover 行始终不被吸收，沿用上述保守排除理由。
```

- [ ] **Step 3: 在 `expand_train_pool` 分支里追加两路切分**

紧接现有 `fault_baseline_train`/`fault_baseline_eval` 的告警块之后、`train_pool = pd.concat([...])` 之前插入：

```python
        # inject 阶段非目标 endpoint 行（fan-out 残留）。判据 is_endpoint_anomaly==False
        # 兼容两种 label_granularity，见 docstring；case 级标签的 case 在此自动选不出行。
        fault_inject_nontarget_df = anomaly_df[
            (anomaly_df["phase"] == "inject") & (~anomaly_df["is_endpoint_anomaly"])
        ].copy()
        (
            fault_inject_nontarget_train,
            fault_inject_nontarget_eval,
        ) = split_fault_phase_temporal(
            fault_inject_nontarget_df, fraction=fault_inject_nontarget_train_fraction
        )
        fault_inject_nontarget_train["source_phase"] = "fault_inject_nontarget"
        if (
            fault_inject_nontarget_train_fraction > 0
            and len(fault_inject_nontarget_train) == 0
            and len(fault_inject_nontarget_df) > 0
        ):
            LOG.warning(
                "fault_inject_nontarget_train_fraction=%.3f 对全部 %d 行 inject 非目标"
                "endpoint 行贡献了 0 行进训练池——请检查 fraction 是否过小或各 case 的"
                "inject 窗口数是否过少（int(n*fraction) 自然向下取整，不做至少 1 窗兜底）",
                fault_inject_nontarget_train_fraction,
                len(fault_inject_nontarget_df),
            )

        # recover 阶段非目标 endpoint 行。判据必须是 is_target_endpoint==False 且先按
        # label_granularity=="endpoint" 过滤，见 docstring。
        fault_recover_nontarget_df = anomaly_df[
            (anomaly_df["phase"] == "recover")
            & (anomaly_df["label_granularity"] == "endpoint")
            & (~anomaly_df["is_target_endpoint"])
        ].copy()
        (
            fault_recover_nontarget_train,
            fault_recover_nontarget_eval,
        ) = split_fault_phase_temporal(
            fault_recover_nontarget_df, fraction=fault_recover_nontarget_train_fraction
        )
        fault_recover_nontarget_train["source_phase"] = "fault_recover_nontarget"
        if (
            fault_recover_nontarget_train_fraction > 0
            and len(fault_recover_nontarget_train) == 0
            and len(fault_recover_nontarget_df) > 0
        ):
            LOG.warning(
                "fault_recover_nontarget_train_fraction=%.3f 对全部 %d 行 recover 非目标"
                "endpoint 行贡献了 0 行进训练池——请检查 fraction 是否过小或各 case 的"
                "recover 窗口数是否过少",
                fault_recover_nontarget_train_fraction,
                len(fault_recover_nontarget_df),
            )
```

- [ ] **Step 4: 把两路 train 拼进 `train_pool`**

原 `train_pool = pd.concat([train_fit_labeled, fault_baseline_train], ignore_index=True)` 改为：

```python
        train_pool = pd.concat(
            [
                train_fit_labeled,
                fault_baseline_train,
                fault_inject_nontarget_train,
                fault_recover_nontarget_train,
            ],
            ignore_index=True,
        )
```

- [ ] **Step 5: 改写 `eval_all` 的构成**

原来的 `eval_all` 用 `anomaly_df[anomaly_df["phase"] != "baseline"]` 一把捞走全部 inject/recover 行，现在这两个阶段各有一部分被吸收，必须按"未被吸收的补集"精确拼接。原 561-568 行替换为：

```python
        # eval_all 五段拼接。inject/recover 两个阶段各自被拆成"被吸收进 train 的部分"
        # 与"留在 eval 的部分"，这里只收后者；写成显式五段而不是 phase != baseline 的
        # 粗筛，是为了让每一段的归属理由可读、可断言。
        # ① inject 阶段正样本（is_endpoint_anomaly==True）——永不被吸收，是评估的
        #    唯一正样本来源。
        eval_inject_positive = anomaly_df[
            (anomaly_df["phase"] == "inject") & anomaly_df["is_endpoint_anomaly"]
        ]
        # ② recover 阶段未参与新切分的行：case 级标签 case 的全部 recover 行（判据
        #    无法区分目标/非目标，整段保留）+ endpoint 级标签 case 里的目标 endpoint
        #    recover 行（保守排除，不吸收）。
        recover_mask = anomaly_df["phase"] == "recover"
        eval_recover_untouched = anomaly_df[
            recover_mask
            & (
                (anomaly_df["label_granularity"] != "endpoint")
                | anomaly_df["is_target_endpoint"]
            )
        ]
        eval_all = pd.concat(
            [
                eval_inject_positive,
                fault_inject_nontarget_eval,
                eval_recover_untouched,
                fault_recover_nontarget_eval,
                fault_baseline_eval,
                parts["eval_normal_holdout"],
            ],
            ignore_index=True,
        )
        eval_all.to_parquet(out / "eval_all.parquet", index=False)
```

- [ ] **Step 6: 扩写该分支末尾的 `LOG.info`**

原 570-582 行的 `LOG.info` 改为（追加四个新计数）：

```python
        LOG.info(
            "v1 切分（训练池已扩容，baseline_fraction=%.3f inject_nontarget_fraction=%.3f "
            "recover_nontarget_fraction=%.3f）：train_fit=%d holdout=%d "
            "fault_baseline_total=%d fault_baseline_to_train=%d fault_baseline_to_eval=%d "
            "inject_nontarget_total=%d inject_nontarget_to_train=%d inject_nontarget_to_eval=%d "
            "recover_nontarget_total=%d recover_nontarget_to_train=%d "
            "recover_nontarget_to_eval=%d train_pool=%d eval_all=%d",
            fault_baseline_train_fraction,
            fault_inject_nontarget_train_fraction,
            fault_recover_nontarget_train_fraction,
            len(parts["train_fit"]),
            len(parts["eval_normal_holdout"]),
            len(fault_baseline_df),
            len(fault_baseline_train),
            len(fault_baseline_eval),
            len(fault_inject_nontarget_df),
            len(fault_inject_nontarget_train),
            len(fault_inject_nontarget_eval),
            len(fault_recover_nontarget_df),
            len(fault_recover_nontarget_train),
            len(fault_recover_nontarget_eval),
            len(train_pool),
            len(eval_all),
        )
```

- [ ] **Step 7: 手工验证"无行丢失"——新旧 eval_all 在 fraction=0 时必须逐行等同**

这是本 task 最容易出错的地方（五段拼接漏一段就静默丢负样本）。两个新 fraction 取默认 0.0 时，新实现的 `eval_all` 必须与改动前完全一致：

```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v1_expanded_pool.yaml \
  --dataset tests/fixtures/nontarget_split_mini.yaml \
  --out-dir /tmp/ntgt_frac0 --seed 42
conda run -n interface python -c "
import pandas as pd
tr=pd.read_parquet('/tmp/ntgt_frac0/train.parquet')
ev=pd.read_parquet('/tmp/ntgt_frac0/eval_all.parquet')
# fraction 默认 0.0 → 两路新吸收各 0 行，train 只含 normal_case + fault_baseline
print('train source_phase:'); print(tr['source_phase'].value_counts())
assert set(tr['source_phase'].unique()) <= {'normal_case','fault_baseline'}, tr['source_phase'].unique()
# eval_all 应含全部 inject(20 行) + 全部 recover(20 行) + 未吸收 baseline + holdout
print('eval phase x case:'); print(ev.groupby(['phase','case_id']).size())
assert (ev['phase']=='inject').sum()==20, (ev['phase']=='inject').sum()
assert (ev['phase']=='recover').sum()==20, (ev['phase']=='recover').sum()
assert set(tr['sample_id']) & set(ev['sample_id'])==set()
print('OK')
"
```

Expected: 打印 `OK`。`inject` 20 行 = 2 个故障 case × 5 窗 × 2 endpoint；`recover` 同理 20 行。若 inject/recover 计数少于 20，说明五段拼接漏段。

- [ ] **Step 8: 手工验证 fraction=0.2 时两路吸收生效且不吃正样本**

```bash
conda run -n interface python -c "
import yaml, pathlib
raw = yaml.safe_load(pathlib.Path('configs/contract/v1_expanded_pool.yaml').read_text())
raw['fault_inject_nontarget_train_fraction'] = 0.2
raw['fault_recover_nontarget_train_fraction'] = 0.2
pathlib.Path('/tmp/ntgt_cfg_02.yaml').write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
"
conda run -n interface python scripts/build_contract.py \
  --config /tmp/ntgt_cfg_02.yaml \
  --dataset tests/fixtures/nontarget_split_mini.yaml \
  --out-dir /tmp/ntgt_frac02 --seed 42
conda run -n interface python -c "
import pandas as pd
tr=pd.read_parquet('/tmp/ntgt_frac02/train.parquet')
ev=pd.read_parquet('/tmp/ntgt_frac02/eval_all.parquet')
print(tr['source_phase'].value_counts())
# inject 非目标候选池：只有 endpoint 级 case（Lv_E_NTGT_travel）的非目标 endpoint，
# 5 窗 × 1 endpoint = 5 行；int(5*0.2)=1 窗 → 1 行进 train
assert (tr['source_phase']=='fault_inject_nontarget').sum()==1, (tr['source_phase']=='fault_inject_nontarget').sum()
# recover 非目标候选池同理 5 行 → 1 行进 train
assert (tr['source_phase']=='fault_recover_nontarget').sum()==1, (tr['source_phase']=='fault_recover_nontarget').sum()
# 关键：训练池里绝不能有正样本
assert not tr['is_endpoint_anomaly'].any(), '训练池吸收了正样本！'
# 关键：case 级标签 case 一行 recover 都不能被吸收
absorbed = tr[tr['source_phase'].isin(['fault_inject_nontarget','fault_recover_nontarget'])]
assert (absorbed['case_id']=='Lv_E_NTGT_travel').all(), absorbed['case_id'].unique()
assert set(tr['sample_id']) & set(ev['sample_id'])==set()
print('OK: eval_all', len(ev), 'train', len(tr))
"
```

Expected: 打印 `OK: eval_all 58 train 12` 之外的具体数字以实际为准，但四条 assert 必须全过。

- [ ] **Step 9: 清理临时产物**

```bash
rm -rf /tmp/ntgt_frac0 /tmp/ntgt_frac02 /tmp/ntgt_cfg_02.yaml
```

- [ ] **Step 10: 跑既有相关测试确认零回归**

两个新 fraction 此时仍是默认 0.0（Task 6 才在 config 里设 1.0），既有测试行为不应有任何变化：

```bash
conda run -n interface pytest tests/test_contract_v1_train_pool.py tests/test_contract_v1_split.py tests/test_build_contract_v1_endpoint_id.py tests/test_split_fault_phase.py -q
```

Expected: 全部 PASS

- [ ] **Step 11: Commit**

```bash
git add scripts/build_contract.py
git commit -m "[Feature]: _write_v1 接线 inject/recover 非目标 endpoint 行时序切分

expand_train_pool=true 时，除 baseline 阶段行外，额外按两个独立 fraction 吸收
inject 阶段非目标 endpoint 行（判据 is_endpoint_anomaly==False）与 recover 阶段
非目标 endpoint 行（判据 is_target_endpoint==False 且仅 label_granularity==
endpoint 的 case 生效）。eval_all 从'phase != baseline 粗筛'改为五段显式拼接，
精确收纳两阶段未被吸收的补集。_write_v1 签名收拢为传 cfg，避免三个连续 float
位置参数写反不报错。两个新 fraction 默认 0.0，本 commit 不改变任何既有产物。"
```

---

## Task 5: 集成测试锁住新切分行为

**Files:**
- Modify: `tests/test_contract_v1_train_pool.py`

- [ ] **Step 1: 加一个能覆写 fraction 的 config 辅助函数**

现有 `_run_v1_build` 只能传 config 路径。新测试要在同一份 fixture 上试不同 fraction，需要动态产出 config。**不能**用"读原文件 + 字符串追加"的方式（`v1_expanded_pool.yaml` 在 Task 6 之后已含这两个字段，追加会产生重复 key，PyYAML 静默取最后一个，看似能用但脆弱），改为 YAML 往返覆写。追加到 `tests/test_contract_v1_train_pool.py` 顶部（`_run_v1_build` 之后）：

```python
import yaml


def _cfg_with_fractions(tmp_path: Path, name: str, **overrides: float) -> str:
    """基于 v1_expanded_pool.yaml 产出一份覆写了指定 fraction 的临时 config。

    用 YAML 往返（load → 改 dict → dump）而不是字符串追加：v1_expanded_pool.yaml
    本身已声明这些字段，追加会产生重复 key——PyYAML 静默取最后一个，测试看似能过，
    但一旦有人调整字段顺序或加注释就会悄悄读到另一个值。
    """
    raw = yaml.safe_load((REPO_ROOT / "configs/contract/v1_expanded_pool.yaml").read_text())
    raw.update(overrides)
    cfg_path = tmp_path / f"{name}.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))
    return str(cfg_path)
```

- [ ] **Step 2: 写失败的测试（新切分核心行为）**

追加到 `tests/test_contract_v1_train_pool.py` 末尾：

```python
_NTGT_DATASET = "tests/fixtures/nontarget_split_mini.yaml"


def test_inject_nontarget_rows_partially_absorbed_into_train(tmp_path):
    """inject 阶段非目标 endpoint 行按 fraction 部分吸收进训练池。

    nontarget_split_mini 的 Lv_E_NTGT_travel 有 5 个 inject 窗口 × 1 个非目标
    endpoint = 5 行候选，int(5*0.2)=1 窗 → 1 行进 train、4 行留 eval。
    Lv_D_CASELVL_travel（case 级标签）在 inject 阶段所有行 is_endpoint_anomaly
    都 fallback 为 True，不进候选池，故总候选恒为 5 行而非 10 行。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "inject_02", fault_inject_nontarget_train_fraction=0.2
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    absorbed = train[train["source_phase"] == "fault_inject_nontarget"]
    assert len(absorbed) == 1, f"应吸收最早 1 窗 × 1 endpoint = 1 行，实际 {len(absorbed)}"
    assert (absorbed["phase"] == "inject").all()
    assert (absorbed["case_id"] == "Lv_E_NTGT_travel").all()
    assert not absorbed["is_endpoint_anomaly"].any()

    eval_inject_nontarget = eval_all[
        (eval_all["phase"] == "inject") & (~eval_all["is_endpoint_anomaly"])
    ]
    # 剩 4 行来自 Lv_E_NTGT_travel，另 0 行来自 case 级 case（其 inject 行全是正样本）
    assert len(eval_inject_nontarget) == 4, len(eval_inject_nontarget)


def test_inject_target_rows_never_absorbed_regardless_of_fraction(tmp_path):
    """正样本（inject 阶段目标 endpoint 行）不管 fraction 多高都不进训练池。

    这是 is_endpoint_anomaly 判据的核心保护：若实现误用 is_target_endpoint==False
    作为 inject 侧判据，case 级标签 case（is_target_endpoint 全 False）的全部
    inject 行都会被当成"非目标"吸收——而它们的 is_endpoint_anomaly 是 True，
    就是正样本。fraction=1.0 让这个 bug 必然暴露。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "inject_10", fault_inject_nontarget_train_fraction=1.0
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert not train["is_endpoint_anomaly"].any(), "训练池吸收了正样本"
    # 两个故障 case 各 5 窗 inject：endpoint 级 case 贡献 5 个正样本（目标 endpoint），
    # case 级 case 的 10 行 inject 全是正样本（fallback），合计 15 行必须全留 eval
    positives = eval_all[eval_all["is_endpoint_anomaly"]]
    assert len(positives) == 15, f"正样本应恒为 15 行，实际 {len(positives)}"


def test_recover_nontarget_rows_partially_absorbed_into_train(tmp_path):
    """recover 阶段非目标 endpoint 行按独立 fraction 部分吸收进训练池。

    只有 endpoint 级标签 case 的非目标 endpoint 参与：5 窗 × 1 endpoint = 5 行
    候选，int(5*0.2)=1 窗 → 1 行进 train。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "recover_02", fault_recover_nontarget_train_fraction=0.2
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    absorbed = train[train["source_phase"] == "fault_recover_nontarget"]
    assert len(absorbed) == 1, f"应吸收最早 1 窗 × 1 endpoint = 1 行，实际 {len(absorbed)}"
    assert (absorbed["phase"] == "recover").all()
    assert (absorbed["case_id"] == "Lv_E_NTGT_travel").all()
    assert not absorbed["is_target_endpoint"].any()


def test_recover_target_endpoint_rows_never_absorbed(tmp_path):
    """endpoint 级标签 case 的目标 endpoint recover 行永不被吸收——沿用 entry 014
    的保守排除理由（系统未稳定回正常态，分布未验证）。fraction=1.0 时必须仍成立。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "recover_10", fault_recover_nontarget_train_fraction=1.0
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    recover_in_train = train[train["phase"] == "recover"]
    assert not recover_in_train["is_target_endpoint"].any(), "目标 endpoint 的 recover 行被吸收了"
    # Lv_E_NTGT_travel 目标 endpoint 的 5 个 recover 窗必须全留 eval
    target_recover_eval = eval_all[
        (eval_all["phase"] == "recover")
        & (eval_all["case_id"] == "Lv_E_NTGT_travel")
        & eval_all["is_target_endpoint"]
    ]
    assert len(target_recover_eval) == 5, len(target_recover_eval)


def test_case_level_label_recover_rows_never_absorbed(tmp_path):
    """label_granularity=='case' 的 case，其 recover 行不管 fraction 多高都不被吸收。

    这是本轮最容易被静默破坏的行为：case 级标签 case 没有 target_endpoint 字段、
    is_target_endpoint 全为 False，若实现漏掉 label_granularity=='endpoint' 这层
    过滤，这批 recover 行会全部被当成"非目标"吸收进训练池——其中可能包含实际就是
    故障发生地的 endpoint，污染训练池对"正常"的定义。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path, "recover_case_10", fault_recover_nontarget_train_fraction=1.0
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert not (
        (train["case_id"] == "Lv_D_CASELVL_travel") & (train["phase"] == "recover")
    ).any(), "case 级标签 case 的 recover 行被吸收了"
    # 该 case 的 10 行 recover（5 窗 × 2 endpoint）必须整段留在 eval_all
    case_recover_eval = eval_all[
        (eval_all["case_id"] == "Lv_D_CASELVL_travel") & (eval_all["phase"] == "recover")
    ]
    assert len(case_recover_eval) == 10, len(case_recover_eval)


def test_label_granularity_derives_from_target_endpoint_not_anomaly_level(tmp_path):
    """label_granularity 的判据是 target_endpoint 字段是否存在，不是 anomaly_level。

    fixture 里 Lv_D_CASELVL_travel 的 anomaly_level 故意写成 'endpoint' 却没有
    target_endpoint 字段。若实现（现在或将来）改用 anomaly_level 判断，这个 case
    会被误判为 endpoint 级精确标签，其 recover 行会被错误纳入吸收候选。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config="configs/contract/v1_expanded_pool.yaml", dataset=_NTGT_DATASET)

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case_rows = eval_all[eval_all["case_id"] == "Lv_D_CASELVL_travel"]
    assert (case_rows["anomaly_level"] == "endpoint").all(), "fixture 前提变了"
    assert (case_rows["label_granularity"] == "case").all()

    ntgt_rows = eval_all[eval_all["case_id"] == "Lv_E_NTGT_travel"]
    assert (ntgt_rows["label_granularity"] == "endpoint").all()


def test_all_three_fractions_together_keep_sample_id_disjoint(tmp_path):
    """三路吸收（baseline + inject 非目标 + recover 非目标）全部推到 1.0 时，
    train 与 eval_all 的 sample_id 仍必须互斥——这是 Contract v1 存在的理由，
    也是本轮唯一的硬约束。整窗切分天然保证它，本测试防止未来重构破坏。
    """
    out_dir = tmp_path / "contract_v1"
    cfg = _cfg_with_fractions(
        tmp_path,
        "all_10",
        fault_baseline_train_fraction=1.0,
        fault_inject_nontarget_train_fraction=1.0,
        fault_recover_nontarget_train_fraction=1.0,
    )
    _run_v1_build(out_dir, config=cfg, dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()
    # 无行丢失：三路全吸收后，两侧行数之和必须等于 fixture 总行数（12 Normal +
    # 30 + 30 = 72）减去 train_val（Normal 三路切分里既不进 train 也不进 eval_all
    # 的那一份，6 窗 × int(6*0.2)=1 窗 × 2 endpoint = 2 行）
    assert len(train) + len(eval_all) == 72 - 2


def test_nontarget_fractions_are_noop_when_expand_train_pool_false(tmp_path):
    """两个新 fraction 只在 expand_train_pool=true 时被消费。expand_train_pool=false
    时即便都设成 1.0，train.parquet 也必须恒等于 train_fit.parquet、不含 source_phase
    列——照抄 fault_baseline_train_fraction 的既有约定（见该字段的 noop 测试）。
    """
    raw = yaml.safe_load((REPO_ROOT / "configs/contract/v1.yaml").read_text())
    raw["fault_inject_nontarget_train_fraction"] = 1.0
    raw["fault_recover_nontarget_train_fraction"] = 1.0
    cfg_path = tmp_path / "v1_nontarget_noop.yaml"
    cfg_path.write_text(yaml.safe_dump(raw, allow_unicode=True, sort_keys=False))

    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir, config=str(cfg_path), dataset=_NTGT_DATASET)

    train = pd.read_parquet(out_dir / "train.parquet")
    train_fit = pd.read_parquet(out_dir / "train_fit.parquet")
    assert set(train["sample_id"]) == set(train_fit["sample_id"])
    assert "source_phase" not in train.columns

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    # 故障 case 的 inject/recover 行必须全留 eval（各 20 行）
    assert (eval_all["phase"] == "inject").sum() == 20
    assert (eval_all["phase"] == "recover").sum() == 20
```

- [ ] **Step 3: 跑新测试确认全部通过**

Task 4 已实现逻辑，所以这批测试应当直接通过。若有 fail，说明 Task 4 的实现与本设计不符，回去修 Task 4 的实现而不是改断言数字。

```bash
conda run -n interface pytest tests/test_contract_v1_train_pool.py -q
```

Expected: 全部 PASS（既有 7 个 + 新增 8 个）

- [ ] **Step 4: 更新既有测试的 docstring，说明其结论现在是 fixture 相关的**

`test_train_pool_excludes_inject_and_recover_rows`（原第 51 行附近）断言 train 不含 inject/recover 行。这条在 `split_fraction_mini` 上仍成立（该 fixture 的唯一 endpoint 就是目标 endpoint，两个新候选池恒为空），但它已不是全局不变量。给它补 docstring 说明清楚，避免后人误以为"train 永不含 inject/recover 行"：

```python
def test_train_pool_excludes_inject_and_recover_rows(tmp_path):
    """split_fraction_mini 上 train 不含 inject/recover 行。

    注意这**不是**全局不变量：该 fixture 的故障 case 只有 1 个 endpoint 且它就是
    目标 endpoint，故 inject 非目标候选池（is_endpoint_anomaly==False）与 recover
    非目标候选池（is_target_endpoint==False）恒为空，两个 nontarget fraction 取
    任何值都吸收不到行。在有 endpoint 级 fan-out 的 fixture 上（见
    tests/fixtures/nontarget_split_mini），train 合法地会含 inject/recover 行——
    那批行由 test_inject_nontarget_rows_partially_absorbed_into_train 等测试覆盖。
    """
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    assert not (train["phase"] == "recover").any()
    assert not (train["phase"] == "inject").any()
```

- [ ] **Step 5: 跑全量测试确认零回归**

```bash
conda run -n interface pytest tests/ -q
```

Expected: 全部 PASS。特别关注 `test_e2e_reliability_gate_smoke.py` 与 `test_e2e_deviation_weighted_smoke.py`——它们走 `v1_expanded_pool.yaml` + `nan_propagation_mini`（case 级标签、无 `target_endpoint`），两个新候选池均为空，行为不应变化。

- [ ] **Step 6: Commit**

```bash
git add tests/test_contract_v1_train_pool.py
git commit -m "[Test]: 锁住 inject/recover 非目标行切分行为

覆盖：两路按 fraction 部分吸收、正样本永不被吸收（is_endpoint_anomaly 判据的
核心保护）、目标 endpoint 的 recover 行永不被吸收、case 级标签 case 的 recover
行永不被吸收（label_granularity 过滤层）、label_granularity 判据来自
target_endpoint 而非 anomaly_level、三路 fraction 全推 1.0 时 sample_id 仍互斥
且无行丢失、expand_train_pool=false 时两个新字段是 no-op。
同时给 test_train_pool_excludes_inject_and_recover_rows 补 docstring 说明其
结论现在是 fixture 相关的、不再是全局不变量。"
```

---

## Task 6: 配置启用 + 文档同步

**Files:**
- Modify: `configs/contract/v1_expanded_pool.yaml`、`configs/contract/v1_new_ep1.yaml`
- Modify: `CLAUDE.md:109-114`（Commands 注释块）、`CLAUDE.md:161`、`CLAUDE.md:227-228`（已知问题）
- Create: `history/entries/022-inject-recover-nontarget-split.md`
- Modify: `history/index.md`

- [ ] **Step 1: 两份 config 声明新字段 = 1.0**

`configs/contract/v1_expanded_pool.yaml`，在现有 `fault_baseline_train_fraction: 0.2` 之后追加：

```yaml
# 故障 case 的 inject 阶段"非目标 endpoint"行（endpoint 级 fan-out 残留，判据
# is_endpoint_anomaly==False）按时间窗时序切分：最早该比例的窗口进训练池，其余留
# eval_all。设 1.0 = 全部吸收，尽量压低 eval_all 的负样本地板——entry 019 实测
# 这批 fan-out 残留是负样本最大一项（new_ep1 上 2546 行），fault_baseline_train_fraction
# 推到 1.0 也压不动它，正负比封顶 12.24:87.76。见 history/entries/022。
fault_inject_nontarget_train_fraction: 1.0
# 故障 case 的 recover 阶段"非目标 endpoint"行（判据 is_target_endpoint==False，
# 且只对 label_granularity=="endpoint" 的 case 生效——case 级标签的 case 该列全为
# False、无法区分目标/非目标，其 recover 行整段留在 eval_all）。目标 endpoint 自身
# 的 recover 行始终不吸收（系统未稳定回正常态，分布未验证，沿用 entry 014 决策）。
fault_recover_nontarget_train_fraction: 1.0
```

`configs/contract/v1_new_ep1.yaml` 追加同样两个字段与值（注释可精简为一句 + 指向 `history/entries/022`），并在该文件那段长注释（描述 12.24:87.76 上限的那段）末尾补一句：

```yaml
# 【entry 022 更新】上述"需要改变 eval_all 构成定义"的判断已落地：新增
# fault_inject_nontarget_train_fraction / fault_recover_nontarget_train_fraction
# 两个字段，分别吸收 inject/recover 阶段的非目标 endpoint 行。上文 12.24:87.76
# 的上限描述仅对"只有 fault_baseline_train_fraction 一个可调参数"时成立。
```

- [ ] **Step 2: 跑一次两份 config 的加载校验**

```bash
conda run -n interface python -c "
from src.contracts.contract_config import load_contract_config
for p in ('configs/contract/v1_expanded_pool.yaml','configs/contract/v1_new_ep1.yaml','configs/contract/v1.yaml'):
    c=load_contract_config(p)
    print(p, c.expand_train_pool, c.fault_baseline_train_fraction,
          c.fault_inject_nontarget_train_fraction, c.fault_recover_nontarget_train_fraction)
"
```

Expected：
```
configs/contract/v1_expanded_pool.yaml True 0.2 1.0 1.0
configs/contract/v1_new_ep1.yaml True <该文件既有值> 1.0 1.0
configs/contract/v1.yaml False 0.2 0.0 0.0
```

- [ ] **Step 3: 跑受 config 影响的测试**

```bash
conda run -n interface pytest tests/test_dvc_pipeline_v1.py tests/test_contract_v1_train_pool.py tests/test_e2e_reliability_gate_smoke.py tests/test_e2e_deviation_weighted_smoke.py -q
```

Expected: 全部 PASS。`test_v1_and_v1_expanded_configs_share_identical_modalities` 只比较 `modalities`/`window_size_s`/`contract_version`/`expand_train_pool`，新增字段不影响它。两个 e2e 冒烟测试用 `nan_propagation_mini`（case 级标签），两个新候选池为空，行为不变。

- [ ] **Step 4: 更新 `CLAUDE.md` Commands 注释块（第 109-114 行）**

把该注释块里"train 额外吸收故障 case 的 baseline 阶段行扩容训练池（838→6249 行），eval_all 同步摘除这些行防泄漏（13632→8221 行）"这两句，改为反映三路吸收（行数是旧数据集的历史数字，已因 entry 021 的 drift 与本次改动双重失效，改为不写具体数字）：

```
# v1.yaml 默认 expand_train_pool=false（train=train_fit 纯 Normal，eval_all=全部故障阶段+holdout，
# 与 entry 012 既有实验数字可比）。v1_expanded_pool.yaml 显式打开 expand_train_pool=true，
# train 按三个独立 fraction 吸收故障 case 的三类行：baseline 阶段行
# （fault_baseline_train_fraction，默认 0.2）、inject 阶段非目标 endpoint 行
# （fault_inject_nontarget_train_fraction）、recover 阶段非目标 endpoint 行
# （fault_recover_nontarget_train_fraction），被吸收的窗口同步从 eval_all 摘除防泄漏。
# 具体行数随数据集而变，不在此写死（旧数字已因 entry 021 的 MAX_CONTENT_CHARS drift 失效）。
```

- [ ] **Step 5: 更新 `CLAUDE.md` 第 161 行的 `expand_train_pool` 字段说明**

改为：

```
- `configs/contract/v1.yaml` 的 `expand_train_pool: bool` 字段（仅 v1 契约消费，v0 不识别）：`false`（默认）=train 仅纯 Normal `train_fit`；`true`=train 按三个独立 fraction 额外吸收故障 case 的 baseline 阶段行、inject 阶段非目标 endpoint 行、recover 阶段非目标 endpoint 行，被吸收窗口同步从 eval_all 摘除防泄漏。三个 fraction 默认分别是 0.2 / 0.0 / 0.0，`v1_expanded_pool.yaml` 与 `v1_new_ep1.yaml` 把后两个显式设为 1.0
```

- [ ] **Step 6: 更新 `CLAUDE.md` 已知问题第 227-228 行**

第 227 行里的具体行数（13632 / 12683 / 8221）改为不写死（同 Step 4 理由），并把第 228 行整条改写为覆盖三路 fraction 的版本：

```
- **`expand_train_pool=true` 后 eval_all 类别比例由三个 fraction 共同控制**：`fault_baseline_train_fraction`（issue #16 修复，默认 0.2，切 baseline 阶段行）、`fault_inject_nontarget_train_fraction` 与 `fault_recover_nontarget_train_fraction`（entry 022 新增，默认 0.0，`v1_expanded_pool.yaml`/`v1_new_ep1.yaml` 设 1.0）。三者都走同一个整窗时序切分函数 `src/contracts/split_fault_phase.py`，最早该比例的窗口进训练池、其余留 eval_all，`train.sample_id ∩ eval_all.sample_id == ∅` 由整窗切分保证。**两路"非目标"判据不同、不可互换**：inject 侧用 `is_endpoint_anomaly == False`（兼容两种 `label_granularity`；case 级标签下该列 fallback 等于 `is_anomaly`、inject 阶段恒 True，故自动选不出行，不会误吸收正样本）；recover 侧必须用 `is_target_endpoint == False` 且只对 `label_granularity == "endpoint"` 的 case 生效（`is_anomaly` 定义为 `phase == "inject"`，recover 阶段恒 False，导致 `is_endpoint_anomaly` 对 recover 所有 endpoint 含目标恒为 False，拿它筛"非目标"是空操作；而 case 级标签的 case `is_target_endpoint` 全为 False、无法区分目标/非目标，其 recover 行整段留 eval_all）。目标 endpoint 自身的 recover 行始终不吸收。**即使三路全推 1.0 也未必到 1:1**——负样本地板还剩 recover 未吸收部分 + Normal holdout，后两者各有独立保护理由（分布未验证 / Normalizer 防泄漏），具体可达比例需按实际数据集核算。见 history/entries/016、019、022
```

- [ ] **Step 7: 写 history entry 022**

`history/entries/022-inject-recover-nontarget-split.md`，按 `history/entries/_template.md` 的结构。必写内容：
- **做了什么**：新增两个 fraction 字段 + `_write_v1` 三路吸收 + 函数重命名 + 新 fixture。
- **关键决策（不在 commit 里）**：① 复用 `expand_train_pool` 总闸而非新开独立开关（三类吸收语义同属"训练池扩容"）；② 两个 fraction 独立而非共用一个（两类行的规模与对模型的影响机制不同）；③ 默认 0.0 向后兼容，只在两份 expanded config 里显式设 1.0；④ 只吸收非目标 endpoint 的 recover 行，目标 endpoint 的 recover 行沿用 entry 014 保守排除；⑤ 本轮不动 Normal holdout。
- **坑 / 已知问题**（最重要的一条）：**inject 与 recover 的"非目标"判据不能共用同一个字段**。`is_anomaly` 定义为 `phase == "inject"` 这一点导致 `is_endpoint_anomaly` 在 recover 阶段对所有 endpoint 恒为 False，用它筛 recover 非目标是空操作；而 `is_target_endpoint` 在 case 级标签数据上全为 False，直接用它会把"实际就是故障发生地"的 endpoint 的 recover 行也吸收进训练池。两个字段各自的失效场景正好互补，必须分别处理。另记：`eval_all` 从"`phase != baseline` 粗筛"改为五段显式拼接后，漏拼任一段都会静默丢负样本、不报错，因此 Task 4 Step 7/8 的手工核对（fraction=0 时 inject/recover 各 20 行）是必要的验证手段。
- **本轮不重跑既有 baseline 指标**：沿用 entry 021 的判断——数据集待重采（PATCH 类故障需 api_response 响应体才可识别），旧数据集上的新数字不会被任何后续工作引用。因此 `v1_expanded_pool.yaml`/`v1_new_ep1.yaml` 虽已改，但不 `dvc repro`，drift 继续累积并接受。
- **遗留 TODO**：① 三路全推 1.0 后的实际正负比未在真实数据集上核算（`new_ep1` 的 recover 446 行里目标/非目标各占多少未拆分统计）；② recover 未吸收部分 + Normal holdout 是否需要动，留待下一轮；③ entry 019 的另一个独立 TODO（case-aware / per-anomaly-type fraction 分级，针对 PATCH 类训练池污染）仍未做。

- [ ] **Step 8: 更新 `history/index.md`**

在 Entry 列表末尾（entry 021 之后）追加一行：

```
| [022](./entries/022-inject-recover-nontarget-split.md) | 2026-07-29 | Feature | inject/recover 阶段非目标 endpoint 行按两个独立 fraction 吸收进训练池（承接 entry 019 的 12.24:87.76 上限）；split_fault_baseline 重命名为 split_fault_phase；判据踩坑：inject 用 is_endpoint_anomaly、recover 必须用 is_target_endpoint 且仅 endpoint 级标签生效 | `src/contracts/split_fault_phase.py`, `src/contracts/contract_config.py`, `scripts/build_contract.py`, `configs/contract/v1_expanded_pool.yaml`, `configs/contract/v1_new_ep1.yaml`, `tests/fixtures/nontarget_split_mini/`, `CLAUDE.md` |
```

同时在"影响域索引"里 `src/contracts/` 与 `scripts/build_contract.py` 两个条目的 entry 编号列表末尾补上 `022`（该小节是按目录倒排的编号列表，找到对应行追加编号即可）。

- [ ] **Step 9: 跑全量测试 + 确认无遗漏的旧函数名引用**

```bash
conda run -n interface pytest tests/ -q
grep -rn "split_fault_baseline" --include="*.py" --include="*.yaml" . | grep -v "^./docs/superpowers/" | grep -v "^./history/"
```

Expected: 测试全部 PASS；grep 无输出

- [ ] **Step 10: Commit**

```bash
git add configs/contract/v1_expanded_pool.yaml configs/contract/v1_new_ep1.yaml CLAUDE.md history/
git commit -m "[Docs]: 启用两个 nontarget fraction + CLAUDE.md/history 同步

v1_expanded_pool.yaml 与 v1_new_ep1.yaml 把两个新 fraction 设为 1.0（全部吸收，
尽量压低 eval_all 负样本地板）。CLAUDE.md 已知问题条目改写为覆盖三路 fraction，
重点记录两路判据不可互换的原因；行数具体值不再写死（旧数字已因 entry 021 的
MAX_CONTENT_CHARS drift 失效）。新增 history entry 022。

按 entry 021 的既有判断，本轮不 dvc repro 重跑既有 baseline——数据集待重采，
旧数据集上的新数字不会被后续工作引用。"
```

---

## Task 7: 收尾校验

- [ ] **Step 1: 全量测试 + lint**

```bash
conda run -n interface pytest tests/ -q
conda run -n interface pre-commit run --all-files
```

Expected: 测试全部 PASS；pre-commit 全部 Passed 或仅有自动修复（若 black/isort 改了文件，`git add` 后补一个 `[Chore]` commit 或 amend 进 Task 6 的 commit）。

- [ ] **Step 2: 确认 DVC 状态可解释（不重跑）**

```bash
dvc status dvc_reliability_gate/dvc.yaml dvc_new_ep1/dvc.yaml
```

Expected: 显示 deps 已过期（config 与 `scripts/build_contract.py`、`src/contracts/` 都变了）。**这是预期状态，不要 `dvc repro`**——理由见 Task 6 Step 10 的 commit message 与 entry 021。若输出里出现"文件不存在"一类的错误（而不是"过期"），说明 Task 1 的 deps 路径改名有遗漏，回去修。

- [ ] **Step 3: 检查 commit 历史与工作区干净**

```bash
git log --oneline master..HEAD
git status --short
```

Expected: 6 个 commit（Task 1-6 各一个，顺序为 Refactor → Feature(config) → Test(fixture) → Feature(_write_v1) → Test(集成) → Docs）；`git status` 除 `data/new_ep2/`（会话开始时就存在的未跟踪目录，与本次改动无关，不要提交）外无其他改动。

- [ ] **Step 4: 推分支（不自动开 PR，等用户确认）**

```bash
git push -u origin feature/eval-all-nontarget-split
```

之后向用户报告：分支已推、6 个 commit 的内容概要、DVC 处于预期的过期状态且本轮刻意不重跑的理由，询问是否要开 PR。
