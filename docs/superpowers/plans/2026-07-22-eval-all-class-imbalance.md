# 修复 eval_all 类别失衡（issue #16）实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**目标：** 修复 issue #16——`expand_train_pool=true` 时，把每个故障 case 的 `baseline` 阶段行按时间窗时序切分，最早一小部分进训练池、其余留在 `eval_all`，使 `eval_all` 的正负比从被扩容破坏的 ~84:16 拉回接近原始 ~50:50，同时不重新引入 Contract v1 本就为消除的行级 train/eval 泄漏。

**架构：** 现状是 `_write_v1` 的扩容分支把**全部**故障 baseline 行移入 `train.parquet`、并把它们**全部**从 `eval_all.parquet` 摘除——这种"进池即摘除"的全有或全无耦合，掏空了 eval 的负样本类别。我们用"按 case 逐个做时间窗时序切分"取代这种全有或全无的整段搬移：由新配置开关 `fault_baseline_train_fraction`（默认 0.2）控制——每个故障 case 的 baseline 时间窗里，最早 `fraction` 比例的窗口进 train，其余留在 eval。唯一绝不能破坏的不变量——`train.sample_id ∩ eval_all.sample_id == ∅`——被保留并加强。`recover` 阶段的处理保持不动（超出本次范围，遵循 entry 014"不吸收 recover"的决策）。

**技术栈：** Python 3.12、pandas、PyTorch（Deep SVDD，不受影响）、Hydra 配置（YAML）、DVC pipeline、pytest、conda 环境 `interface`。

---

## 背景与约束（写代码前必读）

- **Issue：** GitHub #16（`ready-for-agent`）。根因在 `scripts/build_contract.py::_write_v1` 约 523-541 行：`expand_train_pool=true` 时把故障 baseline 行吸收进 train，同时从 eval 摘除，这两步是互斥关系。
- **只有 `expand_train_pool=true`（RG 专属路径，`configs/contract/v1_expanded_pool.yaml`）受影响。** 默认 `v1.yaml`（`expand_train_pool=false`）与 L0/L1/L2 主链路不受影响——本次修复绝不能改变它们的行为或数字。
- **不变量：** `sample_id` 是行级唯一键（`case_id__endpoint_key__timestamp_window_ms`）。`train.parquet` 与 `eval_all.parquet` 的 `sample_id` 集合必须互斥。这是唯一的硬泄漏约束；只要没有任何一行同时落在两个文件里，把 baseline 行在两者之间切分是安全的。
- **切分单位是时间窗**，不是行：同一 `timestamp_window_ms` 的所有行整体搬移（镜像 `split_normal_rows_temporal`）。这样避免同一个 `service`-`timestamp` 的广播特征同时出现在 train 和 eval。
- **切分是时序的：** 最早的窗口进 train，较晚的窗口进 eval（镜像 Normal 切分"train 窗早于 eval 窗"的纪律）。
- **默认 fraction = 0.2。** 实测能把 eval 负样本占比拉回 ~45:55（接近原始 50:50），同时仍让 838 行纯 Normal 训练池大致翻倍。做成可配置，是因为 Normal 数据集后续会扩充，这个比例会再调。
- **`EndpointBaselineStats` 与 `Normalizer` 的 fit 范围不受影响**——两者始终锁定在 `train_fit`（纯 Normal）。本次修复只改动故障 baseline 行在 `train.parquet` 与 `eval_all.parquet` 之间如何分配，绝不动 fit 集合。

## 文件结构

- **新建** `src/contracts/split_fault_baseline.py`——故障 baseline 行的两路时序切分新函数。单一职责：给定故障 baseline 行 + 一个 fraction，返回 `(train_part, eval_part)`。刻意与 `split_v1.py` 的三路 Normal 切分分开（不同业务概念、不同路数、不同 phase 过滤）。
- **修改** `src/contracts/contract_config.py`——新增 `fault_baseline_train_fraction: float = 0.2` 字段 + `__post_init__` 范围校验 + `load_contract_config` 装配。
- **修改** `scripts/build_contract.py::_write_v1`——用逐 case 时序切分（约 523-541 行）取代全有或全无的 baseline 整段搬移；把新 fraction 参数贯穿传入。
- **修改** `configs/contract/v1_expanded_pool.yaml`——加上 `fault_baseline_train_fraction: 0.2`（显式声明，便于审计）。
- **新建** `tests/fixtures/split_fraction_mini/`（+ `split_fraction_mini.yaml`）——专用 fixture，其故障 case 有 5 个 baseline 窗口（这样 `int(5*0.2)=1`，能产出非零、可验证的切分）。共享的 `mini_data_root` 故障 case 每个只有 1 个 baseline 窗口，无法验证分数切分。
- **新建** `tests/test_split_fault_baseline.py`——新切分函数的单元测试。
- **修改** `tests/test_contract_v1_train_pool.py`——把两个依赖 baseline 行数的测试迁到新 fixture；把泄漏测试的断言从"eval 里 baseline 行数为零"改写为"eval 保留大部分 baseline 行 且 train/eval sample_id 互斥"；新增 fraction 行为测试。
- **修改** `tests/test_contract_config.py`——为新字段加默认值/覆写/范围校验测试。
- **修改** `dvc_reliability_gate/dvc.yaml`——把 `src/contracts/split_fault_baseline.py` 加进 `build_contract_v1_expanded` stage 的 deps。
- **修改** `CLAUDE.md`——把"expand_train_pool=true 后 eval_all 正负比失衡（未修复）"的已知问题条目翻新成描述修复后的行为 + 新开关。
- **新建** `history/entries/016-fix-eval-all-class-imbalance.md` + **修改** `history/index.md`——新 entry（标注 entry 014 的 RG 数字已过期）+ index 列表行 + 影响域索引。

---

### Task 1：新增 `fault_baseline_train_fraction` 配置字段 + 范围校验

**文件：**
- 修改：`src/contracts/contract_config.py`
- 测试：`tests/test_contract_config.py`

- [ ] **Step 1：写失败的测试**

追加到 `tests/test_contract_config.py`：

```python
def test_fault_baseline_train_fraction_defaults_to_0_2(tmp_path):
    """新增开关默认 0.2：config 不写该字段时取默认值。默认值只在
    expand_train_pool=true 时被 _write_v1 消费，但默认值本身必须稳定。"""
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
    assert cfg.fault_baseline_train_fraction == 0.2


def test_fault_baseline_train_fraction_override(tmp_path):
    cfg_path = tmp_path / "c.yaml"
    cfg_path.write_text(
        "contract_version: v1\n"
        "window_size_s: 15\n"
        "fault_baseline_train_fraction: 0.35\n"
        "modalities:\n"
        "  endpoint_red:\n"
        "    preprocessor: TracePreprocessor\n"
        "    preprocessor_version: v0\n"
        "    features: [trace_request_count]\n"
        "    normalization: per_endpoint_min_max\n"
    )
    cfg = load_contract_config(cfg_path)
    assert cfg.fault_baseline_train_fraction == 0.35


def test_fault_baseline_train_fraction_out_of_range_raises(tmp_path):
    """越界值（<0 或 >1）必须在加载时报错，而不是悄悄产出空/全量切分污染实验。"""
    for bad in ("2.0", "-0.1"):
        cfg_path = tmp_path / f"c_{bad}.yaml"
        cfg_path.write_text(
            "contract_version: v1\n"
            "window_size_s: 15\n"
            f"fault_baseline_train_fraction: {bad}\n"
            "modalities:\n"
            "  endpoint_red:\n"
            "    preprocessor: TracePreprocessor\n"
            "    preprocessor_version: v0\n"
            "    features: [trace_request_count]\n"
            "    normalization: per_endpoint_min_max\n"
        )
        with pytest.raises(ValueError, match="fault_baseline_train_fraction"):
            load_contract_config(cfg_path)
```

确认文件顶部已有 `import pytest`（已有）。

- [ ] **Step 2：运行测试确认失败**

运行：`conda run -n interface python -m pytest tests/test_contract_config.py -k fault_baseline_train_fraction -v`
预期：FAIL——`AttributeError: 'ContractConfig' object has no attribute 'fault_baseline_train_fraction'`（且范围校验测试因无校验逻辑而失败）。

- [ ] **Step 3：加字段、校验、loader 装配**

在 `src/contracts/contract_config.py` 里，把字段加到 `ContractConfig` dataclass（放在 `fit_endpoint_baseline_stats` 之后，沿用既有字段注释风格）：

```python
    # v1 专属（仅 expand_train_pool=true 时被 _write_v1 消费）：故障 case 的 baseline
    # 阶段行按时间窗时序切分时，最早 fraction 比例的窗口进训练池，其余留在 eval_all。
    # 默认 0.2——把 eval_all 正负比从扩容导致的 ~84:16 拉回接近原始 ~50:50，同时训练池
    # 仍有实质扩容（见 issue #16 / history/entries/016）。expand_train_pool=false 时
    # 该字段被忽略（但仍参与下方 __post_init__ 范围校验，配错值一律在加载期报错）。
    fault_baseline_train_fraction: float = 0.2
```

给 `ContractConfig` 加一个 `__post_init__`（该 dataclass 当前没有）：

```python
    def __post_init__(self) -> None:
        # 恒校验（不看 expand_train_pool）：越界比例是配置错误，即便当前未被消费也应
        # 在加载期暴露，而不是等到 expand_train_pool 被打开时才悄悄产出空/全量切分。
        if not 0.0 <= self.fault_baseline_train_fraction <= 1.0:
            raise ValueError(
                "fault_baseline_train_fraction 必须落在 [0.0, 1.0]，"
                f"实际 {self.fault_baseline_train_fraction}"
            )
```

装配进 `load_contract_config` 的 `ContractConfig(...)` 构造：

```python
        fault_baseline_train_fraction=raw.get("fault_baseline_train_fraction", 0.2),
```

- [ ] **Step 4：运行测试确认通过**

运行：`conda run -n interface python -m pytest tests/test_contract_config.py -v`
预期：PASS（既有测试 + 3 个新测试）。

- [ ] **Step 5：提交**

```bash
git add src/contracts/contract_config.py tests/test_contract_config.py
git commit -m "[Feature]: ContractConfig 新增 fault_baseline_train_fraction 字段 + 范围校验"
```

---

### Task 2：新时序切分函数 `split_fault_baseline_temporal`

**文件：**
- 新建：`src/contracts/split_fault_baseline.py`
- 测试：`tests/test_split_fault_baseline.py`

- [ ] **Step 1：写失败的测试**

新建 `tests/test_split_fault_baseline.py`：

```python
"""split_fault_baseline_temporal 的单元测试：两路时序切分故障 baseline 行。"""

from __future__ import annotations

import pandas as pd

from src.contracts.split_fault_baseline import split_fault_baseline_temporal

_ENDPOINTS = [f"ep{i}" for i in range(4)]


def _synth_baseline_df(windows_per_case: dict[str, int]) -> pd.DataFrame:
    """合成"故障 case 的 baseline 阶段行"：每个 case 若干时间窗，每窗含全部 endpoint。"""
    rows = []
    for case_id, n_win in windows_per_case.items():
        for w in range(n_win):
            ts = 1_000 + w * 15_000  # 15s 一窗，单调递增
            for ep in _ENDPOINTS:
                rows.append(
                    {
                        "sample_id": f"{case_id}__{ep}__{ts}",
                        "case_id": case_id,
                        "endpoint_key": ep,
                        "timestamp_window_ms": ts,
                        "phase": "baseline",
                        "feat": float(w),
                    }
                )
    return pd.DataFrame(rows)


def test_split_disjoint_and_lossless():
    df = _synth_baseline_df({"F1": 5, "F2": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert set(train["sample_id"]) & set(eval_["sample_id"]) == set()
    assert len(train) + len(eval_) == len(df)


def test_fraction_0_2_takes_earliest_one_of_five_windows():
    """5 窗 × fraction=0.2 → int(5*0.2)=1 窗进 train，其余 4 窗进 eval。
    且进 train 的必须是最早那一窗（时序切分）。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    train_windows = sorted(train["timestamp_window_ms"].unique())
    eval_windows = sorted(eval_["timestamp_window_ms"].unique())
    assert len(train_windows) == 1
    assert len(eval_windows) == 4
    assert max(train_windows) < min(eval_windows)  # train 全部早于 eval


def test_window_not_split_across_train_and_eval():
    """同一时间窗的所有 endpoint 行必须整体归属同一侧，不能拆散。"""
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    train_windows = set(train["timestamp_window_ms"])
    eval_windows = set(eval_["timestamp_window_ms"])
    assert train_windows & eval_windows == set()


def test_single_window_natural_truncation_no_fallback():
    """1 窗 × fraction=0.2 → int(1*0.2)=0 窗进 train，该 case 全部 baseline 行进 eval。
    刻意不做"至少 1 窗给 train"的兜底（不同于 split_normal_rows_temporal）——
    训练池已有纯 Normal 打底，某 case 贡献 0 行不影响训练可行性；强行留 1 窗反而
    会让该 case 在 eval 里的 baseline 覆盖率归零。"""
    df = _synth_baseline_df({"F1": 1})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == len(_ENDPOINTS)


def test_fraction_1_0_all_to_train():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=1.0)
    assert len(train) == len(df)
    assert len(eval_) == 0


def test_fraction_0_0_all_to_eval():
    df = _synth_baseline_df({"F1": 5})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.0)
    assert len(train) == 0
    assert len(eval_) == len(df)


def test_per_case_independent_split():
    """每个 case 独立按自身窗数切分——不同 case 窗数不同时各切各的。"""
    df = _synth_baseline_df({"F1": 5, "F2": 10})
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    f1_train = train[train["case_id"] == "F1"]["timestamp_window_ms"].nunique()
    f2_train = train[train["case_id"] == "F2"]["timestamp_window_ms"].nunique()
    assert f1_train == 1   # int(5*0.2)
    assert f2_train == 2   # int(10*0.2)


def test_empty_input_returns_two_empty_frames():
    """无故障 baseline 行（如数据集里没有故障 case）时返回两个空帧，不抛异常。"""
    df = pd.DataFrame(
        columns=["sample_id", "case_id", "endpoint_key", "timestamp_window_ms", "phase", "feat"]
    )
    train, eval_ = split_fault_baseline_temporal(df, fraction=0.2)
    assert len(train) == 0
    assert len(eval_) == 0
    assert list(train.columns) == list(df.columns)
```

- [ ] **Step 2：运行测试确认失败**

运行：`conda run -n interface python -m pytest tests/test_split_fault_baseline.py -v`
预期：FAIL——`ModuleNotFoundError: No module named 'src.contracts.split_fault_baseline'`。

- [ ] **Step 3：实现切分函数**

新建 `src/contracts/split_fault_baseline.py`：

```python
"""Contract v1 训练池扩容专属：故障 case 的 baseline 阶段行两路时序切分。

修复 issue #16——扩容前的实现把故障 baseline 行整段移入训练池、整段移出 eval_all，
导致 eval_all 负样本骤减、正负比从 ~50:50 崩到 ~84:16。这里改为按时间窗时序切分：
每个 case 内最早 fraction 比例的窗口进训练池，其余留在 eval_all。切分单位是时间窗
（同一窗的所有 endpoint 行不拆散），train 窗全部早于 eval 窗（复用与
split_normal_rows_temporal 一致的防泄漏时序纪律）。

与 split_v1.split_normal_rows_temporal 的差异（故意不共用同一函数）：
- 两路而非三路（train / eval，无 val）
- 对象是故障 case 的 baseline 行，不是 Normal case
- 边界退化不做兜底（见下方注释），而 Normal 切分保证 train_fit 至少 1 窗
"""

from __future__ import annotations

import pandas as pd


def split_fault_baseline_temporal(
    df: pd.DataFrame,
    fraction: float,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把故障 baseline 行切成 (train_part, eval_part) 两个互斥子集。

    调用方须保证 df 仅含故障 case 的 baseline 阶段行（phase == 'baseline'）。
    每个 case 独立按 time_col 排序，最早 int(n * fraction) 个时间窗归 train_part，
    其余归 eval_part。切分单位是时间窗，同一窗的所有行整体归属同一侧。
    """
    if df.empty:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy()

    train_frames: list[pd.DataFrame] = []
    eval_frames: list[pd.DataFrame] = []

    for _case_id, g in df.groupby(case_col, sort=False):
        windows = sorted(g[time_col].unique())
        n = len(windows)
        # int() 自然向下取整，不做"至少 1 窗给 train"的兜底——这与
        # split_normal_rows_temporal 的边界处理刻意不同：那里兜底是因为 train_fit
        # 为空会让 Normalizer 无法 fit；这里训练池已有纯 Normal 的 train_fit 打底，
        # 某个 case 贡献 0 行 baseline 完全不影响训练可行性。反过来，若强行保证
        # 至少 1 窗进 train，对窗数极少的 case 会把仅有的窗吃进 train，让该 case 在
        # eval 里的 baseline 覆盖率归零——正是本次修复要避免的失衡。
        n_train = int(n * fraction)
        train_w = set(windows[:n_train])
        train_frames.append(g[g[time_col].isin(train_w)])
        eval_frames.append(g[~g[time_col].isin(train_w)])

    def _concat(frames: list[pd.DataFrame]) -> pd.DataFrame:
        nonempty = [f for f in frames if len(f)]
        return pd.concat(nonempty, ignore_index=True) if nonempty else df.iloc[0:0].copy()

    return _concat(train_frames), _concat(eval_frames)
```

- [ ] **Step 4：运行测试确认通过**

运行：`conda run -n interface python -m pytest tests/test_split_fault_baseline.py -v`
预期：PASS（全部 8 个测试）。

- [ ] **Step 5：提交**

```bash
git add src/contracts/split_fault_baseline.py tests/test_split_fault_baseline.py
git commit -m "[Feature]: 新增 split_fault_baseline_temporal 两路时序切分函数"
```

---

### Task 3：构建专用测试 fixture `split_fraction_mini`

**为什么需要新 fixture：** 共享的 `mini_data_root` 里每个故障 case 的 baseline 阶段只有 1 个时间窗，`int(1*0.2)=0`，无法验证"最早若干窗进 train、其余进 eval"这个分数切分行为。仿照 `nan_propagation_mini`（曾为 RG e2e 专门新建 fixture）的先例，新建一个 fault case 有 5 个 baseline 窗口的 fixture。为让 `EndpointBaselineStats`（RG 专属，只 fit `train_fit`）不因 endpoint 覆盖不一致抛 KeyError，**Normal case 与 fault case 共用同一个 endpoint_key**（`POST:/api/v1/travelservice/trips/left`），保证 train_fit / train / eval_all 三者 endpoint 覆盖面一致。

**文件：**
- 新建：`tests/fixtures/split_fraction_mini.yaml`
- 新建：`tests/fixtures/split_fraction_mini/Normal/case_metadata.json`
- 新建：`tests/fixtures/split_fraction_mini/Normal/_pipeline_out/tt_endpoint_health_15s.csv`
- 新建：`tests/fixtures/split_fraction_mini/Normal/_pipeline_out/tt_traces_red_15s.csv`
- 新建：`tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/case_metadata.json`
- 新建：`tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/_pipeline_out/tt_endpoint_health_15s.csv`
- 新建：`tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/_pipeline_out/tt_traces_red_15s.csv`

> **说明**：不放 `metric_data` / `log_data`——这些模态缺失会走 build_contract 既有的"填 NaN"分支（`_fill_missing_feature_cols`），fixture 只需驱动 endpoint_red 模态跑通切分逻辑即可。这与 `mini_data_root/Lv_E_HTTPABORT_assurance_mini`（同样只有 trace+api，无 metric/log）一致。

- [ ] **Step 1：写 dataset config**

新建 `tests/fixtures/split_fraction_mini.yaml`：

```yaml
name: split_fraction_mini
roots:
  - tests/fixtures/split_fraction_mini
normal_source: tests/fixtures/split_fraction_mini
fused_window: 15s
```

- [ ] **Step 2：写 Normal case（6 个时间窗，单 endpoint）**

新建 `tests/fixtures/split_fraction_mini/Normal/case_metadata.json`：

```json
{
  "case_id": "Normal",
  "anomaly_type": "Normal",
  "anomaly_level": "none",
  "target_service": null,
  "inject_start_ms": null,
  "inject_end_ms": null
}
```

新建 `tests/fixtures/split_fraction_mini/Normal/_pipeline_out/tt_endpoint_health_15s.csv`：

```csv
case_id,anomaly_type,timestamp_window,endpoint_key,request_count,error_rate,latency_mean,latency_p50,latency_p95,latency_p99,status_2xx_rate,status_4xx_rate,status_5xx_rate,method,normalized_path
Normal,Normal,2026-11-06T21:20:00Z,POST:/api/v1/travelservice/trips/left,8,0.0,23000.0,22800.0,46000.0,46000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Normal,Normal,2026-11-06T21:20:15Z,POST:/api/v1/travelservice/trips/left,10,0.0,21500.0,21000.0,43000.0,43000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Normal,Normal,2026-11-06T21:20:30Z,POST:/api/v1/travelservice/trips/left,6,0.0,23500.0,22500.0,47000.0,47000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Normal,Normal,2026-11-06T21:20:45Z,POST:/api/v1/travelservice/trips/left,9,0.0,22000.0,21500.0,44000.0,44000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Normal,Normal,2026-11-06T21:21:00Z,POST:/api/v1/travelservice/trips/left,7,0.0,24000.0,23000.0,48000.0,48000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Normal,Normal,2026-11-06T21:21:15Z,POST:/api/v1/travelservice/trips/left,11,0.0,21000.0,20500.0,42000.0,42000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
```

新建 `tests/fixtures/split_fraction_mini/Normal/_pipeline_out/tt_traces_red_15s.csv`（`timestamp_window` 用 epoch-ms 整数，与 `mini_data_root/Normal` 的 trace CSV 口径一致；`is_target_endpoint` 列可省略，`TracePreprocessor` 缺失时填 False）：

```csv
case_id,anomaly_type,timestamp_window,window_str,endpoint_key,method,normalized_path,trace_request_count,trace_latency_mean,trace_latency_p50,trace_latency_p95,trace_latency_p99,trace_error_rate,trace_5xx_rate,trace_4xx_rate,trace_status_coverage,weak_is_anomaly,label_confidence,latency_anomaly_signal,error_anomaly_signal,5xx_anomaly_signal,phase,injection_start_ms,injection_end_ms,target_service
Normal,Normal,1794000000000,2026-11-06T21:20:00Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,8,22800.0,22000.0,45500.0,45500.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
Normal,Normal,1794000015000,2026-11-06T21:20:15Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,10,21000.0,20000.0,42000.0,42000.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
Normal,Normal,1794000030000,2026-11-06T21:20:30Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,6,23000.0,22000.0,46000.0,46000.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
Normal,Normal,1794000045000,2026-11-06T21:20:45Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,9,21500.0,21000.0,43500.0,43500.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
Normal,Normal,1794000060000,2026-11-06T21:21:00Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,7,23500.0,22500.0,47000.0,47000.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
Normal,Normal,1794000075000,2026-11-06T21:21:15Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,11,20500.0,20000.0,41000.0,41000.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,normal,,,
```

- [ ] **Step 3：写 fault case（8 个时间窗 = 5 baseline + 1 inject + 2 recover）**

新建 `tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/case_metadata.json`（inject_start = inject_end = win5 的 ms，使 win0-4 为 baseline、win5 为 inject、win6-7 为 recover）：

```json
{
  "case_id": "Lv_E_SPLITTEST_travel",
  "anomaly_type": "Lv_E_SPLITTEST_travel",
  "anomaly_level": "endpoint",
  "target_service": "ts-travel-service",
  "target_endpoint": "POST:/api/v1/travelservice/trips/left",
  "inject_start_ms": 1795000075000,
  "inject_end_ms": 1795000075000
}
```

新建 `tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/_pipeline_out/tt_endpoint_health_15s.csv`：

```csv
case_id,anomaly_type,timestamp_window,endpoint_key,request_count,error_rate,latency_mean,latency_p50,latency_p95,latency_p99,status_2xx_rate,status_4xx_rate,status_5xx_rate,method,normalized_path
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:06:40Z,POST:/api/v1/travelservice/trips/left,8,0.0,23000.0,22800.0,46000.0,46000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:06:55Z,POST:/api/v1/travelservice/trips/left,10,0.0,21500.0,21000.0,43000.0,43000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:07:10Z,POST:/api/v1/travelservice/trips/left,6,0.0,23500.0,22500.0,47000.0,47000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:07:25Z,POST:/api/v1/travelservice/trips/left,9,0.0,22000.0,21500.0,44000.0,44000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:07:40Z,POST:/api/v1/travelservice/trips/left,7,0.0,24000.0,23000.0,48000.0,48000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:07:55Z,POST:/api/v1/travelservice/trips/left,12,0.4,58000.0,56000.0,99000.0,100000.0,0.6,0.0,0.4,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:08:10Z,POST:/api/v1/travelservice/trips/left,8,0.0,23000.0,22500.0,46000.0,46000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,2026-11-18T11:08:25Z,POST:/api/v1/travelservice/trips/left,9,0.0,22500.0,22000.0,45000.0,45000.0,1.0,0.0,0.0,POST,/api/v1/travelservice/trips/left
```

新建 `tests/fixtures/split_fraction_mini/Lv_E_SPLITTEST_travel/_pipeline_out/tt_traces_red_15s.csv`（`is_target_endpoint` 列必须存在且 inject 窗为 True，才能让 `is_endpoint_anomaly` 在 inject 阶段为 True——参考 `mini_data_root` 的 endpoint 级 fixture）：

```csv
case_id,anomaly_type,timestamp_window,window_str,endpoint_key,method,normalized_path,trace_request_count,trace_latency_mean,trace_latency_p50,trace_latency_p95,trace_latency_p99,trace_error_rate,trace_5xx_rate,trace_4xx_rate,trace_status_coverage,is_target_endpoint,weak_is_anomaly,label_confidence,latency_anomaly_signal,error_anomaly_signal,5xx_anomaly_signal,phase,injection_start_ms,injection_end_ms,target_service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000000000,2026-11-18T11:06:40Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,8,22800.0,22000.0,45500.0,45500.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,baseline,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000015000,2026-11-18T11:06:55Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,10,21000.0,20000.0,42000.0,42000.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,baseline,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000030000,2026-11-18T11:07:10Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,6,23000.0,22000.0,46000.0,46000.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,baseline,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000045000,2026-11-18T11:07:25Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,9,21500.0,21000.0,43500.0,43500.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,baseline,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000060000,2026-11-18T11:07:40Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,7,23500.0,22500.0,47000.0,47000.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,baseline,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000075000,2026-11-18T11:07:55Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,12,56000.0,54000.0,98000.0,99000.0,0.4,0.4,0.0,1.0,True,1,strong,1,1,1,inject,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000090000,2026-11-18T11:08:10Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,8,22800.0,22000.0,46000.0,46000.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,recover,1795000075000,1795000075000,ts-travel-service
Lv_E_SPLITTEST_travel,Lv_E_SPLITTEST_travel,1795000105000,2026-11-18T11:08:25Z,POST:/api/v1/travelservice/trips/left,POST,/api/v1/travelservice/trips/left,9,22000.0,21500.0,45000.0,45000.0,0.0,0.0,0.0,1.0,True,0,normal,0,0,0,recover,1795000075000,1795000075000,ts-travel-service
```

- [ ] **Step 4：冒烟验证 fixture 能被 build_contract 消费**

先跑一次未扩容构建，确认 fixture 结构合法、phase 划分正确（此时 Task 3 的 build 逻辑还没改，跑的是既有 `expand_train_pool=false` 分支，故 baseline 全留 eval）：

Run:
```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v1.yaml \
  --dataset tests/fixtures/split_fraction_mini.yaml \
  --out-dir /tmp/split_fraction_smoke --seed 42
```

Run（校验 phase 分布：应有 5 baseline + 1 inject + 2 recover）：
```bash
conda run -n interface python -c "
import pandas as pd
df = pd.read_parquet('/tmp/split_fraction_smoke/eval_all.parquet')
fault = df[df['anomaly_type'] != 'Normal']
print(fault['phase'].value_counts().to_dict())
print('is_endpoint_anomaly inject count:', int(df['is_endpoint_anomaly'].sum()))
"
```
Expected: `{'baseline': 5, 'recover': 2, 'inject': 1}`，且 `is_endpoint_anomaly inject count: 1`。若不符，检查 `inject_start_ms/inject_end_ms` 与 trace CSV 的 `timestamp_window` 是否对齐。

- [ ] **Step 5：提交**

```bash
git add tests/fixtures/split_fraction_mini.yaml tests/fixtures/split_fraction_mini/
git commit -m "[Test]: 新增 split_fraction_mini fixture（5 baseline 窗口故障 case）"
```

---

### Task 4：把新切分接入 `_write_v1`

**文件：**
- 修改：`scripts/build_contract.py`（`_write_v1` 约 489-565 行 + 调用点约 470-478 行）
- 修改：`configs/contract/v1_expanded_pool.yaml`

这一步没有独立单测（`_write_v1` 的行为由 Task 5 的 build_contract 集成测试覆盖），但要先接线、再让 Task 5 的测试驱动验证。

- [ ] **Step 1：在 config 里显式声明 fraction**

在 `configs/contract/v1_expanded_pool.yaml` 的 `fit_endpoint_baseline_stats: true` 行之后追加：

```yaml
# 故障 case 的 baseline 阶段行按时间窗时序切分：最早 20% 窗口进训练池扩容训练，
# 其余 80% 留在 eval_all 维持负样本类别平衡（修复 issue #16 的 ~84:16 失衡）。
# 后续扩充 Normal 数据集后可再调此值。见 history/entries/016。
fault_baseline_train_fraction: 0.2
```

- [ ] **Step 2：改 `_write_v1` 的函数签名与调用点**

在 `scripts/build_contract.py` 顶部 import 区加：

```python
from src.contracts.split_fault_baseline import split_fault_baseline_temporal
```

把 `main()` 里对 `_write_v1` 的调用（约 470-478 行）改为多传一个参数：

```python
    if cfg.contract_version == "v1":
        _write_v1(
            out,
            full,
            normal_mask,
            args.seed,
            cfg.expand_train_pool,
            cfg.fit_endpoint_baseline_stats,
            cfg.fault_baseline_train_fraction,
        )
```

把 `_write_v1` 的签名（约 489-496 行）改为：

```python
def _write_v1(
    out: Path,
    full: pd.DataFrame,
    normal_mask: pd.Series,
    seed: int,
    expand_train_pool: bool,
    fit_endpoint_baseline_stats: bool,
    fault_baseline_train_fraction: float,
) -> None:
```

- [ ] **Step 3：替换 `expand_train_pool` 分支的整段搬移逻辑**

把 `_write_v1` 里 `if expand_train_pool:` 分支体（约 523-541 行，从 `train_fit_labeled = ...` 到 `eval_all.to_parquet(...)` + `LOG.info(...)`）整体替换为：

```python
    if expand_train_pool:
        # 训练池扩容：train_fit（Normal，打标 normal_case）+ 故障 case baseline 阶段行
        # 的最早 fault_baseline_train_fraction 比例窗口（打标 fault_baseline）。
        # 只吸收 baseline，不吸收 recover——系统未稳定回正常态，分布未验证，保守排除。
        train_fit_labeled = parts["train_fit"].copy()
        train_fit_labeled["source_phase"] = "normal_case"

        fault_baseline_df = anomaly_df[anomaly_df["phase"] == "baseline"].copy()
        # issue #16 修复：baseline 行不再整段进训练池、整段摘出 eval，而是按时间窗时序
        # 切分——最早 fraction 比例的窗口进训练池，其余留在 eval_all 维持负样本类别平衡。
        # 唯一硬约束是 train/eval 的 sample_id 互斥（split 按整窗切分天然保证：同一窗不会
        # 既在 train 又在 eval），由 test_contract_v1_train_pool 的防泄漏断言锁定。
        fault_baseline_train, fault_baseline_eval = split_fault_baseline_temporal(
            fault_baseline_df, fraction=fault_baseline_train_fraction
        )
        fault_baseline_train["source_phase"] = "fault_baseline"

        train_pool = pd.concat([train_fit_labeled, fault_baseline_train], ignore_index=True)
        train_pool.to_parquet(out / "train.parquet", index=False)

        # eval_all：故障 case 的 inject/recover 行（全保留）+ 未被吸收进训练池的 baseline
        # 窗口（fault_baseline_eval）+ Normal holdout。inject/recover 用 phase 过滤，
        # baseline 用 split 摘出的 eval 部分，两者不重叠且并集为故障 case 全部非训练行。
        eval_all = pd.concat(
            [
                anomaly_df[anomaly_df["phase"] != "baseline"],
                fault_baseline_eval,
                parts["eval_normal_holdout"],
            ],
            ignore_index=True,
        )
        eval_all.to_parquet(out / "eval_all.parquet", index=False)
        LOG.info(
            "v1 切分（训练池已扩容，fraction=%.3f）：train_fit=%d holdout=%d "
            "fault_baseline_total=%d fault_baseline_to_train=%d fault_baseline_to_eval=%d "
            "train_pool=%d eval_all=%d",
            fault_baseline_train_fraction,
            len(parts["train_fit"]),
            len(parts["eval_normal_holdout"]),
            len(fault_baseline_df),
            len(fault_baseline_train),
            len(fault_baseline_eval),
            len(train_pool),
            len(eval_all),
        )
```

`else` 分支（`expand_train_pool=false`）保持不动。同步更新 `_write_v1` docstring（约 497-514 行）里"eval_all 同步从这些被吸收的行里摘除"那句，改述为"按 fraction 时序切分，最早部分进池、其余留 eval"。

- [ ] **Step 4：运行既有测试确认未破坏未扩容路径 + 新旧接口一致**

Run: `conda run -n interface python -m pytest tests/test_contract_v1_split.py tests/test_dvc_pipeline_v1.py tests/test_build_contract_v1_endpoint_id.py -v`
Expected: PASS（未扩容路径行为不变；`test_build_contract_v1_endpoint_id.py` 走 `v1_expanded_pool.yaml` 但只断言 endpoint_id 列与 sidecar，不断言 baseline 行数，因此仍通过）。

> 若 `test_build_contract_v1_endpoint_id.py` 里的 `test_endpoint_baseline_stats_values_match_train_fit_only_recomputation` 失败，说明改动误动了 `train_fit` 的 fit 范围——但本 Task 只改 `train.parquet`/`eval_all.parquet` 的构成，`parts["train_fit"]` 与 baseline_stats 的 fit 完全没碰，不应失败。若失败必须停下排查，不要绕过。

- [ ] **Step 5：提交**

```bash
git add scripts/build_contract.py configs/contract/v1_expanded_pool.yaml
git commit -m "[Bugfix]: _write_v1 按 fraction 时序切分故障 baseline，修复 eval_all 类别失衡 (#16)"
```

---

### Task 5：迁移并改写 `test_contract_v1_train_pool.py`

**背景：** 该文件现有 6 个测试。其中两个断言"train 里含非零条 baseline 行"，在新默认 `fraction=0.2` 下若继续跑窗口数=1 的 `mini_dataset.yaml` 会失败（`int(1*0.2)=0`），必须迁到新 fixture。泄漏测试的核心断言语义要翻转："eval 不含任何 baseline 行" → "eval 保留大部分 baseline 行 + train/eval sample_id 互斥"。

**文件：**
- 修改：`tests/test_contract_v1_train_pool.py`

- [ ] **Step 1：把 `_run_v1_build` 的默认 fixture 切到新 fixture**

把文件顶部的 `_run_v1_build`（第 10-28 行）改为默认用 `split_fraction_mini.yaml`：

```python
def _run_v1_build(
    out_dir: Path,
    config: str = "configs/contract/v1_expanded_pool.yaml",
    dataset: str = "tests/fixtures/split_fraction_mini.yaml",
) -> None:
    # 这些测试验证训练池扩容 + baseline 时序切分行为，故默认走 v1_expanded_pool.yaml
    # （expand_train_pool=true）+ split_fraction_mini（故障 case 有 5 个 baseline 窗口，
    # int(5*0.2)=1，能验证非零分数切分）。mini_dataset.yaml 的故障 case 每个只有
    # 1 个 baseline 窗口，int(1*0.2)=0，无法验证分数切分，不能用于这批测试。
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / config),
            "--dataset",
            str(REPO_ROOT / dataset),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )
```

- [ ] **Step 2：改写泄漏测试的断言语义**

把 `test_eval_all_excludes_train_pool_rows_no_leakage`（第 68-82 行）整体替换为：

```python
def test_eval_all_and_train_pool_sample_id_disjoint_no_leakage(tmp_path):
    """核心防泄漏不变量（issue #16 修复后仍必须成立且加强）：train_pool 与 eval_all
    的 sample_id 必须互斥。issue #16 前的实现靠"baseline 整段摘出 eval"来保证互斥；
    修复后 baseline 按时间窗时序切分，一部分进 train、其余留 eval，互斥性改由
    split_fault_baseline_temporal 的整窗切分保证（同一窗不会既在 train 又在 eval）。
    这条断言是唯一的硬约束，语义比"eval 不含 baseline 行"更本质。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    assert set(train["sample_id"]) & set(eval_all["sample_id"]) == set()


def test_eval_all_retains_majority_of_fault_baseline_rows(tmp_path):
    """issue #16 修复的正向断言：故障 baseline 行不再被整段摘出 eval，大部分（新 fixture
    里 5 窗中的 4 窗 = 80%）应留在 eval_all 维持负样本类别平衡。train 只吸收最早 1 窗。"""
    out_dir = tmp_path / "contract_v1"
    _run_v1_build(out_dir)
    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    train_baseline = train[(train["anomaly_type"] != "Normal") & (train["phase"] == "baseline")]
    eval_baseline = eval_all[
        (eval_all["anomaly_type"] != "Normal") & (eval_all["phase"] == "baseline")
    ]
    train_windows = train_baseline["timestamp_window_ms"].nunique()
    eval_windows = eval_baseline["timestamp_window_ms"].nunique()
    assert train_windows == 1, "fraction=0.2 × 5 窗应吸收最早 1 窗进 train"
    assert eval_windows == 4, "其余 4 窗 baseline 应留在 eval_all"
```

- [ ] **Step 3：修正剩余测试的 fixture 期望**

`test_train_pool_includes_fault_baseline_rows`（第 31-43 行）、`test_source_phase_column_present_and_correct`（第 54-65 行）、`test_train_pool_excludes_inject_and_recover_rows`（第 46-51 行）现在都走新 fixture，断言语义不变（train 含 baseline 行、source_phase 打标正确、train 不含 inject/recover），但把注释里的 `Lv_P_DISKIO_preserve` 更新为 `Lv_E_SPLITTEST_travel`。这三个测试逻辑无需改动，只更新注释里的 case 名引用。

`test_default_v1_config_does_not_expand_train_pool`（第 85-104 行）：改为显式传 `dataset` 参数保持走 `split_fraction_mini`（也可继续用 mini_dataset——它验证的是 `expand_train_pool=false` 时 baseline 全留 eval，与窗口数无关）。为一致性统一传新 fixture：

```python
    _run_v1_build(out_dir, config="configs/contract/v1.yaml")
```
（`_run_v1_build` 的 `dataset` 默认已是 `split_fraction_mini.yaml`，无需额外传参。）

`test_contract_v1_838_variant_is_pure_normal`（第 107-129 行）：该测试硬编码了 `mini_dataset.yaml`，验证的是"train_fit 纯 Normal"，与本次改动无关，保持不动。

- [ ] **Step 4：运行整个文件确认通过**

Run: `conda run -n interface python -m pytest tests/test_contract_v1_train_pool.py -v`
Expected: PASS（原 6 个 - 1 改名 + 2 新增 = 7 个测试全通过）。

- [ ] **Step 5：提交**

```bash
git add tests/test_contract_v1_train_pool.py
git commit -m "[Test]: 迁移训练池测试到 split_fraction_mini，改写泄漏断言为 sample_id 互斥 (#16)"
```

---

### Task 6：补 DVC 依赖 + 全量测试 + 重跑 RG pipeline

**文件：**
- 修改：`dvc_reliability_gate/dvc.yaml`

- [ ] **Step 1：把新切分模块加进 DVC stage deps**

在 `dvc_reliability_gate/dvc.yaml` 的 `build_contract_v1_expanded` stage 的 `deps:` 列表里，`src/contracts/split_v1.py` 之后追加一行：

```yaml
      - src/contracts/split_fault_baseline.py
```

> 这一步不可省略：DVC 靠 deps 列表判断 stage 是否需要重跑。新文件不在 deps 里，改动它不会触发缓存失效，可能导致重跑时用 stale 产物。

- [ ] **Step 2：跑全量测试套件确认无回归**

Run: `conda run -n interface python -m pytest tests/ -x -q`
Expected: 全绿。重点关注 `test_e2e_reliability_gate_smoke.py`（走 `v1_expanded_pool.yaml`，会经过新切分逻辑）与所有 `test_build_contract*` / `test_dataloader*` / `test_e2e_smoke*`。若 `test_e2e_reliability_gate_smoke.py` 因 `nan_propagation_mini` 的故障 case 只有 1 baseline 窗、`int(1*0.2)=0` 导致 train 不含 fault_baseline 行而失败——检查该测试是否断言了 train 必含 fault_baseline（它只断言 `len(scores)==len(eval_all)`，不涉及 baseline 行数，应不受影响；若确实失败需停下排查）。

- [ ] **Step 3：确认 DVC 会因改动重跑，然后重跑 RG pipeline**

Run（先看 DVC 是否检测到依赖变化）: `dvc status dvc_reliability_gate/dvc.yaml`
Expected: `build_contract_v1_expanded` 因 `scripts/build_contract.py` / `configs/contract/v1_expanded_pool.yaml` / `src/contracts/split_fault_baseline.py` 变化被标记为需重跑。

Run（重跑三段 RG 链路，约 20min）: `dvc repro dvc_reliability_gate/dvc.yaml`
Expected: 三个 stage（build_contract_v1_expanded → train_v1_reliability_gate → eval_v1_reliability_gate）依次重跑成功，产出新的 `artifacts/baseline_v1_reliability_gate/metrics.json`。

Run（记录修复后的新类别比例与指标，供 history entry 引用）:
```bash
conda run -n interface python -c "
import pandas as pd, json
ev = pd.read_parquet('artifacts/contract_v1_expanded/eval_all.parquet')
vc = ev['phase'].value_counts().to_dict()
pos = int(ev['is_endpoint_anomaly'].sum()); neg = len(ev) - pos
print('eval_all phase 分布:', vc)
print(f'eval_all 正样本={pos} 负样本={neg} 正负比={pos/len(ev):.2%}:{neg/len(ev):.2%}')
print('metrics:', json.load(open('artifacts/baseline_v1_reliability_gate/metrics.json')))
"
```
Expected: 负样本占比从修复前的 ~16% 回升（真实 29-case 数据上，baseline 保留 80% 后应显著回升）。**记下实际数字**（phase 分布、正负比、AUROC/AUPRC），Task 7 的 history entry 要引用。

- [ ] **Step 4：提交（含 dvc.lock 与刷新的 metrics）**

```bash
git add dvc_reliability_gate/dvc.yaml dvc_reliability_gate/dvc.lock artifacts/baseline_v1_reliability_gate/metrics.json
git commit -m "[Experiment]: 补 split_fault_baseline DVC 依赖，重跑 RG pipeline 刷新修复后类别比例与指标 (#16)"
```

---

### Task 7：更新 CLAUDE.md 已知问题 + 写 history entry 016

**文件：**
- 修改：`CLAUDE.md`
- 新建：`history/entries/016-fix-eval-all-class-imbalance.md`
- 修改：`history/index.md`

- [ ] **Step 1：翻新 CLAUDE.md 的已知问题条目**

把 `CLAUDE.md` 里 "expand_train_pool=true 后 eval_all 正负比失衡（未修复）" 那条（约 206 行）替换为描述修复后行为的版本：

```markdown
- **`expand_train_pool=true` 后 eval_all 类别比例由 `fault_baseline_train_fraction` 控制（issue #16 已修复）**：修复前 baseline 阶段行整段进训练池、整段摘出 eval，导致 eval_all 正负比崩到 ~84:16。现在故障 baseline 行按时间窗时序切分（`src/contracts/split_fault_baseline.py`）：最早 `fault_baseline_train_fraction` 比例（默认 0.2）的窗口进训练池，其余留在 eval_all，正负比回升至接近原始 ~50:50。`train.sample_id ∩ eval_all.sample_id == ∅` 不变量由整窗切分保证。后续扩充 Normal 数据集后可调该 fraction。跨不同 fraction 取值比较 AUROC/AUPRC 时仍需注明各自的类别比例。见 history/entries/016
```

同时更新约 205 行"`expand_train_pool` 开关会改变 eval_all 行数"那条里的行数（13632 / 8221）——修复后 expanded 的 eval_all 行数会变（baseline 保留 80%），把该条的 8221 更新为 Task 6 Step 3 实测到的新行数，并注明"修复 #16 后"。

- [ ] **Step 2：写 history entry**

新建 `history/entries/016-fix-eval-all-class-imbalance.md`（用 Task 6 Step 3 记录的实测数字填充 XXX 占位）：

```markdown
# 016 · 修复 expand_train_pool 导致的 eval_all 类别失衡（issue #16）

- **日期**: 2026-07-22
- **PR**: #NN · **Commit**: <short-hash>
- **类型**: Bugfix
- **影响域**: `src/contracts/split_fault_baseline.py`, `src/contracts/contract_config.py`, `scripts/build_contract.py`, `configs/contract/v1_expanded_pool.yaml`, `dvc_reliability_gate/dvc.yaml`, `tests/`, `CLAUDE.md`

## 做了什么

修复 entry 014 记录、issue #16 拆分出的 eval_all 类别失衡：`expand_train_pool=true` 时故障 baseline 行不再整段进训练池、整段摘出 eval，改为按时间窗时序切分（新函数 `split_fault_baseline_temporal`），由新配置 `fault_baseline_train_fraction`（默认 0.2）控制——最早 20% 窗口进训练池，其余 80% 留在 eval_all。eval_all 正负比从 ~84:16 回升至约 XXX。重跑了 RG pipeline 刷新指标。

## 关键决策（不在 commit 里）

- **按时间窗时序切分，不按行随机切**：与 `split_normal_rows_temporal` 的切分哲学一致；按整窗切分天然保证 `train.sample_id ∩ eval_all.sample_id == ∅`（同一窗不会既进 train 又进 eval），且避免同一 service-timestamp 的广播特征同时出现在两侧。
- **新写 `split_fault_baseline_temporal` 而非复用/改造三路 Normal 切分**：两路 vs 三路、对象是故障 baseline 而非 Normal、边界退化策略不同（见下），共用一个函数会靠参数分支制造隐晦耦合。
- **边界退化不做兜底**：窗数极少时 `int(n*fraction)` 自然取整（可能 0 窗进 train），刻意不像 Normal 切分那样保证"至少 1 窗"——训练池已有纯 Normal 打底，某 case 贡献 0 行不影响训练；强行留 1 窗反而让该 case 在 eval 的 baseline 覆盖率归零，与修复目标相悖。
- **默认 fraction=0.2**：实测能把 eval 负样本占比拉回接近原始 50:50，同时训练池仍有实质扩容。做成可配置字段，因为 Normal 数据集后续会扩充、比例会再调。
- **recover 阶段不动**：沿用 entry 014"不吸收 recover（分布未验证）"的决策，本次只切 baseline，收紧改动面。

## 坑 / 已知问题

- **entry 014 的 RG 实测数字已因本次修复失效**：eval_all 样本构成变了（baseline 保留 80%），entry 014 表格里的 AUROC/AUPRC（在 ~84:16 失衡 eval 上跑出）不再与修复后可比。本次重跑后的新数字见上方"做了什么"；跨 entry 014/016 比较 RG 指标无意义。
- **共享 mini fixture 无法验证分数切分**：`mini_data_root` 故障 case 每个只有 1 个 baseline 窗，`int(1*0.2)=0`。为此新建 `split_fraction_mini` fixture（故障 case 5 个 baseline 窗），仿 `nan_propagation_mini` 的先例——Normal 与故障 case 共用同一 endpoint，避免 EndpointBaselineStats（只 fit train_fit）查表 KeyError。
- **跨 fraction 取值的 AUROC/AUPRC 不可直接横向比较**：不同 fraction 会改变 eval_all 的类别比例（AUPRC 对此敏感），比较时须注明各自比例。

## 遗留 TODO

- fraction 的取值当前只在真实 29-case 数据上凭"回到 ~50:50"经验定为 0.2，未做敏感性扫描（不同 fraction 对 RG 指标的影响）。若后续要论证 RG 效果，需在固定 fraction 下多 seed 复跑，而非在变动的 eval 口径上比较。
```

- [ ] **Step 3：更新 history/index.md**

在 entry 列表表格末尾（entry 015 行之后）追加：

```markdown
| [016](./entries/016-fix-eval-all-class-imbalance.md) | 2026-07-22 | Bugfix | 修复 expand_train_pool 导致的 eval_all 类别失衡（baseline 按 fraction 时序切分，issue #16） | `src/contracts/split_fault_baseline.py`, `src/contracts/contract_config.py`, `scripts/build_contract.py`, `configs/contract/`, `dvc_reliability_gate/`, `tests/`, `CLAUDE.md` |
```

在"影响域索引"表里，为以下行追加 `, 016`：
- `scripts/build_contract.py（...）` 行
- `src/contracts/` 行
- `CLAUDE.md` 行
- `configs/contract/` 行

在"横切主题"里，把第 107 行"eval 集合的类别比例是否合理，需要独立于训练侵入之外单独核查"那条补一句：

```markdown
（016 落实修复：baseline 行改为按 fraction 时序切分而非整段搬移，eval 类别比例恢复且不再与评估协议冲突）
```

- [ ] **Step 4：提交**

```bash
git add CLAUDE.md history/entries/016-fix-eval-all-class-imbalance.md history/index.md
git commit -m "[Docs]: 更新 CLAUDE.md 已知问题 + 新增 history entry 016 (#16)"
```

- [ ] **Step 5：开 PR**

```bash
git push -u origin bugfix/eval-all-class-imbalance
gh pr create --title "[Bugfix]: 修复 expand_train_pool 导致的 eval_all 类别失衡 (#16)" --body "$(cat <<'EOF'
## Summary
修复 issue #16：`expand_train_pool=true` 时故障 baseline 行整段进训练池、整段摘出 eval，导致 eval_all 正负比从 ~50:50 崩到 ~84:16。改为按时间窗时序切分（`fault_baseline_train_fraction` 默认 0.2），最早 20% 窗口进训练池、其余留 eval，比例回升至接近原始 50:50。

## 关键改动
- 新增 `split_fault_baseline_temporal` 两路时序切分函数
- `ContractConfig` 新增 `fault_baseline_train_fraction` 字段 + 范围校验
- `_write_v1` 扩容分支改用时序切分替代全有或全无搬移
- 新增 `split_fraction_mini` fixture（5 baseline 窗口）
- 重跑 RG pipeline 刷新修复后指标

## 不变量
- `train.sample_id ∩ eval_all.sample_id == ∅` 由整窗切分保证，测试锁定
- 默认 `v1.yaml`（expand_train_pool=false）与 L0/L1/L2 主链路完全不受影响

## Test plan
- [x] `pytest tests/` 全绿
- [x] `dvc repro dvc_reliability_gate/dvc.yaml` 三段重跑成功
- [x] 实测 eval_all 正负比回升（见 history/entries/016）

Closes #16
EOF
)"
```

---

## 自查（写完计划后的复核）

- **Spec 覆盖**：issue #16 根因（`_write_v1` 整段搬移）→ Task 4；可配置比例 → Task 1；切分逻辑 → Task 2；fixture 缺口 → Task 3；测试断言翻转 → Task 5；DVC 依赖 + 重跑 → Task 6；文档/history → Task 7。全覆盖。
- **类型一致性**：`split_fault_baseline_temporal(df, fraction, case_col, time_col) -> tuple[pd.DataFrame, pd.DataFrame]` 在 Task 2 定义、Task 4 调用（传 `fraction=` 关键字），签名一致。`fault_baseline_train_fraction: float` 在 Task 1 定义、Task 4 经 `cfg.fault_baseline_train_fraction` 传入 `_write_v1`，名称一致。
- **无占位符**：所有代码步骤含完整代码；fixture CSV 含真实数值与对齐的时间戳；history entry 里的 XXX 明确标注为 Task 6 实测填充，不是遗漏。
```
