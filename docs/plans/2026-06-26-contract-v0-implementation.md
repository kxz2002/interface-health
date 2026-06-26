# Contract v0 实施计划（多模态数据融合）

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 把 [2026-06-26-multimodal-contract-design.md](./2026-06-26-multimodal-contract-design.md) 设计文档落地为可跑的代码，让 `dvc repro` 端到端跑通：raw → contract parquet → DataLoader → FusionModule → Deep SVDD → 分层 AUROC。

**Architecture:** 三层架构——离线 ModalityPreprocessor（trace/api/metric/log）一次性把 raw 数据写成 contract parquet；在线 ContractDataLoader 读 parquet 返回 per-modality tensor dict；FusionModule 独立插件层把 dict 融合成统一表示供 Detector 消费。Contract 是预处理与建模之间的稳定接口墙。

**Tech Stack:** Python 3.11、PyTorch、Hydra、DVC、pyarrow（chunked metric reading）、drain3-improved（日志解析）、pytest。

**当前分支:** `feature/fusion-contract`（已有设计文档 commit `aa1ac47`）。

---

## 任务总览与依赖图

```
Phase 1: 基础设施
  T1 endpoint_to_service.yaml ──┐
  T2 configs/contract/v0.yaml   ├──→ T3 base.py (抽象基类 + contract_v0 schema)
                                │
Phase 2: 模态预处理器（并行）
  T3 ──→ T4 TracePreprocessor   ──┐
       ──→ T5 ApiPreprocessor    ──┤
       ──→ T6a MetricPreprocessor (perf spike) ──→ T6b MetricPreprocessor (impl)
       ──→ T7 LogPreprocessor    ──┤
                                   │
Phase 3: Contract 装配             ▼
  T4-T7 ──→ T8 Normalization ──→ T9 build_contract.py（端到端写 parquet）
                              ──→ T10 schema.json writer
                                   │
Phase 4: 在线层（并行，依赖 T9 产物）
  T9 ──→ T11 ContractDataLoader PointDataset
      ──→ T12 ContractDataLoader SequenceDataset
      ──→ T13 EarlyConcatFusion
                                   │
Phase 5: Baseline + 评估           ▼
  T11+T13 ──→ T14 Deep SVDD detector
           ──→ T15 train_baseline_v0.py
           ──→ T16 eval_baseline_v0.py（分层 AUROC）
                                   │
Phase 6: 集成与文档                ▼
  T15+T16 ──→ T17 dvc.yaml 三个新 stage
           ──→ T18 e2e smoke test
           ──→ T19 feature-selection-rationale-v0.md
```

**关键依赖**：
- T6a perf spike 决定 T6b 实现策略；T6a 失败需要回到设计文档调整 chunked reading 方案
- T8 归一化必须先于 T9 build_contract（fit 只用 Normal case）
- T11/T12/T13 在 T9 产出真实 parquet 后才能写有意义的集成测试

---

## Phase 1: 基础设施（T1–T3）

### Task 1: 静态映射表 `endpoint_to_service.yaml`

**Files:**
- Create: `configs/contract/endpoint_to_service.yaml`
- Test: `tests/test_endpoint_service_mapping.py`

**Step 1: Write the failing test**

```python
# tests/test_endpoint_service_mapping.py
"""验证 8 个外部 endpoint 都有 service 映射，且 service 名符合 Train-Ticket 命名。"""
from pathlib import Path
import yaml

EXPECTED_ENDPOINTS = {
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelplanservice/travelPlan/cheapest",
    "POST:/api/v1/travelplanservice/travelPlan/minStation",
    "POST:/api/v1/travelplanservice/travelPlan/quickest",
    "GET:/api/v1/routeservice/routes",
}

def test_endpoint_mapping_covers_all_v0_endpoints():
    mapping = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    assert set(mapping.keys()) == EXPECTED_ENDPOINTS

def test_endpoint_mapping_services_follow_ts_naming():
    mapping = yaml.safe_load(Path("configs/contract/endpoint_to_service.yaml").read_text())
    for endpoint, service in mapping.items():
        assert service.startswith("ts-"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"
        assert service.endswith("-service"), f"{endpoint} → {service} 不符合 Train-Ticket 命名"
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_endpoint_service_mapping.py -v`
Expected: FAIL with `FileNotFoundError`.

**Step 3: Write the YAML**

参考 `data/anomod/Normal_planA_*/trace_data/` 中的 `service_to_pod_mapping.csv` 或既有 `_pipeline_out/tt_endpoint_health_15s.csv` 推断映射关系（每个外部 endpoint 第一跳所属的服务）。

```yaml
# configs/contract/endpoint_to_service.yaml
# 8 个 v0 外部 endpoint → Train-Ticket service 静态映射
# 不依赖运行时 trace 解析，避免数据缺失时 contract 失败
"POST:/api/v1/preserveservice/preserve": ts-preserve-service
"POST:/api/v1/orderservice/order/refresh": ts-order-service
"POST:/api/v1/travelservice/trips/left": ts-travel-service
"POST:/api/v1/travel2service/trips/left": ts-travel2-service
"POST:/api/v1/travelplanservice/travelPlan/cheapest": ts-travel-plan-service
"POST:/api/v1/travelplanservice/travelPlan/minStation": ts-travel-plan-service
"POST:/api/v1/travelplanservice/travelPlan/quickest": ts-travel-plan-service
"GET:/api/v1/routeservice/routes": ts-route-service
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_endpoint_service_mapping.py -v`
Expected: PASS 2 tests.

**Step 5: Commit**

```bash
git add configs/contract/endpoint_to_service.yaml tests/test_endpoint_service_mapping.py
git commit -m "[Feature]: 静态 endpoint→service 映射表（contract v0 标识层）"
```

---

### Task 2: 特征字段配置 `configs/contract/v0.yaml`

**Files:**
- Create: `configs/contract/v0.yaml`
- Create: `src/contracts/contract_config.py`（加载并校验 YAML）
- Test: `tests/test_contract_config.py`

**Step 1: Write the failing test**

```python
# tests/test_contract_config.py
from src.contracts.contract_config import load_contract_config

def test_load_v0_config_has_three_modalities():
    cfg = load_contract_config("configs/contract/v0.yaml")
    assert set(cfg.modalities.keys()) == {"endpoint_red", "service_metric", "service_log"}

def test_v0_feature_dim_is_18():
    cfg = load_contract_config("configs/contract/v0.yaml")
    total = sum(len(m.features) for m in cfg.modalities.values())
    assert total == 18, f"feature_dim 必须是 18，当前 {total}"

def test_v0_endpoint_red_has_10_features():
    cfg = load_contract_config("configs/contract/v0.yaml")
    assert len(cfg.modalities["endpoint_red"].features) == 10

def test_normalization_scope_valid():
    cfg = load_contract_config("configs/contract/v0.yaml")
    valid_scopes = {"per_endpoint_min_max", "per_service_min_max", "global_min_max"}
    for m in cfg.modalities.values():
        assert m.normalization in valid_scopes

def test_missing_modality_field_raises():
    import pytest
    with pytest.raises(ValueError, match="modalities"):
        load_contract_config("tests/fixtures/bad_contract_no_modalities.yaml")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_contract_config.py -v`
Expected: FAIL with `ModuleNotFoundError: src.contracts.contract_config`.

**Step 3: Write minimal implementation**

参考设计文档 §9.1 的完整 YAML 内容写入 `configs/contract/v0.yaml`（包含 candidates_pool）。然后写加载器：

```python
# src/contracts/contract_config.py
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import yaml

@dataclass(frozen=True)
class ModalitySpec:
    preprocessor: str
    preprocessor_version: str
    features: list[str]
    normalization: str
    candidates_pool: list[str] = field(default_factory=list)

@dataclass(frozen=True)
class ContractConfig:
    contract_version: str
    window_size_s: int
    modalities: dict[str, ModalitySpec]

def load_contract_config(path: str | Path) -> ContractConfig:
    raw = yaml.safe_load(Path(path).read_text())
    if "modalities" not in raw:
        raise ValueError("contract config 缺少 'modalities' 字段")
    modalities = {
        name: ModalitySpec(**spec) for name, spec in raw["modalities"].items()
    }
    return ContractConfig(
        contract_version=raw["contract_version"],
        window_size_s=raw["window_size_s"],
        modalities=modalities,
    )
```

并写一个故意缺字段的 fixture：

```yaml
# tests/fixtures/bad_contract_no_modalities.yaml
contract_version: v0.0
window_size_s: 15
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_contract_config.py -v`
Expected: PASS 5 tests.

**Step 5: Commit**

```bash
git add configs/contract/v0.yaml src/contracts/contract_config.py tests/test_contract_config.py tests/fixtures/bad_contract_no_modalities.yaml
git commit -m "[Feature]: contract v0 特征字段配置 + 加载器"
```

---

### Task 3: 抽象基类 `ModalityPreprocessor` + `contract_v0` schema 校验器

**Files:**
- Create: `src/preprocessors/__init__.py`
- Create: `src/preprocessors/base.py`
- Create: `src/contracts/contract_v0.py`
- Modify: `src/contracts/__init__.py`（导出新校验器）
- Test: `tests/test_contract_v0_schema.py`

**Step 1: Write the failing test**

```python
# tests/test_contract_v0_schema.py
"""contract v0 schema：18 维特征列 + 6 个标识列 + 8 个标签列。"""
import pandas as pd
import pytest
from src.contracts.contract_v0 import (
    validate_contract_df,
    ContractV0Error,
    REQUIRED_ID_COLUMNS,
    REQUIRED_LABEL_COLUMNS,
)

def _make_valid_row():
    return {
        # ID
        "sample_id": "Normal_planA__POST:/api/v1/preserveservice/preserve__1780972185000",
        "case_id": "Normal_planA",
        "endpoint_key": "POST:/api/v1/preserveservice/preserve",
        "service_name": "ts-preserve-service",
        "timestamp_window_ms": 1780972185000,
        "window_str": "2026-06-09T02:29:45Z",
        # endpoint_red (10)
        "endpoint_red__trace_request_count": 0.5,
        "endpoint_red__trace_latency_p50": 0.3,
        "endpoint_red__trace_latency_p95": 0.4,
        "endpoint_red__trace_error_rate": 0.0,
        "endpoint_red__trace_5xx_rate": 0.0,
        "endpoint_red__client_request_count": 0.5,
        "endpoint_red__client_latency_p95": 0.4,
        "endpoint_red__client_error_rate": 0.0,
        "endpoint_red__client_5xx_rate": 0.0,
        "endpoint_red__latency_divergence": 0.1,
        # service_metric (5)
        "service_metric__cpu_usage_rate": 0.2,
        "service_metric__memory_usage_ratio": 0.3,
        "service_metric__net_rx_error_rate": 0.0,
        "service_metric__net_tx_error_rate": 0.0,
        "service_metric__process_count": 0.5,
        # service_log (3)
        "service_log__event_rate": 0.4,
        "service_log__error_ratio": 0.0,
        "service_log__template_diversity": 0.2,
        # Label
        "phase": "normal",
        "is_anomaly": False,
        "is_train_eligible": True,
        "injection_start_ms": None,
        "injection_end_ms": None,
        "target_service": None,
        "anomaly_type": "Normal",
        "anomaly_level": "none",
    }

def test_valid_contract_passes():
    df = pd.DataFrame([_make_valid_row()])
    validated = validate_contract_df(df, "configs/contract/v0.yaml")
    assert len(validated) == 1

def test_missing_feature_column_raises():
    row = _make_valid_row()
    del row["service_log__event_rate"]
    with pytest.raises(ContractV0Error, match="service_log__event_rate"):
        validate_contract_df(pd.DataFrame([row]), "configs/contract/v0.yaml")

def test_score_out_of_range_for_rate_column_raises():
    """error_rate / 5xx_rate / memory_usage_ratio 必须在 [0, 1]。"""
    row = _make_valid_row()
    row["endpoint_red__trace_error_rate"] = 1.5
    with pytest.raises(ContractV0Error, match=r"trace_error_rate.*\[0"):
        validate_contract_df(pd.DataFrame([row]), "configs/contract/v0.yaml")

def test_duplicate_sample_id_raises():
    df = pd.DataFrame([_make_valid_row(), _make_valid_row()])
    with pytest.raises(ContractV0Error, match="duplicate"):
        validate_contract_df(df, "configs/contract/v0.yaml")

def test_inconsistent_anomaly_label_raises():
    """is_anomaly 必须等价于 phase == 'inject'。"""
    row = _make_valid_row()
    row["phase"] = "inject"
    row["is_anomaly"] = False  # 矛盾
    with pytest.raises(ContractV0Error, match="phase.*is_anomaly"):
        validate_contract_df(pd.DataFrame([row]), "configs/contract/v0.yaml")
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_contract_v0_schema.py -v`
Expected: FAIL with `ModuleNotFoundError: src.contracts.contract_v0`.

**Step 3: Write minimal implementation**

`src/preprocessors/base.py`：

```python
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import pandas as pd

class ModalityPreprocessor(ABC):
    version: str = "v0"

    @abstractmethod
    def fit(self, raw_paths: list[Path]) -> None: ...

    @abstractmethod
    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame: ...

    @abstractmethod
    def get_feature_columns(self) -> list[str]: ...
```

`src/contracts/contract_v0.py`（沿用 `scores_v0.py` 的"一次性收集所有错误"风格）：

```python
"""Contract v0：多模态融合数据表的格式约定。

详见 docs/plans/2026-06-26-multimodal-contract-design.md §4。
"""
from __future__ import annotations
from typing import NewType
import pandas as pd
from src.contracts.contract_config import load_contract_config

ContractV0 = NewType("ContractV0", pd.DataFrame)

REQUIRED_ID_COLUMNS = [
    "sample_id", "case_id", "endpoint_key", "service_name",
    "timestamp_window_ms", "window_str",
]
REQUIRED_LABEL_COLUMNS = [
    "phase", "is_anomaly", "is_train_eligible",
    "injection_start_ms", "injection_end_ms", "target_service",
    "anomaly_type", "anomaly_level",
]
RATE_COLUMNS = [
    "endpoint_red__trace_error_rate", "endpoint_red__trace_5xx_rate",
    "endpoint_red__client_error_rate", "endpoint_red__client_5xx_rate",
    "service_metric__cpu_usage_rate", "service_metric__memory_usage_ratio",
    "service_log__error_ratio",
]

class ContractV0Error(ValueError):
    """Contract v0 校验失败时抛出。错误信息会列出所有发现的问题。"""

def validate_contract_df(df: pd.DataFrame, config_path: str) -> ContractV0:
    cfg = load_contract_config(config_path)
    feature_cols = [
        f"{mod}__{feat}"
        for mod, spec in cfg.modalities.items()
        for feat in spec.features
    ]
    expected = REQUIRED_ID_COLUMNS + feature_cols + REQUIRED_LABEL_COLUMNS

    errors: list[str] = []

    missing = [c for c in expected if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        raise ContractV0Error("contract v0 violations: " + "; ".join(errors))

    if df["sample_id"].duplicated().any():
        errors.append(f"sample_id has {int(df['sample_id'].duplicated().sum())} duplicate values")

    for col in RATE_COLUMNS:
        vals = df[col].dropna()
        if ((vals < 0) | (vals > 1)).any():
            errors.append(f"{col} 越界 [0, 1]，发现 {((vals < 0) | (vals > 1)).sum()} 行")

    inconsistent = (df["phase"] == "inject") != df["is_anomaly"]
    if inconsistent.any():
        errors.append(
            f"phase 与 is_anomaly 不一致：{int(inconsistent.sum())} 行 "
            "（is_anomaly 必须等价于 phase == 'inject'）"
        )

    if errors:
        raise ContractV0Error("contract v0 violations: " + "; ".join(errors))

    return ContractV0(df)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_contract_v0_schema.py -v`
Expected: PASS 5 tests.

**Step 5: Commit**

```bash
git add src/preprocessors/__init__.py src/preprocessors/base.py src/contracts/contract_v0.py src/contracts/__init__.py tests/test_contract_v0_schema.py
git commit -m "[Feature]: ModalityPreprocessor 抽象基类 + contract v0 schema 校验器"
```

---

## Phase 2: 模态预处理器（T4–T7）

> ⚠ **依赖说明**：T4/T5/T7 之间无依赖，可在不同 worktree 并行开发。T6 因数据量大，要先做 perf spike。

### Task 4: TracePreprocessor@v0（复用 `tt_traces_red_15s.csv`）

**Files:**
- Create: `src/preprocessors/trace_preprocessor.py`
- Test: `tests/test_trace_preprocessor.py`
- Test fixture: `tests/fixtures/mini_tt_traces_red_15s.csv`（手造 6 行：2 endpoint × 3 window）

**Step 1: Write the failing test**

```python
# tests/test_trace_preprocessor.py
import pandas as pd
from pathlib import Path
from src.preprocessors.trace_preprocessor import TracePreprocessor

V0_ENDPOINTS = {
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/orderservice/order/refresh",
    # ... 8 个完整列表
}

def test_trace_output_columns_match_contract():
    pre = TracePreprocessor()
    cols = pre.get_feature_columns()
    expected = {
        "endpoint_red__trace_request_count",
        "endpoint_red__trace_latency_p50",
        "endpoint_red__trace_latency_p95",
        "endpoint_red__trace_error_rate",
        "endpoint_red__trace_5xx_rate",
    }
    assert set(cols) == expected

def test_trace_transform_filters_to_v0_endpoints(tmp_path):
    """非 v0 endpoint 的行必须被丢弃。"""
    pre = TracePreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_tt_traces_red_15s.csv"),
        case_meta={"case_id": "Normal_planA"},
    )
    assert set(df["endpoint_key"].unique()).issubset(V0_ENDPOINTS)

def test_trace_transform_preserves_window_alignment(tmp_path):
    pre = TracePreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_tt_traces_red_15s.csv"),
        case_meta={"case_id": "Normal_planA"},
    )
    # 时间戳必须是 15s 的整数倍
    assert (df["timestamp_window_ms"] % 15_000 == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_trace_preprocessor.py -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

先看真实文件确认列名：
```bash
head -1 data/anomod/Normal_planA_*/_pipeline_out/tt_traces_red_15s.csv
```

然后实现：

```python
# src/preprocessors/trace_preprocessor.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import pandas as pd
import yaml
from src.preprocessors.base import ModalityPreprocessor

class TracePreprocessor(ModalityPreprocessor):
    """v0：从已 pipeline 化的 tt_traces_red_15s.csv 抽取 5 维 trace RED 特征。"""

    version = "v0"
    OUTPUT_COLUMNS = [
        "endpoint_red__trace_request_count",
        "endpoint_red__trace_latency_p50",
        "endpoint_red__trace_latency_p95",
        "endpoint_red__trace_error_rate",
        "endpoint_red__trace_5xx_rate",
    ]

    def __init__(self, endpoint_mapping_path: str = "configs/contract/endpoint_to_service.yaml"):
        self._v0_endpoints = set(yaml.safe_load(Path(endpoint_mapping_path).read_text()).keys())

    def fit(self, raw_paths: list[Path]) -> None:
        return  # v0：归一化由独立 Normalization 步骤负责

    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        raw = pd.read_csv(raw_path)
        df = raw[raw["endpoint_key"].isin(self._v0_endpoints)].copy()
        df = df.rename(columns={
            "request_count": "endpoint_red__trace_request_count",
            "latency_p50_ms": "endpoint_red__trace_latency_p50",
            "latency_p95_ms": "endpoint_red__trace_latency_p95",
            "error_rate": "endpoint_red__trace_error_rate",
            "http_5xx_rate": "endpoint_red__trace_5xx_rate",
        })
        return df[["endpoint_key", "timestamp_window_ms"] + self.OUTPUT_COLUMNS]

    def get_feature_columns(self) -> list[str]:
        return list(self.OUTPUT_COLUMNS)
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_trace_preprocessor.py -v`
Expected: PASS 3 tests.

**Step 5: Commit**

```bash
git add src/preprocessors/trace_preprocessor.py tests/test_trace_preprocessor.py tests/fixtures/mini_tt_traces_red_15s.csv
git commit -m "[Feature]: TracePreprocessor v0（trace RED 5 维抽取）"
```

---

### Task 5: ApiPreprocessor@v0（合并入 endpoint_red）

**Files:**
- Create: `src/preprocessors/api_preprocessor.py`
- Test: `tests/test_api_preprocessor.py`
- Test fixture: `tests/fixtures/mini_tt_endpoint_health_15s.csv`

**Step 1: Write the failing test**

```python
# tests/test_api_preprocessor.py
from pathlib import Path
from src.preprocessors.api_preprocessor import ApiPreprocessor

def test_api_output_columns():
    pre = ApiPreprocessor()
    assert set(pre.get_feature_columns()) == {
        "endpoint_red__client_request_count",
        "endpoint_red__client_latency_p95",
        "endpoint_red__client_error_rate",
        "endpoint_red__client_5xx_rate",
        "endpoint_red__latency_divergence",
    }

def test_api_transform_computes_latency_divergence():
    """latency_divergence = client_p95 − trace_p95，要求 trace 数据 join 后存在。"""
    pre = ApiPreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_tt_endpoint_health_15s.csv"),
        case_meta={"case_id": "Normal_planA"},
    )
    assert "endpoint_red__latency_divergence" in df.columns
    # 至少一行非空
    assert df["endpoint_red__latency_divergence"].notna().any()

def test_api_transform_inner_joins_8_endpoints():
    """tt_endpoint_health 已经是 client+trace inner join 的产物，行数 ≤ 8 × n_windows。"""
    pre = ApiPreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_tt_endpoint_health_15s.csv"),
        case_meta={"case_id": "Normal_planA"},
    )
    assert df["endpoint_key"].nunique() <= 8
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_api_preprocessor.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

参考 T4 的写法，加上 `latency_divergence` 派生列计算（`client_latency_p95 - trace_latency_p95`）。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_api_preprocessor.py -v`
Expected: PASS 3 tests.

**Step 5: Commit**

```bash
git add src/preprocessors/api_preprocessor.py tests/test_api_preprocessor.py tests/fixtures/mini_tt_endpoint_health_15s.csv
git commit -m "[Feature]: ApiPreprocessor v0（client RED + latency_divergence）"
```

---

### Task 6a: MetricPreprocessor 性能 Spike（验证 chunked reading 内存 < 4GB）

**Why a spike first**: 单 case metric CSV 1.3–3.0GB，全量 read_csv 会爆内存。设计文档要求 pyarrow chunked + window 预过滤，但实测可行性必须先用一个最大 case 验证。

**Files:**
- Create: `scripts/spikes/metric_perf_spike.py`（一次性脚本，不进 src）
- Create: `tests/test_metric_perf_spike.py`（验证脚本的 memory_profiler 输出）

**Step 1: Write the failing test**

```python
# tests/test_metric_perf_spike.py
"""Perf spike：在最大的 metric_data CSV 上跑 chunked reading，验证峰值内存 < 4GB。"""
import json
import subprocess
import sys
from pathlib import Path
import pytest

# 找到 metric_data 下最大的 csv
def _largest_metric_csv() -> Path:
    candidates = list(Path("data/anomod").glob("*/metric_data/*.csv"))
    if not candidates:
        pytest.skip("没有 metric_data CSV，跳过 perf spike")
    return max(candidates, key=lambda p: p.stat().st_size)

@pytest.mark.slow
def test_metric_chunked_reading_under_4gb():
    csv = _largest_metric_csv()
    out = Path("artifacts/_spike_metric.json")
    out.unlink(missing_ok=True)
    subprocess.run(
        [sys.executable, "scripts/spikes/metric_perf_spike.py",
         "--input", str(csv),
         "--window-start-ms", "0",
         "--window-end-ms", "9999999999999",
         "--out", str(out)],
        check=True,
    )
    report = json.loads(out.read_text())
    assert report["peak_memory_mb"] < 4096, (
        f"chunked reading 峰值 {report['peak_memory_mb']:.1f} MB 超过 4GB 上限"
    )
    assert report["rows_kept"] > 0
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metric_perf_spike.py -v -m slow`
Expected: FAIL (spike 脚本不存在)。

**Step 3: Write the spike script**

```python
# scripts/spikes/metric_perf_spike.py
"""Metric chunked reading 性能 spike。

读 metric_data CSV，按 window 预过滤，返回峰值内存 & 保留行数。
仅用于验证 4GB 上限，不进 src。
"""
import argparse
import json
import resource
from pathlib import Path
import pyarrow.csv as pacsv
import pyarrow.compute as pc

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--window-start-ms", type=int, required=True)
    parser.add_argument("--window-end-ms", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows_kept = 0
    reader = pacsv.open_csv(args.input, read_options=pacsv.ReadOptions(block_size=64 << 20))
    for batch in reader:
        ts_col_name = "datetime"  # 探查报告确认
        ts = pc.cast(batch[ts_col_name], "int64")
        mask = pc.and_(pc.greater_equal(ts, args.window_start_ms),
                       pc.less_equal(ts, args.window_end_ms))
        rows_kept += int(pc.sum(mask).as_py() or 0)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mb = peak_kb / 1024 if peak_kb > 100_000 else peak_kb  # macOS bytes, linux kb

    Path(args.out).write_text(json.dumps({
        "peak_memory_mb": peak_mb,
        "rows_kept": rows_kept,
        "input": args.input,
    }))

if __name__ == "__main__":
    main()
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metric_perf_spike.py -v -m slow`
Expected: PASS with peak memory report logged.

**Step 5: Commit**

```bash
git add scripts/spikes/metric_perf_spike.py tests/test_metric_perf_spike.py
git commit -m "[Chore]: metric chunked reading 性能 spike（验证 <4GB 内存上限）"
```

> **决策点**：若 spike 失败（>4GB），不要直接进 T6b，先和决策者讨论是否改用 DuckDB / 列裁剪策略，更新设计文档 §5.4 后再继续。

---

### Task 6b: MetricPreprocessor@v0 正式实现

**Files:**
- Create: `src/preprocessors/metric_preprocessor.py`
- Test: `tests/test_metric_preprocessor.py`
- Test fixture: `tests/fixtures/mini_metric_data.csv`（手造 ~200 行覆盖 5 个指标 × 2 service × 多窗口）

**Step 1: Write the failing test**

```python
# tests/test_metric_preprocessor.py
from pathlib import Path
import pandas as pd
import pytest
from src.preprocessors.metric_preprocessor import MetricPreprocessor

def test_metric_output_columns():
    pre = MetricPreprocessor()
    assert set(pre.get_feature_columns()) == {
        "service_metric__cpu_usage_rate",
        "service_metric__memory_usage_ratio",
        "service_metric__net_rx_error_rate",
        "service_metric__net_tx_error_rate",
        "service_metric__process_count",
    }

def test_metric_transform_aggregates_to_service_level():
    """同 service 多 pod 必须聚合（mean），输出按 (service, window) 唯一。"""
    pre = MetricPreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_metric_data.csv"),
        case_meta={
            "case_id": "Normal_planA",
            "trace_window_start_ms": 1_780_972_185_000,
            "trace_window_end_ms": 1_780_972_290_000,
        },
    )
    # (service_name, timestamp_window_ms) 唯一
    assert not df.duplicated(["service_name", "timestamp_window_ms"]).any()

def test_metric_uses_intermediate_cache(tmp_path):
    """同一 case 第二次 transform 应直接读 artifacts/intermediate/。"""
    pre = MetricPreprocessor(intermediate_dir=tmp_path)
    case_meta = {"case_id": "Normal_planA", "trace_window_start_ms": 0, "trace_window_end_ms": 10**13}
    pre.transform(Path("tests/fixtures/mini_metric_data.csv"), case_meta)
    assert (tmp_path / "metrics_filtered_Normal_planA.parquet").exists()

def test_metric_window_floor_alignment():
    """metric 时间戳必须 floor 对齐到 trace window 起点的 15s 桶。"""
    pre = MetricPreprocessor()
    df = pre.transform(
        Path("tests/fixtures/mini_metric_data.csv"),
        case_meta={"case_id": "Normal_planA",
                   "trace_window_start_ms": 1_780_972_185_000,
                   "trace_window_end_ms": 1_780_972_290_000},
    )
    assert (df["timestamp_window_ms"] % 15_000 == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_metric_preprocessor.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

按 spike 验证的方式：pyarrow chunked + window 预过滤 + 中间 parquet 缓存 + counter 类指标做 rate + (service, window) groupby mean。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_metric_preprocessor.py -v`
Expected: PASS 4 tests.

**Step 5: Commit**

```bash
git add src/preprocessors/metric_preprocessor.py tests/test_metric_preprocessor.py tests/fixtures/mini_metric_data.csv
git commit -m "[Feature]: MetricPreprocessor v0（5 指标 + rate + service 聚合 + 中间缓存）"
```

---

### Task 7: LogPreprocessor@v0（Drain3 + 频率统计）

**Files:**
- Create: `src/preprocessors/log_preprocessor.py`
- Test: `tests/test_log_preprocessor.py`
- Test fixture: `tests/fixtures/mini_logs/ts-route-service/sample.log`（手造 50 行，含 ERROR/INFO 混合）

**Step 1: Write the failing test**

```python
# tests/test_log_preprocessor.py
from pathlib import Path
from src.preprocessors.log_preprocessor import LogPreprocessor

def test_log_output_columns():
    pre = LogPreprocessor()
    assert set(pre.get_feature_columns()) == {
        "service_log__event_rate",
        "service_log__error_ratio",
        "service_log__template_diversity",
    }

def test_log_fit_only_uses_normal_logs(tmp_path):
    """Drain3 模板必须只从 Normal case 训练。"""
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([Path("tests/fixtures/mini_logs/ts-route-service/sample.log")])
    assert (tmp_path / "drain.bin").exists()

def test_log_transform_outputs_service_window_rows(tmp_path):
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([Path("tests/fixtures/mini_logs/ts-route-service/sample.log")])
    df = pre.transform(
        Path("tests/fixtures/mini_logs"),  # 整个 case 的 log 根目录
        case_meta={"case_id": "Normal_planA"},
    )
    assert "service_name" in df.columns
    assert "timestamp_window_ms" in df.columns

def test_log_error_ratio_when_no_events_is_zero(tmp_path):
    """无事件窗口的 error_ratio 必须是 0 而非 NaN。"""
    pre = LogPreprocessor(drain3_state_path=tmp_path / "drain.bin")
    pre.fit([Path("tests/fixtures/mini_logs/ts-route-service/sample.log")])
    df = pre.transform(
        Path("tests/fixtures/mini_logs"),
        case_meta={"case_id": "Normal_planA"},
    )
    empty_rows = df[df["service_log__event_rate"] == 0]
    assert (empty_rows["service_log__error_ratio"] == 0).all()
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_log_preprocessor.py -v`
Expected: FAIL.

**Step 3: Write minimal implementation**

依赖 drain3-improved，按 service 目录遍历 `.log` 文件，逐行抽时间戳和等级，过 Drain3 拿模板 id，再按 (service, 15s window) 聚合。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_log_preprocessor.py -v`
Expected: PASS 4 tests.

**Step 5: Commit**

```bash
git add src/preprocessors/log_preprocessor.py tests/test_log_preprocessor.py tests/fixtures/mini_logs/
git commit -m "[Feature]: LogPreprocessor v0（Drain3 模板 + 频率/错误率/多样性）"
```

---

## Phase 3: Contract 装配（T8–T10）

### Task 8: Normalization（per-endpoint / per-service min-max fit/transform）

**Files:**
- Create: `src/data/__init__.py`
- Create: `src/data/normalization.py`
- Test: `tests/test_normalization.py`

**Step 1: Write the failing test**

```python
# tests/test_normalization.py
import json
from pathlib import Path
import pandas as pd
import pytest
from src.data.normalization import Normalizer

@pytest.fixture
def normal_df():
    return pd.DataFrame({
        "endpoint_key": ["ep1"] * 4 + ["ep2"] * 4,
        "endpoint_red__trace_request_count": [0, 10, 20, 30, 100, 200, 300, 400],
    })

def test_per_endpoint_minmax_fit_transform(normal_df):
    norm = Normalizer(
        rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")},
    )
    norm.fit(normal_df)
    out = norm.transform(normal_df)
    # 每个 endpoint 独立缩放到 [0, 1]
    assert out.loc[normal_df["endpoint_key"] == "ep1", "endpoint_red__trace_request_count"].min() == 0
    assert out.loc[normal_df["endpoint_key"] == "ep1", "endpoint_red__trace_request_count"].max() == 1

def test_anomaly_case_uses_normal_stats(normal_df):
    """故障 case 用 Normal 拟合的参数，超过 1 的值允许出现。"""
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    anomaly_df = pd.DataFrame({
        "endpoint_key": ["ep1"],
        "endpoint_red__trace_request_count": [60],  # 超出 Normal 的 30
    })
    out = norm.transform(anomaly_df)
    assert out["endpoint_red__trace_request_count"].iloc[0] == 2.0  # (60 - 0) / (30 - 0)

def test_stats_roundtrip_json(tmp_path, normal_df):
    norm = Normalizer(rules={"endpoint_red__trace_request_count": ("per_endpoint", "min_max")})
    norm.fit(normal_df)
    norm.save(tmp_path / "stats.json")
    norm2 = Normalizer.load(tmp_path / "stats.json")
    pd.testing.assert_frame_equal(norm.transform(normal_df), norm2.transform(normal_df))
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_normalization.py -v`
Expected: FAIL with `ModuleNotFoundError`.

**Step 3: Write minimal implementation**

```python
# src/data/normalization.py
from __future__ import annotations
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd

Scope = Literal["per_endpoint", "per_service", "global"]
Method = Literal["min_max"]

@dataclass
class _Stats:
    scope: Scope
    method: Method
    # key=group_value, value=(min, max)；scope=global 时 key 为 "__global__"
    by_group: dict[str, tuple[float, float]]

class Normalizer:
    def __init__(self, rules: dict[str, tuple[Scope, Method]]):
        self.rules = rules
        self._stats: dict[str, _Stats] = {}

    def fit(self, df: pd.DataFrame) -> None:
        for col, (scope, method) in self.rules.items():
            group_col = {"per_endpoint": "endpoint_key",
                         "per_service": "service_name",
                         "global": None}[scope]
            by_group: dict[str, tuple[float, float]] = {}
            if group_col is None:
                by_group["__global__"] = (float(df[col].min()), float(df[col].max()))
            else:
                for g, sub in df.groupby(group_col):
                    by_group[str(g)] = (float(sub[col].min()), float(sub[col].max()))
            self._stats[col] = _Stats(scope=scope, method=method, by_group=by_group)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        for col, stats in self._stats.items():
            group_col = {"per_endpoint": "endpoint_key",
                         "per_service": "service_name",
                         "global": None}[stats.scope]
            if group_col is None:
                lo, hi = stats.by_group["__global__"]
                out[col] = (out[col] - lo) / max(hi - lo, 1e-9)
            else:
                def _scale(row):
                    lo, hi = stats.by_group.get(str(row[group_col]), (0.0, 1.0))
                    return (row[col] - lo) / max(hi - lo, 1e-9)
                out[col] = out.apply(_scale, axis=1)
        return out

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({
            col: {"scope": s.scope, "method": s.method, "by_group": s.by_group}
            for col, s in self._stats.items()
        }, indent=2))

    @classmethod
    def load(cls, path: Path) -> "Normalizer":
        raw = json.loads(Path(path).read_text())
        rules = {col: (spec["scope"], spec["method"]) for col, spec in raw.items()}
        norm = cls(rules)
        norm._stats = {
            col: _Stats(spec["scope"], spec["method"],
                       {k: tuple(v) for k, v in spec["by_group"].items()})
            for col, spec in raw.items()
        }
        return norm
```

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_normalization.py -v`
Expected: PASS 3 tests.

**Step 5: Commit**

```bash
git add src/data/__init__.py src/data/normalization.py tests/test_normalization.py
git commit -m "[Feature]: Normalizer（per-endpoint/per-service min-max fit/transform）"
```

---

### Task 9: `build_contract.py` 端到端离线 pipeline

**Files:**
- Create: `scripts/build_contract.py`
- Test: `tests/test_build_contract_smoke.py`

**Step 1: Write the failing test**

```python
# tests/test_build_contract_smoke.py
"""build_contract.py 端到端 smoke test：跑一个 mini case，检查产物结构。"""
import subprocess
import sys
from pathlib import Path
import pandas as pd
from src.contracts.contract_v0 import validate_contract_df

def test_build_contract_produces_valid_parquet(tmp_path):
    out_dir = tmp_path / "contract_v0"
    subprocess.run(
        [sys.executable, "scripts/build_contract.py",
         "--config", "configs/contract/v0.yaml",
         "--data-root", "tests/fixtures/mini_data_root",  # 含 Normal + 1 个故障 case 的子集
         "--out-dir", str(out_dir),
         "--seed", "42"],
        check=True,
    )
    assert (out_dir / "train.parquet").exists()
    assert (out_dir / "eval_all.parquet").exists()
    assert (out_dir / "normalization_stats.json").exists()
    assert (out_dir / "schema.json").exists()

    train = pd.read_parquet(out_dir / "train.parquet")
    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    validate_contract_df(train, "configs/contract/v0.yaml")
    validate_contract_df(eval_all, "configs/contract/v0.yaml")
    # train 只含 Normal
    assert (train["case_id"].str.startswith("Normal")).all()
    # eval_all 含至少 2 个 case
    assert eval_all["case_id"].nunique() >= 2

def test_build_contract_is_reproducible(tmp_path):
    """同 seed 跑两次，parquet 内容 hash 一致。"""
    import hashlib
    hashes = []
    for run in range(2):
        out_dir = tmp_path / f"run_{run}"
        subprocess.run(
            [sys.executable, "scripts/build_contract.py",
             "--config", "configs/contract/v0.yaml",
             "--data-root", "tests/fixtures/mini_data_root",
             "--out-dir", str(out_dir),
             "--seed", "42"],
            check=True,
        )
        hashes.append(hashlib.sha256((out_dir / "train.parquet").read_bytes()).hexdigest())
    assert hashes[0] == hashes[1]
```

**Step 2: Run test to verify it fails**

Run: `pytest tests/test_build_contract_smoke.py -v`
Expected: FAIL with `FileNotFoundError`.

**Step 3: Write minimal implementation**

```python
# scripts/build_contract.py
"""离线 pipeline：raw data → contract parquet。

流程：
1. 枚举 data_root 下所有 case 目录
2. 每个 case 跑 4 个 preprocessor，按 (case, endpoint, window) join
3. 用 Normal case fit Normalizer，对所有 case transform
4. 拼标识列 + 特征列 + 标签列 → DataFrame
5. 校验 contract，写 train.parquet (Normal) + eval_all.parquet (all)
"""
from __future__ import annotations
import argparse, json, logging
from pathlib import Path
import numpy as np
import pandas as pd
from src.contracts.contract_config import load_contract_config
from src.contracts.contract_v0 import validate_contract_df
from src.data.normalization import Normalizer
from src.preprocessors.trace_preprocessor import TracePreprocessor
from src.preprocessors.api_preprocessor import ApiPreprocessor
from src.preprocessors.metric_preprocessor import MetricPreprocessor
from src.preprocessors.log_preprocessor import LogPreprocessor
from src.utils.seed import set_seed  # 已有

LOG = logging.getLogger(__name__)

def _enumerate_cases(data_root: Path) -> list[Path]: ...
def _process_one_case(case_dir: Path, cfg, mapping, ...) -> pd.DataFrame: ...
def _build_normalizer_rules(cfg) -> dict: ...

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = load_contract_config(args.config)
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    # 1. 跑所有 case 的预处理
    all_frames = []
    for case_dir in _enumerate_cases(Path(args.data_root)):
        df = _process_one_case(case_dir, cfg, ...)
        all_frames.append(df)
    full = pd.concat(all_frames, ignore_index=True).sort_values(["case_id", "endpoint_key", "timestamp_window_ms"])

    # 2. Normal-only fit normalizer
    normal_mask = full["case_id"].str.startswith("Normal")
    normalizer = Normalizer(_build_normalizer_rules(cfg))
    normalizer.fit(full[normal_mask])
    full = normalizer.transform(full)
    normalizer.save(out / "normalization_stats.json")

    # 3. 校验 + 写 parquet
    validate_contract_df(full, args.config)
    full[normal_mask].to_parquet(out / "train.parquet", index=False)
    full.to_parquet(out / "eval_all.parquet", index=False)

    # 4. schema.json
    (out / "schema.json").write_text(json.dumps({
        "contract_version": cfg.contract_version,
        "window_size_s": cfg.window_size_s,
        "feature_dim": sum(len(m.features) for m in cfg.modalities.values()),
        "feature_groups": {
            name: {
                "columns": [f"{name}__{f}" for f in m.features],
                "preprocessor": f"{m.preprocessor}@{m.preprocessor_version}",
                "normalization_scope": m.normalization,
            }
            for name, m in cfg.modalities.items()
        },
        "evaluation_strata": ["overall", "by_anomaly_type", "by_anomaly_level"],
    }, indent=2))

if __name__ == "__main__":
    main()
```

> 创建 `tests/fixtures/mini_data_root/` 时只需要 2 个 case（Normal + 1 故障），每个 case 含 4 个模态的极简文件（~10 行）。

**Step 4: Run test to verify it passes**

Run: `pytest tests/test_build_contract_smoke.py -v`
Expected: PASS 2 tests.

**Step 5: Commit**

```bash
git add scripts/build_contract.py tests/test_build_contract_smoke.py tests/fixtures/mini_data_root/
git commit -m "[Feature]: build_contract.py 离线 pipeline（raw → contract parquet）"
```

---

### Task 10: schema.json 独立校验测试

> T9 已经写 schema.json，此处单独加测试确保字段稳定性，避免下游消费者被破坏。

**Files:**
- Test: `tests/test_schema_json_contract.py`

```python
# tests/test_schema_json_contract.py
import json
from pathlib import Path

REQUIRED_KEYS = {"contract_version", "window_size_s", "feature_dim", "feature_groups", "evaluation_strata"}

def test_schema_json_keys_stable(tmp_path):
    import subprocess, sys
    out = tmp_path / "contract_v0"
    subprocess.run([sys.executable, "scripts/build_contract.py",
                    "--config", "configs/contract/v0.yaml",
                    "--data-root", "tests/fixtures/mini_data_root",
                    "--out-dir", str(out), "--seed", "1"], check=True)
    schema = json.loads((out / "schema.json").read_text())
    assert REQUIRED_KEYS.issubset(schema.keys())
    assert schema["feature_dim"] == 18
    assert set(schema["feature_groups"].keys()) == {"endpoint_red", "service_metric", "service_log"}
```

Run / commit 同 T9 风格。

```bash
git add tests/test_schema_json_contract.py
git commit -m "[Test]: schema.json 字段稳定性校验"
```

---

## Phase 4: 在线层（T11–T13）

> ⚠ 这三个任务可在 T9 产出真实 contract parquet 后并行开发。

### Task 11: `ContractDataLoader` PointDataset 模式

**Files:**
- Create: `src/data/contract_dataloader.py`
- Test: `tests/test_dataloader_point.py`

**Step 1: Write the failing test**

```python
# tests/test_dataloader_point.py
import torch
import pandas as pd
import pytest
from src.data.contract_dataloader import ContractDataset

@pytest.fixture
def tiny_contract(tmp_path):
    import subprocess, sys
    out = tmp_path / "contract"
    subprocess.run([sys.executable, "scripts/build_contract.py",
                    "--config", "configs/contract/v0.yaml",
                    "--data-root", "tests/fixtures/mini_data_root",
                    "--out-dir", str(out), "--seed", "1"], check=True)
    return out

def test_point_dataset_returns_modality_dict(tiny_contract):
    ds = ContractDataset(
        parquet_path=tiny_contract / "train.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="point",
    )
    sample = ds[0]
    assert set(sample.keys()) >= {"endpoint_red", "service_metric", "service_log", "label", "meta"}
    assert sample["endpoint_red"].shape == (10,)
    assert sample["service_metric"].shape == (5,)
    assert sample["service_log"].shape == (3,)
    assert isinstance(sample["endpoint_red"], torch.Tensor)

def test_point_nan_mean_impute(tiny_contract):
    """NaN 用训练均值填补，输出不再有 NaN。"""
    df = pd.read_parquet(tiny_contract / "train.parquet")
    df.loc[0, "service_log__event_rate"] = float("nan")
    df.to_parquet(tiny_contract / "train.parquet", index=False)
    ds = ContractDataset(
        parquet_path=tiny_contract / "train.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="point", nan_strategy="mean",
    )
    sample = ds[0]
    assert not torch.isnan(sample["service_log"]).any()

def test_point_label_carries_phase_and_anomaly(tiny_contract):
    ds = ContractDataset(
        parquet_path=tiny_contract / "eval_all.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="point",
    )
    sample = ds[0]
    assert "phase" in sample["label"]
    assert "is_anomaly" in sample["label"]
```

**Step 2/3/4: 实现 → 测试通过**

`ContractDataset` 是 `torch.utils.data.Dataset`，构造时从 schema.json 拿到三个 modality 的列名分组。

```bash
git add src/data/contract_dataloader.py tests/test_dataloader_point.py
git commit -m "[Feature]: ContractDataset Point 模式（per-modality tensor dict）"
```

---

### Task 12: `ContractDataLoader` SequenceDataset 模式

**Files:**
- Modify: `src/data/contract_dataloader.py`
- Test: `tests/test_dataloader_sequence.py`

**Step 1: Write the failing test**

```python
# tests/test_dataloader_sequence.py
import torch
from src.data.contract_dataloader import ContractDataset

def test_sequence_dataset_shape(tiny_contract):
    ds = ContractDataset(
        parquet_path=tiny_contract / "eval_all.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="sequence", sequence_length=4,
    )
    sample = ds[0]
    assert sample["endpoint_red"].shape == (4, 10)
    assert sample["service_metric"].shape == (4, 5)
    assert sample["service_log"].shape == (4, 3)

def test_sequence_never_crosses_case_boundary(tiny_contract):
    """序列窗口不能跨 (case_id, endpoint_key) 边界。"""
    ds = ContractDataset(
        parquet_path=tiny_contract / "eval_all.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="sequence", sequence_length=4,
    )
    for i in range(len(ds)):
        sample = ds[i]
        assert len(set(sample["meta"]["case_id_per_step"])) == 1
        assert len(set(sample["meta"]["endpoint_key_per_step"])) == 1

def test_sequence_skips_groups_shorter_than_length(tiny_contract):
    """组长度 < sequence_length 必须被跳过，不能 padding。"""
    ds = ContractDataset(
        parquet_path=tiny_contract / "eval_all.parquet",
        schema_path=tiny_contract / "schema.json",
        mode="sequence", sequence_length=1000,  # 故意设很大
    )
    assert len(ds) == 0
```

**Step 2/3/4**: 用 `(case_id, endpoint_key)` groupby + 组内滑窗实现，不跨组、不 padding。

```bash
git add src/data/contract_dataloader.py tests/test_dataloader_sequence.py
git commit -m "[Feature]: ContractDataset Sequence 模式（组内滑窗，禁止跨 case 边界）"
```

---

### Task 13: `EarlyConcatFusion`（v0 baseline 融合）

**Files:**
- Create: `src/fusion/__init__.py`
- Create: `src/fusion/base.py`
- Create: `src/fusion/early_concat.py`
- Test: `tests/test_early_concat_fusion.py`

**Step 1: Write the failing test**

```python
# tests/test_early_concat_fusion.py
import torch
from src.fusion.early_concat import EarlyConcatFusion

def test_concat_output_dim_18():
    fusion = EarlyConcatFusion(modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3})
    assert fusion.output_dim == 18

def test_concat_forward_shape():
    fusion = EarlyConcatFusion(modality_dims={"endpoint_red": 10, "service_metric": 5, "service_log": 3})
    batch = {
        "endpoint_red": torch.randn(4, 10),
        "service_metric": torch.randn(4, 5),
        "service_log": torch.randn(4, 3),
    }
    out = fusion(batch)
    assert out.shape == (4, 18)

def test_concat_order_is_deterministic():
    """输出顺序必须固定，否则下游模型层会错位。"""
    fusion = EarlyConcatFusion(modality_dims={"endpoint_red": 2, "service_metric": 1, "service_log": 1})
    batch = {
        "endpoint_red": torch.tensor([[1., 2.]]),
        "service_metric": torch.tensor([[3.]]),
        "service_log": torch.tensor([[4.]]),
    }
    out = fusion(batch)
    assert torch.equal(out, torch.tensor([[1., 2., 3., 4.]]))
```

**Step 2/3/4**:

```python
# src/fusion/base.py
import torch.nn as nn
from abc import abstractmethod

class FusionModule(nn.Module):
    @abstractmethod
    def forward(self, modality_dict): ...
    @property
    @abstractmethod
    def output_dim(self): ...

# src/fusion/early_concat.py
class EarlyConcatFusion(FusionModule):
    MODALITY_ORDER = ("endpoint_red", "service_metric", "service_log")

    def __init__(self, modality_dims):
        super().__init__()
        self._dims = modality_dims

    def forward(self, modality_dict):
        return torch.cat([modality_dict[m] for m in self.MODALITY_ORDER], dim=-1)

    @property
    def output_dim(self):
        return sum(self._dims[m] for m in self.MODALITY_ORDER)
```

```bash
git add src/fusion/ tests/test_early_concat_fusion.py
git commit -m "[Feature]: FusionModule 抽象基类 + EarlyConcatFusion v0 baseline"
```

---

## Phase 5: Baseline + 评估（T14–T16）

### Task 14: Deep SVDD detector

**Files:**
- Create: `src/models/__init__.py`
- Create: `src/models/deep_svdd.py`
- Test: `tests/test_deep_svdd.py`

**Step 1: Write the failing test**

```python
# tests/test_deep_svdd.py
import torch
from src.models.deep_svdd import DeepSVDD

def test_deep_svdd_forward_returns_distance():
    model = DeepSVDD(input_dim=18, hidden_dim=32)
    x = torch.randn(4, 18)
    dist = model(x)
    assert dist.shape == (4,)
    assert (dist >= 0).all()  # L2 距离

def test_deep_svdd_center_initialized_from_data():
    """center 必须从训练集前向均值初始化（论文做法）。"""
    model = DeepSVDD(input_dim=18, hidden_dim=32)
    train_x = torch.randn(32, 18)
    model.init_center(train_x)
    assert model.center is not None
    assert model.center.shape == (32,)  # hidden_dim
```

**Step 2/3/4**: 极简 Deep SVDD：MLP encoder + center 距离作为异常分数。

```bash
git add src/models/deep_svdd.py tests/test_deep_svdd.py
git commit -m "[Feature]: Deep SVDD detector（MLP encoder + center 距离）"
```

---

### Task 15: `train_baseline_v0.py`

**Files:**
- Create: `scripts/train_baseline_v0.py`
- Test: `tests/test_train_baseline_v0.py`

**Step 1: Write the failing test**

```python
# tests/test_train_baseline_v0.py
import subprocess, sys
from pathlib import Path
import pandas as pd
from src.contracts import validate_scores_df

def test_train_baseline_v0_writes_scores_contract(tmp_path):
    contract_dir = tmp_path / "contract"
    subprocess.run([sys.executable, "scripts/build_contract.py",
                    "--config", "configs/contract/v0.yaml",
                    "--data-root", "tests/fixtures/mini_data_root",
                    "--out-dir", str(contract_dir), "--seed", "1"], check=True)

    out = tmp_path / "scores.parquet"
    subprocess.run([sys.executable, "scripts/train_baseline_v0.py",
                    "--contract-dir", str(contract_dir),
                    "--out", str(out), "--seed", "42",
                    "--epochs", "2"], check=True)
    df = pd.read_parquet(out)
    validate_scores_df(df)  # 必须符合现有 scores_v0 契约
    assert "case_id" in df.columns
    assert "anomaly_type" in df.columns
```

**Step 2/3/4**: 训练 Deep SVDD on `train.parquet`，推理 `eval_all.parquet`，写符合 `scores_v0` 契约的 parquet（带 case_id/anomaly_type 等诊断列供分层评估）。

```bash
git add scripts/train_baseline_v0.py tests/test_train_baseline_v0.py
git commit -m "[Feature]: train_baseline_v0 训练脚本（Deep SVDD + EarlyConcat）"
```

---

### Task 16: `eval_baseline_v0.py`（分层 AUROC）

**Files:**
- Create: `scripts/eval_baseline_v0.py`
- Test: `tests/test_eval_baseline_v0_stratified.py`

**Step 1: Write the failing test**

```python
# tests/test_eval_baseline_v0_stratified.py
import json
import subprocess, sys
import pandas as pd

def test_metrics_has_stratified_keys(tmp_path):
    # 简化 fixture：手造一个 scores.parquet
    df = pd.DataFrame({
        "sample_id": [f"s{i}" for i in range(20)],
        "score": [0.1] * 10 + [0.9] * 10,
        "y_true": [0] * 10 + [1] * 10,
        "case_id": ["Normal"] * 10 + ["Lv_P_cpu"] * 10,
        "anomaly_type": ["Normal"] * 10 + ["Lv_P_cpu"] * 10,
        "anomaly_level": ["none"] * 10 + ["performance"] * 10,
    })
    scores = tmp_path / "scores.parquet"
    df.to_parquet(scores)

    out = tmp_path / "metrics.json"
    subprocess.run([sys.executable, "scripts/eval_baseline_v0.py",
                    "--scores", str(scores), "--out", str(out)], check=True)
    metrics = json.loads(out.read_text())
    assert "overall" in metrics
    assert "by_anomaly_type" in metrics
    assert "by_anomaly_level" in metrics
    assert "Lv_P_cpu" in metrics["by_anomaly_type"]
    assert "performance" in metrics["by_anomaly_level"]
    assert 0 <= metrics["overall"]["auroc"] <= 1
```

**Step 2/3/4**: 按 `anomaly_type` 和 `anomaly_level` 分组各算一次 AUROC，组里只有一类时跳过并记录原因。

```bash
git add scripts/eval_baseline_v0.py tests/test_eval_baseline_v0_stratified.py
git commit -m "[Feature]: eval_baseline_v0 分层 AUROC（by_anomaly_type / by_anomaly_level）"
```

---

## Phase 6: 集成与文档（T17–T19）

### Task 17: 扩展 `dvc.yaml` 加三个新 stage

**Files:**
- Modify: `dvc.yaml`
- Test: `tests/test_dvc_pipeline_v0.py`

**Step 1: Write the failing test**

```python
# tests/test_dvc_pipeline_v0.py
import subprocess, pytest, yaml
from pathlib import Path

def test_dvc_yaml_has_v0_stages():
    pipeline = yaml.safe_load(Path("dvc.yaml").read_text())
    stages = set(pipeline["stages"].keys())
    assert {"build_contract", "train_v0", "eval_v0"}.issubset(stages)

def test_dvc_repro_runs_v0_pipeline():
    # 假设运行时有 data/anomod/ 真实数据
    if not Path("data/anomod/Normal_planA_20260609T022635Z").exists():
        pytest.skip("无真实数据集，跳过 dvc repro")
    subprocess.run(["dvc", "repro", "eval_v0"], check=True)
    assert Path("artifacts/baseline_v0/metrics.json").exists()
```

**Step 2/3/4**: 在 `dvc.yaml` 加 stage：

```yaml
  build_contract:
    cmd: python scripts/build_contract.py --config configs/contract/v0.yaml --data-root data/anomod --out-dir artifacts/contract_v0 --seed 42
    deps:
      - scripts/build_contract.py
      - src/preprocessors
      - src/contracts/contract_v0.py
      - src/data/normalization.py
      - configs/contract/v0.yaml
      - configs/contract/endpoint_to_service.yaml
    outs:
      - artifacts/contract_v0/train.parquet
      - artifacts/contract_v0/eval_all.parquet
      - artifacts/contract_v0/normalization_stats.json
      - artifacts/contract_v0/schema.json

  train_v0:
    cmd: python scripts/train_baseline_v0.py --contract-dir artifacts/contract_v0 --out artifacts/baseline_v0/scores.parquet --seed 42 --epochs 50
    deps:
      - scripts/train_baseline_v0.py
      - src/models/deep_svdd.py
      - src/fusion/early_concat.py
      - src/data/contract_dataloader.py
      - artifacts/contract_v0/train.parquet
      - artifacts/contract_v0/eval_all.parquet
    outs:
      - artifacts/baseline_v0/scores.parquet

  eval_v0:
    cmd: python scripts/eval_baseline_v0.py --scores artifacts/baseline_v0/scores.parquet --out artifacts/baseline_v0/metrics.json
    deps:
      - scripts/eval_baseline_v0.py
      - src/contracts
      - artifacts/baseline_v0/scores.parquet
    metrics:
      - artifacts/baseline_v0/metrics.json:
          cache: false
```

```bash
git add dvc.yaml tests/test_dvc_pipeline_v0.py
git commit -m "[Chore]: 扩展 dvc.yaml 加 build_contract / train_v0 / eval_v0 三个 stage"
```

---

### Task 18: E2E smoke test（验收标准 #1–#7）

**Files:**
- Test: `tests/test_e2e_contract_pipeline.py`

```python
# tests/test_e2e_contract_pipeline.py
"""完整 e2e 验收：dvc repro 跑通 + 全部 8 个验收标准。"""
import json, hashlib
import subprocess
import pytest
from pathlib import Path
import pandas as pd
from src.contracts.contract_v0 import validate_contract_df
from src.contracts import validate_scores_df

DATA_AVAILABLE = Path("data/anomod/Normal_planA_20260609T022635Z").exists()

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_1_dvc_repro_succeeds():
    subprocess.run(["dvc", "repro", "eval_v0"], check=True)

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_2_schema_passes():
    df = pd.read_parquet("artifacts/contract_v0/eval_all.parquet")
    validate_contract_df(df, "configs/contract/v0.yaml")

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_3_row_counts_in_range():
    train = pd.read_parquet("artifacts/contract_v0/train.parquet")
    eval_all = pd.read_parquet("artifacts/contract_v0/eval_all.parquet")
    assert 400 <= len(train) <= 1500, f"train 行数 {len(train)} 不在 ~800 附近"
    assert 6000 <= len(eval_all) <= 15000, f"eval_all 行数 {len(eval_all)} 不在 ~10K 附近"

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_4_reproducibility():
    subprocess.run(["dvc", "repro", "-f", "build_contract"], check=True)
    h1 = hashlib.sha256(Path("artifacts/contract_v0/train.parquet").read_bytes()).hexdigest()
    subprocess.run(["dvc", "repro", "-f", "build_contract"], check=True)
    h2 = hashlib.sha256(Path("artifacts/contract_v0/train.parquet").read_bytes()).hexdigest()
    assert h1 == h2

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_6_auroc_in_reasonable_range():
    m = json.loads(Path("artifacts/baseline_v0/metrics.json").read_text())
    auroc = m["overall"]["auroc"]
    assert 0.6 <= auroc <= 1.0, f"baseline AUROC {auroc} 退化为随机水平"

@pytest.mark.skipif(not DATA_AVAILABLE, reason="缺少真实数据集")
def test_acceptance_7_stratified_metrics_present():
    m = json.loads(Path("artifacts/baseline_v0/metrics.json").read_text())
    assert len(m["by_anomaly_type"]) >= 5  # 至少 5 类故障 + Normal
    assert set(m["by_anomaly_level"].keys()) & {"service", "performance", "database"}
```

```bash
git add tests/test_e2e_contract_pipeline.py
git commit -m "[Test]: e2e 验收测试覆盖 6 项验收标准"
```

---

### Task 19: 特征选择论据文档

**Files:**
- Create: `docs/agent-docs/feature-selection-rationale-v0.md`

> **此文档只在 user 明确要求时创建**。内容是设计文档 §9 的展开版，把每个 v0 入选特征对应到 RED 方法学 / 文献引用 / 数据可用性三个论据维度。是论文"特征工程"小节的素材。

如果决策者要求文档，按设计文档 §9 + 候选特征表 + 论文引用扩写到 ~500 行。

```bash
git add docs/agent-docs/feature-selection-rationale-v0.md
git commit -m "[Docs]: v0 特征选择论据（RED + 文献 + 数据可用性）"
```

---

## 任务依赖矩阵（执行顺序参考）

| Phase | Task | 阻塞依赖 | 可并行 |
|------|------|---------|--------|
| 1 | T1 | — | T2 |
| 1 | T2 | — | T1 |
| 1 | T3 | T1, T2 | — |
| 2 | T4 | T3 | T5, T7 |
| 2 | T5 | T3 | T4, T7 |
| 2 | T6a | T3 | T4, T5, T7 |
| 2 | T6b | T6a | — |
| 2 | T7 | T3 | T4, T5 |
| 3 | T8 | T3 | — |
| 3 | T9 | T4–T8 | — |
| 3 | T10 | T9 | — |
| 4 | T11 | T9 | T12, T13 |
| 4 | T12 | T9, T11 | T13 |
| 4 | T13 | T9 | T11, T12 |
| 5 | T14 | T13 | — |
| 5 | T15 | T11, T13, T14 | — |
| 5 | T16 | T15 | — |
| 6 | T17 | T15, T16 | — |
| 6 | T18 | T17 | — |
| 6 | T19 | —（独立写作） | 全程 |

---

## 不变量与跨任务保证（Code review 检查清单）

下面这些是设计文档 §8 列出的全局不变量，每个 PR 必须保证不被破坏：

1. **归一化只用 Normal 拟合**：T8 fit 调用必须只接受 Normal case 行。code review 时确认。
2. **Sequence 不跨 case 边界**：T12 测试已覆盖。
3. **时间戳 floor 对齐**：T6b 测试已覆盖。
4. **NaN ≠ 0**：T11 测试覆盖填补策略；T6b/T7 测试覆盖 raw 缺失写 NaN 不写 0。
5. **`endpoint_to_service.yaml` 与 contract parquet 同时存在**：T9 必须显式 require T1 配置。
6. **MetricPreprocessor 内存 < 4GB**：T6a perf spike + T18 在真实数据上的回归。

---

## 执行交接

**Plan complete and saved to `docs/plans/2026-06-26-contract-v0-implementation.md`. Two execution options:**

**1. Subagent-Driven (this session)** - 我在本 session 中按任务派发 fresh subagent，每个任务完成后做 code review，迭代快、便于实时调整。

**2. Parallel Session (separate)** - 你在 worktree 中开新 session，用 executing-plans skill 批量执行，带 checkpoint，适合长时间无人值守跑完整 phase。

**哪种方式？**
