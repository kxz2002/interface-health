# Per-Endpoint 精确标签接入评估 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `endpoint_raw2` 数据源里 `is_target_endpoint` 精确标签从原始 CSV 穿透到 contract parquet、scores.parquet，最终在 `metrics.json` 里产出统一口径的 per-endpoint AUROC，不新增双轨评估体系。

**Architecture:** 在现有 Contract v0 pipeline（`build_contract.py` → `train_baseline_v0.py` → `eval_baseline_v0.py`）上做纯增量改动：新增两列（`label_granularity`、`is_endpoint_anomaly`），显式贯穿三个脚本，`eval` 四层分层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增 `by_endpoint`）统一改用新标签列计算。不改 `is_anomaly`/`y_true` 既有语义，不改 contract v0 现有强校验。

**Tech Stack:** Python, pandas, pytest, PyYAML — 与现有代码库一致，不引入新依赖。

---

## 背景速览（供不熟悉本仓库的工程师）

这是一个异常检测研究项目。原始 chaos 注入实验数据经 4 个 `ModalityPreprocessor` 子类处理后，拼成一张 "contract parquet" 宽表，每一行是"某个 endpoint 在某个 15 秒时间窗口"的一条样本。训练脚本 `train_baseline_v0.py` 读这张表训练 One-Class 模型，产出 `scores.parquet`（每行一个异常分数）；`eval_baseline_v0.py` 读 `scores.parquet` 算分层 AUROC，写 `metrics.json`。

现有标签 `is_anomaly` 是"case 级"的粗粒度标签：只要这一行的时间窗口落在这个 case 的故障注入窗口内，这个 case 里**所有** endpoint 都被标记为异常，不管这个 endpoint 是否真的被打了故障。这是因为大多数历史数据（`anomod_v1`）确实是整个 service 级故障，没有更细的粒度。

新的数据源 `endpoint_raw2` 不同：它的每个 case 都精确记录了 `target_endpoint`（这次故障具体打的是哪个 endpoint），原始 CSV `tt_traces_red_15s.csv` 里有 `is_target_endpoint` 布尔列逐行标记。但这个字段目前在数据处理链路中被丢弃，完全没有使用。

本计划要做的事：让这个精确字段流转全程，产出一个新标签 `is_endpoint_anomaly`（= 是目标 endpoint AND 处于注入窗口），同时保留一个 `label_granularity` 列标记"这一行的标签精确到什么程度"（`"endpoint"` = 精确知道具体哪个 endpoint；`"case"` = 只知道整个 case 处于故障期，是历史遗留的近似粒度）。所有分层评估统一用新标签算，不因数据源不同分裂成两套指标口径。

设计背景详见 `docs/superpowers/specs/2026-07-08-per-endpoint-label-eval-design.md`，本计划严格按该 spec 落地。

---

## 关键事实（写代码前务必知道，避免重蹈历史踩坑）

1. **`is_target_endpoint` 在原始 CSV 里全程为 `True`/`False`，不随 `phase` 变化**：它标记的是"这个 endpoint 是不是本次注入的目标"这个静态身份，baseline/inject/recover 三个阶段都不变。因此 `is_endpoint_anomaly` 必须是 `is_target_endpoint AND phase=='inject'` 的组合，不能直接用 `is_target_endpoint`。
2. **该字段只存在于 `tt_traces_red_15s.csv`（server 侧），不存在于 `tt_endpoint_health_15s.csv`（client 侧）**，且目前会在 `TracePreprocessor.transform()` 里被丢弃（该方法最后一行只选择 `OUTPUT_COLUMNS` 里列出的 5 个特征列）。
3. **`anomod_v1`（以及所有 mini fixture 里没有 `target_endpoint` 字段的 case）的 `case_metadata.json` 没有 `target_endpoint` 键**，`case_meta.get("target_endpoint")` 会返回 `None`。这是判断"这个 case 该走精确标签还是 fallback"的唯一依据——不要用 `anomaly_level == "endpoint"` 代替判断，那是故障类型描述，语义不同，未来可能失配。
4. **`train_baseline_v0.py` 的 `out_df` 是显式列出的字段字典**（不是"把 `eval_df` 全部列都写进去"），新增列必须显式写进这个字典，否则会被静默丢弃——这正是当年 `is_target_endpoint` 在 `build_contract.py` 里被弄丢的同一种错误，本计划的 Task 5 专门针对这一步写回归测试防止再犯。
5. **`y_true`（`scores_v0.py` 契约要求的必需列）不变，继续等于 `is_anomaly`**（case 级）。四层 `stratified` 报告改用新的 `is_endpoint_anomaly`，但 `y_true` 保持契约兼容，不做替换。

---

### Task 1: `TracePreprocessor` 新增保留 `is_target_endpoint` 列

**Files:**
- Modify: `src/preprocessors/trace_preprocessor.py`
- Test: `tests/test_trace_preprocessor.py`

- [ ] **Step 1: 写失败测试 — 有 `is_target_endpoint` 列时应被保留**

在 `tests/test_trace_preprocessor.py` 末尾新增：

```python
def test_trace_transform_preserves_is_target_endpoint_when_present(tmp_path):
    """endpoint_raw2 类数据源的 tt_traces_red_15s.csv 带 is_target_endpoint 列，
    该列必须原样传递到输出，供后续 contract 层判断 per-endpoint 精确标签。"""
    csv_path = tmp_path / "tt_traces_red_15s.csv"
    csv_path.write_text(
        "case_id,anomaly_type,timestamp_window,endpoint_key,method,normalized_path,"
        "trace_request_count,trace_latency_mean,trace_latency_p50,trace_latency_p95,"
        "trace_latency_p99,trace_error_rate,trace_5xx_rate,trace_4xx_rate,"
        "trace_status_coverage,weak_is_anomaly,label_confidence,latency_anomaly_signal,"
        "error_anomaly_signal,5xx_anomaly_signal,phase,injection_start_ms,injection_end_ms,"
        "target_service,target_endpoint,anomaly_level,is_target_endpoint\n"
        "c1,Lv_E_HTTPABORT_assurance,1783150000000,"
        "GET:/api/v1/assuranceservice/assurances/types,GET,"
        "/api/v1/assuranceservice/assurances/types,9,22000.0,21500.0,44000.0,44000.0,"
        "0.0,0.0,0.0,1.0,0,normal,0,0,0,inject,1783150000000,1783150015000,"
        "ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,True\n"
    )
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(csv_path, case_meta={"case_id": "c1"})
    assert "is_target_endpoint" in df.columns
    assert bool(df["is_target_endpoint"].iloc[0]) is True


def test_trace_transform_fills_false_when_is_target_endpoint_absent():
    """anomod_v1 类数据源没有 is_target_endpoint 列时，不应报错，应填 False 占位。"""
    pre = TracePreprocessor(
        endpoint_mapping_path=REPO_ROOT / "configs/contract/endpoint_to_service.yaml"
    )
    df = pre.transform(
        REPO_ROOT / "tests/fixtures/mini_tt_traces_red_15s.csv",
        case_meta={"case_id": "Normal_planA"},
    )
    assert "is_target_endpoint" in df.columns
    assert not df["is_target_endpoint"].any()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_trace_preprocessor.py -k is_target_endpoint -v`
Expected: FAIL —— `AssertionError: assert 'is_target_endpoint' in df.columns`（因为当前 `transform()` 还没保留该列）

- [ ] **Step 3: 实现最小改动**

编辑 `src/preprocessors/trace_preprocessor.py`，把 `transform()` 方法（原第48-52行）改为：

```python
    def transform(self, raw_path: Path, case_meta: dict[str, Any]) -> pd.DataFrame:
        raw = pd.read_csv(raw_path)
        df = raw[raw["endpoint_key"].isin(self._v0_endpoints)].copy()
        df = df.rename(columns=self._RENAME)
        if "is_target_endpoint" not in df.columns:
            df["is_target_endpoint"] = False
        return df[
            ["endpoint_key", "timestamp_window_ms", "is_target_endpoint"] + self.OUTPUT_COLUMNS
        ]
```

不改 `OUTPUT_COLUMNS`（它是 5 维特征列的契约，`is_target_endpoint` 不是"特征"，是标签相关的原始字段，不应计入 `get_feature_columns()` 的返回值，因此不放进 `OUTPUT_COLUMNS`，单独在返回列表里追加）。

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_trace_preprocessor.py -v`
Expected: 全部 PASS（包括原有的 `test_trace_output_columns_match_contract` 等——`get_feature_columns()` 没变，不受影响）

- [ ] **Step 5: 提交**

```bash
git add src/preprocessors/trace_preprocessor.py tests/test_trace_preprocessor.py
git commit -m "feat: TracePreprocessor 保留 is_target_endpoint 列，缺列时填 False"
```

### Task 2: 新增 endpoint 级 mini fixture case（供后续 build_contract 测试用）

**Files:**
- Create: `tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/trace_data/case_metadata.json`
- Create: `tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/_pipeline_out/tt_traces_red_15s.csv`
- Create: `tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/_pipeline_out/tt_endpoint_health_15s.csv`

这个新 fixture case 模拟 `endpoint_raw2` 的数据形态：带 `target_endpoint` 的 `case_metadata.json` + 带 `is_target_endpoint` 列的 `tt_traces_red_15s.csv`。窗口对齐现有 fixture 惯例（15000ms 步长），故障目标是白名单里的
`GET:/api/v1/assuranceservice/assurances/types`（对应 `ts-assurance-service`），同一 case 里再放一个非目标 endpoint（`POST:/api/v1/users/login`，对应 `ts-auth-service`）验证"case 内非目标 endpoint 仍标 `label_granularity=endpoint` 但 `is_endpoint_anomaly=False`"。

- [ ] **Step 1: 创建目录结构**

Run: `mkdir -p tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/trace_data tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/_pipeline_out`

- [ ] **Step 2: 写 `case_metadata.json`**

```json
{
  "case_id": "Lv_E_HTTPABORT_assurance_mini",
  "anomaly_type": "Lv_E_HTTPABORT_assurance",
  "anomaly_level": "endpoint",
  "target_service": "ts-assurance-service",
  "target_endpoint": "GET:/api/v1/assuranceservice/assurances/types",
  "inject_start_ms": 1800000015000,
  "inject_end_ms": 1800000015000
}
```

（三个窗口时间戳为 `1800000000000`/`1800000015000`/`1800000030000`，`inject_start_ms == inject_end_ms == 1800000015000` 只覆盖中间那个 15s 窗口，对应 baseline/inject/recover，与现有 `Lv_E_HTTPABORT_travel_mini` fixture 惯例一致。）

- [ ] **Step 3: 写 `_pipeline_out/tt_traces_red_15s.csv`**

三个时间窗口 × 两个 endpoint（目标 + 非目标），目标 endpoint 的 `is_target_endpoint` 全程为 `True`（三行都是），非目标 endpoint 全程为 `False`：

```csv
case_id,anomaly_type,timestamp_window,window_str,endpoint_key,method,normalized_path,trace_request_count,trace_latency_mean,trace_latency_p50,trace_latency_p95,trace_latency_p99,trace_error_rate,trace_5xx_rate,trace_4xx_rate,trace_status_coverage,weak_is_anomaly,label_confidence,latency_anomaly_signal,error_anomaly_signal,5xx_anomaly_signal,phase,injection_start_ms,injection_end_ms,target_service,target_endpoint,anomaly_level,is_target_endpoint
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000000000,2027-01-15T08:00:00Z,GET:/api/v1/assuranceservice/assurances/types,GET,/api/v1/assuranceservice/assurances/types,9,300.0,290.0,600.0,650.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,baseline,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,True
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000015000,2027-01-15T08:00:15Z,GET:/api/v1/assuranceservice/assurances/types,GET,/api/v1/assuranceservice/assurances/types,11,900.0,880.0,1800.0,1900.0,0.4,0.3,0.0,1.0,1,strong,1,1,1,inject,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,True
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000030000,2027-01-15T08:00:30Z,GET:/api/v1/assuranceservice/assurances/types,GET,/api/v1/assuranceservice/assurances/types,10,310.0,300.0,610.0,660.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,recover,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,True
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000000000,2027-01-15T08:00:00Z,POST:/api/v1/users/login,POST,/api/v1/users/login,7,150.0,145.0,300.0,320.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,baseline,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,False
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000015000,2027-01-15T08:00:15Z,POST:/api/v1/users/login,POST,/api/v1/users/login,8,155.0,150.0,305.0,325.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,inject,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,False
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,1800000030000,2027-01-15T08:00:30Z,POST:/api/v1/users/login,POST,/api/v1/users/login,9,148.0,143.0,298.0,318.0,0.0,0.0,0.0,1.0,0,normal,0,0,0,recover,1800000015000,1800000015000,ts-assurance-service,GET:/api/v1/assuranceservice/assurances/types,endpoint,False
```

注意第二行(inject 窗口、目标 endpoint)`trace_error_rate=0.4` 是刻意设的高值，方便后续测试断言"这一行应该被判定为异常"时有区分度；非目标 endpoint 全程数值平稳，不异常。

- [ ] **Step 4: 写 `_pipeline_out/tt_endpoint_health_15s.csv`**

client 侧数据，两个 endpoint 三个窗口，与 trace 侧 join 键（`endpoint_key` + 转换后的 `timestamp_window_ms`）对齐：

```csv
case_id,anomaly_type,timestamp_window,endpoint_key,request_count,error_rate,latency_mean,latency_p50,latency_p95,latency_p99,status_2xx_rate,status_4xx_rate,status_5xx_rate,method,normalized_path
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:00Z,GET:/api/v1/assuranceservice/assurances/types,9,0.0,310.0,300.0,610.0,660.0,1.0,0.0,0.0,GET,/api/v1/assuranceservice/assurances/types
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:15Z,GET:/api/v1/assuranceservice/assurances/types,11,0.4,910.0,890.0,1810.0,1910.0,0.6,0.0,0.4,GET,/api/v1/assuranceservice/assurances/types
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:30Z,GET:/api/v1/assuranceservice/assurances/types,10,0.0,320.0,310.0,620.0,670.0,1.0,0.0,0.0,GET,/api/v1/assuranceservice/assurances/types
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:00Z,POST:/api/v1/users/login,7,0.0,155.0,150.0,305.0,325.0,1.0,0.0,0.0,POST,/api/v1/users/login
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:15Z,POST:/api/v1/users/login,8,0.0,160.0,155.0,310.0,330.0,1.0,0.0,0.0,POST,/api/v1/users/login
Lv_E_HTTPABORT_assurance_mini,Lv_E_HTTPABORT_assurance,2027-01-15T08:00:30Z,POST:/api/v1/users/login,9,0.0,153.0,148.0,303.0,323.0,1.0,0.0,0.0,POST,/api/v1/users/login
```

- [ ] **Step 5: 验证 fixture 本身没有语法错误（可被 pandas 正常读取）**

Run:
```bash
conda run -n interface python -c "
import pandas as pd
df1 = pd.read_csv('tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/_pipeline_out/tt_traces_red_15s.csv')
df2 = pd.read_csv('tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/_pipeline_out/tt_endpoint_health_15s.csv')
assert len(df1) == 6 and len(df2) == 6
assert df1['is_target_endpoint'].tolist() == [True, True, True, False, False, False]
print('fixture OK')
"
```
Expected: 打印 `fixture OK`，无异常

- [ ] **Step 6: 提交**

这一步暂不提交（fixture 会在 Task 3 里被引用验证），先继续 Task 3，两者一起提交。

### Task 3: Contract 层新增 `label_granularity` / `is_endpoint_anomaly` 两列

**Files:**
- Modify: `scripts/build_contract.py:233-253`（`_attach_label_columns`）
- Modify: `src/contracts/contract_v0.py:19-28`（`REQUIRED_LABEL_COLUMNS`）
- Test: `tests/test_build_contract_endpoint_labels.py`（新建）
- Test: `tests/test_contract_v0_schema.py`（扩）

- [ ] **Step 1: 写失败测试 — 新建 `tests/test_build_contract_endpoint_labels.py`**

这个测试用 Task 2 新建的 `Lv_E_HTTPABORT_assurance_mini` fixture（带精确标签）和已有的 `Lv_P_DISKIO_preserve` / `Normal`（无 `target_endpoint`，走 fallback）验证两条路径：

```python
"""per-endpoint 精确标签接入验证：label_granularity / is_endpoint_anomaly 两列。

对应 docs/superpowers/specs/2026-07-08-per-endpoint-label-eval-design.md。
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[1]
CONFIG = REPO_ROOT / "configs/contract/v0.yaml"


def _build(out_dir: Path, dataset_yaml: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(CONFIG),
            "--dataset",
            str(dataset_yaml),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_precise_case_label_granularity_and_is_endpoint_anomaly(tmp_path):
    """带 target_endpoint 的 case：目标 endpoint 只在 inject 窗口 is_endpoint_anomaly=True，
    非目标 endpoint 全程 False；两者 label_granularity 都是 'endpoint'。"""
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/merged_mini.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case = eval_all[eval_all["case_id"] == "Lv_E_HTTPABORT_assurance_mini"]
    assert not case.empty, "fixture 未产出该 case，检查 Task 2 的 fixture 是否正确落地"

    target = case[case["endpoint_key"] == "GET:/api/v1/assuranceservice/assurances/types"]
    assert set(target["label_granularity"]) == {"endpoint"}
    target_by_phase = target.set_index("phase")["is_endpoint_anomaly"]
    assert target_by_phase["baseline"] == False  # noqa: E712
    assert target_by_phase["inject"] == True  # noqa: E712
    assert target_by_phase["recover"] == False  # noqa: E712

    non_target = case[case["endpoint_key"] == "POST:/api/v1/users/login"]
    assert set(non_target["label_granularity"]) == {"endpoint"}
    assert not non_target["is_endpoint_anomaly"].any()


def test_fallback_case_is_endpoint_anomaly_matches_is_anomaly(tmp_path):
    """没有 target_endpoint 的 case（如 Lv_P_DISKIO_preserve）：
    label_granularity 全部为 'case'，is_endpoint_anomaly 严格等于 is_anomaly。"""
    out_dir = tmp_path / "contract_v0"
    _build(out_dir, REPO_ROOT / "tests/fixtures/mini_dataset.yaml")

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")
    case = eval_all[eval_all["case_id"] == "Lv_P_DISKIO_preserve"]
    assert not case.empty

    assert set(case["label_granularity"]) == {"case"}
    pd.testing.assert_series_equal(
        case["is_endpoint_anomaly"].astype(bool).reset_index(drop=True),
        case["is_anomaly"].astype(bool).reset_index(drop=True),
        check_names=False,
    )
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_build_contract_endpoint_labels.py -v`
Expected: FAIL —— `KeyError: 'label_granularity'`（`_attach_label_columns` 还没产出这一列；且 build_contract 目前还会因为 `merged_mini.yaml` 里没有 Task 2 新 fixture case 被自动纳入而缺少该 case——由于 `merged_mini.yaml` 的 `roots` 直接指向整个 `mini_data_root` 目录，新增的 case 目录会被 `_enumerate_cases` 自动发现，不需要改 yaml）

- [ ] **Step 3: 修改 `contract_v0.py`，先声明新列**

编辑 `src/contracts/contract_v0.py`，把 `REQUIRED_LABEL_COLUMNS`（第19-28行）改为：

```python
REQUIRED_LABEL_COLUMNS = [
    "phase",
    "is_anomaly",
    "is_train_eligible",
    "injection_start_ms",
    "injection_end_ms",
    "target_service",
    "anomaly_type",
    "anomaly_level",
    "label_granularity",
    "is_endpoint_anomaly",
]
```

- [ ] **Step 4: 修改 `build_contract.py` 的 `_attach_label_columns`**

编辑 `scripts/build_contract.py`，把 `_attach_label_columns` 函数（原第233-253行）末尾追加新逻辑，改为：

```python
def _attach_label_columns(ep_df: pd.DataFrame, case_meta: dict) -> None:
    inject_start = case_meta.get("inject_start_ms")
    inject_end = case_meta.get("inject_end_ms")
    ts = ep_df["timestamp_window_ms"]

    if inject_start is not None and inject_end is not None:
        phase = pd.Series("baseline", index=ep_df.index, dtype="object")
        phase[(ts >= inject_start) & (ts <= inject_end)] = "inject"
        phase[ts > inject_end] = "recover"
        ep_df["phase"] = phase
    else:
        ep_df["phase"] = "normal"

    ep_df["is_anomaly"] = ep_df["phase"] == "inject"
    # is_train_eligible 标记 non-inject 窗口；train.parquet 目前只取 Normal case（更严格）
    ep_df["is_train_eligible"] = ~ep_df["is_anomaly"]
    ep_df["injection_start_ms"] = inject_start
    ep_df["injection_end_ms"] = inject_end
    ep_df["target_service"] = case_meta.get("target_service")
    ep_df["anomaly_type"] = case_meta.get("anomaly_type", "Normal")
    ep_df["anomaly_level"] = case_meta.get("anomaly_level", "none")

    target_endpoint = case_meta.get("target_endpoint")
    if target_endpoint is not None:
        ep_df["label_granularity"] = "endpoint"
        is_target = ep_df.get("is_target_endpoint", False)
        if isinstance(is_target, pd.Series):
            is_target = is_target.fillna(False)
        ep_df["is_endpoint_anomaly"] = is_target & ep_df["is_anomaly"]
    else:
        ep_df["label_granularity"] = "case"
        ep_df["is_endpoint_anomaly"] = ep_df["is_anomaly"]
```

`ep_df.get("is_target_endpoint", False)` 处理了一种边界情况：某些模态 join 路径（如非 endpoint_red 特征来源）可能不经过 `TracePreprocessor`，理论上 `ep_df` 应该总有这一列（因为 `_process_one_case` 里 `trace_df` 总是第一个被 merge 进来的），但用 `.get()` 兜底比假设列一定存在更安全，不会在未来数据源变化时抛 `KeyError`。

- [ ] **Step 5: 运行测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_build_contract_endpoint_labels.py -v`
Expected: 两个测试都 PASS

- [ ] **Step 6: 运行全量已有测试确认没有破坏现有行为**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: 全部 PASS（此时 `test_contract_v0_schema.py` 里 `_make_valid_row()` 缺少新增的两个必需列，预期会失败——如果失败，进入 Step 6b）

- [ ] **Step 6b: 修复 `test_contract_v0_schema.py` 的 fixture 行**

编辑 `tests/test_contract_v0_schema.py`，在 `_make_valid_row()`（第18-57行）的 Label 部分末尾（`"anomaly_level": "none",` 之后）新增两个字段：

```python
        "anomaly_level": "none",
        "label_granularity": "case",
        "is_endpoint_anomaly": False,
    }
```

- [ ] **Step 7: 再次运行全量测试确认通过**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: 全部 PASS

- [ ] **Step 8: 提交（含 Task 2 的 fixture）**

```bash
git add tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini \
        tests/test_build_contract_endpoint_labels.py \
        tests/test_contract_v0_schema.py \
        src/contracts/contract_v0.py \
        scripts/build_contract.py
git commit -m "feat: contract 层新增 label_granularity/is_endpoint_anomaly 精确标签列"
```

### Task 4: `train_baseline_v0.py` 显式传递新列到 `scores.parquet`

**Files:**
- Modify: `scripts/train_baseline_v0.py:148-160`
- Test: `tests/test_train_baseline_v0.py`（扩）

- [ ] **Step 1: 写失败测试**

在 `tests/test_train_baseline_v0.py` 末尾新增：

```python
def test_train_baseline_v0_scores_carry_endpoint_label_columns(tmp_path):
    """scores.parquet 必须显式带上 is_endpoint_anomaly / label_granularity，
    否则四层 eval 分层无法计算——这是历史上 is_target_endpoint 被静默丢弃的同一种坑。"""
    contract_dir = tmp_path / "contract"
    _build_contract(contract_dir)

    out = tmp_path / "scores.parquet"
    subprocess.run(
        [
            sys.executable,
            "scripts/train_baseline_v0.py",
            "--contract-dir",
            str(contract_dir),
            "--out",
            str(out),
            "--seed",
            "42",
            "--epochs",
            "2",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    df = pd.read_parquet(out)
    eval_all = pd.read_parquet(contract_dir / "eval_all.parquet")
    assert "is_endpoint_anomaly" in df.columns
    assert "label_granularity" in df.columns

    merged = df.merge(
        eval_all[["sample_id", "is_endpoint_anomaly", "label_granularity"]],
        on="sample_id",
        suffixes=("_out", "_src"),
    )
    assert (
        merged["is_endpoint_anomaly_out"].astype(bool)
        == merged["is_endpoint_anomaly_src"].astype(bool)
    ).all()
    assert (merged["label_granularity_out"] == merged["label_granularity_src"]).all()
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_train_baseline_v0.py -k carry_endpoint_label -v`
Expected: FAIL —— `AssertionError: assert 'is_endpoint_anomaly' in df.columns`

- [ ] **Step 3: 修改 `scripts/train_baseline_v0.py`**

编辑 `main()` 函数里构造 `out_df` 的部分（原第148-160行），改为：

```python
    eval_df = pd.read_parquet(eval_pq)
    out_df = pd.DataFrame(
        {
            "sample_id": eval_df["sample_id"].astype(str),
            "score": eval_df["sample_id"].map(score_map).astype(float),
            "y_true": eval_df["is_anomaly"].astype(int),
            "case_id": eval_df["case_id"],
            "endpoint_key": eval_df["endpoint_key"],
            "phase": eval_df["phase"],
            "anomaly_type": eval_df["anomaly_type"],
            "anomaly_level": eval_df["anomaly_level"],
            "is_endpoint_anomaly": eval_df["is_endpoint_anomaly"].astype(int),
            "label_granularity": eval_df["label_granularity"],
        }
    )
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_train_baseline_v0.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 提交**

```bash
git add scripts/train_baseline_v0.py tests/test_train_baseline_v0.py
git commit -m "feat: scores.parquet 显式携带 is_endpoint_anomaly/label_granularity"
```

### Task 5: `eval_baseline_v0.py` 四层统一改用 `is_endpoint_anomaly`，新增 `by_endpoint`

**Files:**
- Modify: `scripts/eval_baseline_v0.py:35-69`（`compute_stratified_metrics`）
- Test: `tests/test_eval_baseline_v0_stratified.py`（扩）

这是本计划最核心的一步。当前 `compute_stratified_metrics` 全部用 `y_true` 计算；改动后四层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增 `by_endpoint`）全部改用 `is_endpoint_anomaly`，且 `by_endpoint` 对全部数据按 `endpoint_key` 分组（不按 `label_granularity` 过滤、不拆分数据源），与其余三层完全对称。

- [ ] **Step 1: 写失败测试 — 验证四层都用新标签而非 `y_true`**

在 `tests/test_eval_baseline_v0_stratified.py` 末尾新增：

```python
def test_stratified_layers_use_is_endpoint_anomaly_not_y_true(tmp_path):
    """构造 y_true 与 is_endpoint_anomaly 取值相反的数据，断言四层输出的数值
    对应 is_endpoint_anomaly，不是 y_true——验证标签口径切换正确。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "score": [0.1] * 10 + [0.9] * 10,
            # y_true 与 is_endpoint_anomaly 故意完全相反
            "y_true": [1] * 10 + [0] * 10,
            "is_endpoint_anomaly": [0] * 10 + [1] * 10,
            "case_id": ["Normal"] * 10 + ["Lv_E_HTTPABORT_assurance_mini"] * 10,
            "anomaly_type": ["Normal"] * 10 + ["Lv_E_HTTPABORT_assurance"] * 10,
            "anomaly_level": ["none"] * 10 + ["endpoint"] * 10,
            "endpoint_key": ["ep_a"] * 10 + ["ep_a"] * 10,
            "label_granularity": ["case"] * 10 + ["endpoint"] * 10,
        }
    )
    scores = tmp_path / "scores.parquet"
    df.to_parquet(scores)

    out = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, "scripts/eval_baseline_v0.py", "--scores", str(scores), "--out", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    metrics = json.loads(out.read_text())
    # score 越高越异常；用 is_endpoint_anomaly 时高分组(score=0.9)对应 is_endpoint_anomaly=1，
    # 完全可分，AUROC 应为 1.0。若代码仍误用 y_true，会因高分组 y_true=0 而得到 AUROC=0.0。
    assert metrics["stratified"]["overall"]["auroc"] == pytest.approx(1.0)
    assert metrics["auroc"] == pytest.approx(1.0)


def test_by_endpoint_stratum_groups_by_endpoint_key(tmp_path):
    """by_endpoint 新增分层：按 endpoint_key 分组算 AUROC，不区分 label_granularity。"""
    df = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "score": [0.1, 0.9] * 10,
            "y_true": [0, 1] * 10,
            "is_endpoint_anomaly": [0, 1] * 10,
            "case_id": ["c1"] * 10 + ["c2"] * 10,
            "anomaly_type": ["Lv_X"] * 20,
            "anomaly_level": ["endpoint"] * 20,
            # 前 10 行是 endpoint A（label_granularity=endpoint），
            # 后 10 行是 endpoint B（label_granularity=case，即 fallback 数据源）
            "endpoint_key": ["ep_A"] * 10 + ["ep_B"] * 10,
            "label_granularity": ["endpoint"] * 10 + ["case"] * 10,
        }
    )
    scores = tmp_path / "scores.parquet"
    df.to_parquet(scores)

    out = tmp_path / "metrics.json"
    subprocess.run(
        [sys.executable, "scripts/eval_baseline_v0.py", "--scores", str(scores), "--out", str(out)],
        check=True,
        cwd=str(REPO_ROOT),
    )
    metrics = json.loads(out.read_text())
    assert "by_endpoint" in metrics["stratified"]
    by_ep = metrics["stratified"]["by_endpoint"]
    # 两个 endpoint 都要出现，不因 label_granularity 不同被排除或拆分成两组
    assert set(by_ep.keys()) == {"ep_A", "ep_B"}
    assert by_ep["ep_A"]["n_samples"] == 10
    assert by_ep["ep_B"]["n_samples"] == 10
    assert by_ep["ep_A"]["auroc"] == pytest.approx(1.0)
    assert by_ep["ep_B"]["auroc"] == pytest.approx(1.0)
```

`tests/test_eval_baseline_v0_stratified.py` 顶部需要 `import pytest`（检查现有文件是否已导入——若没有，一并加上 `import pytest`）。

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_eval_baseline_v0_stratified.py -v`
Expected: FAIL —— `test_stratified_layers_use_is_endpoint_anomaly_not_y_true` 报 `AssertionError`（当前用 `y_true` 算出 AUROC=0.0 而非 1.0）；`test_by_endpoint_stratum_groups_by_endpoint_key` 报 `KeyError: 'by_endpoint'`

- [ ] **Step 3: 修改 `compute_stratified_metrics`**

编辑 `scripts/eval_baseline_v0.py`，把整个 `compute_stratified_metrics` 函数（原第35-69行）改为：

```python
def compute_stratified_metrics(df: pd.DataFrame) -> dict:
    stratified: dict = {}
    label_col = df["is_endpoint_anomaly"].astype(int)

    stratified["overall"] = {"auroc": _safe_auroc(label_col.values, df["score"].values)}

    stratified["by_anomaly_type"] = {}
    for atype, grp in df.groupby("anomaly_type"):
        grp_label = grp["is_endpoint_anomaly"].astype(int).values
        stratified["by_anomaly_type"][atype] = {
            "auroc": _safe_auroc(grp_label, grp["score"].values),
            "n_samples": len(grp),
        }

    stratified["by_anomaly_level"] = {}
    if "anomaly_level" in df.columns:
        for level, grp in df.groupby("anomaly_level"):
            grp_label = grp["is_endpoint_anomaly"].astype(int).values
            stratified["by_anomaly_level"][level] = {
                "auroc": _safe_auroc(grp_label, grp["score"].values),
                "n_samples": len(grp),
            }

    stratified["by_endpoint"] = {}
    if "endpoint_key" in df.columns:
        for ep, grp in df.groupby("endpoint_key"):
            grp_label = grp["is_endpoint_anomaly"].astype(int).values
            stratified["by_endpoint"][ep] = {
                "auroc": _safe_auroc(grp_label, grp["score"].values),
                "n_samples": len(grp),
            }

    auroc = _safe_auroc(label_col.values, df["score"].values)
    auprc = _safe_auprc(label_col.values, df["score"].values)
    # metrics_v0 契约允许 has_labels=False：单类数据无法计算 AUROC/AUPRC，
    # 此时 auroc/auprc 均为 None，契约要求两者也为 None，逻辑自洽。
    has_labels = auroc is not None

    return {
        "protocol_version": "v0",
        "higher_is_more_anomalous": True,
        "n_samples": len(df),
        "has_labels": has_labels,
        "auroc": auroc,
        "auprc": auprc,
        "stratified": stratified,
    }
```

`label_granularity` 列本次只需存在于输入 `df` 里（Task 4 已确保），`compute_stratified_metrics` 不读取它——按 spec §3.5 决定，噪声占比字段本次不做，留作遗留 TODO。

- [ ] **Step 4: 运行新增测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_eval_baseline_v0_stratified.py -v`
Expected: 全部 PASS

- [ ] **Step 5: 运行该文件原有测试确认没有破坏**

Run: `conda run -n interface python -m pytest tests/test_eval_baseline_v0_stratified.py -v -k "not is_endpoint_anomaly and not by_endpoint"`
Expected: 全部 PASS——注意原有 `test_metrics_has_stratified_keys` 和 `test_metrics_skips_single_class_group` 的 fixture DataFrame 里没有 `is_endpoint_anomaly` 列，会因 `KeyError` 失败，需要进入 Step 5b 修复

- [ ] **Step 5b: 修复原有两个测试的 fixture，补上 `is_endpoint_anomaly` 列**

编辑 `tests/test_eval_baseline_v0_stratified.py`：

`test_metrics_has_stratified_keys` 里的 DataFrame 构造（原第14-23行），在 `"anomaly_level": [...]` 之后新增一行：

```python
            "anomaly_level": ["none"] * 10 + ["performance"] * 10,
            "is_endpoint_anomaly": [0] * 10 + [1] * 10,
```

`test_metrics_skips_single_class_group` 里的 DataFrame 构造（原第49-57行），在 `"anomaly_level": [...]` 之后新增一行：

```python
            "anomaly_level": ["none"] * 10,
            "is_endpoint_anomaly": [0] * 10,
```

- [ ] **Step 6: 运行全量测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_eval_baseline_v0_stratified.py -v`
Expected: 全部 PASS

- [ ] **Step 7: 提交**

```bash
git add scripts/eval_baseline_v0.py tests/test_eval_baseline_v0_stratified.py
git commit -m "feat: eval 四层分层统一改用 is_endpoint_anomaly，新增 by_endpoint 分层"
```

### Task 6: `schema.json` 的 `evaluation_strata` 同步 + 文档补记

**Files:**
- Modify: `scripts/build_contract.py:393`（`_write_schema`）
- Modify: `tests/test_schema_json_contract.py`
- Modify: `docs/agent-docs/dataset-guide.md`

`schema.json` 里的 `evaluation_strata` 字段是给下游消费者（未来的融合策略/检测器）看的"这份 contract 支持哪些分层评估视角"的清单，Task 5 新增了 `by_endpoint` 分层，这个清单要同步，否则会出现"代码已经支持但 schema 没声明"的口径漂移——正是 entry 006 里教训过的"配置与实际口径漂移"同类问题。

- [ ] **Step 1: 写失败测试**

编辑 `tests/test_schema_json_contract.py`，在 `test_schema_json_keys_stable` 末尾新增断言：

```python
    assert schema["evaluation_strata"] == [
        "overall",
        "by_anomaly_type",
        "by_anomaly_level",
        "by_endpoint",
    ]
```

- [ ] **Step 2: 运行测试确认失败**

Run: `conda run -n interface python -m pytest tests/test_schema_json_contract.py -v`
Expected: FAIL —— 断言失败，实际值缺少 `"by_endpoint"`

- [ ] **Step 3: 修改 `scripts/build_contract.py`**

编辑 `_write_schema` 函数（原第393行）：

```python
                "evaluation_strata": [
                    "overall",
                    "by_anomaly_type",
                    "by_anomaly_level",
                    "by_endpoint",
                ],
```

- [ ] **Step 4: 运行测试确认通过**

Run: `conda run -n interface python -m pytest tests/test_schema_json_contract.py tests/test_e2e_smoke.py -v`
Expected: 全部 PASS（`test_e2e_smoke.py` 里同样断言 `schema["feature_dim"] == 18`，不受影响；未断言 `evaluation_strata` 具体值，不冲突）

- [ ] **Step 5: 补记 `docs/agent-docs/dataset-guide.md`**

在 §4.2（`trace_data`，第110-139行）的 "关键字段" 说明之后，新增一个小节记录 `target_endpoint`/`is_target_endpoint`/`anomaly_level`（endpoint 场景）的口径——用 Edit 工具在第139行（`关键字段：inject_start_ms / inject_end_ms 是毫秒级时间戳...`）之后插入：

```markdown

**endpoint 级精确标签字段**（仅 `endpoint_raw2` 等 endpoint 级故障注入数据源具备）：

| 字段 | 位置 | 说明 |
|---|---|---|
| `target_endpoint` | `case_metadata.json` | 本次故障注入的目标 endpoint（字符串，`METHOD:/path` 格式）。case 级故障注入数据源（如 `anomod_v1`）没有这个字段 |
| `anomaly_level` | `case_metadata.json` | 故障类型描述（如 `"endpoint"`/`"performance"`/`"database"`），**不等价于**"标签精确度"，不要用它判断该 case 是否有精确 per-endpoint 标签 |
| `is_target_endpoint` | `tt_traces_red_15s.csv`（server 侧，逐行） | 标记"这一行的 endpoint 是不是本次注入的目标"，**全程**（baseline/inject/recover）保持该 case 的判定结果不变，不随 phase 变化。要得到"此刻是否异常"需要与 `phase=='inject'` 做 AND |

Contract v0 pipeline 据此产出两个衍生列（见 `_attach_label_columns`）：

- `label_granularity`：`"endpoint"`（该 case 有 `target_endpoint`，标签精确到具体 endpoint）或 `"case"`（无该字段，fallback 为 case 级近似标签，历史遗留粒度）
- `is_endpoint_anomaly`：精确 case 为 `is_target_endpoint AND phase=='inject'`；fallback case 直接等于 `is_anomaly`

`is_anomaly`/`y_true` 语义不受影响，继续是 case 级"该时间窗口是否处于注入期"判断，供需要该语义的历史逻辑使用。
```

- [ ] **Step 6: 提交**

```bash
git add scripts/build_contract.py tests/test_schema_json_contract.py docs/agent-docs/dataset-guide.md
git commit -m "docs: schema evaluation_strata 补 by_endpoint + dataset-guide 记录 endpoint 标签字段口径"
```

### Task 7: 全量测试 + 端到端验证 + history entry

**Files:**
- Create: `history/entries/008-per-endpoint-label-eval.md`
- Modify: `history/index.md`

- [ ] **Step 1: 跑全量测试套件**

Run: `conda run -n interface python -m pytest tests/ -v`
Expected: 全部 PASS，无 skip 之外的异常（`test_sequence_dataset_shape` 可能仍 skip，属于既有行为，与本次改动无关）

- [ ] **Step 2: 用真实 `endpoint_raw2` 数据跑一次 `build_contract.py`，抽查精确标签正确性**

Run:
```bash
conda run -n interface python scripts/build_contract.py \
  --config configs/contract/v0.yaml \
  --dataset configs/data/merged_v1.yaml \
  --out-dir /tmp/contract_v0_verify \
  --seed 42
```
Expected: 命令成功退出（exit 0），产出 `train.parquet`/`eval_all.parquet`/`schema.json`/`normalization_stats.json`

- [ ] **Step 3: 抽查目标 endpoint 的 `is_endpoint_anomaly` 只在 inject 窗口为 True**

Run:
```bash
conda run -n interface python -c "
import pandas as pd
df = pd.read_parquet('/tmp/contract_v0_verify/eval_all.parquet')
precise = df[df['label_granularity'] == 'endpoint']
assert len(precise) > 0, '没有任何精确标签行，检查 endpoint_raw2 是否被 dataset config 正确纳入'
case = precise[precise['case_id'].str.startswith('Lv_E_HTTPABORT_assurance')].iloc[:1]
print(case[['case_id', 'endpoint_key', 'phase', 'is_target_endpoint', 'is_endpoint_anomaly']] if not case.empty else '未找到该 case，请换一个 case_id 前缀核对')
one_case_id = precise['case_id'].unique()[0]
one_case = precise[precise['case_id'] == one_case_id]
target_ep = one_case.loc[one_case['is_endpoint_anomaly'] & (one_case['phase']=='inject'), 'endpoint_key']
assert not target_ep.empty, f'case {one_case_id} 在 inject 窗口没有任何 is_endpoint_anomaly=True 的行'
ep = target_ep.iloc[0]
ep_rows = one_case[one_case['endpoint_key'] == ep]
non_inject = ep_rows[ep_rows['phase'] != 'inject']
assert not non_inject['is_endpoint_anomaly'].any(), f'{ep} 在非 inject 窗口仍被标记异常，检查 phase AND 逻辑'
print(f'验证通过：case={one_case_id} endpoint={ep} 只在 inject 窗口被标记 is_endpoint_anomaly=True')
"
```
Expected: 打印"验证通过"，没有 `AssertionError`

- [ ] **Step 4: 跑一次完整 train + eval，确认 `metrics.json` 里 `by_endpoint` 非空**

Run:
```bash
conda run -n interface python scripts/train_baseline_v0.py \
  --contract-dir /tmp/contract_v0_verify \
  --out /tmp/scores_verify.parquet \
  --seed 42 --epochs 5

conda run -n interface python scripts/eval_baseline_v0.py \
  --scores /tmp/scores_verify.parquet \
  --out /tmp/metrics_verify.json

conda run -n interface python -c "
import json
m = json.load(open('/tmp/metrics_verify.json'))
by_ep = m['stratified']['by_endpoint']
assert len(by_ep) > 0, 'by_endpoint 为空'
non_null = {k: v for k, v in by_ep.items() if v['auroc'] is not None}
print(f'by_endpoint 共 {len(by_ep)} 个 endpoint，其中 {len(non_null)} 个有非 None 的 AUROC')
for k, v in list(non_null.items())[:5]:
    print(f'  {k}: auroc={v[\"auroc\"]:.4f}, n_samples={v[\"n_samples\"]}')
"
```
Expected: 打印非空的 `by_endpoint` 统计，无异常抛出

- [ ] **Step 5: 清理临时验证产物**

Run: `rm -rf /tmp/contract_v0_verify /tmp/scores_verify.parquet /tmp/metrics_verify.json`

- [ ] **Step 6: 写 history entry**

创建 `history/entries/008-per-endpoint-label-eval.md`（参考 `history/entries/_template.md` 格式，`git log --oneline` 查实际 PR 号和最新 commit hash 后填入 `<PR号>`/`<commit-hash>` 占位）：

```markdown
# 008 · per-endpoint 精确标签接入评估

- **日期**: 2026-07-08
- **PR**: #<PR号> · **Commit**: <commit-hash>
- **类型**: Feature
- **影响域**: `src/preprocessors/trace_preprocessor.py`, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `scripts/eval_baseline_v0.py`, `src/contracts/contract_v0.py`

## 做了什么

承接 entry 006 遗留 TODO③：让 `endpoint_raw2` 的 `is_target_endpoint` 精确标签从原始 CSV 一路穿透到 `metrics.json`，产出真正意义上的 per-endpoint AUROC。新增两个 contract 列：`label_granularity`（`"endpoint"` = 精确标签 / `"case"` = case 级近似 fallback）、`is_endpoint_anomaly`（= `is_target_endpoint AND phase=='inject'`；fallback 时等于 `is_anomaly`）。`eval_baseline_v0.py` 的四层分层（`overall`/`by_anomaly_type`/`by_anomaly_level`/新增 `by_endpoint`）全部统一改用 `is_endpoint_anomaly`，不因数据源不同分裂成两套评估口径。

## 关键决策（不在 commit 里）

- **新增列而不是改 `is_anomaly` 语义**：`is_anomaly`/`y_true` 保持 case 级语义不变，满足 `scores_v0.py` 契约与既有强校验（`contract_v0.py` 的 `is_anomaly == phase=='inject'`），新标签走纯增量列，不需要动 contract v0 现有校验规则。
- **`by_endpoint` 对全部数据统一分组，不按 `label_granularity` 过滤**：最初设计考虑过"精确标签行才算 by_endpoint"、"双轨 AUROC 分开报告"等方案，最终否定——因为下游会有多个融合策略消融 + 多个检测器横向对比，标签可信度必须是 contract 层的路由规则（`label_granularity` 列），不能是 eval 脚本里的临时过滤逻辑，否则每新增一个下游组合都要重新决策一次。
- **`label_granularity` 判定依据是 case 是否有 `target_endpoint` 字段，不是 `anomaly_level`**：`anomaly_level` 是故障类型描述（如 `"endpoint"`/`"performance"`/`"database"`），恰好 `endpoint_raw2` 的 `anomaly_level` 取值是 `"endpoint"`，但这是巧合不是语义保证，用它判断标签精度会在未来数据源变化时失配。
- **`is_target_endpoint` 必须 AND `phase=='inject'`**：实测确认该字段在原始 CSV 里全程（baseline/inject/recover）为同一个值，标记的是"身份"不是"此刻状态"，直接用会把该 endpoint 的正常时段也误标为异常。

## 坑 / 已知问题

- **`train_baseline_v0.py` 的 `out_df` 是显式字段字典**：新增列不会自动从 `eval_all.parquet` 带过去，必须手动加进字段列表——这是当年 `is_target_endpoint` 在 `build_contract.py` 里被静默丢弃的同一种模式，本次专门写了 `test_train_baseline_v0_scores_carry_endpoint_label_columns` 做回归防线。

## 遗留 TODO

- `metrics.json` 未新增噪声构成字段（如 `n_precise`/`n_fallback`），`label_granularity` 本次只落盘不汇总，若后续需要在论文里量化标签噪声对 AUROC 的影响，需要单独设计。
- 训练侧仍是纯 One-Class（`build_contract.py` 的 train split 固定只用 Normal），双标签并存（`is_anomaly` + `is_endpoint_anomaly`）已满足监督学习对比的标签前提，但 train/eval split 逻辑尚未参数化为"全量带标签进训练"。
- Contract v0 → v1 正式版本升级仍未做，本次全部改动是 v0 之上的增量列。
```

- [ ] **Step 7: 更新 `history/index.md`**

在"影响域索引"表格末尾追加新行，并在"横切主题"里视情况补充——先用 `Read` 工具查看 `history/index.md` 当前的表格结构和横切主题小节的现有条目，再用 `Edit` 追加：

在影响域表格末尾新增：
```
| `src/preprocessors/trace_preprocessor.py`（is_target_endpoint 传递） | 008 |
| `scripts/eval_baseline_v0.py`（by_endpoint 分层） | 008 |
```

在横切主题小节末尾新增一条：
```
- **标签精度分级**：数据集里不同数据源标签粒度不一致（case 级近似 vs endpoint 级精确）时，用 contract 层的显式列（如 `label_granularity`）标记精度，而不是靠数据源名字或已有的故障类型字段（如 `anomaly_level`）隐性判断——后者语义上不等价，未来数据源变化会静默失配（008）
```

- [ ] **Step 8: 提交**

```bash
git add history/entries/008-per-endpoint-label-eval.md history/index.md
git commit -m "docs: 记录 per-endpoint 精确标签接入 history entry 008"
```

---

## 完成后自查清单

- [ ] `conda run -n interface python -m pytest tests/ -v` 全绿
- [ ] `is_anomaly`/`y_true` 语义和既有强校验完全未变
- [ ] `by_endpoint` 对全部数据统一分组，没有出现"两套 AUROC 体系"
- [ ] `train_baseline_v0.py` 的 `out_df` 显式带上了两个新列
- [ ] `docs/agent-docs/dataset-guide.md` 和 `history/index.md` 都已更新
- [ ] 遗留 TODO（噪声构成字段、训练侧监督学习前瞻、contract v1）已写入 history entry，没有丢失
