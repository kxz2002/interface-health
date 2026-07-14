# endpoint_raw2 接入 Contract v0 Pipeline 实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `build_contract.py` 通过一份受版本控制的数据集 config 同时消费 anomod_v1 + endpoint_raw2 两个数据源，合并产出统一 contract parquet，跑通 `dvc repro` 得到合理 One-Class AUROC/AUPRC。

**Architecture:** 新增 `src/data/dataset_config.py` 纯配置加载模块；`build_contract.py` 入口由 `--data-root <单路径>` 改为 `--dataset <config路径>`，对 config 里每个 root 分别 `_enumerate_cases` 后合并，其余流程（concat → fit-on-Normal 归一化 → 校验 → 写 parquet）不变；Normal 只来自 anomod_v1。

**Tech Stack:** Python 3.11, pandas, PyYAML, pytest, DVC, PyTorch (Deep SVDD)。

---

## 相对原始 design doc（2026-07-03）的偏差修正

本 plan 基于 `docs/superpowers/specs/2026-07-03-endpoint-raw-ingestion-design.md`，但探查后修正了以下几点，**执行时以本 plan 为准**：

1. **数据源用 `data/endpoint_raw2/`（07-04 重采版），不是 `data/endpoint_raw/`。**
   raw2 是 design doc §2.6 / §5 里"log 因 `fsnotify: too many open files` 全崩，需重采"之后的成果：log_data 从 45M/case 增至 931M/case，`log_data/<run-id>/<service-pod>/` 三层结构完整，`case_metadata.json` 带 `target_endpoint` + `anomaly_level: endpoint`。16 个 case，命名同 raw。**未被 git 追踪**（`?? data/endpoint_raw2/`），符合 design doc §3.6"本次不 `dvc add`"。
2. **anomod 侧真实目录是 `data/anomod_v1/`（12 case），不存在 `data/anomod`。** design doc §3.1 已用 `data/anomod_v1`，但现有 `configs/data/anomod.yaml` 里写的是 `root: data/anomod`（旧路径），改造时一并修正。
3. **8 个 endpoint→service 映射已用 log pod 目录名核对，design doc §3.2 的初值全部正确**（含两个不显然的：`users/login → ts-auth-service`、`contactservice/.../{uuid} → ts-contacts-service`）。trace CSV 无 `endpoint_service` 列（只有 `target_service`），故 §3.2 说的"以 endpoint_service 列为准"不可行，改用 log_data pod 目录名核对。
4. **log 前提反转**：raw2 log 已修复，design doc §2.6/§3.3"走 NaN 填 0 降级、log 无贡献"对 raw2 **不再成立**。Task 8 收尾验证从"接受 log 近似常量"改为"确认 log join 命中率显著高于 raw 的 2%–27%"。
5. **现有 `test_endpoint_service_mapping.py` 用的是旧错误名单**（route/travelplan 组），Task 2 是**改写**该测试而非新增。
6. **mini fixture 连锁影响**：现有 `tests/fixtures/mini_data_root` 的 endpoint_key 只有 `POST:/api/v1/travelservice/trips/left` 和 `GET:/api/v1/routeservice/routes`。名单修正后 `routeservice/routes` 不再在白名单内——Task 6 需保证合并 e2e 的 endpoint_key 落在新的 8 个白名单内，否则 `ep_to_svc.map()` 得到 NaN service_name。

## 关联历史（project-history）

- **entry 005**（Contract v0）：`_REPO_ROOT = Path(__file__)` 路径约定、fit-on-Normal 防泄漏、`_enumerate_cases` 以 `_pipeline_out/` 为锚点、LogPreprocessor 三层目录下探 + `Asia/Shanghai` 时区修复——本次全部保持不动，依赖其成果。
- **遗留 TODO 划界**：per-endpoint 精确标签 + 评估（用 `is_target_endpoint`）拆入后续 PR（③），本次 endpoint_raw2 仍用 case 级"脏"标签，与 anomod_v1 一致。

---

## 文件结构（File Structure）

| 文件 | 动作 | 职责 |
|---|---|---|
| `src/data/dataset_config.py` | 新建 | 纯配置加载：读 yaml → `DatasetConfig` dataclass（`name`/`roots`/`normal_source`/`fused_window`），字段缺失显式 raise。无副作用，独立可测。 |
| `configs/data/merged_v1.yaml` | 新建 | 声明合并数据集：`roots: [data/anomod_v1, data/endpoint_raw2]`，`normal_source: data/anomod_v1`。 |
| `configs/data/anomod.yaml` | 改造 | 改为同格式（`roots: [data/anomod_v1]`, `normal_source: data/anomod_v1`），修正旧 `root: data/anomod` 路径。 |
| `configs/contract/endpoint_to_service.yaml` | 整体替换 | 8 个真实 client 入口（dataset-guide §6）→ ts-service 映射。 |
| `scripts/build_contract.py` | 修改 | `--data-root` → `--dataset`；用 `DatasetConfig.roots` 多 root 枚举合并；Normal 来源按 `normal_source`。 |
| `dvc.yaml` | 修改 | `build_contract` stage 的 `cmd` 改用 `--dataset`，`deps` 补 merged_v1.yaml。 |
| `tests/test_dataset_config.py` | 新建 | dataset_config loader 单测。 |
| `tests/test_endpoint_service_mapping.py` | 改写 | 断言 key 集合 == §6 的 8 个真实 endpoint（现有断言用的是旧错误名单）。 |
| `tests/test_build_contract_multi_root.py` | 新建 | 多 root 枚举合并测试。 |
| `tests/test_dvc_pipeline_v0.py` | 扩充 | build_contract stage 用 `--dataset` 且 config 在 deps 里。 |
| `tests/test_e2e_smoke.py` | 修改 | `--data-root` → `--dataset`；扩成合并 e2e。 |
| `tests/fixtures/mini_data_root/<endpoint_raw2 case>/` | 新建 | 1 个 endpoint_raw2 mini case fixture。 |
| `tests/fixtures/merged_mini.yaml` | 新建 | 指向 mini_data_root 的合并 config，供 e2e 用。 |

---

## Task 1: dataset_config 加载模块

**Files:**
- Create: `src/data/dataset_config.py`
- Test: `tests/test_dataset_config.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_dataset_config.py
from pathlib import Path

import pytest

from src.data.dataset_config import DatasetConfig, load_dataset_config


def test_load_merged_config(tmp_path):
    cfg_path = tmp_path / "merged.yaml"
    cfg_path.write_text(
        "name: merged_v1\n"
        "roots:\n"
        "  - data/anomod_v1\n"
        "  - data/endpoint_raw2\n"
        "normal_source: data/anomod_v1\n"
        "fused_window: 15s\n"
    )
    cfg = load_dataset_config(cfg_path)
    assert isinstance(cfg, DatasetConfig)
    assert cfg.name == "merged_v1"
    assert cfg.roots == [Path("data/anomod_v1"), Path("data/endpoint_raw2")]
    assert cfg.normal_source == Path("data/anomod_v1")
    assert cfg.fused_window == "15s"


def test_missing_roots_raises(tmp_path):
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text("name: x\nnormal_source: data/anomod_v1\n")
    with pytest.raises(ValueError, match="roots"):
        load_dataset_config(cfg_path)


def test_normal_source_must_be_in_roots(tmp_path):
    cfg_path = tmp_path / "bad2.yaml"
    cfg_path.write_text(
        "name: x\nroots:\n  - data/anomod_v1\nnormal_source: data/not_listed\n"
    )
    with pytest.raises(ValueError, match="normal_source"):
        load_dataset_config(cfg_path)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `conda run -n interface pytest tests/test_dataset_config.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'src.data.dataset_config'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/data/dataset_config.py
"""数据集组成配置加载：把"数据集由哪些 root 组成"沉淀成受版本控制的 config。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    roots: list[Path]
    normal_source: Path
    fused_window: str


def load_dataset_config(path: str | Path) -> DatasetConfig:
    raw = yaml.safe_load(Path(path).read_text())

    name = raw.get("name")
    if not name:
        raise ValueError(f"dataset config {path} 缺少 name 字段")

    roots_raw = raw.get("roots")
    if not roots_raw:
        raise ValueError(f"dataset config {path} 缺少 roots 字段（至少一个数据源目录）")
    roots = [Path(r) for r in roots_raw]

    normal_raw = raw.get("normal_source")
    if not normal_raw:
        raise ValueError(f"dataset config {path} 缺少 normal_source 字段")
    normal_source = Path(normal_raw)
    if normal_source not in roots:
        raise ValueError(
            f"dataset config {path} 的 normal_source={normal_source} 不在 roots {roots} 中"
        )

    return DatasetConfig(
        name=name,
        roots=roots,
        normal_source=normal_source,
        fused_window=raw.get("fused_window", "15s"),
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `conda run -n interface pytest tests/test_dataset_config.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add src/data/dataset_config.py tests/test_dataset_config.py
git commit -m "$(cat <<'EOF'
[Feature]: 新增 dataset_config 多数据源配置加载模块

把"数据集由哪些 root 组成"沉淀成受版本控制的 config，
补上 CLAUDE.md "数据集版本记录在 config" 一直未落实的约定。
EOF
)"
```

---

## Task 2: endpoint 名单修正 + 回归防线

**Files:**
- Modify: `configs/contract/endpoint_to_service.yaml`（整体替换）
- Modify: `tests/test_endpoint_service_mapping.py`（改写 EXPECTED_ENDPOINTS）

> **背景**：现有名单是 route/travelplan 组（其中 4 个在两数据集 client 侧都不出现 = 死条目），导致 anomod_v1 baseline 实际只有 4 个活 endpoint。dataset-guide §6 记录的 8 个真实 client 入口才是正确口径。service 名已用 log_data pod 目录核对无误。

- [ ] **Step 1: 改写测试到新的真实名单（RED）**

```python
# tests/test_endpoint_service_mapping.py 中替换 EXPECTED_ENDPOINTS 及断言
EXPECTED_ENDPOINTS = {
    "GET:/api/v1/assuranceservice/assurances/types",
    "GET:/api/v1/contactservice/contacts/account/{uuid}",
    "POST:/api/v1/inside_pay_service/inside_payment",
    "POST:/api/v1/orderservice/order/refresh",
    "POST:/api/v1/preserveservice/preserve",
    "POST:/api/v1/travel2service/trips/left",
    "POST:/api/v1/travelservice/trips/left",
    "POST:/api/v1/users/login",
}
```

新增一个"钉死口径"回归测试（把配置与 dataset-guide §6 漂移这个坑钉死）：

```python
def test_endpoint_mapping_matches_dataset_guide(mapping):
    """回归防线：endpoint_to_service.yaml 的 key 必须与 dataset-guide §6 的 8 个真实
    client 入口完全一致。任何一方漂移都会在此断裂，防止配置与真实数据口径悄悄背离。"""
    guide = (REPO_ROOT / "docs/agent-docs/dataset-guide.md").read_text()
    section = guide.split("## 六、Endpoint 清单")[1].split("## 七")[0]
    guide_endpoints = {
        line.strip()
        for line in section.splitlines()
        if line.strip().startswith(("GET:", "POST:", "PUT:", "DELETE:", "PATCH:"))
    }
    assert set(mapping.keys()) == guide_endpoints
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n interface pytest tests/test_endpoint_service_mapping.py -v`
Expected: FAIL（`test_endpoint_mapping_covers_all_v0_endpoints` 与新 `..._matches_dataset_guide` 均因旧 yaml 内容不符而失败）

- [ ] **Step 3: 整体替换 endpoint_to_service.yaml（GREEN）**

```yaml
# configs/contract/endpoint_to_service.yaml
# 8 个真实 client 外部入口 → Train-Ticket service 静态映射
# 口径来源：docs/agent-docs/dataset-guide.md §6（由 test_endpoint_mapping_matches_dataset_guide 钉住）
# service 名以 log_data pod 目录核对（排除 gateway 视角）
"GET:/api/v1/assuranceservice/assurances/types": ts-assurance-service
"GET:/api/v1/contactservice/contacts/account/{uuid}": ts-contacts-service
"POST:/api/v1/inside_pay_service/inside_payment": ts-inside-payment-service
"POST:/api/v1/orderservice/order/refresh": ts-order-service
"POST:/api/v1/preserveservice/preserve": ts-preserve-service
"POST:/api/v1/travel2service/trips/left": ts-travel2-service
"POST:/api/v1/travelservice/trips/left": ts-travel-service
"POST:/api/v1/users/login": ts-auth-service
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n interface pytest tests/test_endpoint_service_mapping.py -v`
Expected: PASS（3 passed：covers / ts-naming / matches-dataset-guide）

- [ ] **Step 5: Commit**

```bash
git add configs/contract/endpoint_to_service.yaml tests/test_endpoint_service_mapping.py
git commit -m "$(cat <<'EOF'
[Data]: 修正 endpoint 白名单为 dataset-guide §6 真实 8 入口

旧名单 route/travelplan 组含 4 个死条目（client 侧不出现），
使 anomod_v1 baseline 只有 4 个活 endpoint。改为真实 client 入口，
并加回归测试钉住与 dataset-guide 的口径一致性。
EOF
)"
```

---

## Task 3: build_contract.py 入口改 `--dataset` + 多 root 枚举

**Files:**
- Modify: `scripts/build_contract.py`（`main()` 的 argparse 与 case 枚举、Normal 判定）
- Test: `tests/test_build_contract_multi_root.py`（新建）

> **改动限于编排层**：`_process_one_case` / 归一化 / 校验 / 写 parquet 全部不动。只改①入口参数 `--data-root`→`--dataset`，②多 root 枚举合并，③Normal 来源由 config `normal_source` 界定（而非仅靠 `anomaly_type.startswith("Normal")`——后者保留作为二次保险）。

- [ ] **Step 1: Write the failing test（多 root 枚举合并）**

```python
# tests/test_build_contract_multi_root.py
from pathlib import Path

from scripts.build_contract import _enumerate_cases_multi


def _make_case(root: Path, name: str):
    (root / name / "_pipeline_out").mkdir(parents=True)


def test_enumerate_cases_multi_root_merges_and_sorts(tmp_path):
    root_a = tmp_path / "ds_a"
    root_b = tmp_path / "ds_b"
    _make_case(root_a, "Normal")
    _make_case(root_a, "Lv_S_HTTPABORT_preserve")
    _make_case(root_b, "Lv_E_HTTPABORT_assurance")

    cases = _enumerate_cases_multi([root_a, root_b])

    names = [c.name for c in cases]
    assert names == ["Lv_E_HTTPABORT_assurance", "Lv_S_HTTPABORT_preserve", "Normal"]
    assert len(cases) == len(set(cases))  # 无重复


def test_enumerate_cases_multi_root_empty_raises_upstream(tmp_path):
    # 空 root 返回空列表（由 main() 负责 raise，此处只验证枚举本身不抛）
    assert _enumerate_cases_multi([tmp_path / "empty"]) == []
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n interface pytest tests/test_build_contract_multi_root.py -v`
Expected: FAIL with `ImportError: cannot import name '_enumerate_cases_multi'`

- [ ] **Step 3: 在 build_contract.py 加多 root 枚举函数**

在 `_enumerate_cases` 下方新增（保留原单 root 函数，被多 root 复用）：

```python
def _enumerate_cases_multi(roots: list[Path]) -> list[Path]:
    """跨多个数据源 root 枚举 case，合并后按 case 名排序。

    每个 root 独立 _enumerate_cases 再 concat；不同 root 的 case 目录名在本数据集
    里天然唯一（时间戳后缀），故不做跨 root 去重。返回按 dir.name 排序保证稳定。
    """
    merged: list[Path] = []
    for root in roots:
        merged.extend(_enumerate_cases(root))
    return sorted(merged, key=lambda p: p.name)
```

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n interface pytest tests/test_build_contract_multi_root.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: 改 main() 入口参数与枚举调用**

替换 `main()` 中 argparse 的 `--data-root` 与 case 枚举段。将：

```python
    parser.add_argument("--data-root", required=True)
```

改为：

```python
    parser.add_argument("--dataset", required=True, help="数据集组成 config（configs/data/*.yaml）")
```

将（约 278–281 行）：

```python
    cases = _enumerate_cases(Path(args.data_root))
    LOG.info("发现 %d 个 case", len(cases))
    if not cases:
        raise RuntimeError(f"data_root {args.data_root} 下未找到任何含 _pipeline_out/ 的 case 目录")
```

改为：

```python
    dataset_cfg = load_dataset_config(args.dataset)
    LOG.info("数据集 %s，roots=%s", dataset_cfg.name, [str(r) for r in dataset_cfg.roots])
    cases = _enumerate_cases_multi(dataset_cfg.roots)
    LOG.info("发现 %d 个 case", len(cases))
    if not cases:
        raise RuntimeError(
            f"dataset {args.dataset} 的 roots {dataset_cfg.roots} 下未找到任何含 _pipeline_out/ 的 case 目录"
        )
```

在文件顶部 import 段加：

```python
from src.data.dataset_config import load_dataset_config
```

- [ ] **Step 6: Normal 来源按 config 加保险**

`normal_mask`（约 307 行）仍以 `anomaly_type.startswith("Normal")` 为主判据（endpoint_raw2 无 Normal，天然只匹配 anomod_v1），无需改逻辑。但把 `normal_source` 作为断言保险，防止误配。在 `normalizer.fit` 前插入：

```python
    normal_cases = full.loc[normal_mask, "case_id"].unique()
    LOG.info("Normal case 数=%d，来自 %s", len(normal_cases), dataset_cfg.normal_source)
```

（不引入 case→root 反查，保持改动最小；normal_source 主要在 config 层表达意图，Task 6 e2e 会实证 Normal 只来自 anomod 侧。）

- [ ] **Step 7: 更新受影响的既有测试引用**

`test_build_contract_smoke.py`、`test_e2e_toy_pipeline.py` 若引用 `--data-root` 需同步（Task 6 统一处理 e2e_smoke；此处先跑冒烟确认 import 不破）。

Run: `conda run -n interface pytest tests/test_build_contract_multi_root.py tests/test_dataset_config.py -v`
Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add scripts/build_contract.py tests/test_build_contract_multi_root.py
git commit -m "$(cat <<'EOF'
[Feature]: build_contract 入口改 --dataset，支持多数据源合并枚举

从 --data-root 单路径切到 --dataset config 入口，按 config.roots
逐个枚举 case 后合并。单 case 处理/归一化/校验流程不变。
EOF
)"
```

---

## Task 4: 数据集 config 文件（merged_v1 + anomod 改造）

**Files:**
- Create: `configs/data/merged_v1.yaml`
- Modify: `configs/data/anomod.yaml`（改造成同格式）

- [ ] **Step 1: 新建 merged_v1.yaml**

```yaml
# configs/data/merged_v1.yaml
# anomod_v1 (service 级故障) + endpoint_raw2 (endpoint 级故障，log 重采版) 合并数据集
# Normal case 只来自 anomod_v1（endpoint_raw2 无 Normal）
name: merged_v1
description: anomod_v1 (12 case, 含 Normal) + endpoint_raw2 (16 endpoint 级故障 case)
roots:
  - data/anomod_v1
  - data/endpoint_raw2
normal_source: data/anomod_v1
fused_window: 15s
```

- [ ] **Step 2: 改造 anomod.yaml 为同格式**

```yaml
# configs/data/anomod.yaml
# 单数据源：仅 anomod_v1（service 级故障注入）。字段口径详见 dataset-guide.md
name: anomod
description: Train-Ticket service 级故障注入数据集（12 case, 含 Normal）
roots:
  - data/anomod_v1
normal_source: data/anomod_v1
fused_window: 15s
```

- [ ] **Step 3: 验证两个 config 能被 loader 解析**

Run:
```bash
conda run -n interface python -c "from src.data.dataset_config import load_dataset_config; \
print(load_dataset_config('configs/data/merged_v1.yaml')); \
print(load_dataset_config('configs/data/anomod.yaml'))"
```
Expected: 两行 `DatasetConfig(...)`，无异常，roots 路径正确。

- [ ] **Step 4: Commit**

```bash
git add configs/data/merged_v1.yaml configs/data/anomod.yaml
git commit -m "$(cat <<'EOF'
[Data]: 新增 merged_v1 合并数据集 config，改造 anomod.yaml 为多 root 格式

merged_v1 声明 anomod_v1 + endpoint_raw2 两个 root，Normal 仅来自
anomod_v1。anomod.yaml 从旧 root: data/anomod 改为 roots 列表格式。
EOF
)"
```

---

## Task 5: dvc.yaml build_contract stage 改用 `--dataset` + 静态一致性测试

**Files:**
- Modify: `dvc.yaml`（build_contract stage 的 `cmd` 与 `deps`）
- Test: `tests/test_dvc_pipeline_v0.py`（扩充）

> **DVC 测试划界**（design doc §3.5）：本次只加"stage 参数/依赖一致性"静态测试，**不**上"真跑 dvc repro"集成测试（CI 脆弱、成本高）。真跑留 Task 8 手动收尾。

- [ ] **Step 1: Write the failing test（扩 test_dvc_pipeline_v0.py）**

```python
# tests/test_dvc_pipeline_v0.py 末尾追加
def test_build_contract_uses_dataset_config():
    """build_contract stage 必须用 --dataset 指向真实存在的 config，
    且该 config 列入 deps（防"config 改了但没进 deps 导致 dvc 不重跑"的隐蔽坑）。"""
    pipeline = yaml.safe_load((REPO_ROOT / "dvc.yaml").read_text())
    stage = pipeline["stages"]["build_contract"]

    cmd = stage["cmd"]
    assert "--dataset" in cmd
    assert "--data-root" not in cmd

    # 从 cmd 抽出 --dataset 后面的路径，断言文件存在
    tokens = cmd.split()
    ds_path = tokens[tokens.index("--dataset") + 1]
    assert (REPO_ROOT / ds_path).exists(), f"{ds_path} 不存在"

    # 该 config 必须在 deps 里
    deps = stage.get("deps", [])
    assert ds_path in deps, f"{ds_path} 未列入 build_contract 的 deps"
```

- [ ] **Step 2: Run to verify it fails**

Run: `conda run -n interface pytest tests/test_dvc_pipeline_v0.py::test_build_contract_uses_dataset_config -v`
Expected: FAIL（当前 cmd 仍是 `--data-root data/anomod`）

- [ ] **Step 3: 改 dvc.yaml build_contract stage**

将 `build_contract` stage 的 `cmd`（19–21 行）与 `deps` 改为：

```yaml
  build_contract:
    cmd: python scripts/build_contract.py --config configs/contract/v0.yaml --dataset
      configs/data/merged_v1.yaml --out-dir artifacts/contract_v0 --seed 42
    deps:
      - scripts/build_contract.py
      - src/preprocessors
      - src/contracts/contract_v0.py
      - src/data/normalization.py
      - src/data/dataset_config.py
      - configs/contract/v0.yaml
      - configs/contract/endpoint_to_service.yaml
      - configs/data/merged_v1.yaml
    outs:
      - artifacts/contract_v0/train.parquet
      - artifacts/contract_v0/eval_all.parquet
      - artifacts/contract_v0/normalization_stats.json
      - artifacts/contract_v0/schema.json
```

> 注：`deps` 里补了 `src/data/dataset_config.py` 和 `configs/data/merged_v1.yaml`，二者变更都应触发 stage 重跑。**不**把 `data/anomod_v1`/`data/endpoint_raw2` 列入 deps——它们未被 DVC 追踪（design doc §3.6），列入会导致 dvc 因无 .dvc 记录而报错。

- [ ] **Step 4: Run to verify it passes**

Run: `conda run -n interface pytest tests/test_dvc_pipeline_v0.py -v`
Expected: PASS（含原有 stage 存在性测试）

- [ ] **Step 5: Commit**

```bash
git add dvc.yaml tests/test_dvc_pipeline_v0.py
git commit -m "$(cat <<'EOF'
[Chore]: dvc build_contract stage 切到 --dataset merged_v1，补 deps

cmd 用 --dataset configs/data/merged_v1.yaml，deps 补 dataset_config.py
与 merged_v1.yaml。加静态测试断言 cmd/deps 一致性。
EOF
)"
```

---

## Task 6: endpoint_raw2 mini fixture + 合并 e2e

**Files:**
- Create: `tests/fixtures/mini_data_root/Lv_E_HTTPABORT_travel_mini/`（1 个 endpoint 级故障 mini case）
- Create: `tests/fixtures/merged_mini.yaml`（指向 mini_data_root 的合并 config）
- Modify: `tests/test_e2e_smoke.py`（`--data-root`→`--dataset`）

> **关键约束**（偏差修正 6）：mini case 的 `endpoint_key` 必须落在新的 8 个白名单内，且要在 mini Normal 里有同 endpoint 的 Normal 统计量供 per_endpoint 归一化。现有 mini Normal 只有 `POST:/api/v1/travelservice/trips/left` 和 `GET:/api/v1/routeservice/routes`——后者已不在白名单。**故 mini endpoint_raw2 case 用 `POST:/api/v1/travelservice/trips/left`**（白名单内 + Normal 有基线），target 设为该 endpoint。

- [ ] **Step 1: 造 mini endpoint_raw2 case 目录结构**

```bash
CASE=tests/fixtures/mini_data_root/Lv_E_HTTPABORT_travel_mini
mkdir -p "$CASE/_pipeline_out" "$CASE/trace_data" "$CASE/metric_data" \
         "$CASE/log_data/run0/ts-travel-service-6f467bc998-jt5dd"
```

- [ ] **Step 2: 写 case_metadata.json（含 target_endpoint，模拟 endpoint 级故障）**

```json
{
  "case_id": "Lv_E_HTTPABORT_travel_mini",
  "anomaly_type": "Lv_E_HTTPABORT_travel",
  "anomaly_level": "endpoint",
  "target_service": "ts-travel-service",
  "target_endpoint": "POST:/api/v1/travelservice/trips/left",
  "inject_start_ms": 1783150000000,
  "inject_end_ms": 1783150015000
}
```
写到 `$CASE/trace_data/case_metadata.json`（Layout B：真实数据 metadata 在 trace_data/ 下，验证 `_load_case_meta` 的 Layout B 分支）。

- [ ] **Step 3: 造最小 traces + endpoint_health CSV**

复用现有 mini CSV 的列结构（见 `tests/fixtures/mini_data_root/Normal/_pipeline_out/*.csv` 表头）。`tt_traces_red_15s.csv` 与 `tt_endpoint_health_15s.csv` 各造 2–3 行，`endpoint_key=POST:/api/v1/travelservice/trips/left`，`timestamp_window` 跨 inject 窗口（如 `1783149990000`(baseline) / `1783150005000`(inject) / `1783150020000`(recover)），保证 inner join 有交集且三 phase 都覆盖。具体列取值参照 Normal fixture 同列量级填即可（本 case 只验证 pipeline 跑通与 join，不验证数值）。

- [ ] **Step 4: 写 merged_mini.yaml**

```yaml
# tests/fixtures/merged_mini.yaml
name: merged_mini
roots:
  - tests/fixtures/mini_data_root
normal_source: tests/fixtures/mini_data_root
fused_window: 15s
```

> mini 的 Normal 和 endpoint_raw2 case 都在同一个 `mini_data_root` 下（fixture 无需真造两个物理 root——多 root 枚举逻辑已由 Task 3 的 `test_build_contract_multi_root` 单测覆盖，此处 e2e 重点验证"合并后 Normal 来源正确 + endpoint 级 case 进 eval"）。`normal_source` 指向同一 root。

- [ ] **Step 5: 改 test_e2e_smoke.py 用 `--dataset`（GREEN 前先 RED）**

把 `pipeline_out` fixture 与 `test_pipeline_reproducible` 里两处 build_contract 调用的：

```python
            "--data-root",
            str(MINI_DATA_ROOT),
```

改为：

```python
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/merged_mini.yaml"),
```

新增一个断言 endpoint 级 case 进 eval 且 Normal 来源正确的测试：

```python
def test_endpoint_level_case_in_eval_and_normal_source(pipeline_out):
    """V8: 合并后 eval_all 含 endpoint 级 mini case；train(Normal) 不含它。"""
    eval_df = pd.read_parquet(pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(pipeline_out["contract_dir"] / "train.parquet")
    assert "Lv_E_HTTPABORT_travel_mini" in set(eval_df["case_id"])
    assert (train_df["anomaly_type"] == "Normal").all()
    assert "Lv_E_HTTPABORT_travel_mini" not in set(train_df["case_id"])
```

- [ ] **Step 6: Run to verify it passes**

Run: `conda run -n interface pytest tests/test_e2e_smoke.py -v`
Expected: PASS（原有 V1–V7 + 新 V8）

- [ ] **Step 7: Commit**

```bash
git add tests/fixtures/mini_data_root/Lv_E_HTTPABORT_travel_mini tests/fixtures/merged_mini.yaml tests/test_e2e_smoke.py
git commit -m "$(cat <<'EOF'
[Test]: 合并 e2e——endpoint 级 mini case + merged_mini config

e2e_smoke 切到 --dataset merged_mini，新增 endpoint 级故障 mini case，
断言合并后 endpoint 级 case 进 eval、Normal 仅来自 anomod 侧。
EOF
)"
```

---

## Task 7: 全量测试回归

**Files:** 无（仅运行）

- [ ] **Step 1: 跑全量测试**

Run: `conda run -n interface pytest tests/ -v`
Expected: 全绿。重点确认这些不因本次改动破裂：
- `test_endpoint_service_mapping.py`（新名单）
- `test_e2e_smoke.py` / `test_e2e_toy_pipeline.py`（`--dataset` 入口）
- `test_build_contract_smoke.py`（若引用 `--data-root` 需在此暴露并修）

- [ ] **Step 2: 若有失败，逐个修复后重跑**

常见连锁点：任何硬编码 `--data-root` 的测试。全仓搜索确认无残留：

Run: `rg -n "data-root|data_root" scripts/ tests/`
Expected: 除本 plan 预期的注释/历史外无活跃 `--data-root` 调用。

- [ ] **Step 3: Commit（若 Step 2 有修复）**

```bash
git add -A
git commit -m "[Test]: 修复 --data-root→--dataset 迁移的残留测试引用"
```

---

## Task 8: 真实数据 dvc repro 收尾验证（手动）

**Files:** 无（运行 + 观测，产物写 artifacts/ 已 gitignore）

> **这是 design doc §3.6 收尾验证 + 偏差修正 4 的 log 命中率复核。** 用真实 28 case（anomod_v1 12 + endpoint_raw2 16）跑通全链路。

- [ ] **Step 1: 跑全链路**

Run: `conda run -n interface dvc repro build_contract train_v0 eval_v0`
Expected: 三 stage 成功。观测 build_contract 日志：
- `发现 28 个 case`（12 + 16）
- 每个 endpoint_raw2 case 的 `log join 命中率` **显著高于 2%–27%**（raw2 log 已重采修复；若仍很低说明时区/对齐或 fixture 外的新问题，需排查而非接受）
- 无 case 因 endpoint join 空而被跳过

- [ ] **Step 2: 看指标合理性**

Run: `conda run -n interface dvc metrics show`
Expected: `auroc` / `auprc` 非 NaN、非恰好 0.5。合并后样本类别分布合理（Normal 全程为负 + inject 窗为正）。

- [ ] **Step 3: 确认合并正确性**

Run:
```bash
conda run -n interface python -c "
import pandas as pd
ev = pd.read_parquet('artifacts/contract_v0/eval_all.parquet')
tr = pd.read_parquet('artifacts/contract_v0/train.parquet')
print('eval cases:', ev['case_id'].nunique(), '| train all Normal:', (tr['anomaly_type']=='Normal').all())
print('endpoint_raw2 cases in eval:', sum(ev['anomaly_type'].str.startswith('Lv_E_')==True))
print('Lv_E case count:', ev.loc[ev['anomaly_type'].str.startswith('Lv_E_'),'case_id'].nunique())
"
```
Expected: 16 个 `Lv_E_*` case 全在 eval；train 全为 Normal。

- [ ] **Step 4: 记录验证结论**（供后续 history entry 006 引用，暂记在本 plan 或 PR 描述）

记录：实际 case 数、log 命中率区间（raw vs raw2 对比）、AUROC/AUPRC、发现的任何新问题。

---

## Self-Review 结论

**Spec 覆盖**（对照 design doc §3）：
- §3.1 数据集配置层（方案 B）→ Task 1, 3, 4 ✓
- §3.2 endpoint 名单修正 → Task 2 ✓（service 名已实证核对）
- §3.3 Normal 与归一化不改代码 → Task 3 Step 6（加 config 保险，不动 fit 逻辑）✓
- §3.4 标签/schema 不动 → 全程未触碰 `_attach_label_columns` / contract schema ✓
- §3.5 五类测试 → Task 1（config loader）/ Task 2（名单回归）/ Task 3（多 root 枚举）/ Task 6（合并 e2e）/ Task 5（DVC 静态）✓
- §3.6 落地清单 + 收尾验证 → Task 5(dvc.yaml) + Task 8 ✓
- §5 遗留 TODO（③ per-endpoint 标签、log 重采、DVC 追踪）→ 明确划界不做，Task 8 记录 log 复核 ✓

**偏差留痕**：6 处偏差已在顶部"偏差修正"节列明，执行以本 plan 为准。

**类型一致性**：`DatasetConfig`(name/roots/normal_source/fused_window)、`load_dataset_config`、`_enumerate_cases_multi` 在 Task 1/3/4/6 中签名一致。

**无占位符**：所有代码步骤含完整代码或明确的列结构指引（Task 6 Step 3 CSV 因依赖现有 fixture 表头，给出取值规则而非逐字节内容——执行时照 Normal fixture 同列填）。
