# 30/60 分钟重采 Normal 数据接入 DVC pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 用两份重采 Normal 数据（`data/normal_0711_30`、`data/normal_0711_60`）完全替换 `anomod_v1/Normal` 作为训练/评估用的正常样本来源，并接入 DVC 的 `build_contract` 阶段。

**Architecture:** 新建 `data/normal_v2/` wrapper 目录承载两份重采数据（使其满足 `_enumerate_cases` 对"root 下一层是具名 case 目录"的假设），归档旧 `anomod_v1/Normal` 到 `anomod_v1/_archive/Normal`（多套一层目录深度使其不再被扫描到，不用改代码）。新建 `configs/data/merged_v2.yaml` 声明三 root 合并（`anomod_v1` + `endpoint_raw2` + `normal_v2`），`dvc.yaml` 的 `build_contract` 阶段切换过去。不改动任何 Python 代码。

**Tech Stack:** Python 3 / pandas / pytest / DVC / YAML

**参考 spec:** `docs/superpowers/specs/2026-07-13-normal-v2-ingestion-design.md`

---

## File Structure Overview

本次改动涉及的文件（无新增 Python 代码，只有数据、配置、测试 fixture、文档）：

- **数据（物理移动，不进版本控制）**
  - `data/normal_v2/normal_0711_30/`（原 `data/normal_0711_30/` 整体移入）
  - `data/normal_v2/normal_0711_60/`（原 `data/normal_0711_60/` 整体移入）
  - `data/anomod_v1/_archive/Normal/`（原 `data/anomod_v1/Normal/` 整体移入）
- **配置**
  - `configs/data/merged_v2.yaml`（新建）
  - `dvc.yaml`（修改：`build_contract` 阶段的 `cmd` 与 `deps`）
- **测试**
  - `tests/fixtures/merged_v2_mini/`（新建：模拟三 root + wrapper 结构的最小 fixture）
  - `tests/fixtures/merged_v2_mini.yaml`（新建：对应的 dataset config fixture）
  - `tests/test_dataset_config.py`（修改：新增一个验证"root 下一层是 wrapper，wrapper 下才是具名 case"这种结构能正常被 `_enumerate_cases_multi` 处理的测试；这个假设本来就成立，用测试钉住，防止未来有人改动枚举逻辑破坏它）
  - `tests/test_build_contract_multi_root.py`（修改：新增验证归档目录深度足够、不会被 `_enumerate_cases_multi` 扫到的测试）
  - `tests/test_e2e_smoke.py`（修改：`merged_pipeline_out` fixture 切到新的多 root fixture，验证 Normal case 数与 case_id 集合）

  `tests/test_dvc_pipeline_v0.py` 的 `test_build_contract_uses_dataset_config` 是从 `dvc.yaml` 的 `cmd` 里动态解析 `--dataset` 路径并断言其存在、且在 `deps` 里，不硬编码文件名，因此切到 `merged_v2.yaml` 后该测试无需修改，会自动覆盖新路径。
- **文档**
  - `CLAUDE.md`（修改：数据集描述、目录结构注释）
  - `history/entries/009-normal-v2-ingestion.md`（新建）
  - `history/index.md`（修改：新增一行 + 影响域索引）

---

## Task 1: 移动数据到 wrapper / 归档旧 Normal（数据层，无代码改动）

**Files:**
- Move: `data/normal_0711_30/` → `data/normal_v2/normal_0711_30/`
- Move: `data/normal_0711_60/` → `data/normal_v2/normal_0711_60/`
- Move: `data/anomod_v1/Normal/` → `data/anomod_v1/_archive/Normal/`

这一步是纯文件系统操作。`data/anomod_v1/` 整体命中 `.gitignore:58`（`data/anomod_v1/`），`data/normal_0711_30`、`data/normal_0711_60`、`data/endpoint_raw2` 均未被 `.gitignore` 排除但也从未 `git add` 过（保持 untracked，符合项目现有惯例）。三者都不受 git 版本控制，因此本任务用 `mv` 而非 `git mv`，也不需要 git commit。

- [ ] **Step 1: 移动前核对当前目录结构**

```bash
find data/normal_0711_30 data/normal_0711_60 -maxdepth 1
find data/anomod_v1 -maxdepth 1
```

Expected: 前者输出两个目录各自的 5 个子目录（`api_responses/ log_data/ metric_data/ _pipeline_out/ trace_data/`）；后者输出 `data/anomod_v1` 下的 12 个 case 子目录（含 `Normal`）。

- [ ] **Step 2: 新建 wrapper 目录并移动两份重采数据**

```bash
mkdir -p data/normal_v2
mv data/normal_0711_30 data/normal_v2/
mv data/normal_0711_60 data/normal_v2/
```

- [ ] **Step 3: 归档旧 Normal case**

```bash
mkdir -p data/anomod_v1/_archive
mv data/anomod_v1/Normal data/anomod_v1/_archive/
```

- [ ] **Step 4: 验证移动后的结构符合预期**

```bash
find data/normal_v2 -maxdepth 2
find data/anomod_v1/_archive -maxdepth 2
ls data/anomod_v1 | grep -v _pipeline_out
```

Expected:
- `data/normal_v2` 下恰好两个子目录：`normal_0711_30`、`normal_0711_60`，各自内部结构与移动前一致（5 个子目录不变）。
- `data/anomod_v1/_archive` 下恰好一个子目录：`Normal`，内部结构不变（含 `case_metadata.json` 在 `Normal/trace_data/` 下）。
- `data/anomod_v1` 顶层列表不再包含 `Normal`，改为包含 `_archive`。

- [ ] **Step 5: 验证归档目录不会被现有枚举逻辑扫到（用真实代码路径核对，不改代码）**

```bash
python3 -c "
from pathlib import Path
root = Path('data/anomod_v1')
cases = sorted(p.parent for p in root.glob('*/_pipeline_out') if p.is_dir())
print([c.name for c in cases])
assert 'Normal' not in [c.name for c in cases]
assert '_archive' not in [c.name for c in cases]
print('OK: 归档后的 Normal 不会被 _enumerate_cases 扫到')
"
```

Expected: 打印 11 个 case 名（`Lv_D_cachelimit` 等，不含 `Normal`），末尾打印 `OK: ...`。这一步复现的正是 `scripts/build_contract.py:47-51` 的 `_enumerate_cases` 逻辑（`root.glob("*/_pipeline_out")`），用来在改配置之前先确认数据层改动本身是有效的。

- [ ] **Step 6: 验证 normal_v2 能被同一逻辑扫到 2 个 case**

```bash
python3 -c "
from pathlib import Path
root = Path('data/normal_v2')
cases = sorted(p.parent for p in root.glob('*/_pipeline_out') if p.is_dir())
names = [c.name for c in cases]
print(names)
assert names == ['normal_0711_30', 'normal_0711_60']
print('OK: normal_v2 下两个 case 都能被扫到')
"
```

Expected: 打印 `['normal_0711_30', 'normal_0711_60']`，末尾打印 `OK: ...`。

无 git commit（这些目录不受版本控制）。下一个任务（Task 2）会新建 `configs/data/merged_v2.yaml` 引用这里的新路径。

---

## Task 2: 新建 `configs/data/merged_v2.yaml`

**Files:**
- Create: `configs/data/merged_v2.yaml`
- Reference: `configs/data/merged_v1.yaml`（保持不变，不要修改）

`merged_v1.yaml` 的 `roots` 只有 `anomod_v1` + `endpoint_raw2`，`normal_source` 指向 `anomod_v1`。`merged_v2.yaml` 在此基础上把 `anomod_v1` 换成三 root（`anomod_v1` 保留但其 Normal 已归档、不再贡献 Normal case；新增 `normal_v2`），并把 `normal_source` 指向 `normal_v2`。

- [ ] **Step 1: 创建 merged_v2.yaml**

```yaml
# configs/data/merged_v2.yaml
# anomod_v1 (11 个 service 级故障 case，Normal 已归档至 _archive/，不参与枚举)
# + endpoint_raw2 (16 个 endpoint 级故障 case)
# + normal_v2 (2 个重采 Normal case：30min + 60min，替换 anomod_v1 原 Normal
#   —— 原 Normal 因 cadvisor 断流导致 metric 模态窗口内 0 覆盖，详见
#   docs/superpowers/specs/2026-07-13-normal-v2-ingestion-design.md)
name: merged_v2
description: anomod_v1 (11 case) + endpoint_raw2 (16 case) + normal_v2 (2 个重采 Normal case)
roots:
  - data/anomod_v1
  - data/endpoint_raw2
  - data/normal_v2
normal_source: data/normal_v2
fused_window: 15s
```

- [ ] **Step 2: 验证 YAML 能被 `load_dataset_config` 正确解析**

```bash
python3 -c "
from src.data.dataset_config import load_dataset_config
cfg = load_dataset_config('configs/data/merged_v2.yaml')
print(cfg)
assert cfg.name == 'merged_v2'
from pathlib import Path
assert cfg.roots == (Path('data/anomod_v1'), Path('data/endpoint_raw2'), Path('data/normal_v2'))
assert cfg.normal_source == Path('data/normal_v2')
print('OK: merged_v2.yaml 解析正确')
"
```

Expected: 打印 `DatasetConfig(name='merged_v2', roots=(...), normal_source=PosixPath('data/normal_v2'), fused_window='15s')`，末尾打印 `OK: ...`。这一步验证 `normal_source ∈ roots` 的校验（`src/data/dataset_config.py:23-26`）能通过——`data/normal_v2` 确实在 `roots` 列表里。

- [ ] **Step 3: 用真实数据跑一次 `_enumerate_cases_multi`，确认三 root 合计枚举出 29 个 case，且不含被归档的旧 Normal**

```bash
python3 -c "
from src.data.dataset_config import load_dataset_config
from scripts.build_contract import _enumerate_cases_multi

cfg = load_dataset_config('configs/data/merged_v2.yaml')
cases = _enumerate_cases_multi(cfg.roots)
names = sorted(c.name for c in cases)
print(f'总数: {len(names)}')
print(names)
assert len(names) == 29, f'期望 29 个 case，实际 {len(names)}'
assert 'Normal' not in names, '归档后的旧 Normal 不应出现'
assert 'normal_0711_30' in names and 'normal_0711_60' in names
print('OK: 三 root 合计 29 个 case，旧 Normal 已排除，新 Normal 已接入')
"
```

Expected: 打印 `总数: 29`，case 名列表里包含 `normal_0711_30`、`normal_0711_60`，不包含 `Normal`；末尾打印 `OK: ...`。若数量不是 29，先检查 Task 1 的移动是否完整执行（`data/anomod_v1` 应剩 11 个非 `_pipeline_out`/`_archive` 子目录，`data/endpoint_raw2` 应有 16 个，`data/normal_v2` 应有 2 个）。

- [ ] **Step 4: 提交**

```bash
git add configs/data/merged_v2.yaml
git commit -m "$(cat <<'EOF'
[Data]: 新增 merged_v2 数据集配置

三 root 合并：anomod_v1（Normal 已归档）+ endpoint_raw2 + normal_v2
（30/60 分钟重采 Normal，替换因 cadvisor 断流导致 metric 全窗口 0
覆盖的旧 Normal）。merged_v1.yaml 保留作历史快照，不动。
EOF
)"
```

---

## Task 3: 切换 `dvc.yaml` 的 `build_contract` 阶段到 `merged_v2.yaml`

**Files:**
- Modify: `dvc.yaml:19-35`

当前 `build_contract` 阶段（`dvc.yaml:19-35`）：

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

- [ ] **Step 1: 修改 `cmd` 和 `deps` 里的数据集路径**

```yaml
  build_contract:
    cmd: python scripts/build_contract.py --config configs/contract/v0.yaml --dataset
      configs/data/merged_v2.yaml --out-dir artifacts/contract_v0 --seed 42
    deps:
      - scripts/build_contract.py
      - src/preprocessors
      - src/contracts/contract_v0.py
      - src/data/normalization.py
      - src/data/dataset_config.py
      - configs/contract/v0.yaml
      - configs/contract/endpoint_to_service.yaml
      - configs/data/merged_v2.yaml
    outs:
      - artifacts/contract_v0/train.parquet
      - artifacts/contract_v0/eval_all.parquet
      - artifacts/contract_v0/normalization_stats.json
      - artifacts/contract_v0/schema.json
```

只改这两处（`cmd` 里的 `--dataset` 值、`deps` 列表里的文件名），`train_v0`、`eval_v0` 两个阶段的定义不动。

- [ ] **Step 2: 验证 `dvc.yaml` 仍是合法 YAML，且改动符合既有测试的断言**

```bash
python3 -c "
import yaml
pipeline = yaml.safe_load(open('dvc.yaml'))
stage = pipeline['stages']['build_contract']
assert 'configs/data/merged_v2.yaml' in stage['cmd']
assert 'configs/data/merged_v1.yaml' not in stage['cmd']
assert 'configs/data/merged_v2.yaml' in stage['deps']
print('OK: dvc.yaml build_contract 阶段已指向 merged_v2')
"
```

Expected: 打印 `OK: ...`。这一步复现的正是 `tests/test_dvc_pipeline_v0.py::test_build_contract_uses_dataset_config` 的断言逻辑（该测试本身无需修改，因为它是从 `dvc.yaml` 动态读取路径，不硬编码文件名）。

- [ ] **Step 3: 跑 `test_dvc_pipeline_v0.py` 确认现有测试仍然全部通过**

```bash
pytest tests/test_dvc_pipeline_v0.py -v
```

Expected: 3 个测试全部 `PASSED`（`test_dvc_yaml_has_v0_stages`、`test_dvc_v0_stages_have_required_fields`、`test_build_contract_uses_dataset_config`）。

- [ ] **Step 4: 提交**

```bash
git add dvc.yaml
git commit -m "$(cat <<'EOF'
[Data]: dvc build_contract 阶段切到 merged_v2 数据集

Normal 数据来源从 anomod_v1（metric 断流）切换为 normal_v2（30/60
分钟重采，metric 覆盖完整）。
EOF
)"
```

---

## Task 4: 新建 `tests/fixtures/merged_v2_mini/` + `tests/fixtures/merged_v2_mini.yaml`

**Files:**
- Create: `tests/fixtures/merged_v2_mini/anomod_like/Lv_P_DISKIO_preserve/`（复制自 `tests/fixtures/mini_data_root/Lv_P_DISKIO_preserve/`，case 级 fallback 标签场景）
- Create: `tests/fixtures/merged_v2_mini/anomod_like/_archive/Normal_old/_pipeline_out/`（空目录，模拟归档后的旧 Normal，验证深度排除）
- Create: `tests/fixtures/merged_v2_mini/endpoint_like/Lv_E_HTTPABORT_assurance_mini/`（复制自 `tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini/`，endpoint 级精确标签场景）
- Create: `tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_30_mini/`（改编自 `tests/fixtures/mini_data_root/Normal/`）
- Create: `tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_60_mini/`（同上）
- Create: `tests/fixtures/merged_v2_mini.yaml`

这个 fixture 在测试规模上复现真实场景的三层结构：`anomod_like`（对应真实 `anomod_v1`，Normal 已归档、剩 1 个 case 级标签 case）+ `endpoint_like`（对应真实 `endpoint_raw2`，1 个 endpoint 级标签 case）+ `normal_v2_root`（对应真实 `normal_v2` wrapper，2 个 Normal case）。用于 Task 5 中 `tests/test_e2e_smoke.py` 的完整 pipeline 集成测试。

复制时不需要修改 CSV 内部的 `case_id` 列——`TracePreprocessor.transform()`/`ApiPreprocessor.transform()` 只选取 `endpoint_key`/`timestamp_window_ms`/RED 特征列，CSV 里的 `case_id` 列不会被读取（真正的 case 标识来自 `case_metadata.json` 或目录名，见 `scripts/build_contract.py:84-91` 的 `_load_case_meta`）。

- [ ] **Step 1: 复制 `Lv_P_DISKIO_preserve`（case 级标签 case）到 `anomod_like/`**

```bash
mkdir -p tests/fixtures/merged_v2_mini
cp -r tests/fixtures/mini_data_root/Lv_P_DISKIO_preserve tests/fixtures/merged_v2_mini/anomod_like/Lv_P_DISKIO_preserve
```

- [ ] **Step 2: 新建归档目录（空 `_pipeline_out`，验证深度排除机制）**

```bash
mkdir -p tests/fixtures/merged_v2_mini/anomod_like/_archive/Normal_old/_pipeline_out
```

`_enumerate_cases_multi` 对 `anomod_like` 这个 root 执行 `root.glob("*/_pipeline_out")`，只会匹配 `anomod_like/*/_pipeline_out`（一层深）。`Normal_old` 的 `_pipeline_out` 在 `anomod_like/_archive/Normal_old/_pipeline_out`，比这深一层，不会被匹配到——不需要任何真实数据内容，因为这个目录永远不会被读取。

- [ ] **Step 3: 复制 `Lv_E_HTTPABORT_assurance_mini`（endpoint 级标签 case）到 `endpoint_like/`**

```bash
mkdir -p tests/fixtures/merged_v2_mini/endpoint_like
cp -r tests/fixtures/mini_data_root/Lv_E_HTTPABORT_assurance_mini tests/fixtures/merged_v2_mini/endpoint_like/Lv_E_HTTPABORT_assurance_mini
```

- [ ] **Step 4: 复制 `Normal` 两次到 `normal_v2_root/`，模拟 30min/60min 两个重采 Normal case**

```bash
mkdir -p tests/fixtures/merged_v2_mini/normal_v2_root
cp -r tests/fixtures/mini_data_root/Normal tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_30_mini
cp -r tests/fixtures/mini_data_root/Normal tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_60_mini
```

- [ ] **Step 5: 改写两份 `case_metadata.json`，区分 case_id 与 baseline_sec**

```bash
cat > tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_30_mini/case_metadata.json <<'EOF'
{
  "case_id": "normal_0711_30_mini",
  "anomaly_type": "Normal",
  "anomaly_level": "none",
  "target_service": null,
  "inject_start_ms": null,
  "inject_end_ms": null,
  "baseline_sec": 1800
}
EOF

cat > tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_60_mini/case_metadata.json <<'EOF'
{
  "case_id": "normal_0711_60_mini",
  "anomaly_type": "Normal",
  "anomaly_level": "none",
  "target_service": null,
  "inject_start_ms": null,
  "inject_end_ms": null,
  "baseline_sec": 3600
}
EOF
```

- [ ] **Step 6: 新建 `merged_v2_mini.yaml`**

```yaml
# tests/fixtures/merged_v2_mini.yaml
name: merged_v2_mini
roots:
  - tests/fixtures/merged_v2_mini/anomod_like
  - tests/fixtures/merged_v2_mini/endpoint_like
  - tests/fixtures/merged_v2_mini/normal_v2_root
normal_source: tests/fixtures/merged_v2_mini/normal_v2_root
fused_window: 15s
```

- [ ] **Step 7: 验证目录结构**

```bash
find tests/fixtures/merged_v2_mini -maxdepth 3 | sort
```

Expected（关键行，顺序可能不同）：
```
tests/fixtures/merged_v2_mini/anomod_like
tests/fixtures/merged_v2_mini/anomod_like/Lv_P_DISKIO_preserve
tests/fixtures/merged_v2_mini/anomod_like/_archive
tests/fixtures/merged_v2_mini/anomod_like/_archive/Normal_old
tests/fixtures/merged_v2_mini/endpoint_like
tests/fixtures/merged_v2_mini/endpoint_like/Lv_E_HTTPABORT_assurance_mini
tests/fixtures/merged_v2_mini/normal_v2_root
tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_30_mini
tests/fixtures/merged_v2_mini/normal_v2_root/normal_0711_60_mini
```

- [ ] **Step 8: 用真实枚举逻辑验证：4 个 case，2 个 Normal，归档目录不出现**

```bash
python3 -c "
from pathlib import Path
from src.data.dataset_config import load_dataset_config
from scripts.build_contract import _enumerate_cases_multi
import json

cfg = load_dataset_config('tests/fixtures/merged_v2_mini.yaml')
cases = _enumerate_cases_multi(cfg.roots)
names = sorted(c.name for c in cases)
print(names)
assert len(names) == 4, f'期望 4 个 case，实际 {len(names)}: {names}'
assert 'Normal_old' not in names, '归档目录不应被扫到'
assert 'normal_0711_30_mini' in names and 'normal_0711_60_mini' in names

normal_count = 0
for c in cases:
    meta_path = c / 'case_metadata.json'
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {'anomaly_type': 'Normal'}
    if str(meta.get('anomaly_type', '')).startswith('Normal'):
        normal_count += 1
assert normal_count == 2, f'期望 2 个 Normal case，实际 {normal_count}'
print('OK: merged_v2_mini 共 4 case，其中 2 个 Normal，归档目录已排除')
"
```

Expected: 打印 `['Lv_E_HTTPABORT_assurance_mini', 'Lv_P_DISKIO_preserve', 'normal_0711_30_mini', 'normal_0711_60_mini']`，末尾打印 `OK: ...`。

- [ ] **Step 9: 提交**

```bash
git add tests/fixtures/merged_v2_mini tests/fixtures/merged_v2_mini.yaml
git commit -m "$(cat <<'EOF'
[Test]: 新增 merged_v2_mini fixture

三 root 结构（anomod_like + endpoint_like + normal_v2_root wrapper）
在测试规模复现 merged_v2 场景，含归档目录深度排除验证。供
test_e2e_smoke.py 的合并 pipeline 集成测试使用。
EOF
)"
```

---

## Task 5: 扩展测试文件，覆盖 wrapper 结构与归档排除逻辑

**Files:**
- Modify: `tests/test_dataset_config.py`
- Modify: `tests/test_build_contract_multi_root.py`
- Modify: `tests/test_e2e_smoke.py`

三处改动分别验证：(1) `merged_v2.yaml` 式的三 root 配置能被 `load_dataset_config` 正确解析；(2) `_enumerate_cases_multi` 对"归档目录多套一层深度"的排除机制本身（unit 级，用 `tmp_path`，不依赖真实 fixture 文件）；(3) 用 Task 4 的 `merged_v2_mini` fixture 跑一次完整 pipeline，验证 Normal case 数、总 case 数、归档排除在 end-to-end 层面成立。

- [ ] **Step 1: `test_dataset_config.py` 新增三 root 配置解析测试**

在 `tests/test_dataset_config.py` 末尾追加：

```python
def test_load_three_root_config_with_wrapper_normal_source(tmp_path):
    cfg_path = tmp_path / "merged_v2.yaml"
    cfg_path.write_text(
        "name: merged_v2\n"
        "roots:\n"
        "  - data/anomod_v1\n"
        "  - data/endpoint_raw2\n"
        "  - data/normal_v2\n"
        "normal_source: data/normal_v2\n"
        "fused_window: 15s\n"
    )
    cfg = load_dataset_config(cfg_path)
    assert cfg.roots == (
        Path("data/anomod_v1"),
        Path("data/endpoint_raw2"),
        Path("data/normal_v2"),
    )
    assert cfg.normal_source == Path("data/normal_v2")
```

- [ ] **Step 2: 运行确认新测试通过，且不破坏既有测试**

```bash
pytest tests/test_dataset_config.py -v
```

Expected: 6 个测试全部 `PASSED`（原有 5 个 + 新增 1 个）。

- [ ] **Step 3: `test_build_contract_multi_root.py` 新增归档深度排除测试**

在 `tests/test_build_contract_multi_root.py` 末尾追加：

```python
def test_enumerate_cases_multi_root_archived_case_not_scanned(tmp_path):
    # 归档：把 case 从 root 直接子目录挪到 root/_archive/ 下多一层，
    # 验证 _enumerate_cases_multi（root.glob("*/_pipeline_out")，只扫一层）不会扫到它。
    root = tmp_path / "anomod_like"
    _make_case(root, "Lv_P_DISKIO_preserve")
    (root / "_archive" / "Normal_old" / "_pipeline_out").mkdir(parents=True)

    cases = _enumerate_cases_multi([root])

    names = [c.name for c in cases]
    assert names == ["Lv_P_DISKIO_preserve"]
    assert "Normal_old" not in names
```

- [ ] **Step 4: 运行确认新测试通过**

```bash
pytest tests/test_build_contract_multi_root.py -v
```

Expected: 4 个测试全部 `PASSED`（原有 3 个 + 新增 1 个）。

- [ ] **Step 5: `test_e2e_smoke.py` 新增 `merged_v2_pipeline_out` fixture**

在 `merged_pipeline_out` fixture 定义之后（`tests/test_e2e_smoke.py` 现有 `merged_pipeline_out` 函数结束处）追加一个新 fixture：

```python
@pytest.fixture(scope="module")
def merged_v2_pipeline_out(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e_merged_v2")
    contract_dir = out / "contract_v0"

    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(REPO_ROOT / "configs/contract/v0.yaml"),
            "--dataset",
            str(REPO_ROOT / "tests/fixtures/merged_v2_mini.yaml"),
            "--out-dir",
            str(contract_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )

    return {"contract_dir": contract_dir}
```

- [ ] **Step 6: 新增三个断言测试，验证 Normal 数量、总 case 数、归档排除**

紧接 Step 5 的 fixture 之后追加：

```python
def test_merged_v2_normal_case_count(merged_v2_pipeline_out):
    """V9: merged_v2 场景下 Normal case 数为 2，均来自 normal_v2_root。"""
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    assert (train_df["anomaly_type"] == "Normal").all()
    normal_case_ids = set(train_df["case_id"])
    assert normal_case_ids == {"normal_0711_30_mini", "normal_0711_60_mini"}


def test_merged_v2_total_case_count(merged_v2_pipeline_out):
    """V10: merged_v2 场景下全量 case 数为 4（1 case 级 + 1 endpoint 级 + 2 Normal）。"""
    eval_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    all_case_ids = set(eval_df["case_id"]) | set(train_df["case_id"])
    assert all_case_ids == {
        "Lv_P_DISKIO_preserve",
        "Lv_E_HTTPABORT_assurance_mini",
        "normal_0711_30_mini",
        "normal_0711_60_mini",
    }


def test_merged_v2_archived_case_excluded(merged_v2_pipeline_out):
    """V11: 归档的 Normal_old 不出现在任何输出 case_id 中。"""
    eval_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "eval_all.parquet")
    train_df = pd.read_parquet(merged_v2_pipeline_out["contract_dir"] / "train.parquet")
    all_case_ids = set(eval_df["case_id"]) | set(train_df["case_id"])
    assert "Normal_old" not in all_case_ids
```

- [ ] **Step 7: 运行完整 e2e 测试套件确认全部通过**

```bash
pytest tests/test_e2e_smoke.py -v
```

Expected: 全部测试 `PASSED`（原有 8 个 V1-V8 + 新增 V9/V10/V11 共 11 个）。

- [ ] **Step 8: 提交**

```bash
git add tests/test_dataset_config.py tests/test_build_contract_multi_root.py tests/test_e2e_smoke.py
git commit -m "$(cat <<'EOF'
[Test]: 覆盖三 root wrapper 结构与归档排除逻辑

新增：merged_v2 式三 root 配置解析测试、归档深度排除的 unit 测试、
基于 merged_v2_mini fixture 的端到端 Normal 数量/总 case 数/归档
排除断言（V9-V11）。
EOF
)"
```

---

## Task 6: 更新 `CLAUDE.md` 数据集描述与目录结构注释

**Files:**
- Modify: `CLAUDE.md`

`CLAUDE.md` 中有多处引用当前数据集组成（"28 个 case"、`normal_0711_30/60` "尚未接入"等），切换到 `merged_v2` 后这些描述会过时，需要同步更新。

- [ ] **Step 1: 更新数据集总览描述（原第 14 行附近）**

原文：
```
当前数据集：Train-Ticket 微服务系统，合并两个数据源共 28 个 case——`data/anomod_v1/`（12 case，1 Normal + 11 service 级故障注入）+ `data/endpoint_raw2/`（16 case，endpoint 级故障注入），由 `configs/data/merged_v1.yaml` 声明合并（`normal_source` 固定为 anomod_v1）。
```

改为：

```
当前数据集：Train-Ticket 微服务系统，合并三个数据源共 29 个 case——`data/anomod_v1/`（11 个 service 级故障注入 case，原 Normal 因 cadvisor 断流已归档至 `_archive/`，不参与训练评估）+ `data/endpoint_raw2/`（16 case，endpoint 级故障注入）+ `data/normal_v2/`（2 个重采 Normal case，30min+60min），由 `configs/data/merged_v2.yaml` 声明合并（`normal_source` 固定为 `data/normal_v2`）。历史快照 `configs/data/merged_v1.yaml`（Normal 取自 `anomod_v1`）保留不动，仅用于复现旧实验。
```

- [ ] **Step 2: 更新 Directory Structure 注释块**

原文（`anomod_v1`/`endpoint_raw2`/`normal_0711_30,60` 三行）：
```
├── anomod_v1/         # Train-Ticket service 级故障注入数据集（12 case 含 Normal，READ-ONLY，never modify）
│   ├── Normal/
│   ├── Lv_P_*/  Lv_S_*/  Lv_D_*/   # 11 个故障注入 case
│   └── <case>/_pipeline_out/         # pipeline 产物（tt_endpoint_health_15s.csv / tt_traces_red_15s.csv）
├── endpoint_raw2/     # endpoint 级故障注入数据集（16 case，log 重采修复版，READ-ONLY）
│   └── Lv_E_HTTP{ABORT,DELAY,PATCH,REPLACE}_{assurance,order,travel,travel2}/
├── endpoint_raw/      # endpoint_raw2 的旧版本，log 采集因 fsnotify watcher 耗尽而全崩（inject/recover 阶段零日志覆盖），已弃用不参与 pipeline
├── normal_0711_30/, normal_0711_60/  # 新增 Normal 采集，尚未接入任何 configs/data/*.yaml，暂不参与训练/评估
```

改为：
```
├── anomod_v1/         # Train-Ticket service 级故障注入数据集（READ-ONLY，never modify）
│   ├── _archive/Normal/              # 原 Normal case，因 cadvisor 断流导致 metric 模态窗口内 0 覆盖，已归档不参与枚举
│   ├── Lv_P_*/  Lv_S_*/  Lv_D_*/   # 11 个故障注入 case
│   └── <case>/_pipeline_out/         # pipeline 产物（tt_endpoint_health_15s.csv / tt_traces_red_15s.csv）
├── endpoint_raw2/     # endpoint 级故障注入数据集（16 case，log 重采修复版，READ-ONLY）
│   └── Lv_E_HTTP{ABORT,DELAY,PATCH,REPLACE}_{assurance,order,travel,travel2}/
├── endpoint_raw/      # endpoint_raw2 的旧版本，log 采集因 fsnotify watcher 耗尽而全崩（inject/recover 阶段零日志覆盖），已弃用不参与 pipeline
├── normal_v2/         # 30/60 分钟重采 Normal 数据（2 case，metric 15s 桶 100% 覆盖），替换 anomod_v1 原 Normal
│   ├── normal_0711_30/
│   └── normal_0711_60/
```

- [ ] **Step 3: grep 确认没有遗漏的过时引用**

```bash
grep -n "merged_v1\|28 个 case\|尚未接入\|暂不参与训练" CLAUDE.md
```

Expected: 无匹配，或匹配到的行是合理保留的历史说明（如果有，逐条确认是否需要改）。若 `merged_v1` 仍被提到（例如 Commands 章节的示例命令），保留——`merged_v1.yaml` 本身作为历史快照不删除，代码示例引用旧文件名是合理的历史记录，不算过时引用。

- [ ] **Step 4: 提交**

```bash
git add CLAUDE.md
git commit -m "$(cat <<'EOF'
[Docs]: 更新数据集描述为三源合并（29 case）

Normal 来源从 anomod_v1 切换为新的 normal_v2（30/60 分钟重采），
反映 merged_v2.yaml 的接入。
EOF
)"
```

---

## Task 7: 新增 `history/entries/009-normal-v2-ingestion.md` 并更新 `history/index.md`

**Files:**
- Create: `history/entries/009-normal-v2-ingestion.md`
- Modify: `history/index.md`

- [ ] **Step 1: 查看 entry 模板与 index 当前内容**

```bash
cat history/entries/_template.md
tail -30 history/index.md
```

- [ ] **Step 2: 创建 `history/entries/009-normal-v2-ingestion.md`**

按模板结构填写（元数据 / 做了什么 / 关键决策 / 坑与已知问题 / 遗留 TODO），内容基于本次实际改动与 `docs/superpowers/specs/2026-07-13-normal-v2-ingestion-design.md` 的决策依据：

```markdown
# 009: 30/60 分钟重采 Normal 数据接入 DVC pipeline

## 元数据
- 日期: 2026-07-13
- PR: #<待填：合并时回填>
- Commit: <待填：合并时回填>
- 类型: [Data]
- 影响域: 数据集配置 (`configs/data/`)、DVC pipeline (`dvc.yaml`)、测试 fixture (`tests/fixtures/`)

## 做了什么
- 用两份重采 Normal 数据（`data/normal_0711_30`、`data/normal_0711_60`，30min+60min）完全替换 `anomod_v1/Normal` 作为训练/评估用的正常样本来源。
- 新建 `data/normal_v2/` wrapper 目录承载两份重采数据，满足 `_enumerate_cases` 对"root 下一层是具名 case 目录"的假设。
- 归档旧 `anomod_v1/Normal` 到 `anomod_v1/_archive/Normal`，多套一层目录深度使其不再被 `_enumerate_cases` 扫到（数据物理保留，可逆，符合 `anomod_v1` 的 READ-ONLY 约束）。
- 新建 `configs/data/merged_v2.yaml`（三 root：`anomod_v1` + `endpoint_raw2` + `normal_v2`，`normal_source: data/normal_v2`），`dvc.yaml` 的 `build_contract` 阶段切换过去。`merged_v1.yaml` 保留不动，作历史快照。
- 无任何 Python 代码改动。

## 关键决策（不在 commit 里）
1. **为什么归档而非删除**：`anomod_v1/` 是 CLAUDE.md 明确约定的 READ-ONLY 目录；删除违反该约束且不可逆。归档（移动到更深一层目录）既排除了旧 Normal 参与训练，又保持数据物理可追溯、可逆。
2. **为什么不硬化 `normal_source` 为真过滤条件**：探查发现 `normal_source` 字段目前只用于日志（`build_contract.py` 里判定 Normal 的真实逻辑是全局扫描 `anomaly_type.str.startswith("Normal")`，与 `normal_source` 值无关）。本次没有把它升级为真过滤依据，因为归档已经从数据层解决了"旧 Normal 混入"的问题，升级为硬过滤在当前只有一个待排除来源的情况下是过度设计（YAGNI）。
3. **为什么新建 `merged_v2.yaml` 而不改 `merged_v1.yaml`**：`merged_v1.yaml` 被 `dvc.lock` 引用，代表一个已跑过的历史数据集组成快照；直接修改会破坏历史实验的可复现性。
4. **为什么用"多套一层目录"而不是"改名加下划线前缀"来排除旧 Normal**：验证过 `_enumerate_cases` 用的 `root.glob("*/_pipeline_out")` 会匹配任意名字的直接子目录（不区分是否有下划线前缀），只有真正增加一层目录深度才能让它被跳过。

## 坑/已知问题
- `normal_0711_30/60` 与 `anomod_v1/Normal` 是不同批次采集（前者 2026-07-11，后者 2026-06-29，`tt_max_workers` 也不同：4 vs 5），三者时间窗互不重叠，物理上是独立的 run——本次替换认为这对"正常基线"是可接受的（正常运行状态不依赖具体采集批次），但如果未来发现 `tt_max_workers` 影响特征分布，需要重新评估。
- `normal_0711_30/60` 的 `tt_traces_red_15s.csv` 比 `anomod_v1` 多 4 列（`endpoint_service`/`target_endpoint`/`anomaly_level`/`is_target_endpoint`，entry 008 引入的字段）。`TracePreprocessor.transform()` 已对 `is_target_endpoint` 缺失做兼容处理，无需改代码，但如果未来这类新增字段被真正用作特征输入，需要重新检查所有历史 case 是否都有该字段。

## 遗留 TODO
- `normal_source` 字段仍然只是日志用途、不参与真实过滤——如果未来出现第二个需要排除的 Normal 来源，应该考虑把它升级为真正的过滤条件（当前 YAGNI 判断可能需要重新评估）。
```

- [ ] **Step 3: 更新 `history/index.md`**

在条目列表追加一行（紧跟现有格式），并在"影响域"倒排索引中把 `configs/data/`、`dvc.yaml`、`tests/fixtures/` 对应条目补上 `009`。具体格式需对照 `history/index.md` 现有条目的确切写法（追加时保持列对齐/顺序一致，不要臆造新的列结构）。

- [ ] **Step 4: 提交**

```bash
git add history/entries/009-normal-v2-ingestion.md history/index.md
git commit -m "$(cat <<'EOF'
[Docs]: 新增 history entry 009（normal_v2 接入）

记录本次替换 Normal 数据来源的关键决策：归档而非删除、
normal_source 未硬化为过滤条件的理由。
EOF
)"
```

---

## Self-Review

**Spec 覆盖检查**（对照 `docs/superpowers/specs/2026-07-13-normal-v2-ingestion-design.md`）：
- §3.1 数据层改动（wrapper + 归档）→ Task 1 ✓
- §3.2 配置层改动（`merged_v2.yaml`，`normal_source` 不硬化）→ Task 2 ✓
- §3.3 Pipeline 层改动（`dvc.yaml`）→ Task 3 ✓
- §3.4 不改的部分（无代码改动）→ Task 1-3 均未涉及任何 `src/`/`scripts/` 代码改动，符合 ✓
- §3.5 文档更新（CLAUDE.md + history）→ Task 6、Task 7 ✓
- §4 测试（`test_dataset_config.py` 新 fixture、集成测试新增断言：Normal=2、归档排除、总 29 case）→ Task 4（fixture）+ Task 5（test_dataset_config.py / test_build_contract_multi_root.py / test_e2e_smoke.py 断言）✓

**Placeholder 扫描**：全部 Step 均含可直接执行的命令/代码与明确的 Expected 输出，未发现 "TBD"/"类似 Task N"/"添加适当的..." 等占位表述。Task 7 Step 3 中"具体格式需对照现有条目"这一句不是占位符，是提醒执行者去读一个当前未知的现有文件格式再写，属于必要的现场核对指引（`history/index.md` 的确切列结构未在本计划编写时读取，无法提前写死，但已给出明确的操作方式：追加一行、更新倒排索引、保持格式一致）。

**类型/签名一致性检查**：Task 4 产出的 case_id（`normal_0711_30_mini`/`normal_0711_60_mini`/`Lv_P_DISKIO_preserve`/`Lv_E_HTTPABORT_assurance_mini`）与 Task 5 测试断言中引用的 case_id 集合完全一致；`merged_v2_mini.yaml` 的 `name`/`roots`/`normal_source` 字段名与 `src/data/dataset_config.py` 的 `DatasetConfig` 字段定义一致。

---

**Plan 完成。** 已保存到 `docs/superpowers/plans/2026-07-13-normal-v2-ingestion.md`（Task 1-7 + Self-Review）。

两种执行方式：

1. **Subagent-Driven（推荐）** — 我为每个 Task 派发一个新 subagent，两阶段 review，快速迭代
2. **Inline Execution** — 在当前 session 里按 Task 批量执行，每个 checkpoint 停下来给你确认

你想用哪种方式？
