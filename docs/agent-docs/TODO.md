# 最小可复现实验闭环 TODO（迭代 2）

> 目标：在模型尚未确定的前提下，建立稳定的 train → eval 接口骨架，使 CI 可在不依赖真实数据的情况下端到端跑通。
>
> 任务类型：per-endpoint × time-window 的单类异常检测（One-Class），训练只用正常数据，评估用 `phase == inject` 强标注。
>
> scores 格式：Parquet（需补充 pyarrow 依赖）。

---

## 待完成项

- [x] **1. 补充 pyarrow 依赖**
  `pyproject.toml` 和 `environment.yml` 同步添加 `pyarrow>=14.0`。
  原因：`artifacts/scores.parquet` 依赖 pyarrow，当前两个文件均缺失。

- [x] **2. Contract 校验器**
  新增 `src/contracts/scores_v0.py` 和 `src/contracts/metrics_v0.py`。
  - `validate_scores_df(df)`：检查必需列（`sample_id, score, y_true`）、dtype、无空值、score 可转 float
  - `validate_metrics_dict(d)`：检查必需字段存在、类型正确、auroc 在 [0,1] 或 null
  - v0 固定 `higher_is_more_anomalous=True`；schema 留扩展余地（可选字段允许存在）

- [x] **3. 评测脚本 `scripts/eval.py`**
  输入：`artifacts/scores.parquet`（默认路径可覆盖）。
  行为：读取 → contract 校验 → 计算 AUROC/AUPRC（sklearn）→ 写 `artifacts/metrics.json`。
  指标格式符合 metrics contract v0（含 `protocol_version: v0`）。
  score 分布统计（mean/std/quantiles）作为额外诊断字段一并写入（v0 内标签恒在）。

- [x] **4. Toy Baseline 脚本 `scripts/train_baseline.py`**
  仅实现 `--toy` 模式：合成高斯数据（normal vs anomaly 分布不同），score = 距均值 L2，生成 `artifacts/scores.parquet`（含 `y_true`）。
  目的：让 DVC pipeline 和 CI 在没有真实模型时跑通。

- [x] **5. DVC pipeline 骨架 `dvc.yaml`**
  两个 stage：
  - `train`：运行 `train_baseline.py --toy`，产出 `artifacts/scores.parquet`
  - `eval`：运行 `eval.py`，产出 `artifacts/metrics.json`
  明确写出 `deps` 和 `outs`，`dvc repro` 可从头跑通。
  未来替换真实模型只需改 `train` stage 命令，eval 接口不变。

- [x] **6. Tests**
  新增三个测试文件：
  - `tests/test_scores_contract.py`：校验器对合法 df 通过、对缺列失败
  - `tests/test_metrics_contract.py`：校验器对合法 dict 通过
  - `tests/test_e2e_toy_pipeline.py`：运行 toy baseline + eval，断言 `metrics.json` 存在且 `auroc ∈ [0,1]`
  所有测试不依赖真实数据，CI 3 分钟内跑完。

---

## 验收标准

1. `pytest -q` 全通过
2. `python scripts/train_baseline.py --toy` 生成 `artifacts/scores.parquet`
3. `python scripts/eval.py` 生成 `artifacts/metrics.json`，含 `protocol_version: v0`，`auroc ∈ [0,1]`
4. `dvc repro` 能从头跑通 `train → eval`
5. pre-commit lint/format 无红线
