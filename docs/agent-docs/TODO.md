# 仓库前期准备 TODO

> 当前阶段：前期基础设施搭建。目标是在开始任何实验之前，把"可复现、可追踪、可审计"的工程底座建好。

---

## 待完成项

- [ ] **1. 可复现环境**
  一条命令复现 baseline 的最低标准：`conda env create` + `pip install` + `python scripts/train.py --config configs/base.yaml`。
  包括：`environment.yml`、`requirements.txt`、`Makefile`（或 `run.sh`）、seed 统一封装。

- [x] **2. 数据集版本化 + Dataset Card**
  用 DVC 管理 `dataset/` 和 `data/processed/`，`.dvc` 文件入 git，原始数据不入 git。
  撰写 Dataset Card（数据来源、字段说明、已知问题、引用方式）。

- [ ] **3. 实验追踪（MLflow 或 W&B）**
  选型并接入：每次 run 自动记录 config snapshot、git commit hash、数据集版本、random seed、metrics。
  训练脚本与追踪解耦（通过 callback 或 logger wrapper）。

- [x] **4. 配置系统**
  用 Hydra（或纯 YAML + argparse）把所有超参数从代码中剥离到 `configs/`。
  支持命令行覆盖（`--override key=val`），禁止代码内硬编码任何实验参数。

- [ ] **5. GitHub CI**
  配置 `.github/workflows/ci.yml`：push/PR 时自动跑 `pytest tests/`、lint（black + isort）、以及一次 smoke-test（小数据量跑通完整 pipeline）。
  确保仓库主干随时可运行。

- [ ] **6. 可复现承诺清单（NeurIPS Reproducibility Checklist）**
  对照 NeurIPS reproducibility checklist，逐条声明本项目的覆盖情况（代码、数据、超参数、随机性、硬件依赖等），写入 `docs/reproducibility.md`。

---

## 完成记录

_暂无_
