# 002 · 搭建工程基础设施（PR #1）

- **日期**: 2026-06-22
- **PR**: #1 · **Commit**: b3d6f37
- **类型**: Chore（基础设施）
- **影响域**: `pyproject.toml`, `Makefile`, `environment.yml`, `.dvc/`, `configs/`, `src/utils/`, `.github/workflows/`, `CLAUDE.md`

## 做了什么

把仓库从「空骨架」变成「可装可跑可复现」的工程基线。三件事并进：

1. **数据组织**：DVC 接管 `data/`，重组为按数据集为原子单元（`data/anomod/`、`data/lo2-sample/`）。本地 remote 指向 `~/dvc-remote/interface-health`
2. **配置与包**：`pyproject.toml`（可 `pip install -e ".[dev]"`） + Hydra `configs/`（`base.yaml` 为入口）+ 所有 `src/` 子包 `__init__.py`
3. **复现栈**：`environment.yml` + `Makefile`（`setup`/`sync`/`torch-cpu`/`torch-gpu`） + GitHub Actions CI（lint + pytest） + `src/utils/seed.py` + `src/utils/logger.py`（MLflow wrapper）

## 关键决策（不在 commit 里）

- **DVC remote 用本地路径，不挂 S3**：科研单机环境，避免引入云依赖；后续团队协作再切换
- **数据组织从"按处理阶段"（raw/processed）改为"按数据集"**：因为后续融合需要在同一数据集内对齐多模态，跨阶段拷贝反而引发版本错配
- **torch 不进 environment.yml，单独走 Makefile**：torch 的 CPU/GPU/CUDA 版本差异太大，锁死任何一个都会卡设备迁移
- **`drain3` → `drain3-improved`**：原 `drain3` 锁 `cachetools==4.2.1`，与 `mlflow 3.x` 冲突。import 路径相同，无侵入替换
- **mlflow 2.22.1 → 3.13.0**：跟随上面 cachetools 解锁的连锁升级，同时拿到 mlflow 3 的新 UI
- **CI 不做 smoke test**：曾经加过一个跑 toy training 的 smoke job，发现意义不大且拖慢 CI，移除（见 25cde39）

## 坑 / 已知问题

- **Kiro/Claude 在 base conda 环境运行**，验证 `interface` 环境必须 `conda run -n interface python -c ...`，直接 `python` 走的是 base——这一条已写入 CLAUDE.md "Known Gotchas"
- **`tests/__init__.py` 必须存在**：CI lint 工具会扫不到无 `__init__.py` 的 tests 目录，引发奇怪报错（见 2ccdcb0）
- **PR review 提的三处问题**（已在 754a60d 修）：
  1. `ci.yml` lint 范围缺 `scripts/`
  2. `anomod.yaml` 训练数据注释和 CLAUDE.md 任务定义口径不一致
  3. 占位测试改成真正的 `set_seed` 可复现性测试，不能只是 `assert True`

## 遗留 TODO

- pipeline 还不能端到端跑通（只有骨架，没有具体 train/eval 脚本） → 见 entry 003
- contracts 模块尚未引入，train↔eval 接口约定还是隐式的 → 见 entry 003
- pre-commit 配置在 entry 001 里有，但 CI 里没显式跑 pre-commit，仅跑 black/isort——后续考虑统一
