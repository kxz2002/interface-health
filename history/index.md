# Project History · Index

本仓库历次 PR/迭代的轻量索引。**开发新功能前先扫这里**，按影响域找相关 entry，再决定是否打开详细文档。

详细文档放 [`entries/`](./entries/)。

## 使用约定

### 对 agent
1. **开发新功能前**：浏览下方 entry 列表与影响域索引，识别本次任务可能影响到的过去工作。如有相关 entry，打开对应文档查阅"关键决策 / 坑 / 遗留 TODO"。
2. **PR merge 前**：在 `entries/` 下新增一篇 `NNN-<slug>.md`，并在本 index 里追加一行（含影响域、一句话摘要）。entry 编号严格递增。

### 对 entry 模板
每篇 entry 须包含以下结构（参考 [`entries/_template.md`](./entries/_template.md)）：

- 元数据：日期、PR/Commit、类型、影响域
- **做了什么**：1–2 段
- **关键决策（不在 commit 里）**：commit message 不会写的"为什么这么选 / 否定了什么选项"
- **坑 / 已知问题**：踩过的雷、PR review 抓到的问题、与文档约定相悖的细节
- **遗留 TODO**：本次未做但与本次紧密相关、未来必然要处理的事

**写作原则**：commit message 已经讲过的"做了什么"压缩成一两句，把 commit 不会写的"为什么 / 哪里坑 / 留下了什么"放大成主体。这是 history 相对 git log 的真正价值。

---

## Entry 列表

| # | 日期 | 类型 | 标题 | 影响域 |
|---|------|------|------|--------|
| [001](./entries/001-project-init.md) | 2026-05-10 | Feature | 初始化项目结构与基础设施 | 仓库骨架, CLAUDE.md, scripts/lo2-scripts/ |
| [002](./entries/002-infrastructure.md) | 2026-06-22 | Chore (PR #1) | 搭建工程基础设施 | DVC, Hydra, pyproject, Makefile, environment.yml, CI, src/utils/, CLAUDE.md |
| [003](./entries/003-experiment-loop.md) | 2026-06-24 | Chore (PR #2) | 最小可复现实验闭环（迭代 2） | src/contracts/, scripts/train_baseline.py, scripts/eval.py, dvc.yaml, artifacts/, tests/, CLAUDE.md |
| [004](./entries/004-project-history.md) | 2026-06-24 | Feature | 引入 Project History 历史记忆系统 | history/, skills-local/, Makefile, CLAUDE.md |
| [005](./entries/005-contract-v0.md) | 2026-06-26 | Feature | Contract v0 多模态数据融合接口墙（全链路 18-dim + Deep SVDD baseline） | src/contracts/, src/preprocessors/, src/fusion/, src/models/, src/data/, scripts/, configs/contract/, dvc.yaml, tests/ |

---

## 影响域索引（按目录/主题倒排）

> 改动某个目录时，按这张表找过往相关 entry。同一域可能有多条，按时间排序。

| 目录 / 主题 | 相关 entries |
|-------------|--------------|
| `CLAUDE.md` | 001, 002, 003, 004 |
| `data/` 组织与 DVC | 002 |
| `configs/` (Hydra) | 002 |
| `src/utils/` (seed, logger) | 002 |
| `src/contracts/` | 003 |
| `scripts/` (train, eval) | 003 |
| `scripts/lo2-scripts/` | 001 |
| `docs/agent-docs/` | 001 |
| `docs/plans/` | 001 |
| `artifacts/` | 003 |
| `dvc.yaml` / DVC pipeline | 002 (初始化), 003 (定义 stage) |
| `environment.yml` / `Makefile` | 002 |
| `.github/workflows/ci.yml` | 002, 004 |
| `tests/` | 002 (占位), 003 (契约/e2e) |
| `pyproject.toml` | 002 |
| `history/` + `skills-local/` | 004 |
| `Makefile` | 002 (环境), 004 (install-skills) |
| 多模态融合（`src/fusion/`） | 005 |
| 模型实现（`src/models/`） | 005 |
| 数据 loader（`src/data/`） | 005 |
| 评估指标细化（per-endpoint, phase 对齐） | 005 |
| `src/preprocessors/` | 005 |
| `configs/contract/` | 005 |

---

## 横切主题（cross-cutting）

> 不绑定某个目录、但反复出现的约束或踩坑模式。

- **可复现性**：seed 同时设 PyTorch + NumPy（002）；DVC stage 必须显式传 seed（003）
- **契约层**：train↔eval 解耦，字段口径必须经 contract 校验（003）；contract 版本即接口版本，破坏字段口径必须升版
- **CLI 合约**：脚本失败必须 `exit 1` 而不是只 log；JSON 输出必须 `allow_nan=False`（003）
- **PR review 反复抓到的类别**：silent failure、测试覆盖不足、文档/代码口径漂移
- **环境管理**：Kiro/Claude 默认 base conda，验证 `interface` 环境用 `conda run -n interface`（002）
