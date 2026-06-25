# 001 · 初始化项目结构与基础设施

- **日期**: 2026-05-10
- **Commit**: 1d8eb25
- **类型**: Feature
- **影响域**: `docs/`, `scripts/lo2-scripts/`, `CLAUDE.md`, 仓库骨架

## 做了什么

仓库第一次提交。建目录骨架 + 写 CLAUDE.md + 引入 LO2 数据集相关辅助脚本。

- 建立 `src/ configs/ tests/ docs/ scripts/` 等空目录
- 写第一版 `CLAUDE.md`：确立研究背景（Train-Ticket + One-Class 异常检测方向）
- 配置 pre-commit（isort + black）、`.gitignore`
- 引入 `scripts/lo2-scripts/`：LO2 数据集相关的一次性脚本（csv 合并、降采样、PCA、size 分布等）
- 引入 `docs/agent-docs/`：LO2 数据集说明（`lo2-dataset-readme.md`、`lo2-scripts.md`）
- 引入 `docs/plans/2026-05-08-lit-review-and-research-directions.md`：文献调研笔记

## 关键决策（不在 commit 里）

- **研究方向定位（写进 CLAUDE.md）**：
  - 论文创新点 1（主）：per-endpoint × time-window 单类异常检测——文献中几乎空白
  - 论文创新点 2（次）：多模态 RED 特征融合（客户端 api_responses + 服务端 traces）
- **两个走向**：场景 A（SAT 项目拿到）→ 真实业务数据 / 场景 B（拿不到）→ Train-Ticket。当前默认按 B 推进
- **为什么选 One-Class**：审稿人会质疑监督方法依赖故障注入标注，无法泛化到工业场景

## 坑 / 已知问题

- LO2 相关脚本是从外部环境直接搬进来的，**不属于主 pipeline**，未来主流程是 anomod（Train-Ticket）方向。LO2 只是备选数据源占位
- 没有 Python 包结构（没有 `pyproject.toml`、`__init__.py`），`src.*` 还不能 import——这一步留到 PR #1 解决

## 遗留 TODO

- 包结构、Hydra、DVC、CI、可复现性工具链 → 见 entry 002
