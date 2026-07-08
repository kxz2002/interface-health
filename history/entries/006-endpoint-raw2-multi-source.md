# 006 · endpoint_raw2 接入 Contract v0：多数据源合并 pipeline

- **日期**: 2026-07-08
- **PR**: #6 · **Commit**: 05a5afe
- **类型**: Feature
- **影响域**: `src/data/dataset_config.py`, `scripts/build_contract.py`, `configs/data/`, `configs/contract/endpoint_to_service.yaml`, `dvc.yaml`, `tests/`

## 做了什么

补记：本 entry 是 merge 后补写的，PR #6 当时未按约定写 history entry。

把 `build_contract.py` 从单一 `--data-root` 入口改为 `--dataset <config>` 入口，支持按 `configs/data/*.yaml` 声明的多个 root 枚举 case 后合并（方案 B，经 brainstorming 确定）。新增 `endpoint_raw2` 数据源，`merged_v1.yaml` 声明 `anomod_v1 + endpoint_raw2` 两个 root，Normal 仅取自 `anomod_v1`。同时修正了 `endpoint_to_service.yaml` 白名单为 dataset-guide §6 真实 8 个 client 入口（旧名单含 4 个死条目，使 baseline 只有 4 个活 endpoint）。

## 关键决策（不在 commit 里）

- **本次不动标签/schema**：per-endpoint 标签 + 评估拆为独立后续 PR，本次只做数据源合并，避免一次改动同时耦合"数据来源"和"评估粒度"两个维度。
- **Normal 只来自 anomod_v1**：因为 `endpoint_raw2` 的 client 集合 ⊂ anomod_v1 client 集合，归一化逻辑不用改，避免引入跨数据源的 Normal 基准不一致问题。
- **`_enumerate_cases_multi` 遇到不存在的 root 显式 raise**：原来 glob 遇到打错的路径会静默返回空，多 root 配置里一个路径错了会静默丢失半个数据集。改为显式报错并按 root 记录 case 数、对贡献为 0 的 root 告警。

## 坑 / 已知问题

- **endpoint 白名单口径漂移**：`endpoint_to_service.yaml` 的 route/travelplan 组存量 4 个死条目在 client 侧从未出现，使 `anomod_v1` baseline 实际只有 4 个活 endpoint 参与训练/评估，属于长期存在但直到本次才被发现的口径漂移。修正后新增回归测试钉住与 dataset-guide 的一致性。
- **6 个测试文件同用 `--data-root` 模式**：build_contract 入口改名后，原计划只提到 2 个测试受影响，实际排查发现所有子进程调用 build_contract.py 的测试都用旧入口，需统一迁移到 `--dataset` + 新建 fixture config。
- **`_enumerate_cases_multi` 类型注解滞后**：`DatasetConfig.roots` 改成 tuple 之后，函数形参注解仍是 `list[Path]`，PR review 抓到后改为 `Sequence[Path]`。
- **真实 28-case dvc repro 暴露 metric 模态全 NaN**：整体 AUROC 0.591 被 metric 模态拖累（当时判断是"时区/时间对齐 bug"），拆到独立分支 `bugfix/fix-metric` 处理，见 entry 007。

## 遗留 TODO

- per-endpoint 标签 + 评估（多数据源合并后的③）：本次未做，是独立后续 PR。
- metric 模态全 NaN 问题：已在 entry 007 中修复代码层根因，但数据层的 Normal 采集缺口需要重采数据，本 entry 遗留。
