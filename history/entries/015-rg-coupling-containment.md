# 015 · RG 耦合收束：从 `is_reliability_gate` 分支到 `from_contract` 钩子

- **日期**: 2026-07-21
- **PR**: `bugfix/rg-coupling-containment`（待开 PR）· **Commit**: ace9a85..94b5176（代码+设计文档 3 commit，不含本次文档同步 commit）
- **类型**: Refactor
- **影响域**: `src/fusion/`, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `src/contracts/contract_config.py`, `configs/contract/`, `dvc.yaml`, `dvc_reliability_gate/`, `CLAUDE.md`

## 做了什么

entry 014（Reliability Gate Fusion 实验）为了打通 RG 这条实验链路，在三处**共享/通用代码路径**里插入了 RG 专属的耦合逻辑，被记录为 entry 014 与 `CLAUDE.md` Known Gotchas 里的技术债待清理项。本次 PR 收束这三处耦合，不改动 RG 门控算法本身：

1. `FusionModule` 新增 `from_contract` classmethod 钩子（默认=今天的 `hydra.utils.instantiate` 行为），`ReliabilityGatedFusion` 覆写它自行加载 `EndpointBaselineStats` 并派生 `id_to_endpoint_key`；基类 `forward` 统一新增 `endpoint_id: torch.Tensor | None = None` 参数（L0/L1/L2 接受即忽略）。`train_baseline_v0.py` 的 `is_reliability_gate` 字符串匹配分支（构造/center 初始化/`_train`/`_infer` 共 4 处）全部消失。
2. `ContractConfig` 新增 `fit_endpoint_baseline_stats: bool = False` 开关（照抄 `expand_train_pool` 模式），取代 `contract_version == "v1"` 作为"是否产出 `endpoint_id` 列与 `endpoint_baseline_stats.json`"的判据。`v1.yaml` 设为 `false`，`v1_expanded_pool.yaml` 设为 `true`。
3. 3 个 RG 专属 DVC stage（`build_contract_v1_expanded`/`train_v1_reliability_gate`/`eval_v1_reliability_gate`）从根 `dvc.yaml` 移到 `dvc_reliability_gate/dvc.yaml`（子目录，每 stage 带 `wdir: ..`），裸 `dvc repro` 不再触发这 3 个 stage（原本约 20min 不必要的 rebuild）。触发方式改为 `dvc repro dvc_reliability_gate/dvc.yaml`。

## 关键决策（不在 commit 里）

- **DVC 隔离文件放子目录而非根级同名文件**：原计划是根级新建 `dvc_reliability_gate.yaml` 作为 `dvc.yaml` 的同级文件。实测发现 DVC 3.67.1 在任意位置都强制 pipeline 文件字面量命名为 `dvc.yaml` 或 `*.dvc` 后缀，`dvc_reliability_gate.yaml` 会被直接拒绝（`ERROR: bad DVC file name`）。改为子目录 `dvc_reliability_gate/dvc.yaml`，每个 stage 加 `wdir: ..` 让原有的（相对 repo 根书写的）`cmd`/`deps`/`outs` 路径字符串不需要改写就能继续解析——这个约束是通过直接实验验证的，不是假设。
- **config 开关取值提前到 commit 2（而非按最初设计放在 commit 3）**：`fit_endpoint_baseline_stats` 默认 `False`，若把两份 v1 系 config 的取值声明推迟到 commit 3（DVC 拆分那一步）才写，commit 2 落地后到 commit 3 之前，`v1_expanded_pool.yaml` 会短暂走默认值 `False` → 不产出 `endpoint_baseline_stats.json` → RG e2e smoke test 会变红。为保证每个 commit 下 `pytest tests/` 全绿，config 取值随代码改动同一个 commit 落地。4 个 commit 的数量与类型（`[Refactor]`×3 + `[Docs]`×1）不变。
- **不修 eval_all 类别失衡问题**：entry 014 记录的 `expand_train_pool=true` 后 eval_all 正负比从 ~50:50 漂移至 ~84:16 是独立问题，已拆到 GitHub issue #16（`ready-for-agent`），与本次耦合收束无关，明确排除在本 PR 范围外。
- **不动 `scripts/analyze_gate_weights.py`**：该脚本是一次性分析脚本（非训练/评估管线的一部分），直接构造 `ReliabilityGatedFusion` 绕过 Hydra 是预期用法，保持原样。

## 坑 / 已知问题

- **DVC 文件命名约束不是文档假设，是实测确认的**：见上方"关键决策"。任何未来想再拆分 DVC stage 到独立文件的尝试都要记住这条约束，不要重复踩这个坑去尝试根级同名文件方案。
- **`_write_v1` 的 6 参数签名（2 个同类型 bool）**：`expand_train_pool` 与 `fit_endpoint_baseline_stats` 都是 bool，位置参数传递没有类型系统保护误传顺序的风险。当前测试覆盖足够捕获这类回归，但如果未来再加第 3 个独立开关，建议改成关键字参数或拆出一个小 dataclass。

## 遗留 TODO

- eval_all 类别失衡（issue #16）仍未处理，需要单独一个 PR。
- `dvc.lock` 中残留 `train_v1_reliability_gate:`/`eval_v1_reliability_gate:` 的无前缀旧 key（stage 搬迁前遗留），现已与两份 `dvc.yaml` 都不对应，是死重量但不影响 `dvc status`/`dvc repro` 正确性；下次任何人touch `dvc.lock` 时可以顺手清理。
