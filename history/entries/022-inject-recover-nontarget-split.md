# 022 · inject/recover 阶段非目标 endpoint 行吸收进训练池

- **日期**: 2026-07-29
- **PR**: 待开 PR（branch: `feature/eval-all-nontarget-split`）· **Commit**: `1c44d28^..ccb9279`（9 commit，含起点 `1c44d28`：2 `[Docs]` 设计/计划 + 1 `[Refactor]` 重命名 + 1 `[Feature]` 加字段 + 1 `[Test]` 新 fixture + 1 `[Feature]` 接线 + 1 `[Refactor]` 审查修复 + 1 `[Test]` 集成测试 + 1 `[Refactor]` 审查修复，不含本次文档同步 commit）
- **类型**: Feature
- **影响域**: `src/contracts/split_fault_phase.py`, `src/contracts/contract_config.py`, `scripts/build_contract.py`, `configs/contract/v1_expanded_pool.yaml`, `configs/contract/v1_new_ep1.yaml`, `tests/fixtures/nontarget_split_mini/`, `tests/test_split_fault_phase.py`, `tests/test_contract_v1_train_pool.py`, `CLAUDE.md`

## 做了什么

entry 019 记下的 12.24:87.76 类别比例上限——只调 `fault_baseline_train_fraction` 压不动的负样本地板，最大一块是 inject 窗口内"非目标 endpoint 的 fan-out 残留"（同一时间窗里 8 个 endpoint 只有 1 个是真正的注入目标，其余 7 个仍算负样本）——本次给出了新的可调参数。`ContractConfig` 新增两个独立字段：`fault_inject_nontarget_train_fraction`（吸收 inject 阶段非目标 endpoint 行，默认 `0.0`）与 `fault_recover_nontarget_train_fraction`（吸收 recover 阶段非目标 endpoint 行，默认 `0.0`）。两者都复用整窗时序切分的通用函数，本次把它从 `split_fault_baseline_temporal` 重命名为 `split_fault_phase_temporal`（`src/contracts/split_fault_baseline.py` → `split_fault_phase.py`），因为函数已不再只处理 baseline 阶段。`scripts/build_contract.py::_write_v1` 在 `expand_train_pool=True` 分支里追加两路切分调用；新增 `tests/fixtures/nontarget_split_mini/` fixture（含目标/非目标 endpoint 混合的 inject/recover 窗口）配合新增/扩充的单元测试与集成测试锁住行为。`configs/contract/v1_expanded_pool.yaml` 与 `configs/contract/v1_new_ep1.yaml` 把两个新字段都设为 `1.0`（全部吸收，尽量压低 eval_all 负样本地板）。

## 关键决策（不在 commit 里）

- **复用 `expand_train_pool` 作总开关，不新增独立开关**：三类吸收（baseline 行、inject 非目标行、recover 非目标行）语义上都是"训练池扩容"的子集，各自用独立 fraction 精细调节比例即可，没有必要在 `expand_train_pool` 之外再加一层"是否启用非目标吸收"的开关——那只会制造两层布尔状态的组合爆炸。
- **两个 fraction 保持独立，不共享一个参数**：inject 与 recover 的非目标行规模、对模型的影响不同（inject 侧是同窗 fan-out 残留，recover 侧是系统恢复期行为），合并成一个参数会丧失分别调节的能力，代价很小（多一个字段）却换来更细的控制粒度。
- **两个字段默认都是 `0.0`，向后兼容**：`v1.yaml`（`expand_train_pool=false`）和历史上未显式设置这两个字段的任何配置不受影响；只有主动打开的 `v1_expanded_pool.yaml`/`v1_new_ep1.yaml` 显式设为 `1.0`。
- **只吸收非目标 endpoint 的 recover 行，目标 endpoint 自身的 recover 行继续排除**：沿用 entry 014 的既有决策——目标 endpoint 在 recover 阶段是否已经稳定回到正常分布未经验证，继续保守地把这部分行留在 eval_all，不纳入训练池。
- **本轮不动 Normal holdout**：三个 fraction 全部推到上限后，负样本地板仍包含 Normal holdout 部分，这部分由 `Normalizer`/`EndpointBaselineStats` 的防泄漏纪律保护，不在本次改动范围。

## 坑 / 已知问题

- **inject 与 recover 的"非目标"判据不能共用同一个字段，两者的失效模式正好互补**：`is_anomaly` 定义为 `phase == "inject"`，这导致 `is_endpoint_anomaly`（= `is_target_endpoint AND phase=='inject'`）在 recover 阶段对所有 endpoint（包括目标 endpoint 自身）恒为 `False`——这意味着用它筛 recover 的"非目标"行不是"选不出行"，而是"全部选中"：该判据对所有 endpoint 都成立，等于筛不掉任何行，会把目标 endpoint 自身的 recover 行也一起误判为非目标吸收进训练池，违反 entry 014 的既有保护决策。反过来，`is_target_endpoint` 只在 `label_granularity == "endpoint"` 的 case（`endpoint_raw2` 等有 `target_endpoint` 字段的数据源）上有意义；`label_granularity == "case"` 的 case（如 `anomod_v1`）该列全部是 `0.0` 或 `NaN`（NaN 经 `fillna(True)` 视为目标），无法区分目标/非目标——如果 recover 侧用 `is_target_endpoint == False` 不加 `label_granularity` 限定，会把 case 级标签的 case 的 recover 行整段错误吸收（这些 case 根本没有"非目标 endpoint"这个概念，它的整个 case 就是异常单元）。因此 inject 侧固定用 `is_endpoint_anomaly == False`（天然兼容两种粒度、不会误吸收正样本），recover 侧必须额外加 `label_granularity == "endpoint"` 限定再用 `is_target_endpoint == False`，两条判据不可互换、不可合并成一条。
- **`eval_all` 从粗粒度的 `phase != 'baseline'` 过滤改成显式 6 段拼接后，漏掉任意一段会静默丢样本、不报错**：过去 eval_all 的构成逻辑简单（"不是 baseline 就进 eval"），本次改成对多路切分结果显式 concat（Normal holdout + inject 目标 + inject 非目标未吸收部分 + recover 目标 + recover 非目标未吸收部分 + baseline 未吸收部分），任何一段忘记拼接进去都只是让 eval_all 行数变少、负样本比例悄悄漂移，不会抛异常或让测试失败得很明显。核对 `fraction=0` 时 inject/recover 各等于 fixture 里 20 行（parity check，即新逻辑必须完全退化回旧行为的行数，对不上说明拼接漏了一段）不是可省略的一次性验证，而是必要的静默丢样本防护——这批不变量现由 `tests/test_contract_v1_train_pool.py` 里 `test_nontarget_fractions_are_noop_when_expand_train_pool_false` 等测试的 `==20` 断言锁定，不再是需要人工重复执行的步骤。
- **`is_target_endpoint` 是 float64（含 NaN），不是干净 bool，直接 `~` 取反会抛 TypeError**：真实 contract 数据里该列取值集合是 `{0.0, 1.0, NaN}`，`~is_target_endpoint` 对 float64 NaN 列会抛 `ufunc 'invert' not supported`；`&` 两侧又会被完整求值，不能靠短路规避，必须先 `.fillna(...).astype(bool)` 显式转成干净 bool 再取反/比较。第一次修复选了 `fillna(False)`（未知值默认"不是目标"），看似合理但方向反了——它会让未知目标状态的行被误判为"非目标"，从而在 recover 侧被非目标吸收逻辑吞进训练池，恰好违反本 entry"目标 endpoint 自身的 recover 行始终不吸收"的核心保守诉求。跟进修复（commit `e31b931`）把方向改为 `fillna(True)`：未知值默认"是目标"，这样它在训练吸收侧被正确排除、在 eval 补集侧被正确保留。教训是遇到"未知值该往哪边默认"这类选择时，要反过来问"哪个方向出错的代价更小/更符合保守设计意图"，不能凭直觉选一个"看起来更常见的默认值"。

## 本轮不重跑既有 baseline 指标

沿用 entry 021 的既有判断：这批数据集（`merged_v2` = anomod_v1 + endpoint_raw2 + normal_v2，以及 `new_ep1`）本身待重采——PATCH 类故障对现有 endpoint RED 特征全隐形（entry 017），旧数据集上产出的新数字不会被任何后续工作引用。因此尽管 `v1_expanded_pool.yaml`/`v1_new_ep1.yaml` 两份 contract 配置本次改动了字段取值，我们不执行 `dvc repro` 重新构建 `artifacts/contract_v1_expanded`/`artifacts/contract_new_ep1_expanded` 及其下游 baseline/metrics——drift 继续累积，接受现状不处理，与 entry 021 的做法一致。

## 遗留 TODO

- 三个 fraction 全部推到 `1.0` 后，真实数据集上 eval_all 的正负比没有实测——例如 `new_ep1` 的 446 行 recover 里目标 vs 非目标各占多少还没拆解统计过。
- 三路 fraction 全推上限后 eval_all 剩余的负样本地板（未吸收的 recover 目标行 + Normal holdout）是否需要进一步处理，留给下一轮判断。
- entry 019 提出的另一条独立 TODO——case-aware / 按故障类型分级调节 fraction，专门针对 PATCH 类故障的训练池污染问题——本次仍未处理。
