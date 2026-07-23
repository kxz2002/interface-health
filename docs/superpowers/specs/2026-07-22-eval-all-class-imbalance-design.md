# Eval-All Class Imbalance Fix — Design Spec

**Date:** 2026-07-22
**Branch:** `bugfix/eval-all-class-imbalance`
**Issue:** GitHub #16（`ready-for-agent`）
**Status:** Design approved（经 `/grilling` 逐决策访谈确认），pending spec review before implementation-plan.

## 1. 背景与动机

entry 014（Reliability Gate Fusion 实验，PR#14）落地 `expand_train_pool` 开关时，为了扩容 RG 的训练池，让 `build_contract.py::_write_v1` 在 `expand_train_pool=true` 时把故障 case 的 `baseline` 阶段行**整段**吸收进训练池，并**整段**从 `eval_all` 摘除以防泄漏。这个"吸收进 train ⟺ 从 eval 摘除"的互斥逻辑有一个副作用：`eval_all` 的负样本被大量掏空。

实测（entry 014）：`expand_train_pool=false`（`v1.yaml`）时 `eval_all` 负样本 6757 行、正负比约 50:50；`expand_train_pool=true`（`v1_expanded_pool.yaml`）时负样本骤降到 1346 行、正负比崩到约 84:16。这违反 `CLAUDE.md`/`docs/agent-docs/dataset-guide.md` 明确写着的评估协议——"负样本（训练+评估）：Normal case 全程 + 异常 case 的 baseline/recover 阶段"——现在的实现把 baseline 阶段整段移出了 eval。

**影响**：AUPRC 对类别比例极敏感，entry 014 记录的 RG 实验 AUPRC 普涨（baseline_v1 0.327→0.483）很可能是类别比例漂移的假象，而非模型真实提升。这个问题在 entry 014 已记录、在 entry 015（RG 耦合收束）里被明确拆分为独立 issue #16 处理，本分支专门解决它。

**范围边界**：只有 `expand_train_pool=true`（RG 专属路径）受影响；默认 `v1.yaml`（`expand_train_pool=false`）与 L0/L1/L2 主链路不受影响，本次修复绝不能改变它们的行为或数字。

## 2. 核心方案

把"故障 baseline 行整段进池/整段摘出 eval"的全有或全无逻辑，替换为**按时间窗时序切分**：每个故障 case 的 baseline 阶段窗口按时间排序，最早 `fault_baseline_train_fraction` 比例的窗口进训练池，其余留在 `eval_all`。这样训练池仍能扩容，同时 eval 保留大部分 baseline 负样本，类别比例回到接近原始 50:50。

唯一绝不能破坏的硬约束是 `train.sample_id ∩ eval_all.sample_id == ∅`（Contract v1 存在的理由，修复 v0 的 train⊆eval 泄漏）。按整窗切分天然保证这一点：同一时间窗的所有行整体归属同一侧，不会有任何一行既在 train 又在 eval。

### 2.1 切分粒度：时间窗，非行

切分单位是 `timestamp_window_ms`，同一窗的所有 endpoint 行整体搬移，不按行随机打散。理由：

- **一致性**：整个项目对 Normal 行的切分（`split_normal_rows_temporal`）已经是这个粒度，故障 baseline 行本质上也是"正常运行阶段"数据，用同一套切分哲学。
- **避免隐蔽泄漏**：`service_metric`/`service_log` 是按 service 粒度 join 进来的，同一 service 在同一时间窗被多个 endpoint 行共享。按行随机切会让同一个 service-timestamp 的 service 侧特征同时出现在 train 和 eval，比按窗切更隐蔽。

### 2.2 切分方式：时序，非随机

每个故障 case 内，baseline 窗口按 `timestamp_window_ms` 升序排列，**最早的**窗口进 train、较晚的留 eval。镜像 `split_normal_rows_temporal` 的"train 窗早于 eval 窗"纪律，保证 eval 侧看到的是训练时未见过的"未来"窗口。

### 2.3 切分比例：可配置，默认 0.2

新增配置字段 `fault_baseline_train_fraction: float = 0.2`。选择 0.2 的依据（真实 29-case 数据实测倒推）：

| fraction | train 新增 baseline 行 | eval 正负比（约） |
|----------|----------------------|------------------|
| 0.2（默认）| ~1082（训练池 838→~1920，翻倍）| 45:55 |
| 0.3 | ~1623 | 43:57 |
| 0.5 | ~2706 | 37:63 |

0.2 让 eval 比例基本恢复到接近原始设计的 50:50，同时训练池仍有实质扩容。做成**可配置字段**而非硬编码，因为用户后续会继续扩充 Normal 数据集，这个比例在未来迭代里会再调。

**这是一个真实 tradeoff，不存在两者都最优的解**：要让 eval 完全回到 50:50，几乎要把 baseline 全部（≈100%）留给 eval，那样训练池扩容就落空了。fraction 本质是在"训练池扩容幅度"和"eval 平衡程度"之间选折中点。

### 2.4 recover 阶段：不动

`recover` 阶段行维持现状全部留在 `eval_all`，不纳入本次范围。沿用 entry 014 的既定决策——"不吸收 recover（系统未稳定回正常态，分布未验证，保守排除）"。recover 本来就是负样本、一直在 eval 里，不是本问题的根因（根因是 baseline 被整段挪走）。收紧改动面到 baseline 切分这一处，便于审查。

## 3. 组件设计

### 3.1 新函数 `split_fault_baseline_temporal`

新建 `src/contracts/split_fault_baseline.py`，**不复用/不改造** `split_v1.split_normal_rows_temporal`。理由：两者是不同业务概念——两路（train/eval）vs 三路（train_fit/train_val/holdout）、对象是故障 baseline vs Normal case、边界退化策略不同（见 3.3）。共用一个函数会靠参数分支制造隐晦耦合，且三路命名对故障 baseline 场景没有语义。

签名：

```python
def split_fault_baseline_temporal(
    df: pd.DataFrame,
    fraction: float,
    case_col: str = "case_id",
    time_col: str = "timestamp_window_ms",
) -> tuple[pd.DataFrame, pd.DataFrame]:  # (train_part, eval_part)
```

调用方（`_write_v1`）保证 `df` 仅含故障 case 的 baseline 阶段行。每个 case 独立按 `time_col` 排序，最早 `int(n * fraction)` 个时间窗归 train_part，其余归 eval_part。

### 3.2 配置字段 `fault_baseline_train_fraction`

给 `ContractConfig`（`src/contracts/contract_config.py`）新增字段，照抄现有 `expand_train_pool`/`fit_endpoint_baseline_stats` 的模式（dataclass 字段 + `load_contract_config()` 里 `raw.get(...)`）：

```python
fault_baseline_train_fraction: float = 0.2
```

**与 `expand_train_pool` 正交、独立生效**：`expand_train_pool` 是开关（要不要扩容），`fault_baseline_train_fraction` 是幅度（扩容多少）。`expand_train_pool=false` 时该字段被 `_write_v1` 忽略。选 A（两个独立字段）而非 B（把 `expand_train_pool` 从 bool 改成 float），是为避免破坏性字段类型变更——所有 `if expand_train_pool:` 的 truthy 判断和历史 config 都得跟着改语义。

**范围校验**：给 `ContractConfig` 加 `__post_init__`，校验 `0.0 <= fault_baseline_train_fraction <= 1.0`，越界抛 `ValueError`。这是**恒校验**（不看 `expand_train_pool`）：越界值是配置错误，即便当前未被消费也应在加载期暴露，而不是等到开关打开时才悄悄产出空/全量切分污染实验。这是一个用户会频繁调整的新参数，配错的代价是悄悄产出错误实验结果而非报错，值得加这道低成本护栏。

### 3.3 边界退化：不做兜底

窗数极少时 `int(n * fraction)` 自然向下取整（如 1 窗 × 0.2 = 0 窗进 train），**刻意不做**"至少 1 窗给 train"的兜底——这与 `split_normal_rows_temporal` 的边界处理不同。理由：

- Normal 切分兜底的动机是"train_fit 不能为空"（否则 Normalizer 无法 fit）；这里训练池已有 838 行纯 Normal 的 `train_fit` 打底，某个 case 贡献 0 行 baseline 完全不影响训练可行性。
- 反过来，强行保证"至少 1 窗进 train"对窗数极少的 case 会把仅有的窗吃进 train，让该 case 在 eval 里的 baseline 覆盖率归零——正是本次修复要避免的失衡。

代码里必须写清楚注释解释为什么这里不像 Normal 切分那样兜底。

### 3.4 `_write_v1` 接线

`_write_v1` 的 `expand_train_pool=true` 分支（约 523-541 行）改为：

1. `train_fit`（打标 `source_phase=normal_case`）不变。
2. 从 `anomaly_df` 筛出 `phase == 'baseline'` 的行，喂给 `split_fault_baseline_temporal(fraction=cfg.fault_baseline_train_fraction)`，得到 `(fault_baseline_train, fault_baseline_eval)`。
3. `train.parquet = train_fit + fault_baseline_train`（后者打标 `source_phase=fault_baseline`）。
4. `eval_all = 故障 case 的 inject/recover 行 + fault_baseline_eval + Normal holdout`。

`else` 分支（`expand_train_pool=false`）完全不动。`EndpointBaselineStats`/`Normalizer` 的 fit 范围也完全不碰——两者始终锁定 `parts["train_fit"]`（纯 Normal），本次只改 `train.parquet`/`eval_all.parquet` 的行构成。

## 4. 组件依赖关系

```
configs/contract/v1_expanded_pool.yaml（声明 fault_baseline_train_fraction: 0.2）
  └─ ContractConfig.fault_baseline_train_fraction（新字段 + 范围校验，§3.2）
       └─ build_contract.py::_write_v1（expand 分支接线，§3.4）
            └─ split_fault_baseline_temporal（新切分函数，§3.1/§3.3）
                 └─ 产出 train.parquet / eval_all.parquet（sample_id 互斥不变量）
```

单向依赖链，实现顺序自然是 §3.2 字段 → §3.1 切分函数 → §3.4 接线。

## 5. 测试策略

- **`split_fault_baseline_temporal` 单测**（新建 `tests/test_split_fault_baseline.py`，纯内存合成 DataFrame）：sample_id 互斥、无行丢失、最早窗进 train 的时序性、整窗不拆散、单窗自然取整（0 窗进 train，不兜底）、fraction=0/1 边界、逐 case 独立切分、空输入返回两空帧。
- **`ContractConfig` 字段测试**（`tests/test_contract_config.py`）：默认值 0.2、覆写生效、越界（2.0 / -0.1）抛 `ValueError`。照抄 `test_fit_endpoint_baseline_stats_*` 的模式。
- **`_write_v1` 集成测试**（`tests/test_contract_v1_train_pool.py`，需新 fixture 见 §6）：
  - **泄漏不变量翻转**：现有 `test_eval_all_excludes_train_pool_rows_no_leakage` 的核心断言"eval 不含任何 baseline 行"在新逻辑下不再成立，改写为"`train.sample_id ∩ eval_all.sample_id == ∅`"（这条更本质、必须保留）+ 新增"eval 保留大部分 baseline 行（5 窗中 4 窗）"的正向断言。
  - 现有 `test_train_pool_includes_fault_baseline_rows` / `test_source_phase_column_present_and_correct` / `test_train_pool_excludes_inject_and_recover_rows` 断言语义不变，只迁移到新 fixture、更新注释里的 case 名。
- **既有测试不回归**：`test_contract_v1_split.py`（未扩容路径）、`test_dvc_pipeline_v1.py`（config 绑定）、`test_build_contract_v1_endpoint_id.py`（baseline_stats fit 范围锁定 train_fit——本次不碰 fit，应仍通过）、`test_e2e_reliability_gate_smoke.py`（走 expanded 路径，只断言 `len(scores)==len(eval_all)`，不涉及 baseline 行数）。
- **回归验证（端到端，人工跑一遍确认修复生效）**：`dvc repro dvc_reliability_gate/dvc.yaml` 重跑 RG 三段链路，实测修复后 `eval_all` 的 phase 分布、正负比、AUROC/AUPRC，确认负样本占比从 ~16% 显著回升，数字记入 history entry 016。

## 6. 测试 Fixture 缺口（阻塞项，必须先解决）

**现有共享 fixture 无法验证分数切分**：`tests/fixtures/mini_data_root` 里每个故障 case（`Lv_P_DISKIO_preserve` 等）的 baseline 阶段**只有 1 个时间窗**，`int(1 * 0.2) = 0`——新默认值下这些 case 的 baseline 会 100% 留在 eval、0 行进 train。这不仅让新增的比例测试无法验证，还会打破现有两个断言"train 含非零条 baseline 行"的测试（`test_train_pool_includes_fault_baseline_rows`、`test_source_phase_column_present_and_correct`）。这些 case 被 15+ 个测试文件共用且对行数有精确断言，不能贸然扩充。

**方案**（仿 `nan_propagation_mini` 先例）：新建专用 fixture `tests/fixtures/split_fraction_mini/`：

- 1 个 Normal case（6 个时间窗）+ 1 个故障 case（8 个时间窗 = 5 baseline + 1 inject + 2 recover）。`int(5 * 0.2) = 1`，能验证"最早 1 窗进 train、其余 4 窗留 eval"。
- Normal 与故障 case **共用同一个 endpoint_key**（`POST:/api/v1/travelservice/trips/left`），保证 `train_fit`/`train`/`eval_all` 三者 endpoint 覆盖面一致——否则 `EndpointBaselineStats`（RG 专属，只 fit `train_fit`）查表会抛 KeyError（正是 `test_e2e_reliability_gate_smoke.py` 当年换用 `nan_propagation_mini` 的原因）。
- 只放 trace + api 两个模态（无 metric/log），缺失模态走 `_fill_missing_feature_cols` 填 NaN，与 `mini_data_root/Lv_E_HTTPABORT_assurance_mini` 一致。

**受影响的现有测试一并迁移**：`test_contract_v1_train_pool.py` 里依赖 baseline 行数的两个测试，连同新增的比例测试，全部切到 `split_fraction_mini`。不涉及 baseline 行数的测试（如 `test_train_pool_excludes_inject_and_recover_rows`，断言"train 不含 inject/recover"在窗数少时依然成立）保持不动。

## 7. DVC 与文档同步

- **DVC 依赖**：`dvc_reliability_gate/dvc.yaml` 的 `build_contract_v1_expanded` stage 的 `deps` 列表必须新增 `src/contracts/split_fault_baseline.py`——否则 DVC 认不出这个新文件是依赖，改动它不会触发缓存失效，重跑可能用 stale 产物。`configs/contract/v1_expanded_pool.yaml` 和 `scripts/build_contract.py` 已在 deps 里，config 新增字段与脚本改动会正常触发级联重跑。
- **重跑范围**：只重跑 `dvc repro dvc_reliability_gate/dvc.yaml`（RG 专属三段），刷新 entry 014 的过期数字。默认主链路（L0/L1/L2）不受影响、无需重跑。
- **CLAUDE.md**：把 "expand_train_pool=true 后 eval_all 正负比失衡（未修复）" 那条已知问题翻新为描述修复后行为 + 新开关 `fault_baseline_train_fraction`；同步更新"expand_train_pool 开关会改变 eval_all 行数"条目里的行数（修复后 expanded eval_all 行数会变）。
- **history entry 016**：新建，须显式标注 **entry 014 的 RG 实测数字已因本次修复失效**（eval 样本构成变了，AUROC/AUPRC 不再与修复后可比），需下一轮在固定 fraction 下重跑才能得到可比数字。更新 `history/index.md` 列表 + 影响域索引 + 横切主题第 107 行（补一句"016 已落实修复"）。

## 8. Out of Scope

- **RG 门控算法本身**（softmax gate collapse、`independent_sigmoid` 消融、高方差等既有问题）——entry 014 已如实记录，本次只改 eval 样本构成，不碰算法。
- **recover 阶段的处理**（§2.4）——维持现状全留 eval，沿用 entry 014 决策。
- **fraction 的敏感性扫描**——本次只把 0.2 作为经验默认落地，不做"不同 fraction 对 RG 指标影响"的系统扫描。若后续要论证 RG 效果，需在固定 fraction 下多 seed 复跑，而非在变动的 eval 口径上比较，列为遗留 TODO。
- **`expand_train_pool=false` 默认路径与 L0/L1/L2**——完全不改，本次改动对它们零影响是必须验证的不变量，不是 scope。

## 9. 实现拆分（预计 7 个 commit）

1. `[Feature]`：`ContractConfig.fault_baseline_train_fraction` 字段 + `__post_init__` 范围校验 + loader 装配 + 单测。
2. `[Feature]`：`split_fault_baseline_temporal` 两路时序切分函数 + 单测。
3. `[Test]`：`split_fraction_mini` fixture（5 baseline 窗口故障 case）。
4. `[Bugfix]`：`_write_v1` expand 分支改用时序切分 + `v1_expanded_pool.yaml` 声明 fraction。
5. `[Test]`：迁移 `test_contract_v1_train_pool.py` 到新 fixture + 泄漏断言翻转为 sample_id 互斥。
6. `[Experiment]`：补 `split_fault_baseline.py` 到 DVC deps + 全量测试 + 重跑 RG pipeline 刷新指标。
7. `[Docs]`：CLAUDE.md 已知问题翻新 + history entry 016 + index 更新。

> Commit 顺序 1→2→4 是依赖链，3 可与 1/2 并行；5 依赖 3+4；6 依赖 5；7 依赖 6（需要重跑后的实测数字）。
