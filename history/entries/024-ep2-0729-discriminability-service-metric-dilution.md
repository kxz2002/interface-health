# 024 · ep2_0729 特征判别力分析 + service_metric 判别力异常根因定位

- **日期**: 2026-08-15
- **PR**: #23（工作分支 `exp/feature-discriminability-analysis`）。原计划编号为 023，因与 master 上并行合并的 PR #22（entry 023 · new_merge 数据集接入）撞号，rebase 时改为 024
- **类型**: Experiment
- **影响域**: `notebooks/feature_discriminability.py`, `configs/data/ep2_0729.yaml`, `configs/contract/v1_ep2_0729.yaml`, `scripts/build_contract.py`（case 级 `is_endpoint_anomaly` 标签构造，发现问题未改代码）, `configs/contract/endpoint_to_service.yaml`（覆盖范围限制，发现问题未改代码）

## 做了什么

在新采集的单批次数据集 `ep2_0729`（2026-07-28/29，28 case，见 `configs/data/ep2_0729.yaml`；相对历史批次消掉了 entry 017 记录的跨 run 混杂）上，用 `notebooks/feature_discriminability.py`（jupytext 配对 `.ipynb`）对 18 维契约特征做了判别力分析：within-case 协议（同 case baseline/recover 做负样本）+ 以故障子类型（family）为聚合单元的稳定性判定，产出 8 张图 + 4 张汇总 CSV（`outputs/figures/feature_discriminability/`，被 `.gitignore` 排除不入库）。

分析发现 `service_metric` 模态（cpu_usage_rate/memory_usage_ratio）对全部已测故障类型 AUROC 均接近 0.5，包括理论上应直接冲击资源指标的 `Lv_P`(CPU/DISKIO)、`Lv_S`(KILLPOD) 这类 case 级故障。随后单独排查该异常，定位到根因：不是数据或预处理问题，是 `build_contract.py` 里 case 级故障的 `is_endpoint_anomaly` 标签构造丢弃了已有的 `target_service` 信息，导致评估时把大量不相关 service 的行错误地当成正样本，稀释了本来很强的信号。

## 关键决策（不在 commit 里）

- **判别力度量不用跨 case 直接平均，改用 family 内 signed AUROC 均值 + 标准差**：跨 16 个 `Lv_E` case 直接算 `|mean(AUROC)-0.5|` 或 `mean(|AUROC-0.5|)` 都会产出误导性结论——前者让方向相反的强信号（`client_error_rate` 在 ABORT/REPLACE 上 AUROC≈0.93、DELAY/PATCH 上≈0.43）互相抵消到 0.681 被埋没；后者把跨 service 不稳定的噪声（`service_log__event_rate` 在 4 个 ABORT case 上 AUROC 0.000/0.789/0.008/0.022 来回翻转）误判成判别力第一名。改成以"同一故障打在 4 个不同 service 上"为可重复实验单元：先看 family 内标准差（`within_fam_std`，可信度），再看最强 family 的效应量（`best_fam_effect`，强度），两者缺一不可，只有左上区（效应量大、标准差小）才可信。
- **判别力度量在原始量纲（`raw_features.parquet`）上算，不用归一化表**：Normalizer 对零方差/全 NaN group 会跳过归一化保留原值，同一张表内量纲混杂，分布图必须在原量纲读——承袭 entry 007/013 的既有教训，本次严格执行。
- **发现 `service_metric` 异常后没有立即改代码，先做根因验证**：用 contract 里已有的 `target_service` 列手工重算了 case 级故障的 AUROC（把正样本限定为 `service_name==target_service`），确认信号本身极强（多数 case 强度 >0.8~0.99）之后才下结论"评估方法问题，非数据/预处理问题"——避免了直接怀疑 `MetricPreprocessor` 或数据采集质量、走错排查方向。

## 坑 / 已知问题

- **`service_metric` 判别力评估被 case 级标签稀释，根因在 `build_contract.py:305-306`**：`ep_df["is_endpoint_anomaly"] = ep_df["is_anomaly"]`（case 级 fallback）对整个 case 内所有 endpoint/service 一视同仁标正样本，不看 `target_service` 是谁。`service_metric` 是纯 service 级特征，不相关 service 在故障期间数值不变，混进正样本后把本来很强的信号拉回接近随机。实测把正样本限定到 `service_name==target_service` 重算后：
  - `Lv_P_CPU_preserve`：`cpu_usage_rate` AUROC 0.534 → 0.906
  - `Lv_S_KILLPOD_preserve`/`order`：`cpu_usage_rate` 0.536/0.555 → 反向强度 0.950/0.989（KILLPOD 杀 pod 导致资源指标骤降而非升高，方向相反但信号极强）
  - `Lv_S_DNSFAIL`/`HTTPABORT`：`memory_usage_ratio` 强度 0.977/0.866
  `MetricPreprocessor`（差分计数器 → core rate → 跨 pod mean → 15s 窗口聚合）本身经检查没有问题。
- **`Lv_D`（target=`tsdb-mysql`）与 `Lv_S_KILLPOD_gateway`（target=`ts-gateway-service`）4 个 case 结构性不可评估，不是"信号弱"**：`configs/contract/endpoint_to_service.yaml` 只映射 8 个客户端入口 service，mysql/gateway 都不在映射范围内，`service_name==target_service` 筛不出任何行（`n_pos=0`）。这是当前 endpoint↔service 映射架构上看不到这些基础设施层 target，与 CLAUDE.md 已记录的"metrics 只能做 service 级"约束是同一类问题的延伸，但更具体——连 service 级都覆盖不到非客户端入口组件。
- **图表本身两处瑕疵，读图或复用绘图代码时需注意**：
  - `01_data_quality.png` 右子图（取值多样性）代码里漏调 `invert_yaxis()`，且右子图无 y 轴标签，导致左右两个子图行顺序镜像颠倒——不能跨图对比同一行。
  - `05_top_feature_distributions.png` 的小提琴图数据是 4 种 HTTP 故障子类型混池（未按 `http_kind` 过滤），但标题标注的 AUROC 是"表现最好的那个故障子类型"单独数值，两者不是同一子集，视觉分离度和标题数字可能对不上（`latency_divergence` 尤其明显：标题写 0.998，混池后的小提琴形状明显分不开）。
- **`rank.within_fam_std` 的平均方式会埋掉"偏科型"特征**：该值是对 4 种故障类型分别算 std 后取平均。`service_log__template_diversity` 单独在 ABORT 上 std=0.014 极稳、效应量 -0.49 很强，但另外 3 种故障下不稳定，平均后 within_fam_std=0.253 被误判为"不可信"。用图 4 散点图筛特征时，右侧的点不能直接判死刑，要回查 `n_fam_stable`（4 种里有几种稳）——只对某一种故障强且稳的特征是"故障类型专用检测器"，该做条件加权而非全局淘汰。
- **两个 contract 产物目录内没有 DVC 自动生成的 `.gitignore`**：`artifacts/contract_ep2_0729/`、`artifacts/contract_ep2_0729_withlog/` 是手动跑 `build_contract.py` 产出（数据源仍在仓库外的 `/tmp` 路径，故意不进 `dvc.yaml` 默认 DAG），不像走 `dvc repro` 的目录会自动生成只放行 `.gitignore` 自身的忽略规则。提交前需仿照 `artifacts/contract_v0/.gitignore` 补一份，否则 parquet 大文件会被 `git add` 原样吃进去。

## 遗留 TODO

- **`label_granularity` 的二元设计（`"endpoint"` vs `"case"`）本身有粒度缺口，需要单独拉分支修**：`target_service` 从数据采集起就存在于 `case_metadata.json`（chaos 注入配置本身就是打向某个 service 的 pod，不是后补字段），`build_contract.py:277` 也早就把它原样搬进了 contract 的每一行；缺的不是数据，是 entry 008 引入 `label_granularity` 时的二分类心智模型——"有 `target_endpoint` 就精确，没有就退化到 case 级近似"，中间漏掉了"没有 `target_endpoint` 但有 `target_service`"这一档 service 级粒度。修复方向是 case 级分支（`build_contract.py:305-306`）改用 `is_anomaly & (service_name == target_service)`，需要先决定 `target_service` 落在 `tsdb-mysql`/`ts-gateway-service`（不在 `endpoint_to_service.yaml` 映射范围内）时的行为——是显式产出 `n_pos=0` 并 warning，还是需要先扩展映射表（见下一条）。
- **上述修复不是孤立改动，会连锁作废多条历史实验数字，需要完整重跑闭环**：`is_endpoint_anomaly` 同时是 `_write_v1`（entry 022）里 inject 阶段"非目标行"吸收判据的输入——`build_contract.py:538-552` 的注释明确写着"case 级标签的 case 在此自动选不出任何行"，这个不变量目前是靠当前 bug 撑住的，修复后会被打破：`v1_expanded_pool.yaml` 把 `fault_inject_nontarget_train_fraction` 设为 1.0，之前贡献 0 行的 case 级故障（`anomod_v1` 11 个 + `ep2_0729` 11 个）会突然有大量非目标 service 的 inject 行涌入训练池，训练池构成发生实质变化。需要重建/重跑的范围：
  - `contract_v0`/`contract_v1`（含 `anomod_v1` case 级 case）：重建
  - `contract_v1_expanded`（RG 专属，`dvc_reliability_gate/dvc.yaml`）：重建 + 重新训练评估，entry 014/016/022 的 AUROC/gate 权重数字全部作废
  - `dvc_deviation_weighted/dvc.yaml`（DWF，复用 `contract_v1_expanded`）：重建 + 重跑，entry 018 数字作废
  - `contract_ep2_0729*`：重建，本次分析的 §7 图重新出
  - `new_ep1` **不受影响**（纯 endpoint 级数据集，全部有 `target_endpoint`，case 级分支是死代码）
  - 文档同步：`build_contract.py:538-552` 长注释、CLAUDE.md Known Gotchas 对应段落、`tests/fixtures/nontarget_split_mini/`（entry 022）大概率需要补一个带 `target_service` 的 case 级 fixture
  - 需要新写一篇 history entry 专门记录"哪些历史数字因这次修复被作废"，不能让 014/016/018/022 的旧数字继续被引用
- **notebook §7（`08_endpoint_vs_case_level.png`）需要用 `target_service` 重新算一版**：当前图上 `service_metric` 两行显示接近灰白（无信号），按本次修正后的口径重算会大幅改观（判定应反转为"对资源型故障判别力很强"）——与上面两条属于同一次修复的一部分，不单独处理。
- **`endpoint_to_service.yaml` 是否要扩展覆盖 mysql/gateway 类基础设施 target，未决**：要评估 `Lv_D`/`KILLPOD_gateway` 这类故障，需要先决定这些组件的 service 级指标怎么接入 endpoint 粒度的 schema（它们本来就不对应任何客户端 endpoint），是更大的架构讨论，本次只记录现象，与上述修复同属一个后续分支的范围。
- **`01_data_quality.png` 右子图的 `invert_yaxis()` 未修**，若要复用这张图做汇报需要先修。
- **两个 `artifacts/contract_ep2_0729*` 目录的 `.gitignore` 未补**，提交前需要处理。
- **分支目前无 commit**，本 entry 先落盘，具体提交/PR 时机由用户决定。
