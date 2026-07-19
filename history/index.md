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
| [006](./entries/006-endpoint-raw2-multi-source.md) | 2026-07-08 | Feature (PR #6) | endpoint_raw2 接入 Contract v0：多数据源合并 pipeline | src/data/dataset_config.py, scripts/build_contract.py, configs/data/, configs/contract/endpoint_to_service.yaml, dvc.yaml, tests/ |
| [007](./entries/007-fix-normalizer-nan-propagation.md) | 2026-07-08 | Bugfix | 修复 Normalizer 全 NaN group 的 NaN 传染 bug | src/data/normalization.py, tests/test_normalization.py |
| [008](./entries/008-per-endpoint-label-eval.md) | 2026-07-11 | Feature | per-endpoint 精确标签接入评估 | `src/preprocessors/trace_preprocessor.py`, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `scripts/eval_baseline_v0.py`, `src/contracts/contract_v0.py` |
| [009](./entries/009-normal-v2-ingestion.md) | 2026-07-13 | Data | 30/60 分钟重采 Normal 数据接入 DVC pipeline | `configs/data/`, `dvc.yaml`, `dvc.lock`, `artifacts/`, `tests/fixtures/merged_v2_mini/`, `tests/test_dataset_config.py`, `tests/test_build_contract_multi_root.py`, `tests/test_e2e_smoke.py`, `CLAUDE.md` |
| [010](./entries/010-gated-fusion-novelty-check.md) | 2026-07-14 | Docs | Gated Conditional Fusion 立项前 novelty-check 结论（暂缓门控实现，转向补基础设施） | `history/`, `src/fusion/`（后续实施方向） |
| [011](./entries/011-fusion-ablation-infra.md) | 2026-07-15 | Feature/Refactor | Fusion 消融基础设施：Hydra 可插拔化 + Contract v1 时序切分 + 参数量对齐工具 | `src/fusion/`, `src/contracts/`, `src/utils/`, `configs/`, `scripts/train_baseline_v0.py`, `scripts/build_contract.py`, `dvc.yaml`, `CLAUDE.md` |
| [012](./entries/012-fusion-l1-l2-gated-ablation.md) | 2026-07-15 | Experiment | L1/L2 融合消融：独立编码器 + 门控条件融合实测 | `src/fusion/`, `configs/fusion/`, `tests/` |
| [013](./entries/013-fix-normalizer-zero-variance-explosion.md) | 2026-07-16 | Bugfix | 修复 Normalizer 零方差 group 除零放大 bug（endpoint_red latency 列 1e13 异常值） | `src/data/normalization.py`, `scripts/build_contract.py`, `tests/test_normalization.py`, `CLAUDE.md` |
| [014](./entries/014-reliability-gate-fusion.md) | 2026-07-19 | Experiment | Reliability Gate Fusion：偏离量门控路由实测（多seed主对比+2x2训练池归因+路由消融+可解释性分析+坍缩根因定位+训练池扩容影响归因），门控坍缩与高方差问题如实记录 | `src/fusion/`, `src/data/endpoint_baseline_stats.py`, `src/contracts/`, `scripts/build_contract.py`, `scripts/train_baseline_v0.py`, `scripts/analyze_gate_weights.py`, `configs/fusion/`, `artifacts/contract_v1*` |

---

## 影响域索引（按目录/主题倒排）

> 改动某个目录时，按这张表找过往相关 entry。同一域可能有多条，按时间排序。

| 目录 / 主题 | 相关 entries |
|-------------|--------------|
| `CLAUDE.md` | 001, 002, 003, 004, 009, 011 |
| `data/` 组织与 DVC | 002 |
| `configs/` (Hydra) | 002, 011（fusion/model config-group、base.yaml 必填字段）, 014（reliability_gate 路由消融配置、`fusion_checkpoint` 可选字段） |
| `src/utils/` (seed, logger) | 002, 011（param_budget） |
| `src/contracts/` | 003, 011（Contract v1 时序切分）, 014（`endpoint_id` 列、训练池扩容吸收 fault baseline 行） |
| `scripts/` (train, eval) | 003, 011（train_baseline_v0.py 改 Hydra entrypoint） |
| `scripts/lo2-scripts/` | 001 |
| `docs/agent-docs/` | 001 |
| `docs/plans/` | 001 |
| `artifacts/` | 003 |
| `dvc.yaml` / DVC pipeline | 002 (初始化), 003 (定义 stage), 009 (切换 merged_v2), 011 (新增 build_contract_v1/train_v1/eval_v1) |
| `environment.yml` / `Makefile` | 002 |
| `.github/workflows/ci.yml` | 002, 004 |
| `tests/` | 002 (占位), 003 (契约/e2e), 009 (merged_v2_mini fixture + 归档排除/多 root 测试) |
| `pyproject.toml` | 002 |
| `history/` + `skills-local/` | 004 |
| `Makefile` | 002 (环境), 004 (install-skills) |
| 多模态融合（`src/fusion/`） | 005, 010, 012, 014（ReliabilityGatedFusion，偏离量门控路由） |
| 模型实现（`src/models/`） | 005 |
| 数据 loader（`src/data/`） | 005, 014（`endpoint_baseline_stats.py`, `endpoint_id` 全链路打通） |
| 评估指标细化（per-endpoint, phase 对齐） | 005 |
| `src/preprocessors/` | 005 |
| `configs/contract/` | 005 |
| `configs/data/`（多数据源配置） | 006, 009 |
| `src/data/dataset_config.py` | 006 |
| `configs/contract/endpoint_to_service.yaml` | 006 |
| `src/data/normalization.py`（Normalizer） | 007, 013（零方差 group 除零放大） |
| `src/preprocessors/trace_preprocessor.py`（is_target_endpoint 传递） | 008 |
| `scripts/build_contract.py`（label_granularity/is_endpoint_anomaly 标签路由，`_attach_label_columns`） | 006, 008, 014（训练池扩容吸收 fault baseline 行） |
| `scripts/train_baseline_v0.py`（out_df 显式字段字典，新增列需手动传递） | 008, 014（`endpoint_id` 传递给 fusion，`fusion_checkpoint` 可选落盘） |
| `scripts/eval_baseline_v0.py`（by_endpoint 分层） | 008 |

---

## 横切主题（cross-cutting）

> 不绑定某个目录、但反复出现的约束或踩坑模式。

- **可复现性**：seed 同时设 PyTorch + NumPy（002）；DVC stage 必须显式传 seed（003）
- **契约层**：train↔eval 解耦，字段口径必须经 contract 校验（003）；contract 版本即接口版本，破坏字段口径必须升版
- **CLI 合约**：脚本失败必须 `exit 1` 而不是只 log；JSON 输出必须 `allow_nan=False`（003）
- **PR review 反复抓到的类别**：silent failure、测试覆盖不足、文档/代码口径漂移
- **环境管理**：Kiro/Claude 默认 base conda，验证 `interface` 环境用 `conda run -n interface`（002）
- **log 模态信号**：Train-Ticket 数据集中 `_previous_*.log` 是断掉的 K8s 符号连结，log 采集仅覆盖实验末尾几分钟；log 特征大面积 NaN 属数据采集限制，非代码 bug；重采时需在 pod 存活期间拷贝历史文件（005）
- **NaN 传染**：`Normalizer` 等只在 Normal 上 fit 的组件，遇到某 group 全 NaN（数据采集缺口）时不能直接算统计量再 transform 全量数据——NaN 统计量会通过减法/除法把其他数据完好的 case 一起污染。修复方式是在 transform 端对 NaN 统计量做跳过判断，保留原值不做运算（007）。同类模式：`ContractDataset` 的全 NaN 列填 0 兜底（005 Bug 4）
- **零方差同样要跳过归一化，不能只处理 NaN**：min-max 归一化在某 group 的 fit 集合只有单一取值时 `hi-lo=0`，若用 `max(hi-lo, eps)` 托底除数而不是跳过，eval 侧任何非零差值都会被放大 `1/eps` 倍，产出 1e9~1e13 量级的"有值但离谱"的数（区别于 NaN 传染，NaN 传染是把数据抹掉，这个是把数据放大到失真）。判定条件要从"仅 NaN"扩展成"NaN 或零方差"，两者统一走跳过路径（013，是 007 约定的直接延伸）
- **标签精度分级**：数据集里不同数据源标签粒度不一致（case 级近似 vs endpoint 级精确）时，用 contract 层的显式列（如 `label_granularity`）标记精度，而不是靠数据源名字或已有的故障类型字段（如 `anomaly_level`）隐性判断——后者语义上不等价，未来数据源变化会静默失配（008）
- **门控/条件融合类机制不足以单独构成创新点**：FiLM/GS-Fuse 已是成熟的"表征级门控条件融合"机制类别，套用到新场景不算创新；"service 级特征广播复制到 endpoint 行"这一问题定义本身也对应统计学 hierarchical/panel data 框架，非全新问题。融合类工作若要立论，需要论证"具体场景下的增量价值"而非机制新颖性本身，且消融设计要用参数量对齐 baseline 隔离贡献，避免 confounding variable（010）
- **One-Class 评估集切分：时序切分优于 leave-service-out**：当 endpoint→service 接近 1:1 映射时，leave-service-out 会把"对未见 endpoint 的泛化能力"错误地测成"异常检测能力"，holdout 正常样本仅因训练时没见过该 endpoint 就被打高分。按时间窗时序切分（同一 case 内早窗训练、晚窗评估）既保证每个 endpoint 在训练/评估都出现，又能让 service 级广播特征的泄漏担忧成立（特征随时间变化，不是常量）——但前提是特征本身有时间变异性，若某场景的 service 级特征在整个观测窗内几乎不变，这条论证不成立，需重新评估（011，翻案 010 的遗留 TODO）
- **简单门控条件融合的实测收益存在但有限**：单纯引入非线性容量（L0→L1）不带来稳定提升，引入门控（L1→L2）才有正向增量，但幅度不大，且随训练池/eval 集合口径变化。**具体数字已两次被后续 contract_v1 重建取代，本行不再引用具体 AUROC 值，避免与最新实测脱节**——entry 012 首次发布的数字（n_samples=13632，L1 均值 AUROC=0.6103/AUPRC=0.3241，L2 均值 AUROC=0.6297/AUPRC=0.3354）已被 entry 014（n_samples=8221，Task 4/6 contract_v1 重建后）取代；本行早前写的 AUROC=0.6169/0.6082/0.6316 既不匹配 entry 012 单 seed 数字也不匹配其多 seed 均值，来源不明（疑似早期草稿遗留），已在本次（014）更正删除，具体数字以最新 history entry（当前为 014）的表格为准，不在本横切主题行里固化任何版本的具体数值。这提示简单 sigmoid/softmax 门控调制已接近该类机制在当前特征/数据规模下的天花板，后续设计新融合模型应把增量来源转向门控没有利用到的信息（如 endpoint×time-window 时序结构），而不是在门控公式本身继续做复杂变体（010 novelty-check 已指出机制本身不构成创新点，012/014 补充了"收益有限/不稳定"的实测证据）
- **竞争性 softmax 门控比独立 sigmoid 更容易训练坍缩**：Reliability Gate Fusion 的路由消融显示，默认 `softmax`（权重和恒为1的二选一）训练后在全部 27 种故障类型上坍缩为几乎恒定的 `w_svc≈1.0`，未体现"按故障类型动态路由"的设计假设；而去掉竞争性归一化的 `independent_sigmoid`（两个门控开关独立取值，不强制和为1）反而 AUROC 更高（0.6598 vs 默认 0.6024）。提示：门控输入维度低、监督信号只能通过下游 loss 间接传导（如 One-Class SVDD 无监督门控）时，softmax 的强制竞争约束可能让训练更容易早期卡进某一端主导的鞍点；同类机制设计如需门控随输入变化，优先验证独立参数化（sigmoid）而非竞争归一化（softmax）是否更不容易坍缩，坍缩现象本身需要专项训练过程可视化才能定位根因，不能仅凭最终权重分布下结论（014）
- **多 seed 方差本身就是证据，不是可忽略的噪声**：Reliability Gate Fusion 4 个 seed 的 AUROC 标准差（0.0792）达到 baseline 机制（L0/L1/L2，0.0161~0.0334）的 2.5~5 倍，且 seed 范围跨越了所有 baseline 均值区间（4 个 seed 里 2 个优于 L2 均值、2 个明显劣于全部 baseline）。单 seed 或均值对比若不同时报告方差，会掩盖"新机制训练不稳定"这一独立于"新机制是否更优"的问题；如实报告的正确姿态是承认"当前数据不支持方向性结论"，而不是挑一个有利 seed 或只报均值（014）
- **门控坍缩根因定位为"初始尺度温和不对称+softmax竞争性放大"，不是鞍点**：逐 epoch 追踪显示 `w_svc` 全程单调爬升、`frac(w_svc<0.01)` 恒为 0，不是"早期坍缩卡死"的鞍点形态；根因是 `dev_ep`/`dev_svc` 中位数级仅 1.2 倍的温和初始不对称，被 softmax 竞争性归一化在无监督 SVDD loss 下持续放大。训练池扩容（838→6249 行）会显著加速/加剧这个放大过程（`frac(w_svc>0.99)` 从稳定 2.9% 升到 50.7%，已排除梯度步数不足的混杂因素），是坍缩的实质性放大因素，不是无关旁枝改动（014）
- **eval 集合的类别比例是否合理，需要独立于训练侵入之外单独核查**：门控坍缩的梯度只来自训练循环，`eval_all` 从不参与反向传播，"eval 类别失衡导致坍缩"这条因果链机制上不成立；但反过来，训练池扩容改 `eval_all` 摘除逻辑时，容易在无意中破坏 eval 集合本身的类别平衡（本例中负样本占比从约50%降到16%，且与 `CLAUDE.md` 既定的评估协议冲突），这是独立于训练动态之外必须单独核查的正确性问题，不能因为"不影响训练"就忽略（014）
