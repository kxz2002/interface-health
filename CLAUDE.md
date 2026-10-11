# Interface-Health

接口健康度多模态融合与时序异常检测研究（PyTorch）：融合 Train-Ticket 微服务的 traces/api_responses/metrics/logs 四模态，做 per-API-endpoint × time-window 粒度的单类（One-Class）异常检测。

## 研究背景与方向

- **定位**：为民航 SAT（航空票务核心系统，6000+ APIs、5 亿次调用/天）潜在项目做技术储备。项目拿到就在 SAT 专有数据上做（周期性、动态基线）；落空就在 Train-Ticket 上专注异常检测创新
- **创新方向**：① 主：per-endpoint × time-window 的 One-Class 检测模型，只用正常数据训练，强标注只用于评估（工业场景标注贵，依赖故障注入标注的监督方法难泛化，审稿人会质疑）；② 次：客户端 RED（api_responses）+ 服务端 RED（traces）特征融合，作为消融维度。差异化点：现有工作几乎都是 service 级粒度

### 当前研究状态（entry 027→031，引用任何 AUROC 前先读）
- **平凡基线打败全部学习模型**（contract 主链路，per-case 宏平均 AUROC）：rel_pos（只看窗口在 case 内的相对时间位置，不看任何特征）0.9656、v2 zscore_l2 0.9853；最强学习模型 v1 DWF 0.9507、v2 最强臂 independent_concat 0.9465（均为 4-seed 均值）。macro AUPRC 下 v2 concat/L1 反超三条平凡基线，结论随指标口径分叉。canonical 表：`artifacts/contract_v2_acceptance/main_table.md`（2026-10-07，数字以 `scripts/collect_v2_acceptance_table.py` 的输出为准）；zscore 基线对 fit 口径与退化哨兵敏感，0.9853 在机制分解前不读成上限（entry 031 遗留 TODO）。entry 028 的下游 benchmark（`scripts/benchmark_downstream.py`，AnoMod 返修材料，直读 `tt_fused_15s.csv`、另一套特征与切分）结论不同：expanded 池 DeepSVDD 0.957 远高于 rel_pos 0.579，但 normal_only 池 rel_pos 0.800 仍打败全部学习模型，两套数字不要混用
- **根因在数据集**：采集用固定时序模板（inject 起始 rel_pos 0.586±0.015），且只有 1 个 Normal case、位于 17 小时采集窗口首位。v2 修好了归一化参照系，但 rel_pos 0.9656 三轮不变，归一化侧已无手段
- **待办与影响**：P1 重采（inject 起点随机化、多 Normal 交错采集）是唯一根治，因 AnoMod 论文锚定 new_merge 而延期；P2 论文转向"benchmark 缺陷 + 规范评估协议"的证据链已齐（entry 031）。在此之前，上面两个创新方向里模型/融合机制之间的 AUROC 排名都不能脱离平凡基线单独当结论

### 核心任务定义（单类异常检测）
- 训练池（全部视为 label=0）：现行 `v1_new_merge`/`v2_new_merge`（`expand_train_pool=true`）= 纯 Normal `train_fit` ∪ 故障 case baseline 阶段最早 20% ∪ inject/recover 阶段非目标行（fraction 0.2/1.0/1.0）
- 阶段标签：`is_anomaly = (phase == 'inject')`，来自 `case_metadata.json` 的 `inject_start_ms`/`inject_end_ms`（毫秒精度），**不依赖 `weak_is_anomaly`**
- 评估正样本：`is_endpoint_anomaly` = inject 阶段 ∩ 注入目标（三档，见 Known Gotchas）；eval_all 其余行均为负样本。推理输出偏离分数，阈值化得 0/1
- 主指标：per-case 宏平均 AUROC/AUPRC（`per_case_auroc_macro`/`per_case_auprc_macro`，单类 case 跳过，new_merge 上计入 23 个 case，entry 029），pooled 只作附注；每份 metrics.json 必须与三条平凡基线（rel_pos/zscore_l2/zscore_max）并列报告

## 数据集

- **主数据集 `data/new_merge/`**（唯一口径正确且可重建的数据集）：Train-Ticket 27 case，2026-07-28/29 单批次采集，READ-ONLY，由 `configs/data/new_merge.yaml` 声明。原始采集 28 case，已剔除 `Lv_S_KILLPOD_gateway`（inject 阶段 trace 缺失，AUROC 无定义，entry 023）。不与其他批次合并：entry 017 证实跨 run 合并会污染评估。189 个 `_pipeline_out` 漂移文件已于 entry 032 `dvc commit` 承认现状，cache 内容完整
- **外部依赖（AnoMod）**：new_merge 也是 `~/Code/Repos/AnoMod`（数据集论文 "Beyond RED"，投 CBASE 2026）的真数据源。论文统计基于 entry 032 承认的漂移后状态，那次漂移就是 AnoMod 侧管线重写 `_pipeline_out` 造成的；`tt_fused_*.csv` 不是 contract 输入，但它是论文统计和 `scripts/benchmark_downstream.py` 的直接输入。所以：① 改 new_merge 任何内容（含重生成 `_pipeline_out`）前先问用户；② `dvc status dvc_new_merge/dvc.yaml` 若再报 `data/new_merge` 变化，先查 AnoMod 侧是否重跑过管线，不要直接 repro。口径差异：论文用 26 case（另剔除 inject 期仅 4 行/1 端点的 `Lv_D_TRANSACTION_timeout`），本仓库 contract 用 27 case
- `AnoMod/processed_data/2026_0729/` 是同一次采集的另一份拷贝（28 case 含 gateway）：原始文件与 new_merge 一致，但 `_pipeline_out` 停在 07-29 版本，含 contract 输入 `tt_endpoint_health_*` 在内 10 类文件不同，不能用来恢复 new_merge 或核对论文数字。entry 024 的 `ep2_0729` 就源于它，其 config 指向的 `/tmp/ep2_0729_nolog` 已不存在
- **已永久丢失**（磁盘、DVC cache、DVC remote 三处皆无）：`endpoint_raw2`、`lo2-sample`（`326b885` 物理删除）、`normal_v2`、`new_ep1`（entry 025 核实）、`new_ep2`。依赖它们的 `merged_v1`/`merged_v2`/`new_ep1` 链路无法重建；entry 014/016/018/022 的数字只能作废（同时建立在 entry 025 修掉的错误标签上），已提交的 `metrics.json` 只当"当时如此"的记录，不在其上继续实验，不与 entry 025 之后的数字并列
- **其余目录**：`data/anomod_v1/`（11 个 service 级故障 case + Normal，历史数据，READ-ONLY；Normal 有 cadvisor 断流导致的 metric 覆盖缺陷，没有 `_archive/` 目录，见 entry 025）；`data/anomod/`（项目最早期单批次，无现行 config 引用）
- `docs/agent-docs/dataset-guide.md` 正文描述的是已弃用的 `data/anomod/` 老 pipeline；现行字段口径与 pipeline 以本文件和 `configs/data/*.yaml`、`configs/contract/*.yaml` 的注释为准

## 代码结构

- `src/contracts/` 数据契约（schema 校验、切分，train↔eval 经此解耦）；`src/data/` 数据加载与 Normalizer；`src/preprocessors/` 四模态预处理器；`src/fusion/` 融合层（L0 early_concat / L1 independent_concat / L2 gated / DWF deviation_weighted / v2 deviation_weighted_selfref；`reliability_gate.py` 已于 entry 031 退役，仅留档）；`src/models/deep_svdd.py`
- `src/training/`、`src/evaluation/` 是空 stub：训练在 `scripts/train_baseline_v0.py`，评估在 `scripts/eval_baseline_v0.py`
- `dvc_new_merge/dvc.yaml` 是唯一可用的 pipeline；`artifacts/` 放产物，`outputs/` 是 Hydra 运行目录，`history/` 是决策记录（见 Project History）

## Commands

```bash
make setup && make torch-cpu   # 新设备；GPU 用 make torch-gpu CUDA_VERSION=cu121
make sync                      # 已有环境，拉新代码后
pip install -e ".[dev]"        # 可编辑安装，使 src.* 可 import

# dvc / pytest / 训练脚本都在 interface 环境执行（base 环境没有 dvc）
conda run -n interface dvc status dvc_new_merge/dvc.yaml   # 验收只看这个文件
conda run -n interface pytest tests/

# ⚠️ 历史管线（数据已永久丢失，不可重建，勿运行）：根 dvc.yaml（v0/v1 + entry 003 的 toy stage）、
# dvc_reliability_gate/、dvc_deviation_weighted/、dvc_new_ep1/。原命令见 git 历史与 entry 011/014/015/018/019。
# 不要在仓库根目录跑无 target 的 `dvc repro`：它执行根 dvc.yaml，`--dry` 即报错；真跑会先删 stage outs 再因缺数据失败，
# 而 artifacts/contract_v0/ 等历史产物不在 cache，删了无法恢复。全项目 `dvc status` 的 changed / not in cache 是这些死链
# 加 data/{normal_v2,new_ep1}.dvc 的预期噪音。analyze_gate_weights.py / analyze_modality_observability.py 依赖的 contract_v1*/ 已清空。

# === new_merge 主链路（dvc_new_merge/dvc.yaml，只注册 seed42）===
# v1 四臂 concat/independent_concat/gated/deviation_weighted，v2 四臂（DWF 换成 deviation_weighted_selfref），
# 外加 v1/v2 各三条平凡基线 rel_pos/zscore_l2/zscore_max。stage 名为 train_new_merge_[v2_]<arm>、score_trivial_[v2_]<name>，
# 对应 eval stage 把 train_/score_ 前缀换成 eval_。RG 两个 stage 已于 entry 031 移除。
# entry 032 之后整文件 repro 安全（status 全绿、--dry 零触发），也可 --single-item 单跑：
dvc repro dvc_new_merge/dvc.yaml
dvc repro --single-item dvc_new_merge/dvc.yaml:train_new_merge_v2_concat

# v2 contract 仍【未注册】dvc build stage（entry 031/032 遗留 TODO）：7 个 v2 train/score stage 依赖
# artifacts/contract_new_merge_v2/*，但没有 stage 产出它，改了 build_contract.py / 归一化代码 / v2 config 后 DVC 不会察觉，
# 必须先手动重建再 repro。v1 contract 由 build_contract_new_merge_expanded stage 产出
# （手动等价：--config configs/contract/v1_new_merge.yaml --out-dir artifacts/contract_new_merge_expanded）。
python scripts/build_contract.py --config configs/contract/v2_new_merge.yaml --dataset configs/data/new_merge.yaml --out-dir artifacts/contract_new_merge_v2 --seed 42

# seed{1,2,3} 不在 dvc.yaml 里，按臂手动跑：v2 用 contract_new_merge_v2、目录 baseline_new_merge_v2_<arm>_seed<s>；
# v1 用 contract_new_merge_expanded、目录 baseline_new_merge_<arm>_seed<s>。
python scripts/train_baseline_v0.py contract_dir=artifacts/contract_new_merge_v2 out=artifacts/baseline_new_merge_v2_concat_seed1/scores.parquet seed=1 training.epochs=50 fusion=concat model=deep_svdd
python scripts/eval_baseline_v0.py --scores artifacts/baseline_new_merge_v2_concat_seed1/scores.parquet --out artifacts/baseline_new_merge_v2_concat_seed1/metrics.json
```

## Config System
- 配置都在 `configs/`（YAML，入口 `configs/base.yaml`），超参数不得硬编码在代码里
- `scripts/train_baseline_v0.py` 是 `@hydra.main` 入口：`contract_dir`/`out` 是 `configs/base.yaml` 的必填字段（`???`），用 `key=value` override，不是 argparse flag。学习率在 `training.optimizer.lr`（不是 `training.lr`）；加 `--cfg job` 只打印组合后的 config、不训练
- 配置组：`fusion=<configs/fusion/ 下的文件名>`，`model=deep_svdd`（目前唯一）；两者都经 `hydra.utils.instantiate` 按 `_target_` 实例化
- `configs/data/*.yaml`：`roots`（合并哪些数据源目录）+ `normal_source`（Normal 只取自哪个 root），由 `build_contract.py --dataset` 消费
- `configs/contract/*.yaml`：契约版本（v1 min-max / v2 per-case z-score）、`expand_train_pool` 与三个吸收 fraction。fraction 按数据集实测校准，换数据集不能照搬（见 `contract-and-labels.md`）

## 约定
- **数据**：数据集目录 READ-ONLY；所有数据变换都要代码化（`src/data/`），不手工处理；每个数据集是独立原子单元（原始数据与 pipeline 产物在同一目录），版本记录在对应 config
- **实验追踪**：每次运行记录 config 快照 + git commit + 数据集版本 + seed。Hydra 快照自动落在 `outputs/<date>/<time>/.hydra/`；MLflow 未接入（`src/utils/logger.py` 的 `ExperimentLogger` 没有调用方，`outputs/mlruns` 不存在）
- **产物目录**：`artifacts/baseline_<dataset>_[<variant>_]<arm>[_seed<s>]/`（variant 如 `v2`/`inject0`，seed42 无后缀），平凡基线为 `artifacts/trivial_baseline_[v2_]<name>/`。`metrics.json` 与 `.gitignore` 入 git，`scores.parquet` 不入 git。只有 dvc stage 的产出进 dvc cache：手动跑的（其余 seed、inject0 等对照）只在本地磁盘，entry 030/031 的对照脚本要读它们现算指标，删了只能重跑
- **可复现**：PyTorch 和 NumPy 的 seed 必须同时设，统一用 `src/utils/seed.py`
- **代码风格**：pre-commit 跑 isort + black；函数签名写 type hints，不强制全量标注；模型类继承 `torch.nn.Module`，训练逻辑与模型定义分离；日志用 `logging`，不用 `print`
- **测试**：`src/data/`、`src/fusion/`、`src/contracts/` 必须有单元测试（契约层用来防 train↔eval 接口被静默破坏）；文件放 `tests/test_<module>.py`；实验流程要能重复验证，但不强制 TDD
- **环境**：conda 环境名 `interface`；torch 不在 `environment.yml` 里，用 `make torch-cpu` / `make torch-gpu` 手动装
- **分支**：`feature/xxx` 基础设施（数据脚本、配置、工具、可复现性），`baseline/xxx` 经典基线，`exp/xxx` 实验迭代（调参、ablation）。前期基础设施阶段统一走 `feature/`

## Project History
- `history/index.md`（entry 列表 + 影响域倒排）+ `history/entries/NNN-*.md`，记录每次 PR 的决策、坑与遗留 TODO，是 commit log 与 CLAUDE.md 之外的第三层文档
- **开发新功能 / 重构 / 修 bug 前必读 `history/index.md`**，按影响域找相关 entry，避免重复踩坑或推翻已有决策
- **PR merge 前必写一篇新 entry**，并更新 index 的列表与影响域索引，写前先查重。流程见 `skills-local/project-history/SKILL.md`（clone 后运行 `make install-skills` 链接到 `.claude/skills/`）

## Known Gotchas（跨模块；模块细节见下方"按需加载的规则"）
- Kiro/Claude 跑在 base conda 环境：直接 `python` 走的是 base，`dvc` 只装在 interface 环境（base 下 command not found），`dvc`/`pytest`/训练脚本都用 `conda run -n interface ...`
- **特征粒度是采集端的架构约束**：traces/api_responses 有 endpoint 级特征；metrics/logs 只有 service 级，重采也不会变。融合设计按 endpoint + service 两层 join 键来
- **标签三档**（entry 025）：按 `case_metadata.json` 里有哪些 target 字段定 `label_granularity`——有 `target_endpoint` → endpoint 档（两者都有时优先）；只有 `target_service` → service 档；都没有 → case 档（实际只有 Normal）。判据是 target 字段，不是 `anomaly_level`。`is_endpoint_anomaly` 恒为 `is_anomaly` 的子集。entry 025 之前的 AUROC/AUPRC 建立在错误标签上，引用前先确认口径
- **eval_all 构成不同的 AUROC 不能直接比**：`expand_train_pool` 与三个 fraction 都会改变 eval_all 的行；比较前先限定到共有 `sample_id` 子集（entry 030 中全量口径凭空叠加了 −0.01~−0.04 的偏差）
- **高方差机制不能只看单 seed**：RG 只有 seed42 时被判为"唯一退化"，补齐 4 个 seed 后发现 seed42 恰是塌陷点（entry 025）。结论写"方向 + 方差量级"
- **`trace_error_rate`/`trace_5xx_rate` 名不副实**：服务端状态码只出现在 5xx span 上（`trace_status_coverage>0` 仅 10.8% 行），`trace_5xx_rate` 非空即 1.0，是退化列。错误信号以 `client_error_rate`/`client_5xx_rate` 为准
- **接入新数据集先查"被杀目标是否承担上报职责"**：`Lv_S_KILLPOD_gateway` 杀掉的正是 trace 上报通道，inject 阶段 0 行、AUROC 无定义（entry 016/023 两次复现）；同批 order/preserve 的 trace 完好。判据是上报职责，不是故障类型
- **DVC 整目录 dep 会被派生文件作废**：build stage 以整个 `data/new_merge` 为 dep，目录内非输入文件（`_pipeline_out`）一漂移就作废整条链（entry 029/031 三次恢复成本，032 处置）。新数据集接入时，派生物应拆成独立 DVC 输出或排除在 dep 外

## 按需加载的规则（`.claude/rules/`）
带 `paths:` frontmatter，Claude 读写匹配文件时才自动加载；只跑命令、不读相关代码时不会加载，需要时直接 Read。代码注释里"见 CLAUDE.md ×× gotcha"指的就是这些条目。
- `contract-and-labels.md`（build_contract / src/contracts / configs/contract / eval 脚本）：per-endpoint 标签精度三档、`label_target_observable` 闸门列、`expand_train_pool` 可比性、三个 fraction 的吸收判据与校准、`is_target_endpoint` 的 float64/NaN 陷阱、v2 退役 rate clip 改量级 sanity
- `normalization.md`（normalization / endpoint_baseline_stats）：Normalizer 退化 group（全 NaN / 零方差）处理与 v2 z-score 回退链
- `fusion-and-models.md`（src/fusion / src/models / train_baseline_v0 / contract_dataloader）：optimizer 必须覆盖全部子模块、eval NaN 用 train 均值填、RG 坍缩与耦合收束（已退役）、DWF 列顺序校验、POT/GPD 阈值
- `preprocessors-and-raw-data.md`（src/preprocessors / configs/data / 直读原始 csv 的脚本）：Drain3 相关、log join 静默 NaN、`MAX_CONTENT_CHARS` 截断损耗、metrics 降采样、特征粒度、`trace_*` 错误列与 KILLPOD 的实测细节

## Git Commit Convention
MUST: 提交信息必须严格遵守下面的格式；PR 标题必须以 type 开头。

格式为 `<type>: <description>`，空一行后可写 body。type 取值：`[Feature]` 新功能/模块、`[Bugfix]` 修复、`[Experiment]` 实验配置/结果/调参、`[Data]` 数据集、`[Model]` 模型/checkpoint、`[Refactor]` 重构、`[Docs]` 文档、`[Test]` 测试、`[Chore]` 构建/工具/环境。Commit attribution 已全局禁用，不加 Co-Authored-By 行。

## Agent skills

### Issue tracker

Issues tracked in GitHub Issues (kxz2002/interface-health), via `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Default five canonical roles (needs-triage / needs-info / ready-for-agent / ready-for-human / wontfix). See `docs/agents/triage-labels.md`.

### Domain docs

Single-context layout: root `CONTEXT.md` + `docs/adr/`. See `docs/agents/domain.md`.
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 81 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into the ARIS repo recorded as `repo_root` in `.aris/installed-skills.txt`, e.g. `~/Code/Repos/Auto-claude-code-research-in-sleep`).
Update with: `bash <repo_root>/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills; regenerates this block).
<!-- ARIS:END -->
