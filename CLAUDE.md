# Interface-Health

接口健康度多模态数据融合与退化轨迹预测研究。基于 PyTorch，融合微服务环境中异构多源数据（metrics/KPI/logs），实现 per-API-endpoint 粒度的健康度建模与时序异常检测。

## 研究背景与方向

### 项目定位

为潜在的民航 SAT（航空票务核心系统，6000+ APIs、5 亿次调用/天）行业项目做技术储备，存在两个走向：

- **场景 A（项目拿到）**：在 SAT 专有数据上做论文，考虑周期性、动态基线等真实业务场景
- **场景 B（项目落空）**：在 Train-Ticket 数据集上做，专注异常检测创新

当前数据集：Train-Ticket 微服务系统，合并三个数据源共 29 个 case——`data/anomod_v1/`（11 个 service 级故障注入 case，原 Normal 因 cadvisor 断流已归档至 `_archive/`，不参与训练评估）+ `data/endpoint_raw2/`（16 case，endpoint 级故障注入）+ `data/normal_v2/`（2 个重采 Normal case，30min+60min），由 `configs/data/merged_v2.yaml` 声明合并（`normal_source` 固定为 `data/normal_v2`）。历史快照 `configs/data/merged_v1.yaml`（Normal 取自 `anomod_v1`）保留不动，仅用于复现旧实验。数据集字段口径、pipeline 逻辑与已知问题详见 `docs/agent-docs/dataset-guide.md`，接触数据相关代码前必读。

### 两个论文创新方向

1. **新的检测模型**（主，小论文重点）：per-endpoint × time-window 的**单类分类（One-Class）**任务，只用正常数据训练，学习"正常边界"，推理时偏离边界的即为异常。
2. **多模态数据融合**（次，特征层面补充贡献）：将客户端 RED（api_responses）+ 服务端 RED（traces）融合为统一特征向量，融合策略作为消融实验的对比维度。

**为什么选 One-Class**：学术界主流方法（VAE/OCSVM/Deep SVDD）均为无监督/半监督——工业场景标注贵、迁移性差，审稿人会质疑依赖故障注入标注的监督方法无法泛化。One-Class 只需正常运行数据训练，强标注只用于**评估**，不进入训练。

**文献差异化点**：现有工作几乎全是 service-level 粒度；本研究做 **per-endpoint × time-window** 粒度，结合多模态 RED 特征输入 Deep SVDD，在文献中几乎是空白。

### 核心任务定义

**单类异常检测**：
- 训练输入：Normal case 下各 endpoint 各时间窗的多模态 RED 特征向量（全部 label=0）
- 推理输出：偏离正常边界的程度分数，阈值化后得到 0/1
- 评估标签：`phase == 'inject'`（来自 `case_metadata.json` 的 `inject_start_ms`/`inject_end_ms`，精确到毫秒），**不依赖 `weak_is_anomaly`**
- 正样本（评估用）：inject 阶段内的 endpoint × 时间窗
- 负样本（训练+评估）：Normal case 全程 + 异常 case 的 baseline/recover 阶段

## Directory Structure
```
data/                  # 所有数据集根目录，每个数据集为独立原子单元
├── anomod_v1/         # Train-Ticket service 级故障注入数据集（READ-ONLY，never modify）
│   ├── _archive/Normal/              # 原 Normal case，因 cadvisor 断流导致 metric 模态窗口内 0 覆盖，已归档不参与枚举
│   ├── Lv_P_*/  Lv_S_*/  Lv_D_*/   # 11 个故障注入 case
│   └── <case>/_pipeline_out/         # pipeline 产物（tt_endpoint_health_15s.csv / tt_traces_red_15s.csv）
├── endpoint_raw2/     # endpoint 级故障注入数据集（16 case，log 重采修复版，READ-ONLY）
│   └── Lv_E_HTTP{ABORT,DELAY,PATCH,REPLACE}_{assurance,order,travel,travel2}/
├── endpoint_raw/      # endpoint_raw2 的旧版本，log 采集因 fsnotify watcher 耗尽而全崩（inject/recover 阶段零日志覆盖），已弃用不参与 pipeline
├── normal_v2/         # 30/60 分钟重采 Normal 数据（2 case），替换 anomod_v1 原 Normal；不再有 cadvisor 断流导致的全窗口 0 覆盖，但 latency_p50/p95/p99 仍有 ~80% 行 NaN（样本量不足触发分位数计算门限），非 100% 覆盖
│   ├── normal_0711_30/
│   └── normal_0711_60/
├── lo2-sample/        # LO2 数据集样本（logs + metrics）
└── external/          # 外部/公开数据集

docs/agent-docs/       # pipeline 与数据集参考文档（接触数据代码前必读）
├── dataset-guide.md   # 字段口径、pipeline 逻辑、已知问题
└── lo2-scripts.md     # LO2 数据集脚本说明

configs/               # 实验配置 (YAML)
scripts/               # 训练、评估、预处理脚本
src/                   # 核心源码
├── contracts/         # 数据契约校验器（scores/metrics 接口约定，train↔eval 解耦）
├── data/              # 数据加载与预处理 (Drain3 log parsing, 对齐, 归一化)
├── preprocessors/     # 各模态预处理器（trace/api/log/metric → 特征 DataFrame）
├── models/            # 模型定义 (融合模块, 检测器, 预测器)
├── fusion/            # 多模态融合层 (L1-L4 层级实现)
├── training/          # 训练逻辑占位（当前为空 stub，trainer 逻辑实际在 scripts/train_baseline_v0.py）
├── evaluation/        # 评估指标占位（当前为空 stub，评估逻辑实际在 scripts/eval_baseline_v0.py）
└── utils/             # 通用工具 (seed, logging, io)

artifacts/             # DVC pipeline 产物（scores.parquet, metrics.json，大文件 gitignore）
outputs/               # 实验结果 (logs, figures, checkpoints)
models/                # 保存的模型 checkpoint
tests/                 # 单元测试
docs/                  # 文档与实验计划
notebooks/             # Jupyter notebook 探索性分析
```
## Commands

```bash
# 建立环境（新设备）
make setup && make torch-cpu   # 或 make torch-gpu CUDA_VERSION=cu121

# 同步环境（已有环境，拉新代码后）
make sync

# MLflow UI（查看实验结果）
mlflow ui --backend-store-uri outputs/mlruns

# 安装项目包（可编辑模式，使 src.* 可 import）
pip install -e ".[dev]"

# === 当前主链路（Contract v0 + Deep SVDD baseline）===
dvc repro                        # 一键跑通全链路（build_contract → train_v0 → eval_v0）
dvc metrics show                 # 查看 artifacts/baseline_v0/metrics.json（AUROC/AUPRC）
dvc repro build_contract         # 只重跑预处理（原始数据 → contract parquet）
dvc repro train_v0               # 只重跑训练（contract parquet → scores）
dvc repro eval_v0                # 只重跑评估（scores → metrics）

# 单独运行（不走 DVC 缓存）
python scripts/build_contract.py --config configs/contract/v0.yaml --dataset configs/data/merged_v2.yaml --out-dir artifacts/contract_v0 --seed 42
python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v0 out=artifacts/baseline_v0/scores.parquet seed=42 training.epochs=50 fusion=concat model=deep_svdd
python scripts/eval_baseline_v0.py --scores artifacts/baseline_v0/scores.parquet --out artifacts/baseline_v0/metrics.json

# === Contract v1（时序切分，修复 train/eval 行重叠）===
dvc repro build_contract_v1 train_v1 eval_v1   # 只重跑 v1 链路
dvc metrics show                               # 同时展示 baseline_v0/v1 的 metrics.json

# 单独运行（不走 DVC 缓存）
python scripts/build_contract.py --config configs/contract/v1.yaml --dataset configs/data/merged_v2.yaml --out-dir artifacts/contract_v1 --seed 42
python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1 out=artifacts/baseline_v1/scores.parquet seed=42 training.epochs=50 fusion=concat model=deep_svdd
python scripts/eval_baseline_v0.py --scores artifacts/baseline_v1/scores.parquet --out artifacts/baseline_v1/metrics.json

# === Contract v1 训练池扩容开关（expand_train_pool，RG 专属）===
# v1.yaml 默认 expand_train_pool=false（train=train_fit 纯 Normal 838 行，eval_all=13632 行，
# 与 entry 012 既有实验数字可比）。v1_expanded_pool.yaml 显式打开 expand_train_pool=true，
# train 额外吸收故障 case 的 baseline 阶段行扩容训练池（838→6249 行），eval_all 同步摘除
# 这些行防泄漏（13632→8221 行）。两者产物目录互相隔离（contract_v1 / contract_v1_expanded），
# L0/L1/L2 对比基线只走前者，ReliabilityGatedFusion（RG）专属实验走后者，见 history/entries/014。
dvc repro build_contract_v1_expanded train_v1_reliability_gate eval_v1_reliability_gate

# 单独运行（不走 DVC 缓存）
python scripts/build_contract.py --config configs/contract/v1_expanded_pool.yaml --dataset configs/data/merged_v2.yaml --out-dir artifacts/contract_v1_expanded --seed 42
python scripts/train_baseline_v0.py contract_dir=artifacts/contract_v1_expanded out=artifacts/baseline_v1_reliability_gate/scores.parquet seed=42 training.epochs=50 fusion=reliability_gate model=deep_svdd
python scripts/eval_baseline_v0.py --scores artifacts/baseline_v1_reliability_gate/scores.parquet --out artifacts/baseline_v1_reliability_gate/metrics.json

# 查看 RG gate 权重分布（实验后分析用）
python scripts/analyze_gate_weights.py --scores artifacts/rg_seed42/scores.parquet

# 运行测试
pytest tests/
```

## Config System
- 所有配置文件放 `configs/`，YAML 格式，入口为 `configs/base.yaml`
- 超参数禁止硬编码在代码中，必须通过配置文件指定
- 命令行覆盖语法（Hydra）: `python scripts/train.py model.hidden_dim=256 training.lr=1e-4`
- 切换配置组: `python scripts/train.py model=vae`
- 模型通过 `hydra.utils.instantiate(cfg.model)` 实例化，融合机制同理通过 `hydra.utils.instantiate(cfg.fusion)` 实例化，`_target_` 指向具体类
- `train_baseline_v0.py` 是 `@hydra.main` 入口：`contract_dir`/`out` 是 `configs/base.yaml` 里的必填 Hydra 字段（`???`），不是 argparse flag，调用改用 `contract_dir=... out=...` override 语法
- `configs/data/*.yaml` 声明数据集组成：`roots`（合并哪些数据源目录）+ `normal_source`（Normal 只取自哪个 root）。如 `merged_v1.yaml` = anomod_v1 + endpoint_raw2；`build_contract.py --dataset <该文件>` 消费
- `configs/contract/v1.yaml` 新增 `expand_train_pool: bool` 字段（仅 v1 契约消费，v0 不识别该字段）：`false`（默认）=train 仅纯 Normal `train_fit`；`true`=train 额外吸收故障 case 的 `baseline` 阶段行（不含 `inject`/`recover`），eval_all 同步摘除防泄漏。两种取值分别对应 `configs/contract/v1.yaml` 与 `configs/contract/v1_expanded_pool.yaml`

## Data Rules
- `data/anomod_v1/` 等数据集目录是 READ-ONLY，绝不修改原始数据
- 所有数据变换必须代码化（`src/data/`），不可手动处理
- 每个数据集为独立原子单元，原始数据与 pipeline 产物均在同一目录下
- 数据集版本记录在对应的 config 文件中
- 大文件和敏感数据使用 `.gitignore` 排除

## Experiment Tracking
- 所有实验运行必须记录: config snapshot + git commit hash + 数据集版本 + random seed
- 实验命名格式: `<model>-<task>-<date>` (例: `anofusion-ablation-20260508`)
- 实验结果输出到 `outputs/` 对应子目录

## Reproducibility
- 每次实验必须设置 random seed，同时设置 PyTorch 和 NumPy
- 使用 `src/utils/seed.py` 中的统一 seed 设置函数
- 训练脚本默认记录完整 config 到输出目录

## Code Style
- 使用 pre-commit 提交前自动格式化（isort + black）
- Type hints 用于函数签名，不强制全量标注
- 模型类继承 `torch.nn.Module`，训练逻辑与模型定义分离
- 日志使用 `logging` 模块，不用 `print`

## Branch Rules
| 分支 | 用途 |
|------|------|
| `feature/xxx` | 基础设施：数据脚本、配置、工具、可复现性 |
| `baseline/xxx` | 实现经典基线方法（论文对比基线） |
| `exp/xxx` | 日常实验迭代：调参、ablation、对比分析 |

前期完善基础设施阶段，统一走 `feature/` 分支。

## Project History
- `history/` 维护项目历次 PR/迭代的决策、坑、遗留 TODO，作为 commit log / CLAUDE.md 之外的第三层文档
- 结构：`history/index.md`（轻量索引 + 影响域倒排）+ `history/entries/NNN-*.md`（详细文档）
- **开发新功能 / 重构 / 修 bug 前必读 `history/index.md`**，按影响域定位相关 entry，避免重复踩坑或破坏已有决策
- **PR merge 前必写一篇新 entry** 到 `history/entries/`，并更新 index 的列表与影响域索引；写之前先查重，避免重复
- 详细流程见 `skills-local/project-history/SKILL.md`（clone 后运行 `make install-skills` 将 skill 链接到 `.claude/skills/`）

## Python Environment
- 统一使用 conda 虚拟环境，环境名为 `interface`
- 多设备训练/推理时确保环境一致
- torch 不在 environment.yml 中，需手动安装：`make torch-cpu`（无 GPU）或 `make torch-gpu`（有 GPU）
- 新设备初次建环境：`make setup` → `make torch-cpu/gpu`；已有环境同步：`make sync`

## Testing
- 核心数据处理逻辑 (`src/data/`) 和融合模块 (`src/fusion/`) 必须有单元测试
- 数据契约层 (`src/contracts/`) 必须有单元测试，确保 train↔eval 接口不被静默破坏
- 测试文件放在 `tests/`，命名 `test_<module>.py`
- 实验流程需有可重复性验证，但不强制 TDD

## Known Gotchas
- PyTorch 和 NumPy 的 seed 必须同时设置，只设一个会导致不可复现
- Log parsing 使用 `drain3-improved`（import 路径与原 drain3 相同：`from drain3 import TemplateMiner`）；原 drain3 包因 cachetools==4.2.1 锁定与 mlflow 3.x 不兼容，已弃用
- Drain log parsing 结果依赖输入顺序，不同顺序可能产生不同模板
- Kiro/Claude 运行在 base conda 环境，验证 interface 环境中的包须用 `conda run -n interface python -c ...`，直接 `python` 走的是 base
- 多模态时间对齐时，降采样会丢失 metrics 高频 spike 信号，需谨慎选择对齐策略
- POT/GPD 动态阈值只需调 q 一个参数，但极度依赖异常分数分布假设
- **多模态粒度架构约束**：traces/api_responses 可做 endpoint 级特征；metrics（Prometheus）和 logs（原始文本）只能做 service 级特征，无 per-endpoint 粒度——这是采集端的架构性限制，重采也不会变。融合设计的真实形态是 endpoint 级 + service 级两层 join 键，需提前接受。**标签精度**（区别于特征粒度）取决于数据源是否提供 `target_endpoint`：`endpoint_raw2` 有 → per-endpoint 精确标签；`anomod_v1` 无 → case 级近似标签（见下方 per-endpoint 标签精度 gotcha）
- **eval NaN 填补用 train 均值**：`ContractDataset(fit_on_parquet=train.parquet)` 用训练集统计量填 eval 的 NaN，严禁用 eval 自身均值，否则 eval 统计泄漏到特征
- **Log join 静默 NaN**：log 特征经 left join 接入，若某 service 在该时间窗口无日志（如 service 挂了），特征为 NaN，后续均值填补会静默抹掉这个"service 无响应"信号
- **Drain3 输出是聚合统计量，不是 raw token**：LogPreprocessor 产出的是 event_rate/error_ratio/template_diversity 三列，template_id 只作为分组 key 计算多样性，不直接进入特征
- **Normalizer 退化 group（全 NaN 或零方差）不做归一化，保留原值**：某 group（或 global scope）在 fit 集合（仅 Normal）里全 NaN，或只出现过单一取值（`hi-lo=0`）时，`transform()` 必须跳过该 group 的归一化运算——全 NaN 会把 `[nan, nan]` 统计量通过减法/除法扩散污染其他数据完好的 case；零方差若不跳过、只是把除数下限到 `1e-9`，会把 eval 侧任何非零差值放大 1e9 倍，产出 1e9~1e13 量级的离谱数值（`endpoint_red__client_latency_p95`/`latency_divergence` 曾实际命中，2026-07-16 修复，见 history 013）。跳过后保留原始量纲；若某列同时受 `RATE_COLUMNS` 的 `clip(0,1)` 约束，需确认原始量纲天然落在 `[0,1]`，否则 clip 会把正常值误判为异常
- **per-endpoint 标签精度**：contract pipeline 按 case 是否含 `target_endpoint`（`case_metadata.json`）产出 `label_granularity`（`"endpoint"` 精确 / `"case"` 近似 fallback）和 `is_endpoint_anomaly`（= `is_target_endpoint AND phase=='inject'`）。判定依据是 `target_endpoint` 字段是否存在，**不是** `anomaly_level`（那是故障类型描述，语义不等价，未来数据源变化会失配）。`is_anomaly`/`y_true` 语义不变（仍是 case 级）；`eval_baseline_v0.py` 的四层分层（overall/by_anomaly_type/by_anomaly_level/by_endpoint）统一用 `is_endpoint_anomaly`
- **`expand_train_pool` 开关会改变 eval_all 行数，不同取值的 AUROC 不可直接横向比较**：`false`（`contract_v1`）eval_all=13632 行，`true`（`contract_v1_expanded`）eval_all=8221 行——后者摘除了被吸收进训练池的故障 baseline 行。L0/L1/L2 与 RG 分别固定用其中一种取值（见上方 Commands 小节），不要把两者的 AUROC 直接摆在一张表里比大小，除非明确注明各自的 eval 样本数和口径差异。`EndpointBaselineStats`/`Normalizer` 的 fit 范围不受该开关影响，始终锁定纯 Normal 的 `train_fit`
- **`expand_train_pool=true` 后 eval_all 正负比失衡（未修复）**：baseline 阶段行从 eval 移入 train 后，eval_all 正负样本比从 ~50:50 漂移至 ~84:16（inject 行相对比例上升）。AUPRC 在 expanded-pool 实验中的改善很可能是类比例变化的假象而非模型提升，见 history/entries/014。跨口径横向比较 AUPRC 时需注明各自的类比例
- **RG softmax gate collapse**：`ReliabilityGatedFusion` 默认 `gate_normalization=softmax` 在所有 27 种故障类型下均收敛到 `w_svc≈1.0`（完全信任 service 分支，endpoint 分支权重归零）。根因：两分支初始偏差尺度 ~1.2× 不对称 + softmax 竞争归一化形成正反馈放大。`independent_sigmoid` 消融（`configs/fusion/reliability_gate_ablation_indep_sigmoid.yaml`）可规避此问题，AUROC 0.6598 vs softmax 0.6024
- **组合模型 optimizer 必须覆盖全部子模块**：`instantiate(optimizer_cfg, params=svdd.parameters())` 只传 SVDD 参数时，fusion encoder 层永远停在随机初始化——L1/L2 在 entry 012 实际命中此 bug（消融结果与 L0 无差异直到修复）。任何新增 fusion+model 组合上线前须验证 optimizer 参数集覆盖所有 `nn.Module`
- **RG coupling tech debt（3 处，待下一 PR 清理）**：① `scripts/train_baseline_v0.py:142-156` 字符串匹配 `cfg.fusion._target_` 做 RG 专属 kwargs 注入，需抽象进 `FusionModule` 接口；② `scripts/build_contract.py:342` 无条件 fit `EndpointBaselineStats`（L0/L1/L2 不消费）；③ `dvc.yaml` RG 专属 stage 在默认 DAG 中，裸 `dvc repro` 会触发 ~20min 不必要的 rebuild

## Git Commit Convention
MUST: 撰写提交信息__必须__严格遵守提交格式。
MUST: 撰写PR信息的标题__必须__在开头带上type信息
```
<type>: <description>

<optional body>
```
Types:
- `[Feature]`: 新功能/新模块
- `[Bugfix]`: Bug 修复
- `[Experiment]`: 实验相关（新实验配置、实验结果、超参数调整）
- `[Data]`: 数据集相关（新增、预处理、版本更新）
- `[Model]`: 模型相关（新模型、模型修改、checkpoint）
- `[Refactor]`: 重构
- `[Docs]`: 文档更新
- `[Test]`: 测试相关
- `[Chore]`: 构建/工具/环境配置

> Commit attribution 已全局禁用，无需添加 Co-Authored-By 行。

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
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/liu/Code/Repos/aris-llm-chat-mirror`).
Update with: `bash /home/liu/Code/Repos/aris-llm-chat-mirror/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
