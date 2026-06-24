# Interface-Health

接口健康度多模态数据融合与退化轨迹预测研究。基于 PyTorch，融合微服务环境中异构多源数据（metrics/KPI/logs），实现 per-API-endpoint 粒度的健康度建模与时序异常检测。

## 研究背景与方向

### 项目定位

为潜在的民航 SAT（航空票务核心系统，6000+ APIs、5 亿次调用/天）行业项目做技术储备，存在两个走向：

- **场景 A（项目拿到）**：在 SAT 专有数据上做论文，考虑周期性、动态基线等真实业务场景
- **场景 B（项目落空）**：在 Train-Ticket 数据集上做，专注异常检测创新

当前数据集：Train-Ticket 微服务系统，11 个 case（1 Normal + 10 故障注入），数据在 `data/anomod/` 下。数据集字段口径、pipeline 逻辑与已知问题详见 `docs/agent-docs/dataset-guide.md`，接触数据相关代码前必读。

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
├── anomod/            # Train-Ticket 故障注入数据集（44GB，READ-ONLY，never modify）
│   ├── Normal/
│   ├── Lv_P_*/  Lv_S_*/  Lv_D_*/   # 10 个故障注入 case
│   └── <case>/_pipeline_out/         # pipeline 产物（tt_fused_15s.csv 等，建模直接使用）
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
├── models/            # 模型定义 (融合模块, 检测器, 预测器)
├── fusion/            # 多模态融合层 (L1-L4 层级实现)
├── training/          # 训练逻辑 (trainer, callback, scheduler)
├── evaluation/        # 评估指标与可视化
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

# 训练（基础）
python scripts/train.py

# 训练（切换模型）
python scripts/train.py model=vae

# 训练（覆盖超参数）
python scripts/train.py model.hidden_dim=256 training.lr=1e-4

# 多组超参数扫描
python scripts/train.py --multirun model=deep_svdd,vae training.lr=1e-3,1e-4

# === 当前阶段（toy baseline + DVC，真实模型尚未实现）===
dvc repro                        # 一键跑通 train → eval，自动检测变更
dvc metrics show                 # 查看 artifacts/metrics.json（AUROC/AUPRC）
python scripts/train_baseline.py --toy --out artifacts/scores.parquet --seed 42
python scripts/eval.py --scores artifacts/scores.parquet --out artifacts/metrics.json

# 运行测试
pytest tests/
```

## Config System
- 所有配置文件放 `configs/`，YAML 格式，入口为 `configs/base.yaml`
- 超参数禁止硬编码在代码中，必须通过配置文件指定
- 命令行覆盖语法（Hydra）: `python scripts/train.py model.hidden_dim=256 training.lr=1e-4`
- 切换配置组: `python scripts/train.py model=vae`
- 模型通过 `hydra.utils.instantiate(cfg.model)` 实例化，`_target_` 指向具体类

## Data Rules
- `data/anomod/` 等数据集目录是 READ-ONLY，绝不修改原始数据
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
- **多模态粒度架构约束**：traces/api_responses 可做 endpoint 级特征；metrics（Prometheus）和 logs（原始文本）只能做 service 级特征，无 per-endpoint 粒度——这是采集端的架构性限制，重采也不会变。融合设计的真实形态是 endpoint 级 + service 级两层 join 键，需提前接受

## Git Commit Convention
MUST: 撰写提交信息__必须__严格遵守提交格式。
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
<!-- ARIS:BEGIN -->
## ARIS Skill Scope
ARIS skills installed in this project: 80 entries.
Manifest: `.aris/installed-skills.txt` (lists every skill ARIS installed and its upstream target).
For ARIS workflows, prefer the project-local skills under `.claude/skills/` over global skills.
Do not modify or delete files inside any skill that is a symlink (symlinks point into `/home/kxz2002/Code/Repos/Auto-claude-code-research-in-sleep`).
Update with: `bash /home/kxz2002/Code/Repos/Auto-claude-code-research-in-sleep/tools/install_aris.sh`  (re-runnable; reconciles new/removed skills).
<!-- ARIS:END -->
