# Interface-Health

接口健康度多模态数据融合与退化轨迹预测研究。基于 PyTorch，融合微服务环境中异构多源数据（metrics/KPI/logs），实现 per-API-endpoint 粒度的健康度建模与时序异常检测。

## Directory Structure
```
data/
├── raw/               # 原始数据集 (READ-ONLY, never modify)
├── processed/          # 预处理后的数据
└── external/          # 外部/公开数据集

configs/               # 实验配置 (YAML)
scripts/               # 训练、评估、预处理脚本
src/                   # 核心源码
├── data/              # 数据加载与预处理 (Drain3 log parsing, 对齐, 归一化)
├── models/            # 模型定义 (融合模块, 检测器, 预测器)
├── fusion/            # 多模态融合层 (L1-L4 层级实现)
├── training/          # 训练逻辑 (trainer, callback, scheduler)
├── evaluation/        # 评估指标与可视化
└── utils/             # 通用工具 (seed, logging, io)

outputs/               # 实验结果 (logs, figures, checkpoints)
models/                # 保存的模型 checkpoint
tests/                 # 单元测试
docs/                  # 文档与实验计划
notebooks/             # Jupyter notebook 探索性分析
```
## Commands
后续补充

## Config System
- 所有配置文件放 `configs/`，YAML 格式
- 超参数禁止硬编码在代码中，必须通过配置文件指定
- 命令行覆盖: `python scripts/train.py --config configs/base.yaml --batch_size 32`

## Data Rules
- `data/raw/` 是 READ-ONLY，绝不修改原始数据
- 所有数据变换必须代码化（`src/data/`），不可手动处理
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

## Testing
- 核心数据处理逻辑 (`src/data/`) 和融合模块 (`src/fusion/`) 必须有单元测试
- 测试文件放在 `tests/`，命名 `test_<module>.py`
- 实验流程需有可重复性验证，但不强制 TDD

## Known Gotchas
- PyTorch 和 NumPy 的 seed 必须同时设置，只设一个会导致不可复现
- Drain3 log parsing 结果依赖输入顺序，不同顺序可能产生不同模板
- 多模态时间对齐时，降采样会丢失 metrics 高频 spike 信号，需谨慎选择对齐策略
- POT/GPD 动态阈值只需调 q 一个参数，但极度依赖异常分数分布假设

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
