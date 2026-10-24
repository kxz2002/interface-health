# 原GRU八方向探索结果

本轮按用户授权的优先级完成八方向、每方向三个seed（1、2、42），另复现原GRU三个seed，共27组。配置见[冻结协议](gru-eight-directions-protocol.md)及 `configs/service/gru_eight_directions_v1.json`。本轮属于已见测试集后的探索，不是新盲测；未改变原GRU默认配置，未关闭历史Task10完整方法矩阵。

## 主要结论

分组GRU的AUROC和AP均值最好，较原GRU分别提高0.008737和0.006945，但seed1的AP下降0.003969，没有通过预声明的“三seed双指标均超过对应原GRU”替代门槛。保留原GRU默认；分组GRU作为后续独立验证候选。其他方向存在局部收益或退步，不能把本轮说成全部无提升，也不能宣称已稳定打败原模型。

下表为三个seed的均值±样本标准差；相关时间窗不作为独立重复。

| 优先级 | 方向 | AUROC | AP | 神经参数量 |
|---|---|---|---|---|
| 对照 | 原GRU | 0.823970 ± 0.001862 | 0.680061 ± 0.010999 | 6447 |
| 1 | 绝对值/变化量双头 | 0.826016 ± 0.001166 | 0.659586 ± 0.009837 | 6843 |
| 2 | 多尺度历史特征 | 0.823628 ± 0.005568 | 0.682272 ± 0.007149 | 11055 |
| 3 | 资源/执行/日志分组GRU | **0.832708 ± 0.002253** | **0.687006 ± 0.000443** | 7119 |
| 4 | Normal原型记忆 | 0.791129 ± 0.003607 | 0.681083 ± 0.012297 | 6447，另有原型存储 |
| 5 | 机制分数组合 | 0.816509 ± 0.001291 | 0.654676 ± 0.015749 | 6447 |
| 6 | 异方差GRU | 0.765577 ± 0.011136 | 0.534431 ± 0.008578 | 6843 |
| 7 | TCN-GRU | 0.817752 ± 0.005796 | 0.680407 ± 0.004353 | 6735 |
| 8 | 遮蔽预训练 | 0.819694 ± 0.001727 | 0.677594 ± 0.001204 | 6843 |

分组GRU逐seed配对：

| seed | 原GRU AUROC/AP | 分组GRU AUROC/AP | AUROC差 | AP差 |
|---|---|---|---|---|
| 1 | 0.824682 / 0.690464 | 0.833499 / 0.686496 | +0.008817 | -0.003969 |
| 2 | 0.821857 / 0.681168 | 0.830165 / 0.687298 | +0.008308 | +0.006130 |
| 42 | 0.825372 / 0.668550 | 0.834459 / 0.687224 | +0.009087 | +0.018674 |

分组模型参数增加约10.4%，尚未开展等参数或去门控消融，不能将提升全部归因于分组或门控本身。AP跨seed波动减小只是本次三个seed的观察，不代表统计显著性。

## 阈值表现与故障盲区

主评价固定3297窗：445正类、2852负类；完整test为12598窗，独立Normal calibration为2853窗。Unknown不补正常，时序上下文不跨Case、Service、划分或时间缺口。阈值取独立Normal校准集合的全局q99，超过阈值才报警。

| 模型 | TP（seed 1/2/42） | FP（seed 1/2/42） | 平均召回 | 平均FPR |
|---|---|---|---|---|
| 原GRU | 251 / 227 / 220 | 30 / 34 / 42 | 52.28% | 1.24% |
| 分组GRU | 228 / 226 / 223 | 35 / 36 / 32 | 50.71% | 1.20% |

分组GRU的排序均值提升未转化为q99召回提升。POOL的117个已标正窗中，原GRU各seed仅检出0/1/0，八方向均为0；DNS的39个已标正窗，各模型均未检出。现有输入没有独立IO、连接池或DNS直接字段，这些结果不能解释为所有干预均实际造成完整服务异常。

原型方向在CPU、DISKIO部分seed召回更高，但总体AUROC明显下降；异方差在DISKIO检出较多，同时主评价显著退步。不能按单一故障挑选替代模型。

## Normal验证与公平性

外层四个fit Normal Case中，最后一个完整Case作为内层留出，另外三个重拟合数值归一化、日志模板及统计；随后用完整四个Case拟合外层模型。每个方向都完成预声明三seed，不按Fault指标调整参数、组合权重、分数方向或训练轮数。

Normal预测损失只作诊断；双头与异方差改变了优化目标，不能直接将其训练loss与原MSE排名。异方差另保存原坐标MSE。原型和机制复用对应seed原GRU，不重新训练预测器；组合权重固定0.8/0.2，分量尺度来自内层留出Normal q95。该Case同时承担尺度参考，因此其组合分数不是独立组合验证；内外模型及原型坐标的迁移误差也是限制。外层calibration仍独立。

## 产物与复现

远端原dirty仓库保持原状；基础commit经Git bundle同步至独立worktree，新增文件按清单同步，启动前核验四个核心文件SHA256。当前分支 `exp/service-model-improvement-session`，基础HEAD `f3bacdc`，新增实现为有源码hash记录的工作区overlay。

- 独立worktree：`gpu-b:/home/u2024110110/Code/Repos/interface-health-runs/service-improvement-session`。
- 实验套件：该worktree下 `outputs/service/eight-directions-20261024-a/`，27个子run均有完成标记、manifest、内外checkpoint、实际context IDs、逐样本分数、阈值、训练历史及评价。
- MLflow：该worktree下 `outputs/service/goal-20261024-mlflow.db`，完整数据库留远端。
- 基础源码归档：套件下 `source-baseline-f3bacdc.tar.gz`，SHA256为 `bd5da639b25ea603fca3fb03465b3e0575236c6733cc330071193498bd8663d1`；overlay与协议留在 `source/`，核心源码hash见审计摘要。
- 历史基线复放：[逐值比对摘要](gru-eight-directions-baseline-replay.json)。三个seed各12598 test及2853 calibration分数与历史原GRU逐值完全相等，最大差0。
- 独立验收：[审计摘要](gru-eight-directions-audit.json)状态success，27组、513项manifest产物、54份内外checkpoint、2160轮实际训练的MLflow记录均通过；27组共用同一个冻结样本绑定。原型/机制的12份checkpoint为基线权重复用，实际独立训练42份。完整训练、模型与Parquet保留在远端；仓库只存源码及小型证据。

在该worktree重训时必须使用尚不存在的新输出目录，例如下列命令；此处仅记录复现入口，不再次启动训练：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n interface python scripts/service_eight_directions.py \
  --config configs/service/gru_eight_directions_v1.json \
  --prepared "$HOME/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a" \
  --preparation "$HOME/Code/Repos/interface-health/outputs/service/preparation-exploratory-20261020-task4" \
  --label-bundle "$HOME/Code/Repos/interface-health/outputs/service/tt-labels-a-v2-20261020-verified" \
  --tracking-uri sqlite:////home/u2024110110/Code/Repos/interface-health-runs/service-improvement-session/outputs/service/replay-mlflow.db \
  --output outputs/service/eight-directions-replay
```

在独立worktree中可用以下命令重做只读验收，必须指定新的审计输出文件：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false \
conda run -n interface python scripts/audit_service_eight_directions.py \
  --suite outputs/service/eight-directions-20261024-a \
  --prepared "$HOME/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a" \
  --preparation "$HOME/Code/Repos/interface-health/outputs/service/preparation-exploratory-20261020-task4" \
  --label-bundle "$HOME/Code/Repos/interface-health/outputs/service/tt-labels-a-v2-20261020-verified" \
  --tracking-db outputs/service/goal-20261024-mlflow.db \
  --output outputs/service/eight-directions-20261024-a/audit-replay.json
```

验收按冻结配置的CUDA设备恢复模型，以严格容差比较重算分数；跨CPU/CUDA运算曾出现约0.001的差异，不能宣称跨设备逐值一致。首版审计还暴露内存tuple与JSON list表示不同，修复方式为JSON规范化后比较全部字段与hash，真实统计漂移仍拒绝。失败审计日志保留，未因此改变训练产物或放宽分数容差。

最终相关测试在本机和gpu-b各通过112项；black、isort及 `git diff --check` 通过。测试范围为 `tests/test_service_eight_directions.py`、`tests/test_service_model_temporal.py`、`tests/test_service_gru_architectures.py`、`tests/test_service_temporal_tuning.py` 和 `tests/test_gru_architecture_audit.py`，没有运行历史Endpoint全套测试或DVC重建。

本轮探索结束，不自动新增第九方向。若另获授权，优先对分组GRU做等参和去门控消融，并取得新增独立Case验证；POOL/DNS先补可观测性与影响证据。

## 分支收尾

2026-10-24按用户授权，核对本机与gpu-b隔离副本的16个相关文件，收尾前SHA256全部一致；统一提交代码、配置、测试、报告和history044，再推送分支向master提出PR。远端隔离副本先保留限定路径的overlay stash备份，再快进到最终提交；原仓库历史dirty文件、完整实验输出及运行时源码快照均保留。上文的f3bacdc和overlay描述是实验启动时的真实provenance，最终Git提交不反写历史manifest。

收尾重新执行112项相关测试并通过；全仓 `black --check src/ scripts/ tests/`（208文件）、`isort --check-only src/ scripts/ tests/` 和 `git diff --check` 均通过。未重新训练模型。

## PR独立审查

PR #39审查发现原MLflow核验仅遍历history已有记录，若history和数据库同时截断会漏报训练不完整。修复后要求inner/outer均完整遵循冻结阶段及epoch顺序：普通模型各50轮、遮蔽模型各10轮预训练再50轮预测、原型/机制复用分支无训练记录；同时拒绝额外MLflow训练记录和重复最终指标。新增回归覆盖后，相关测试由112项增至125项，格式检查通过。冻结模型、训练入口与配置未改动，原检测结果和逐样本分数保持原样。

使用新检查对gpu-b实际27组manifest、完整训练history及只读MLflow数据库复核，全部通过，训练记录仍为2160轮；见[补充核验摘要](gru-eight-directions-tracking-review.json)。本次未再次运行模型分数重放，原54份checkpoint与分数重放证据仍见前述独立审计。
