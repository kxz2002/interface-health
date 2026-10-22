# Task 10 首轮远端结果

日期：2026-10-22。数据、划分、Normal-only 训练和 calibration 协议均绑定到 `model-task9-prepared-20261022-a`；结果只记录已实际完成的运行。

| 方法 | seed | 主测试 AUROC | 主测试 AP | 主测试行数 |
|---|---:|---:|---:|---:|
| transformer_cls | 42 | 0.473863 | 0.204927 | 3297 |
| numeric_IF | 42 | 0.660570 | 0.311528 | 3297 |
| numeric_OCSVM | 42 | 0.563648 | 0.380567 | 3297 |

Transformer 训练耗时约 86 分 04 秒；模型 checkpoint 已生成。三个方法的主测试评分均完整，主测试包含 2852 个 Normal 窗口和 445 个已知正类窗口。首轮 Transformer 的结果低于两个经典基线，说明当前数据存在可利用的数值结构，但不能据此判断模型架构或全矩阵结论。还需要完成其他方法和重复 seed，并检查分数方向、训练收敛及特征贡献。

远端完整产物目录：

- `outputs/service/model-task10-transformer-cls-seed42-evaluate-fix-20261022`
- `outputs/service/model-task10-baseline-20261022-if-evaluate`
- `outputs/service/model-task10-baseline-20261022-ocsvm-evaluate`

checkpoint、parquet、日志和 MLflow 数据库属于实验产物，保留在远端或研究备份，不提交到 Git。
