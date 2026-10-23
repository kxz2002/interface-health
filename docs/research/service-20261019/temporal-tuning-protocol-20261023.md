# 固定服务先验 BERT–GRU 的 Normal-only 超参数选择

本协议是已看过外层测试结果后的探索性实验，不能宣称新盲测，也不能保证得到全局最优超参数或异常检测增益。

## 冻结内容

方案 A 保持冻结 BERT（固定 revision、正文 token/group mean pooling）、64 维可训练投影、32 维 GRU、4 个 15 秒历史窗、12 项数值/日志统计预测和 3 项模态可用性预测。服务先验仍仅用 Normal fit 的 Beta(1,1) 平滑 logit offset；masked MSE 与 BCE 等权。batch128、梯度裁剪1、末轮 checkpoint、搜索种子42固定。

搜索提前限定为学习率 `[0.0003, 0.001, 0.003]` × weight decay `[0.0001, 0.001]` × epochs `[50, 100]`，共12候选。全部候选执行，按该枚举顺序打破完全相同验证损失的并列。不按故障指标扩展候选或改变架构。

## 内外层隔离

只从既有 outer prepared 的已确认 Normal fit ID 确定四个 Normal Case，按真实 `window_start_ms` 的 Case 起点选择最新完整 Case 为内层验证，其余三个为内层拟合；Case 起点并列时以 Case ID 字典序确定。数值归一化、Drain 模板、日志选择、日志统计归一化和服务先验全部从内层 fit 重新拟合。原始表经既有 hash/版本入口校验后立刻按四个 Case 子集化，外层 calibration、外层 Normal test 和 Fault 均不进入候选训练或排序。原 outer 冻结 ID 不改。

只有原 outer fit 的 known-Normal ID 可作为内层训练/验证目标。Case 中其他窗仅能作为对应分区内因果历史；训练历史依旧必须属于已冻结 inner fit，Unknown 不当健康目标。验证 Case 历史仅来自同 Case/Service/分区的严格过去连续窗，首窗冷启动左补零，不跨 Case。

候选选择指标是验证目标逐窗 `masked value MSE + modality BCE` 的均值，同时保留两项分量、完整训练曲线及验证窗数。不得读取 Fault AP/AUROC、调整异常阈值或选 seed。内层无法观测的服务/字段按现有 mask 语义保留，并记录 fit/validation 观测覆盖、被屏蔽 inner fit ID，不删除验证 Case。该损失选择不保证异常区分最优。

## 缓存与复现

内层重新准备后，只生成一次冻结 BERT 向量，12个候选共用；保存确切 ID、日志缓存绑定、向量 SHA256、配置、预处理状态与 hash、所有训练 ID/验证 ID。内层缓存不能替代外层不同 prepared 的缓存。

最终配置从 `run/selected_config.json` 取得，在原 outer fit 上重新训练 seeds `[1,2,42]`，沿用原独立 Normal calibration 与完整测试集，全部种子报告。最终训练可通过 `--text-cache-run` 复用既有同 outer prepared 的 BERT 向量：先 `check_run` 校验 artifact/父链，再核对精确 prepared 父引用、排序 ID、固定模型 revision/池化配置、日志 cache key、float32 `(N,256)`、有限值及向量原始字节 hash。新运行复制向量及元数据，并记录缓存父引用。

## 命令

在对应代码目录使用 `interface` 环境；以下输出目录必须尚不存在：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
~/miniconda3/envs/interface/bin/python scripts/service_temporal_tune.py \
  --config configs/service/tt_temporal_prior_bert_tuning_v1.json \
  --prepared ~/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a \
  --preparation ~/Code/Repos/interface-health/outputs/service/preparation-exploratory-20261020-task4 \
  --label-bundle ~/Code/Repos/interface-health/outputs/service/tt-labels-a-v2-20261020-verified/labels \
  --output ~/Code/Repos/interface-health/outputs/service/TUNING_RUN_ID \
  --tracking-uri "sqlite:///$HOME/Code/Repos/interface-health/outputs/service/model-task9-mlflow.db"
```

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
~/miniconda3/envs/interface/bin/python scripts/service_temporal_model.py \
  --config ~/Code/Repos/interface-health/outputs/service/TUNING_RUN_ID/run/selected_config.json \
  --prepared ~/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a \
  --text-cache-run ~/Code/Repos/interface-health/outputs/service/temporal-bert-bf5fdfd-gru-seed42-20261023-a \
  --method gru --seed 42 \
  --output ~/Code/Repos/interface-health/outputs/service/FINAL_RUN_ID \
  --tracking-uri "sqlite:///$HOME/Code/Repos/interface-health/outputs/service/model-task9-mlflow.db"
```

上述最终训练命令分别使用种子1、2、42及不同输出目录；完整记录所有结果，不选择最佳 seed。
