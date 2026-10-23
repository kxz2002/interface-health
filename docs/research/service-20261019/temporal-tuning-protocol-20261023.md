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

## 2026-10-23 执行结果

代码 `a89bd1f` 已推送，并在 `gpu-b` 独立目录
`~/Code/Repos/interface-health-runs/temporal-tuning-a89bd1f` 核对 HEAD 一致且 clean 后执行。
针对性测试66项通过，提交前 black/isort 等检查通过。

内层拟合 Case 为 `Normal_20261017T020706Z_95064d21e8204018`、
`Normal_20261017T110632Z_4c14d1c71d664ca9`、`Normal_20261017T115148Z_cc9a6da5e99a4263`，
实际 fit7291窗；完整验证 Case 为 `Normal_20261018T030819Z_9adec05d73684fb6`，2857个已确认 Normal 目标窗。
数值、Drain、日志统计重新拟合的状态摘要 hash 为
`acdff409ca8febef6bd7bd0919175ac0e4c8503c480f5996f3924cd7602297f2`。

|候选序号（从0起）|lr|weight decay|epochs|Normal验证loss|
|---|---|---|---|---|
|0|0.0003|0.0001|50|0.7705083710|
|1|0.0003|0.0001|100|0.8010753610|
|2（选中）|0.0003|0.001|50|0.7704755508|
|3|0.0003|0.001|100|0.8010107699|
|4（原默认）|0.001|0.0001|50|0.8287317312|
|5|0.001|0.0001|100|0.9279790198|
|6|0.001|0.001|50|0.8286405472|
|7|0.001|0.001|100|0.9276217451|
|8|0.003|0.0001|50|0.9048029223|
|9|0.003|0.0001|100|1.0411039569|
|10|0.003|0.001|50|0.9039043146|
|11|0.003|0.001|100|1.0179735089|

全部12组成功发布，候选纯训练累计334.02秒，不含预处理、BERT编码和评价。
按预定Normal验证损失选出lr0.0003、weight decay0.001、50轮；相对原默认验证损失下降7.03%。
候选0与2差异仅0.00003282，不能据此声称weight decay的稳健优势；100轮在所有配对中验证损失更高。
搜索MLflow run为 `e466b7f3679143b2ac202de80d09ee1c`，输出目录为
`outputs/service/temporal-tuning-a89bd1f-seed42-20261023-a`。

随后使用选定配置在原outer fit10148窗上重训全部三个种子，复用已校验同prepared的冻结BERT缓存：

|seed|AUROC|AP|q99 TP/FP/FN/TN|MLflow|
|---|---|---|---|---|
|1|0.8108356840|0.6713820651|219/35/226/2817|e44ee11a866a4421a77592c6a6ef7cab|
|2|0.8125620499|0.6684076624|215/33/230/2819|20a90fd8be8b4e98932daac9f0b43c39|
|42|0.8138645067|0.6704034604|221/33/224/2819|166701e4766f4fc6a9e932e67a3586eb|
|均值|0.8124207469|0.6700643960|不合并相关窗口冒充独立重复|—|

三组输出为 `outputs/service/temporal-bert-tuned-a89bd1f-gru-seed{1,2,42}-20261023-a`。
每组主评价3297窗（2852负、445正）、完整评分12598窗、校准2853窗；
与对应原BERT运行核对精确主集、完整评分和校准ID及prepared父引用一致。
搜索7项及每组最终运行15项声明产物均经 `check_run` 的大小/SHA256/递归父链校验通过。
最终fit阶段（含checkpoint保存）分别26.18、24.03、34.67秒，缓存加载校验约2.2秒/组。
队列正常退出，最终GPU占用回落至1MiB。

原BERT默认配置均值为AUROC/AP=0.8132354718/0.6670200639；
调参后AUROC下降约0.00081、AP提高约0.00304，不能声称整体改善。
无BERT先验GRU均值0.8239703001/0.6800605901仍更高。
本次证明Normal验证损失下降未带来相同幅度的异常区分收益；
选定配置是这12候选在单个Normal验证Case上的最优，不是全局最优或故障评价最优。
