# 两条 GRU 路线的运行与保留

两种模型共用 `scripts/service_temporal_model.py` 和 `TemporalPredictor`，由配置决定文本分支；原配置没有改变。

|路线|配置|历史每步输入|定位|
|---|---|---|---|
|服务先验 GRU|`configs/service/tt_temporal_prior_gpu_b_v1.json`|28 维，无文本投影|当前较强的研究主模型|
|冻结 BERT + 服务先验 GRU|`configs/service/tt_temporal_prior_bert_gpu_b_v1.json`|28 + 64 维|可选语义消融，当前未改善|
|8窗服务先验GRU|`configs/service/tt_temporal_prior_gru_context8_gpu_b_v1.json`|28维，历史8窗|独立探索，尚无稳定整体改善|
|hidden64服务先验GRU|`configs/service/tt_temporal_prior_gru_hidden64_gpu_b_v1.json`|28维，隐藏维度64|独立探索，尚无稳定整体改善|
|因果残差服务先验GRU|`configs/service/tt_temporal_prior_gru_residual_gpu_b_v1.json`|28维，原4窗/hidden32|三seed负向结果；仅保留探索记录|
|服务条件化GRU|`configs/service/tt_temporal_prior_gru_service_bias_gpu_b_v1.json`|28维，4窗，加fit服务数值偏置|三seed未改善，独立探索记录|
|短长历史GRU|`configs/service/tt_temporal_prior_gru_short_long_gpu_b_v1.json`|28维，4/8窗各hidden16|均值略升，seed不稳定，独立探索记录|

在 gpu-b 对应源码目录执行，选择配置并使用尚不存在的输出目录：

```bash
model_config=configs/service/tt_temporal_prior_gpu_b_v1.json
run_id=YOUR_NEW_PRIOR_GRU_RUN_ID
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
~/miniconda3/envs/interface/bin/python scripts/service_temporal_model.py \
  --config "$model_config" --method gru --seed 42 \
  --prepared "$HOME/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a" \
  --output "$HOME/Code/Repos/interface-health/outputs/service/$run_id" \
  --tracking-uri "sqlite:///$HOME/Code/Repos/interface-health/outputs/service/model-task9-mlflow.db"
```

BERT 路线改用表中独立配置，并使用不同 run_id；其余参数一致。BERT 从固定 revision 的本地缓存加载。
可选 `--text-cache-run` 指向同 prepared 的已完成 BERT run，经过声明产物、父引用、ID、维度与 hash 校验后复用，不适用于无 BERT 配置。

入口会训练、评分、校准、评价，已有输出目录拒绝覆盖；不是仅加载 checkpoint 的推理命令。
checkpoint、resolved config、fit IDs、上下文 IDs、分数和 manifest 均留在各自 run 中。
恢复无文本 checkpoint 时使用无文本结构，有文本时还需匹配文本投影结构，不能交叉加载。
残差候选恢复时必须从checkpoint的`config.training.value_prediction`传入`TemporalPredictor`；旧配置未提供该字段时默认`absolute`。残差与绝对预测的参数形状相同，仅加载`state_dict`无法区分预测语义，不能忽略对应配置。残差参考仅来自过去有效观测，当前窗口只用于计算预测误差。
后续提交不自动重写这些输出，但共享源码仍需兼容性回归；历史完整代码保留于 `d3c9116`。

新增架构统一通过 `restore_temporal_model(checkpoint, inputs, device)` 恢复，读取checkpoint配置及 `architecture_metadata`；旧checkpoint仍兼容。服务字典仅来自fit，未知服务零偏置，不能仅用相同shape推断架构。

误差校准与状态/时序融合使用独立后处理入口 `scripts/service_gru_error_calibration.py` 和 `scripts/service_gru_state_temporal.py`，分别选择 `tt_gru_error_calibration_gpu_b_v1.json` / `tt_gru_state_temporal_gpu_b_v1.json`；它们复用原GRU checkpoint，不使用此处训练命令。来源绑定、运行ID及迁移限制见 [四方向报告](gru-four-directions-results.md)。本阶段总体结论与远端文件保留见 [阶段收尾](model-exploration-closeout.md)。

BERT 调参协议、实际12候选和三 seed结果见 [调参记录](temporal-tuning-protocol-20261023.md)；
原模型、BERT和强基线对照见 [实验记录](model-collapse-investigation.md)。
