# 两条 GRU 路线的运行与保留

两种模型共用 `scripts/service_temporal_model.py` 和 `TemporalPredictor`，由配置决定文本分支；原配置没有改变。

|路线|配置|历史每步输入|定位|
|---|---|---|---|
|服务先验 GRU|`configs/service/tt_temporal_prior_gpu_b_v1.json`|28 维，无文本投影|当前较强的研究主模型|
|冻结 BERT + 服务先验 GRU|`configs/service/tt_temporal_prior_bert_gpu_b_v1.json`|28 + 64 维|可选语义消融，当前未改善|

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
后续提交不自动重写这些输出，但共享源码仍需兼容性回归；历史完整代码保留于 `d3c9116`。

BERT 调参协议、实际12候选和三 seed结果见 [调参记录](temporal-tuning-protocol-20261023.md)；
原模型、BERT和强基线对照见 [实验记录](model-collapse-investigation.md)。
