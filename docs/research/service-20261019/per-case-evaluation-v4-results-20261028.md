# v4 pooled/per-case 评测结果

本轮使用新版本 `tt_per_case_balanced_v1_pool_dns_excluded`。旧 v3 bundle 和结果未覆盖，仍可通过原 profile 复现。

新测试集先固定 358 个 eligible 窗口：167 个正类、191 个负类。7 个保留故障 Case 各自按正负较小值确定性匹配，均同时包含两类；独立 Normal Case 选取 24 个负类窗口，只进入 pooled 指标，不参与 per-case 宏平均。POOL、DNS、Unknown 和不支持的 recovery 均排除。

模型训练和 Normal calibration 沿用原协议。由于时序模型需要历史上下文，固定测试清单中有 20 个窗口缺少可用历史，按既有规则填充 0 分并记录；最终每个模型报告 336 行。独立 calibration 仍有 71 个窗口缺少历史，阈值计算规则不变。

| 模型 | pooled AUROC | pooled AP | per-case AUROC macro | per-case AP macro |
|---|---:|---:|---:|---:|
| 先验 GRU | 0.9654 | 0.9290 | 0.9835 | 0.95596 |
| MSTGAD 风格 | 0.9481 | 0.9140 | 0.9763 | 0.9474 |
| AnoFusion 风格 | 0.8325 | 0.8301 | 0.9073 | 0.8852 |
| Triple 风格 | 0.9282 | 0.8758 | 0.9694 | 0.9494 |
| PatchAD | 0.5120 | 0.4773 | 0.5473 | 0.5532 |
| KAN-AD | 0.7859 | 0.8322 | 0.8172 | 0.8338 |
| TimesNet | 0.7221 | 0.7667 | 0.7494 | 0.7758 |
| PatchAD short | 0.4991 | 0.5080 | 0.4009 | 0.4903 |
| KAN-AD short | 0.7670 | 0.8193 | 0.8009 | 0.8050 |
| TimesNet short | 0.5915 | 0.5327 | 0.6825 | 0.6502 |

表中是 seed 1、2、42 的算术平均；short 行同样使用三个 seed。per-case 宏平均只对 7 个同时含正负窗口的故障 Case 计算；Normal Case 的 AUROC/AP 保持为单类、未纳入宏平均。逐 seed 数字、q99 Recall/FPR、样本覆盖和逐 Case 结果见同名 JSON 及 gpu-b 输出目录 `percase-v4-prior-gru-20261028`、`percase-v4-neural-baselines-20261028`、`percase-v4-neural-short-baselines-20261028`、`/tmp/percase_eval_v1e`。

本轮没有重新训练 checkpoint；重新执行了模型推理，并用新固定测试集和原独立 Normal calibration 重新计算指标。新 profile 的选择规则、derived bundle 和验收 receipt 均已固化。
