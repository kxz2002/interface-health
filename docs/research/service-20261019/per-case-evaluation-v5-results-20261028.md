# v2 扩充 Normal 测试集结果

本轮使用 `tt_per_case_balanced_v2_pool_dns_excluded`。训练集和独立 Normal calibration 完全沿用既有协议，分别为 10,600 和 2,975 个窗口；只扩充独立 Normal case 的测试窗口。

测试集现在包含 1,488 个窗口：167 个正类、1,321 个负类，其中独立 Normal case 提供 1,154 个负类窗口，7 个故障 Case 仍各自保持正负窗口匹配。POOL、DNS、Unknown 和不支持 recovery 继续排除。时序历史不足导致每个模型实际评分 1,421 行，67 个窗口按既有缺失历史规则处理。

| 模型 | pooled AUROC | pooled AP | per-case AUROC macro | per-case AP macro |
|---|---:|---:|---:|---:|
| 先验 GRU | 0.9786 | 0.8136 | 0.9835 | 0.9560 |
| MSTGAD | 0.9108 | 0.6647 | 0.9763 | 0.9474 |
| AnoFusion | 0.7571 | 0.5874 | 0.9073 | 0.8852 |
| Triple | 0.8342 | 0.3358 | 0.9694 | 0.9494 |
| PatchAD | 0.5341 | 0.1270 | 0.5473 | 0.5532 |
| KAN-AD | 0.8288 | 0.6584 | 0.8172 | 0.8338 |
| TimesNet | 0.7979 | 0.5637 | 0.7494 | 0.7758 |
| PatchAD short | 0.4903 | 0.1555 | 0.4009 | 0.4903 |
| KAN-AD short | 0.7448 | 0.6053 | 0.8009 | 0.8050 |
| TimesNet short | 0.6907 | 0.2284 | 0.6825 | 0.6502 |

这些是三个 seed 的均值。由于本轮只增加 Normal 负类，per-case 指标与 v1 基本保持不变；pooled AP 明显下降，反映了测试集正类比例从约 46.6% 降至约 11.2%，这是预期的 prevalence 效应。结果没有重新训练模型，只重新使用已有 checkpoint 进行推理和评价。
