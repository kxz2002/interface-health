# C · 模态观测覆盖诊断（去跨run混杂）

Deep SVDD × 4 seeds[1, 2, 3, 42]。正样本=is_endpoint_anomaly。AUROC=mean±std。

## 表1 · within-case 协议（负=同case baseline/recover，去混杂）

| HTTP子类型 | endpoint_only | service_only | all | 单特征oracle |
|---|---|---|---|---|
| ABORT | 0.278±0.026 | 0.822±0.038 | 0.426±0.072 | 0.993 (client_error_rate) |
| DELAY | 0.651±0.002 | 0.437±0.035 | 0.629±0.029 | 0.347 (trace_request_count) |
| PATCH | 0.601±0.017 | 0.344±0.074 | 0.595±0.045 | 0.376 (trace_request_count) |
| REPLACE | 0.419±0.009 | 0.469±0.047 | 0.499±0.056 | 1.000 (client_error_rate) |
| Lv_E_all | 0.489±0.012 | 0.518±0.030 | 0.536±0.022 | — |

## 表2 · cross-run 协议（负=跨run Normal，暴露混杂虚高）

| HTTP子类型 | endpoint_only | service_only | all |
|---|---|---|---|
| ABORT | 0.183±0.011 | 0.922±0.012 | 0.394±0.118 |
| DELAY | 0.619±0.025 | 0.696±0.009 | 0.592±0.011 |
| PATCH | 0.574±0.016 | 0.590±0.093 | 0.554±0.026 |
| REPLACE | 0.339±0.008 | 0.717±0.047 | 0.465±0.069 |
| Lv_E_all | 0.428±0.007 | 0.731±0.037 | 0.501±0.043 |

## 读法
- 单特征oracle >> 全向量SVDD ⇒ 信号存在但被朴素等权距离淹没（信号淹没）。
- 表2 service_only 明显高于表1 ⇒ 该子类型的'检测'部分来自跨run身份，非故障（混杂）。
- oracle≈0.5（PATCH）⇒ 该故障对所有 RED 特征隐形，需响应体校验。
