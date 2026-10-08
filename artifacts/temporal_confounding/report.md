# 时间混淆与朴素基线诊断（entry 027）

## 1. 串行采集布局

- case 数：27，其中 Normal case 1 个
- 采集跨度：20260728T161044 → 20260729T090100
- Normal case 采集序号：[0]

```
                                            case_id    collected_at  order
                            Normal_20260728T161044Z 20260728T161044      0
              Lv_P_CPU_preserve_20260728T164711Z_em 20260728T164711      1
           Lv_P_DISKIO_preserve_20260728T172525Z_em 20260728T172525      2
          Lv_P_NETLOSS_preserve_20260728T180731Z_em 20260728T180731      3
        Lv_S_HTTPABORT_preserve_20260728T184544Z_em 20260728T184544      4
 Lv_S_DNSFAIL_preserve_no_order_20260728T192351Z_em 20260728T192351      5
          Lv_S_KILLPOD_preserve_20260728T200124Z_em 20260728T200124      6
             Lv_S_KILLPOD_order_20260728T203705Z_em 20260728T203705      7
                Lv_D_cachelimit_20260728T214842Z_em 20260728T214842      8
Lv_D_CONNECTION_POOL_exhaustion_20260728T222559Z_em 20260728T222559      9
       Lv_D_TRANSACTION_timeout_20260728T230311Z_em 20260728T230311     10
          Lv_E_HTTPABORT_travel_20260728T234005Z_em 20260728T234005     11
          Lv_E_HTTPDELAY_travel_20260729T001736Z_em 20260729T001736     12
        Lv_E_HTTPREPLACE_travel_20260729T005455Z_em 20260729T005455     13
          Lv_E_HTTPPATCH_travel_20260729T013231Z_em 20260729T013231     14
         Lv_E_HTTPABORT_travel2_20260729T020949Z_em 20260729T020949     15
         Lv_E_HTTPDELAY_travel2_20260729T024658Z_em 20260729T024658     16
       Lv_E_HTTPREPLACE_travel2_20260729T032420Z_em 20260729T032420     17
         Lv_E_HTTPPATCH_travel2_20260729T040139Z_em 20260729T040139     18
           Lv_E_HTTPABORT_order_20260729T043859Z_em 20260729T043859     19
           Lv_E_HTTPDELAY_order_20260729T051623Z_em 20260729T051623     20
         Lv_E_HTTPREPLACE_order_20260729T055342Z_em 20260729T055342     21
           Lv_E_HTTPPATCH_order_20260729T063100Z_em 20260729T063100     22
       Lv_E_HTTPABORT_assurance_20260729T070828Z_em 20260729T070828     23
       Lv_E_HTTPDELAY_assurance_20260729T074559Z_em 20260729T074559     24
     Lv_E_HTTPREPLACE_assurance_20260729T082326Z_em 20260729T082326     25
       Lv_E_HTTPPATCH_assurance_20260729T090100Z_em 20260729T090100     26
```

## 2. inject 阶段的时间位置一致性

跨 case 的 inject 起止 rel_pos 分布（std 越小 → 固定时序模板越强 → shortcut 越强）：

```
           min      max     count
count  26.0000  26.0000   26.0000
mean    0.5858   0.9301   57.3077
std     0.0149   0.0395   67.4186
min     0.5752   0.7611    4.0000
25%     0.5804   0.9286   40.0000
50%     0.5804   0.9286   40.0000
75%     0.5804   0.9375   40.0000
max     0.6346   1.0000  290.0000
```

## 3. pooled vs per-case 宏平均（case 身份 shortcut 检验）

```
                         model  pooled_auroc  macro_case_auroc     gap  pooled_auprc  n_valid_case
                   concat (L0)        0.9095            0.9140 -0.0044        0.6717            23
       independent_concat (L1)        0.8352            0.8333  0.0019        0.6480            23
                    gated (L2)        0.9120            0.9131 -0.0011        0.7155            23
 reliability_gate softmax (RG)        0.7026            0.7042 -0.0015        0.3642            23
reliability_gate indep_sigmoid        0.5000            0.5000  0.0000        0.0646            23
      deviation_weighted (DWF)        0.9200            0.9238 -0.0037        0.7400            23
```

gap ≈ 0 说明判别力不来自跨 case 基线差异（per-case 口径下 case 身份不可利用）。

## 4. rel_pos 平凡基线（时间位置 shortcut）

- 只用 rel_pos 当异常分数：pooled AUROC **0.9340** / per-case 宏平均 **0.9656** (n=23)
- pooled AUPRC 0.3112

特征是否间接携带时间位置（GroupKFold by case）：

- R² = **0.1026**, Spearman ρ = **0.1628**
- 低 R² ⇒ 模型不是靠特征闻到时间，rel_pos 高分是数据集缺陷而非模型作弊

## 5. 无学习基线（模型有没有挣到复杂度）

### 5.1 单特征

```
                                      feature  macro_auroc  signed
                endpoint_red__client_5xx_rate       0.7139  0.7139
              endpoint_red__client_error_rate       0.7139  0.7139
           endpoint_red__client_request_count       0.7105  0.7105
              service_log__template_diversity       0.3093  0.6907
     endpoint_red__client_content_length_mean       0.3659  0.6341
              endpoint_red__trace_latency_p50       0.3708  0.6292
             endpoint_red__latency_divergence       0.6050  0.6050
                     service_log__error_ratio       0.4032  0.5968
            endpoint_red__trace_request_count       0.5947  0.5947
              endpoint_red__trace_latency_p95       0.4073  0.5927
                service_metric__process_count       0.5632  0.5632
                      service_log__event_rate       0.4405  0.5595
           service_metric__memory_usage_ratio       0.5489  0.5489
 endpoint_red__client_body_hash_mismatch_rate       0.5423  0.5423
               service_metric__cpu_usage_rate       0.4624  0.5376
                 endpoint_red__trace_5xx_rate       0.5200  0.5200
               endpoint_red__trace_error_rate       0.5167  0.5167
endpoint_red__client_content_length_rel_shift       0.5158  0.5158
             endpoint_red__client_latency_p95       0.5078  0.5078
            service_metric__net_tx_error_rate       0.5000  0.5000
            service_metric__net_rx_error_rate       0.5000  0.5000
```

### 5.2 per-case baseline 自适应 z-score（零参数、零训练）

```
baseline  macro_case_auroc  n_valid_case
  max|z|            0.9403            23
L2 ||z||            0.9415            23
 mean|z|            0.9306            23
```

## 结论

- 最好的学习模型 per-case 宏平均 AUROC：**0.9238**
- 最好的零参数 z-score 基线：**0.9415**
- rel_pos 平凡基线：**0.9656**

→ 零参数基线超过学习模型（差 +0.0177）。
