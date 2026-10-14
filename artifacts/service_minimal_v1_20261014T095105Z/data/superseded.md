# 首轮开发数据已被替代

保留本目录作为首轮 prepare 审计；不用于 score。
补充 fixture 揭示两项 T2 Schema 问题：证据 independent_label/proxy_label
应保存 nullable Int8；R3 有效样本不应因仅诊断列缺值带有 exclusion_reasons。
最终代码已修正并通过 66 项 T0/T1/T2 测试。主数据将来自新的不可覆盖目录
artifacts/service_minimal_v1_20261014T095447Z/data。
本目录所有既有 parquet、manifest 和成功标记保持首轮实际内容。
