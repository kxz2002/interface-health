# 049 · 扩充独立 Normal 测试窗口的 pooled/per-case 评价 v2

- **日期**: 2026-10-28
- **PR**: 待定 · **Commit**: 待定
- **类型**: Feature / Experiment / Test / Docs
- **影响域**: src/service/labeling/、src/service/modeling/、scripts/、configs/service/、tests/test_service_evaluation_split.py、docs/research/service-20261019/

## 做了什么

新增可配置的 `tt_per_case_balanced_v2_pool_dns_excluded` 评价 profile。训练和独立 Normal calibration 保持原 membership 不变，仅将独立 Normal test Case 按固定时间均匀规则扩充到 1,154 个窗口；故障 Case 继续按每 Case 正负较小值匹配。新 bundle 共 1,488 个评价窗口（167 正、1,321 负），7 个故障 Case 均具备两类，旧 v1/v3 bundle 和结果保持不变。

使用现有 checkpoint 重新评测先验 GRU、三个微服务领域适配基线、PatchAD/KAN-AD/TimesNet 及 short-context 变体，记录 pooled 和 per-case AUROC/AP。v2 的 per-case 指标基本不变，pooled AP 按新的正类 prevalence 下降，结果见 `per-case-evaluation-v5-results-20261028.*`。

## 关键决策

- 不为了形式上的 7:2:1 缩减训练集；保留 10,600 个 Normal fit 窗口和 2,975 个 calibration 窗口，避免已有 checkpoint 失去协议一致性。
- Normal quota 使用显式整数 1,154，而不是模型相关或随机数量；选择排序固定为时间、服务名、sample_id。
- 同一评价 bundle 同时支持 pooled 与 per-case 指标；Normal Case 只进入 pooled，per-case 宏平均只统计同时含正负类的故障 Case。

## 坑 / 已知问题

- 扩充 Normal 后测试集正类比例约 11.2%，因此 pooled AP 不应与 v1 直接按数值比较；必须同时报告 prevalence。
- 1,488 个固定评价窗口中，时序历史不足导致模型实际评分 1,421 行；缺失历史仍保留并按既有规则记录，不能静默删除。
- 领域基线是论文结构的工程适配，不是官方代码的完整复现；per-case 宏平均当前只覆盖 7 个故障 Case。

## 遗留 TODO

- 在论文定稿前决定 v2 是否作为唯一主评价协议，并同步更新 PR/论文中的主表和方法描述。
- 继续保留完整 eligible test coverage 和逐 Case 置信区间/稳定性分析，避免把当前 7 个 Case 的宏平均解释成广泛泛化结论。
