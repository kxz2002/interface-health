# 048 · per-case AUROC/AP 的标签体系缺口

- **日期**: 2026-10-28
- **PR**: 待定 · **Commit**: 待定
- **类型**: Docs / Experiment
- **影响域**: docs/research/service-20261019/、src/service/modeling/evaluation.py、configs/service/tt_labels_a_v3_pool_dns_excluded.json

## 做了什么

核对先验 GRU 和三种微服务领域适配基线的逐 Case 评价需求。当前 v3 `main_test` 中，Normal Case 的窗口全部为负类，每个故障 Case 的窗口全部为正类，因此每个 Case 都是单类集合，per-case AUROC/AP 没有定义；现有评测代码会跳过这些 Case，`n_cases_with_both_classes=0`。

本轮保留 pooled AUROC/AP 作为可计算主结果，并把标签体系修改列为后续独立分支任务。当前领域基线的 pooled 结果和代码已完成整理，未为了得到 per-case 数字而填充单类指标或混用不同标签口径。

## 关键决策

- **不把单类 Case 的 AUROC/AP 填成 0 或 1**：这会制造没有统计定义的指标，并掩盖当前标签粒度问题。
- **后续单独修改标签体系**：要实现有意义的 per-case AUROC，需要使每个评价单元同时包含正常和异常窗口，或重新定义 Case×Service/事件级评价单元；不在本分支改写历史标签包。
- **当前结果继续报告 pooled 指标**：它们仍可在现行 Service×Window 标签协议下计算，但必须注明 pooled 与 per-case 的差异。

## 坑 / 已知问题

- Case 级全正/全负分布使 per-case AUROC/AP 失效，即使 pooled AUROC 很高也不能推出 Case 内排序能力。
- 逐 Case q99 召回可以计算，但不能替代 per-case AUROC/AP；Normal Case 也只能用于误报统计。

## 遗留 TODO

- 新分支设计并验收同时含正负窗口的标签体系，明确标签来源、Unknown 处理、时间边界和训练/校准隔离。
- 在新标签体系上重新生成逐 Case 分数和宏平均 AUROC/AP，并与当前 pooled 结果并列报告。
