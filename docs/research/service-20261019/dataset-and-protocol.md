# 当前17 Case数据与协议边界

## 版本和特征

批次 `service_20261017_004546`；plan SHA256 `d08dcbe0a0fca9ce38e242a48e526525b906409a9dadd8949b52fafe95d0f33a`。正式6 Normal＋11故障，保持4 workers、1800秒主窗和随机onset。额外27个失败attempt只在独立历史留证中，不供拟合、校准或测试。正式11个故障Case不能作为失败垃圾剔除。

以 `data/tt_service_20261017_004546/_handoff/` 的迁移清单为准；原始四模态、metadata、生命周期/控制证据、质量、12份/Case派生CSV、batch_control均保留。`batch_plan.json` 同行复制，原件路径和SHA不改写。source/destination内容hash一致不意味着新模型或标签已验收。

Service输入为 `batch_control/research_service_inputs/<case>/service15s/features.parquet`；source_mode为`raw_scrape_supplement`。补充CSV、source_manifest、`_audit_implementation`及质量文件不是可丢中间件。原文中的绝对源路径/`processed_data/<run>/...`通过path map解析；下一轮adapter须核验绑定和hash，不能直接把旧字符串当新路径。

主键 `(case_id, canonical_service_id, window_start_ms)`，25应用Service×15秒完整网格50,575行。R3共同有效41,501（82.06%），9,074 Unknown；全网格95%资源覆盖未通过。CPU为平均消耗核数，不是百分比，不能再次差分；memory_usage_ratio=usage/positive limit，不是working-set比；process_count为进程数。沿用实例/Counter身份、实际dt、30秒gap/切换语义和质量mask，不跨实例拼差分，不以查询返回时间证明源新鲜。

## 已执行的研究协议与后续冻结

另一agent已完成的诊断/基线采用：按时间排序的前三Normal拟合、第四Normal校准、第五与第六分别留出；异常Case全测试。CPU abs-z、R3 max/L2、IF100树seed42、OCSVM(nu=.1,gamma=scale)，同输入/有效ID，Normal 99%分位数、严格`>`，未事后改阈值救IF。全部60种3fit/1cal/2heldout划分仅作Normal敏感性诊断，未据此反选主划分。

本批早期采集准入报告中的first-fit/second-cal/third-heldout是当时未训练的划分建议，不是上述已执行研究协议。后续模型应显式冻结协议版本，不能混合两套分母或凭结果换划分。旧minimal_v1的单Normal 60/20/20和4 Case配置也不等同新批。

## 标签与可评价范围

主任务输出 `(case_id, service, window, anomaly_score, predicted_binary)`。每个纳入主评价的窗口另存label、label_source、证据和适用范围；Unknown排除对应评价，不当第三预测类。资源、日志、Trace同源阈值只能弱评价；人工看同一源并不自动独立。

受控注入时段/目标≠全部实际异常Service集合；非目标、baseline、recover不自动健康。仅Normal拟合/校准，不吸收Fault baseline/recover；Normal case也不证明隐藏状态全面健康。没有可靠正负类时AUROC/AP/F1应空缺并说明，不用代理任务冒充完整真实二分类性能。目标检索可以辅助报告。

CPU/DISKIO目标各31个有效核心窗，单次kill无持续完整inject_core；不补成15秒正窗。DB目标不在25应用候选内，DNS只证明控制暴露未证明业务影响。cachelimit恢复含撤除后显式MySQL写模式维护，不代表自然自愈；kill-order和CPU曾出现checkpoint passed、terminal failed，不能据工程验收称全recover健康。故障期probe只读，不具有写持久化覆盖。

## 多模态与离线限制

上轮读取17份合并Trace（5,251,409,996 bytes）、429应用日志（2,288,070,982 bytes），完整窗659,346条Entry。Trace最短查询门槛1ms，缺Entry不证明无请求/健康；Entry/Exit/Local区分，Summary不补业务Entry。start归窗为离线闭窗统计，缺真实到达时间不报在线提前量。

日志连续覆盖未证明；只用已观察正向错误，不能将静默/缺流当零错误健康。探索异常类型正则和22对observed_edges不是正式模板词典或完备因果图。Endpoint p95缺失较多，多模态增益须在相同样本及信息预算下独立验证。
