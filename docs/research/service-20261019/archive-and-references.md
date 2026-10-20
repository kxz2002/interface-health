# 研究档案与来源

完整本地备份：`/home/kxz2002/Code/research-backups/20261019/service-closeout-service_20261017_004546/`。源件均复制保留，未移动、删改。逐文件去向/源与目的SHA见 `research_archive_manifest.json`；交接包原件和自身manifest位于 `handoff/`。归属不明 `.RESULTS.md.swp` 未读、未复制、未删除。

|归档相对位置|用途|
|---|---|
|research_sources/service-full-revalidation-20261019-CpImzO/|全17项研究：协议、snapshot/source清单、RESULTS、原始分数、Parquet、4张图、复现脚本和测试|
|research_sources/service-baseline-independent-check-20261019-s8vhy7/|独立裁决、逐案例检查、见证资格、重拟合基线、测试输出|
|research_sources/service-baseline-probe-20261018-VnITUY/|前序probe函数/import、旧抽样与参数依赖，不冒充本轮新增|
|research_sources/service-innovation-study-20261018/|历史方案/文献及round2；原No-Go由当前目标覆盖，实验原件不改|
|handoff/|用户提供的收尾包，含逐文件取舍和失败尝试核查|
|failed_attempts/|failed27诊断/控制/Unknown证据，不进入模型根目录|
|dataset_copy_manifest.jsonl、dataset_copy_summary.json|全部正式数据及失败证据复制校验，原相对路径和去向|
|reference_resolution.json、path_map.json|引用分层与原路径解析；外部/tmp来源登记为仅引用，不冒称全部可迁移闭包|
|closeout_state/|开始时两仓库状态、当前测试及最终验证记录|

原样脚本含绝对/tmp import，保留源目录名以便溯源，**归档不等于自包含或生产入口**。后续先按接口提炼纯函数并测试，不直接复制旧脚本进训练流程。研究文献库保留18篇PDF，51条索引不等于51篇全文精读；本收尾未重新联网查新，不增加论文发表/新颖性结论。

直接前作及阅读深度见归档full_revalidation/literature_update.md与previous_literature/round2原台账。可参考ServiceAnomaly（JSS2024）、Eadro（ICSE2023）、AnoFusion（KDD2023）、ART（ASE2024）等，正式论文引用应回查原件的DOI/作者来源和阅读级别；后续新方法不能把引用台账当已复现前作。

采集验收还参照 `~/Code/research-backups/20261015/research-assets-backup-20261015.tar.gz`，SHA256 `6bbf53783bfdf5ccc348da8ff6a5b3e1769055ce2351ca06a1ef36686dc7e9bc`；旧高新颖性门槛不覆盖2026-10-19用户目标。PDF、分数、Parquet及大manifest不随普通源码提交。

引用核对结果：1941条原文件路径及18条目录引用可映射到复制件；另96条历史外部路径本轮均存在，仅登记原路径，未复制其他agent的/tmp目录。正式17项的必要原始模态、补充源和实现hash绑定均通过、无未解析必需输入；这不等于全部历史脚本/引用已可脱离原机运行。原路径和worktree继续保留。
