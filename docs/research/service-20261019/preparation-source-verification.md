# 数据准备来源核验（Task 1，2026-10-20）

本轮只读核验来源与现有资源网格，未改生产代码、raw或既有产物，未训练。可重算脚本与小证据在 `.superpowers/sdd/training-preparation-plan/verify_sources.py`、`task-1-source-evidence.json`。以下“通过”仅限列明检查；不是五表验收或训练资格确认。

## TT 来源身份与资源网格

selected_cases.json SHA256 为 `17199227f7fd6419ae72f8268c4db4025bc8f9610f5ed24cb1ad69f6d201ff46`。本轮复算核对 selected17、migration.selected_ids、path_map.selected_case_ids 完全相同；run_id、plan_hash一致；migration.verification.status=passed。17项为6 Normal、11故障，失败27未加载。

逐Case实际复核 metadata、source_manifest、research_input_quality 的 case/run/plan/plan_item/attempt身份，metadata与features SHA256、补充CSV实际SHA256及quality绑定、quality引用的历史实现快照SHA256。路径先按 `path_map.exact_files` 精确映射，否则仅移除明示 `relative_batch_prefix`；映射结果必须仍在正式数据根内。未修改原绝对路径，也未把当前源码hash冒充历史实现hash。没有重扫TT大Trace/Log或全部迁移文件。

17份features实际读取核对：每Case 25固定候选、119个连续15秒窗、2,975行；主键无重复、case_id一致；服务集合与quality.candidate_services一致，总计50,575行。各Case实际首末窗、metadata边界、服务清单和来源hash均在JSON。保留既有窗边界，不从metadata重新发明对齐规则；资源值不差分、不重算、不归一化。

## TT 执行和日志给Task 2的入口

只从allowlist的case_root进入。原始Trace在 `<case>/trace_data/`，小证据记录每Case顶层 `*skywalking_traces*.json` 候选；`_slices` 可能是同一采集的切片，不能与总文件叠加计数。最终输入应结合 `_handoff/dataset_copy_manifest.jsonl`、trace来源记录选一条确定链路，保留记录身份；本轮没有读取大Trace正文确认所有schema。

日志入口是 `<case>/log_data/stream_registry.json`。登记包含case/run/attempt/plan_item、namespace、pod_name/pod_uid、container_name/container_id、segment、opened/closed时间及log_file/stderr_file。`log_file`保留历史绝对地址，可经path_map精确映射，禁止直接打开历史工作树。相邻同stem `.jsonl` 保存保真采集记录，可用作消息来源；`.log`与`.jsonl`不能重复计数。原始日志流中还有基础设施，按已固定25应用候选筛选，保留实例身份。

本轮仅抽查首个Normal的一个应用流首行：JSONL含 `source_timestamp`（`2026-10-17T10:08:19.188484893+08:00`）、`received_ms`、`raw_line`、`partial`、`segment`、`replay`、`source_occurrence`；原文前缀明确含UTC偏移。业务消息内另有无偏移时间，优先使用采集包装source_timestamp；不能以这一个样本宣称全部记录都有效。Task 2应逐行保留原文引用、处理continuation/partial与replay，按实际source时间裁窗，缺时间保留质量原因，不默认健康。

归档根 `research-backups/20261019/service-closeout-service_20261017_004546/research_sources/service-full-revalidation-20261019-CpImzO/` 有 `trace_sources.json`、`log_sources.json`、`entries/`、`trace_service_windows.parquet`、`log_level_windows.parquet`、`entry_error_records.parquet`。`full_core.py:77`使用start_timestamp_ms；旧执行窗不能直接满足本Spec按结束时间归窗。`logs_full.py`输出按level计数和exception类别，未找到完整原文消息窗表。旧文件可用于来源/数量回查，不能冒充已完成新五表；Task 2应先查entries是否保留duration/end，再决定复用或定向解析raw。

## SN 版本、服务及运行

[Zenodo固定版本7615394](https://zenodo.org/records/7615394)直接列出SN包MD5 `4c523faf64770b1e9a709cdd610e07c4`；本轮对 `/tmp/SN-Dataset.zip` 实际复算相同。既有详细审计 `/tmp/service-cross-dataset-validation/detailed/AUDIT.json` 提供嵌套成员hash；本轮未重复全包重hash。

输入仅选一个 `SN-all.tgz` 的4个故障运行，加 `no fault/` 下3个tar.xz运行；不同时加入外层展开的同批data。七运行及其metadata、container路径、三模态服务集合已导出JSON。所有运行均覆盖相同12个应用服务名称；[官方util.py](https://raw.githubusercontent.com/BEbillionaireUSD/Eadro/main/codes/preprocess/util.py)定义同一12服务及7个资源字段。采用这些明确服务名作canonical ID，不因单模态缺失删服务。

|运行后缀（均前缀SN.2022-04-）|封装角色|
|---|---|
|17T181245D2022-04-17T183616|fault archive|
|17T183729D2022-04-17T190100|fault archive|
|17T190213D2022-04-17T192544|fault archive|
|17T192658D2022-04-17T195031|fault archive|
|20T182405D2022-04-20T184806|no-fault候选|
|21T105249D2022-04-21T111651|no-fault候选|
|21T153302D2022-04-21T155703|no-fault候选|

## SN 裁窗结果与阻塞

既有inventory报告Normal两对共享Trace ID为167和77，但只检查每Trace最早Span；该摘要不足以判断裁窗后的重复。本轮流式读取三个Normal的全部Span，按不加偏移的 `metadata.start <= (startTime+duration)/1e6 < metadata.end` 逐Span筛选，三个运行分别读取272,696、288,922、254,222条Span，裁后均为0。精确min/max和metadata范围见JSON。空交集不能证明独立运行或有效拆分。

例：20日Normal metadata范围为 `[1650421445.66543,1650422886.9702)`；Span结束数值范围为 `[1650450245578858,1650451685871106]`。数值不对齐是实际观察，不能因此猜一个8小时修正。其余两个Normal含历史Span范围，也不能仅凭全包min/max相交认定运行窗内有Span。

目前阻塞如下，依赖步骤必须保留Unknown及training_eligible=false：

- SN日志时区没有采集环境证据，不猜UTC/UTC+8，不据目录名转换。
- SN资源7字段的物理单位及gauge/counter/预先rate语义未确认；官方util只列名称，[single_process.py](https://raw.githubusercontent.com/BEbillionaireUSD/Eadro/main/codes/preprocess/single_process.py)消费已解析数据，不是来源采集器。不能从字段名推断纳秒、字节或CPU百分比，不二次差分CPU。
- [Jaeger固定版本JSON模型](https://raw.githubusercontent.com/jaegertracing/jaeger/v1.60.0/model/json/model.go)明确startTime为Unix epoch微秒、duration为微秒；该格式解释用于上述诊断，但SN源部署时钟与metadata一致性未证实；正式跨模态聚合被0裁窗结果阻断。
- SN缺少已证实服务端入口识别规则；未解决只能称Span统计，错误tag数不能相加当错误Span数。
- no-fault仅封装角色；历史错误tag、历史Span以及独立健康资格尚未解释，不能生成全0标签或自动进入fit/cal池。

本轮没有采纳第三方补齐脚本的时区假设。Task 2可继续TT适配与共享schema；SN只交结构及明确缺失原因，待来源证据解决后再聚合。
