# POOL / DNS 原始观测与模型漏报审计

## 范围与结论

本次仅只读检查 selected17 中两个指定 Case、注册的相关服务日志、每 Case 唯一 merged Trace、load/api_attempts.jsonl，以及 gpu-b 上已完成的 prepared 与 seed42 评分。未重建数据、未训练、未改变标签/阈值/超参数，也未把异常数据用于参数选择。

目前证据**不支持“已经存在明显的连接池耗尽或 DNS 解析失败，只是 GRU/BERT 没学会”**。两项控制器均报告 AllInjected=True，相关业务请求也确实持续发生，但本次检查的相关日志和 Entry/Exit/Local span 没有出现明确的超时、连接池异常、解析失败或 is_error=True；干预时段的请求耗时分布也未出现与配置强度相称的跃升。因此应先核对故障路径与实际应用影响，再决定模型改进。

这不证明服务完全健康：HTTP200 不等于业务成功；未采集异常、缓存/连接复用、网络规则未命中实际流量等仍可能解释结果。本报告不把 pre/recover 自动标为健康，不据诊断删除困难事件，也不将注入关联标签冒充实际状态真值。

## 数据与复现入口

- 数据根：`data/tt_service_20261017_004546/`。
- 只从 `batch_control/final_acceptance/selected_cases.json` 选取以下两项（已确认属于 selected17），日志严格按相应 `log_data/stream_registry.json` 选择；不读取 failed27 或旧数据。
- POOL：`Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em`。
- DNS：`Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em`。
- 各 Case 的 `trace_data/case_metadata.json` 给出实验与注入边界；仅读取 trace_data 顶层唯一 `_skywalking_traces_*.json`，不混入 `_slices`。Trace 以 node_id（缺失时 trace/segment/span ID）去重，按 span end 时间归属 pre/inject/recover，与闭窗观测方向一致；包含可能跨边界的完整 span 耗时，不声称在线提前检测。
- 日志以 source_timestamp 裁剪实验区间；计数为注册 JSONL 记录，非堆栈展开后的独立事件数。
- Prepared：`~/Code/Repos/interface-health/outputs/service/model-task9-prepared-20261022-a`。
- 只读远端解析使用 `~/Code/Repos/interface-health-runs/temporal-tuning-a89bd1f` 的 `scripts.service_model.read_prepared`，未替换旧模型。
- 三个评分目录均位于上述 outputs/service 根，分别为 `temporal-multiseed-a7e7794-gru-seed42-20261023-a`、`temporal-bert-bf5fdfd-gru-seed42-20261023-a`、`temporal-bert-tuned-a89bd1f-gru-seed42-20261023-a`。
- 本轮临时诊断脚本与汇总：`/tmp/pool_dns_raw_audit.py`、`/tmp/pool_dns_attempt_audit.py`、`/tmp/pool_dns_model_audit.py`，原始审计汇总 JSON 位于同目录。它们未改变源文件；/tmp 不作为长期唯一证据。

## 控制器成功与实际应用影响分开

POOL 的实际执行对象是 NetworkChaos：对 MySQL 与 order/preserve/user 三应用之间配置 `direction=from`、`latency=8s`、`jitter=2s` 的网络延迟；不是直接将连接池容量/占用设置为耗尽。证据为 `control/fault_control.json` 的 document.spec 和 `control/fault_observation.json`。控制器记录四个相关 Pod 的 Apply Succeeded / Injected。

DNS 执行为 preserve Pod 的 DNSChaos `action=error`，仅匹配 `ts-order-service*` 与 `ts-order-other-service*`。控制器记录 Selected=True、AllInjected=True。这只说明控制动作已部署，不证明应用当时发出了新的匹配 DNS 查询。

|Case|注入起止（Unix ms）|load_status successes / failures|
|---|---|---|
|POOL|1792233297729–1792233897609|354 / 0|
|DNS|1792332973409–1792333573134|360 / 0|

load_status 计数是加载器汇总，不能当作逐请求业务语义金标。进一步检查 source=load、event=end 的 preserve API 请求，按 end_ms 分段：

|Case|时段|完成请求数|HTTP200数|耗时中位数 ms|p95 ms|
|---|---|---:|---:|---:|---:|
|POOL|pre|151|151|10300.08|13334.14|
|POOL|inject|118|118|9738.64|12188.84|
|POOL|recover|85|85|9988.12|12545.22|
|DNS|pre|170|170|10584.53|13876.01|
|DNS|inject|119|119|10691.70|13212.92|
|DNS|recover|72|72|11984.91|14038.41|

请求覆盖并非零，但没有在该请求层见到HTTP失败或明显延迟跃升。本轮没有读取/判定响应业务字段，不能把这些200确认成业务成功。

## 原始 Trace：已有数据库与调用侧细粒度，但未见强故障信号

本轮扫描 POOL 的 order/preserve/user，以及 DNS 的 preserve/order/order-other 的原始 spans；下表列出重要项。扫描范围内所有 pre/inject/recover 的 Entry/Exit/Local `is_error=True` 数均为0。

|Case / 服务 / span类别|pre数量|inject数量|pre p95 ms|inject p95 ms|
|---|---:|---:|---:|---:|
|POOL order Entry|2674|2029|3969.15|3607.40|
|POOL order MySQL Exit（全部Exit）|3316|2505|397.00|409.80|
|POOL order Hikari Local（getConnection/close合计）|5352|4059|1.00|1.00|
|POOL preserve Entry|163|118|13280.40|12183.45|
|POOL preserve Exit|1146|826|4538.75|3885.00|
|POOL user Entry|163|118|2.90|3.00|
|POOL user MySQL Exit|163|118|1.00|1.00|
|POOL user Hikari Local（getConnection/close合计）|326|236|1.00|1.00|
|DNS preserve Entry|183|119|13840.70|13206.40|
|DNS preserve Exit|1281|843|5484.00|5986.70|
|DNS order Entry|3000|2053|4781.40|4808.80|
|DNS order-other Entry|527|361|2.00|2.00|

POOL 注入期 order 的 Local 中有2030次 `HikariCP/Connection/getConnection`、2029次 close；user 各118次。order Local 最大94ms、user最大8ms，没有观察到秒级池等待。MYSQL Exit 包括 executeQuery/commit/executeUpdate，确实存在数据库访问，并非只看未触发数据库的空请求。

DNS 注入期 preserve 有119条发往 `/api/v1/orderservice/order` 的 Exit；相关 order POST Entry 也有119条，均无错误。因此“DNS故障使所有该调用无法完成”不符合本次观测。DNS缓存、地址注册发现、HTTP连接复用或规则实际未命中都只是待核查假设；现有文件不能区分它们。

## 原始日志与选择损失

针对日志原文检查 UnknownHost、Name or service not known、name resolution、NXDOMAIN、SERVFAIL、DnsNameResolver、UnresolvedAddress，以及 Hikari、连接池耗尽、连接超时、CannotGetJdbc、SQLTransient、CommunicationsException、timeout/timed out/ConnectException/SocketException/connection refused 和 ERROR/FATAL。

以下注册服务在完整实验区间的上述匹配数全部为0：POOL 的 order/preserve/user；DNS 的 preserve/order/order-other。因此没有找到可明确归因于该故障、但被BERT选样丢弃的错误文本。关键词扫描仍可能漏掉不同表达，不能作为完整语义判定。

|Case / 服务|pre日志记录|inject日志记录|recover日志记录|
|---|---:|---:|---:|
|POOL order|425|46|0|
|POOL preserve|2619|1892|1475|
|POOL user|163|118|92|
|DNS preserve|2932|1924|1205|
|DNS order|795|322|0|
|DNS order-other|1398|962|576|

order 日志明显稀疏，尽管其 Trace 持续存在。stream_registry 显示其 stream 从实验前持续注册到终止、close_reason=supervisor_stop；这不能证明每个无日志窗都具有完整应用日志，亦不能把无日志解释为服务死亡。模型对order的Normal先验本就允许日志稀疏。

在主评价的严格39个关联正窗内，prepared coverage进一步显示：

|组|日志有效窗|source/available events|selected events|未选事件|清洗排除|token排除|unmatched_count中位数|
|---|---:|---:|---:|---:|---:|---:|---:|
|POOL order|1/39|46/46|43|3|0|0|0|
|POOL preserve|39/39|1856/1856|1856|0|0|0|0|
|POOL user|39/39|115/115|115|0|0|0|0|
|DNS preserve|39/39|1879/1879|1879|0|0|0|0|

这里的正窗集合比整个注入区间严格，故事件总数小于上一表，不是记录矛盾。preserve日志典型每窗48事件、13个匹配模板，user每窗约3事件、1个模板。暂未发现日志新颖度或已知错误的明显变化。

## 特征与评分链路

`src/service/modality_preparation.py::entry_observations` 当前只保留 Entry，故 MySQL Exit、HTTP Exit 和 Hikari Local 不进入5项执行统计。这确实是表达边界，但本次查看这些被排除的 spans 后，也未发现明显异常等待或错误，不能据此预设加入它们必然改善。

数值预处理使用仅Normal fit的服务级均值/尺度，退化组按既有规则回退；时序模型进一步使用 asinh 压缩。POOL三服务、DNS preserve 的39正窗执行字段均39/39有观测，error_rate均为0。资源各字段有效窗分别30/33/29/34，缺失通过mask显式保留，不当健康0。标准化后的极值会被asinh减弱，但本次输入没有出现“极强原始故障被裁为0”的证据。

日志统计为四项 event_count、matched_template_count、max_matched_fraction、unmatched_count；BERT使用选后正文token与group mean，最多16条消息、每条128 token，且只进入过去4窗历史。当前窗语义不直接作为异常评分目标；输出仍是12项数值/日志统计与3项可用性。因此该方案本来就不能保证检出只存在于当前日志语义、但未影响统计/可用性的异常。

seed42 原服务先验GRU的q99阈值为2.22611737，严格大于才报警：

|39正窗组|总分中位数|value MSE中位数|availability BCE中位数|总分最大值|q99检出|
|---|---:|---:|---:|---:|---:|
|POOL order|0.38589|0.21233|0.07330|1.68725|0/39|
|POOL preserve|0.25226|0.19985|0.05278|0.84544|0/39|
|POOL user|0.20126|0.11416|0.08209|2.12708|0/39|
|DNS preserve|0.46112|0.35471|0.07372|1.36085|0/39|

分量中位数分别计算，其和不必等于总分中位数。两类分量均没有在这些组出现持续强残差，不能归结为“很高的BCE被数值项抵消”：评分是非负相加，不存在抵消。

|模型seed42|q99阈值|POOL order最大|POOL preserve最大|POOL user最大|DNS preserve最大|
|---|---:|---:|---:|---:|---:|
|原先验GRU|2.22612|1.68725|0.84544|2.12708|1.36085|
|默认BERT|2.33532|1.78405|1.23456|2.10930|1.29057|
|调参BERT|2.29492|1.62916|0.92067|2.12038|1.34968|

三个版本这些组均未过阈值，BERT没有生成新的明显异常信号。本次不试探降低阈值来检出这些测试事件。

具体身份示例（均来自当前prepared的关联正类）：

- POOL order：`a0d461d3c6e06bcac490811c554a6b227af97c8fdf19bca97e212774665ed3a3`。
- POOL preserve：`a705db84d4363f2db76bbb8179cd328728e58c8012d0832462553c480e94132b`。
- POOL user：`12b306ff7c99d0015b737161d70592f48a4cd8cd25a79b6333397234db49eafc`。
- DNS preserve：`2ef86ea258390580af4ac009a713d60dd31bb9bb1d454401879d212b06fd840b`。

## 建议下一步

1. **先验证机制，后增加模型容量。** 新采集可加入独立DNS请求成功率/解析耗时、应用实际使用地址来源、连接复用信息；POOL加入连接池active/idle/pending及真实getConnection等待。对受控请求同时保留业务响应字段与依赖调用关联，避免只以控制器AllInjected证明应用失败。
2. **复查部署路径。** 确认NetworkChaos命中的实际MySQL流量方向、端口与Pod；确认应用调用是否确实触发匹配DNS查询。当前只读资产不能证明这些根因，不应写成已确认的故障注入无效。
3. **现有数据保留原主集。** 当前POOL/DNS按注入关联任务继续如实报告漏报与观测边界；若形成新的影响标签，应另立有证据的版本，不能为提高分数直接移除它们或改为正常。
4. **若后续确认Exit/Local有可区分信号，再做通用特征实验。** 对全部服务统一构建依赖调用耗时/错误与池等待统计，让IF/OCSVM获得相同输入；由Normal训练/验证确定参数，Fault仅用于最终诊断。
5. **BERT暂不扩大搜索。** 在本次目标日志里未找到明确错误语义且选择丢失很少，继续BERT调参缺乏证据。若后续发现仅语义可见的异常，再考虑独立当前窗语义评分，并对照模板新颖度与真实/打乱/零语义消融。

本轮结果支持优先调查故障可观测性与干预路径，不支持承诺通过换模型即可攻克这两类事件。
