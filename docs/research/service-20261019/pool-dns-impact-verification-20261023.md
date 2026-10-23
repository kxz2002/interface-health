# POOL / DNS 干预路径与业务影响复核

## 结论

本轮只读复核进一步排除了两个简单解释：POOL 并没有因为 `tsdb-mysql` 与 `tsdb-mysql-leader` 名称不同而明显选错数据库；DNS 的应用逻辑 URL 也确实使用了规则中的 `ts-order-service` 名称。**这些是路径一致性证据，不是数据包实际被干预的证据。**

两 Case 的已记录请求均有完成记录，注入期间开始的请求也全部返回 HTTP200，没有发现前次 end-only 统计漏掉的已记录超时/悬挂请求。归档加载器源码还显示 booking success 判断包含业务 `status==1`，因此已有 successes 汇总比单纯 HTTP200 更有意义；但是没有逐请求响应正文，而且归档源码不是每 Case 执行版本的强证明，仍不能据此确认所有业务状态健康、故障未生效或可以改标签。

目前未能确认：8秒延迟为何未体现在目标数据库调用耗时中、DNS 查询是否经过故障代理、连接是否复用、是否由服务发现组件将逻辑主机名替换为地址。继续加模型容量前，应在未来采集中直接验证这些机制。

## 范围与方法

仅使用 `data/tt_service_20261017_004546/` 内 selected17 的以下两 Case：

- POOL：`Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em`。
- DNS：`Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em`。

读取它们的 `control/fault_control.json`、`control/runtime_snapshot.json`、`control/load_status.json`、`chaos_executed.yaml`、`trace_data/case_metadata.json`、唯一顶层 merged Trace、`load/api_attempts.jsonl`、`load/load_events.jsonl` 和 `api_responses/api_responses.jsonl`。没有读取其他 Case 原始日志；前一轮日志结论见 `pool-dns-observability-audit-20261023.md`。

关联代码证据来自 `batch_control/collection_evidence_20261018T150158Z/` 的 manifest 注册归档，不使用当前 AnoMod 工作区代替采集时源码。涉及三个文件的 SHA256 均与该 manifest 一致：

|归档 `collection_sources/TT_collection-scripts/train-ticket-auto-query/` 文件|SHA256|
|---|---|
|`atomic_queries.py`|`8b28dd206af99620cc700e3db518600265fbfeee38b60d45680cf6acc4b2967e`|
|`test_all_services.py`|`6604f376a0f879dbfbe94efceb03aaa33bccccd60ca280974e62a9845d033b21`|
|`tt_api_recorder.py`|`2c79a9b8a55a0a37ec5d510e692555ea95c046528408b0380a61ba6db26b86b8`|

manifest 自述为 `partial_batch_evidence_archived`、`Latest repaired code ... not proof all cases used identical source revision`。该限制适用于下文所有源码语义推断。

临时诊断脚本 `/tmp/pool_dns_impact.py`、结果 `/tmp/pool_dns_impact.json`，运行方式为仓库根目录下 `conda run --no-capture-output -n interface python /tmp/pool_dns_impact.py`。逐行读取 attempts，按请求 `attempt_id` 配对 begin/end，并核对同 `(attempt_id,event)` 无重复；以 `start_ms` 归属注入区间，再检查 outcome、http_status、exception_type。流式读取 merged Trace 的 `traces` 数组，按 node_id 去重，以 span end 归属注入区间，统计指定服务 Exit 的 peer。上述 /tmp 文件不作为长期唯一证据；所有原始来源与筛选方法已在本报告列明。

本轮所有证据本机已完整存在，未连接 gpu-b，没有训练、安装、重建、修改原始数据、模型、标签或阈值。

## POOL：目标地址一致，但数据包干预效果仍不明

执行 spec 是 NetworkChaos `action=delay`、`latency=8s`、`jitter=2s`、`direction=from`，selector 为 `app=tsdb-mysql`；target 的应用为 order/preserve/user，没有配置端口过滤。它以网络延迟模拟连接池压力，没有直接修改连接池容量或占用。

`fault_control.json` 记录数据库目标为 `tsdb-mysql-0`、Pod IP `10.42.0.4`；关联应用 Pod/IP 是 order `10.42.0.45`、preserve `10.42.0.44`、user `10.42.0.30`。`runtime_snapshot.json.identity.endpoint_slices` 中 `service=tsdb-mysql-leader` 的唯一 ready endpoint 同样为 `tsdb-mysql-0` / `10.42.0.4`，Pod UID 也一致。

注入区间按 end 统计的数据库 Exit 为：order 2505 条、user 118 条，peer 均为 `tsdb-mysql-leader:3306`；preserve 的 826 条 Exit 为到 order/user/security/contacts/travel/basic/seat 的 HTTP 调用，未见直接数据库 Exit。这说明 preserve 的数据库影响链路在已有 Trace 中经过其他应用，不能把 preserve 当作已有直接 SQL 流量的证明。

因此，**服务发现快照、数据库逻辑 peer 与目标 Pod 是相容的**，没有发现“应用实际访问另一台数据库”的证据。Trace peer 不是逐连接最终 IP，EndpointSlice 也是快照；现有资产缺少注入期数据包、socket 目标、tc 规则命中计数及延迟探针，不能证明目标包确实经过有效延迟。配置的 `from` 值本身也不能替代实际流向与规则命中验证。

前轮已观察到 SQL Exit 与 Hikari Local 无明显池等待跃升。本轮没有找到能解释该现象的确定根因，也没有独立证明发生了连接池耗尽。

## DNS：逻辑名称相符，解析链路无法回溯

DNSChaos 对 preserve Pod 生效，patterns 为 `ts-order-service*`、`ts-order-other-service*`。控制器的记录与 preserve Pod 身份相关联。

注入区间 preserve→order 的119条 Exit 的 peer 为 `ts-order-service:80`，URL tag 为 `http://ts-order-service/api/v1/orderservice/order`，component 为 `SpringRestTemplate`。至少逻辑请求名称与第一个 pattern 前缀一致，没有证据支持“应用写了完全不同的服务名”。该区间没有观察到 preserve→order-other Exit；其他服务到 order-other 的请求不能证明 preserve 触发了第二个 pattern。

但 URL/peer 是追踪记录的逻辑名称，不记录当时是否发出 DNS 查询或最后连接的地址。runtime snapshot 记录 Kubernetes 服务与 Nacos 工作负载存在，并不能证明这个 RestTemplate 使用了何种解析、负载均衡或缓存策略。此次 manifest 注册的代码归档是采集器/加载器代码，没有与这次应用镜像绑定的 preserve Java 源码、JVM DNS TTL、HTTP 连接池配置、DNS查询日志或 socket 复用事件。

故以下仍只是候选解释：已有 DNS 缓存、HTTP 连接复用、服务发现组件替换地址、DNS 代理规则未命中。现有资产不能在它们之间作判定；也不能由119次成功调用反推从未有任何 DNS 干预。

## 客户端：补查 begin/end 与业务响应证据

|检查项|POOL|DNS|
|---|---:|---:|
|全部已记录 begin|3588|3649|
|全部已记录 end|3588|3649|
|begin 缺少匹配 end|0|0|
|end 缺少匹配 begin|0|0|
|重复 `(attempt_id,event)`|0|0|
|end outcome=response、exception_type=null|3588|3649|
|注入期开始的请求|1198|1206|
|其中 HTTP200|1198|1206|
|其中 preserve URL 的 HTTP200|120|120|
|skipped 事件|4|3|

这里“全部”涵盖记录器完整文件，包括准备和停止附近的请求；与仅裁实验区间的前轮汇总口径不同。注入期采用 metadata 的 `[inject_start_ms, inject_end_ms)`。preserve 的 start-based 120次与前轮 end-based 118/119次不同，是归属边界变化，不能直接当重复或丢失。

skipped 没有 begin，是受抑制的未启动请求，不能当成一次网络超时或已完成请求。snapshot 代码在请求被抑制时退出 load loop；iterations 与 successes 也有差值（POOL 358 对354，DNS363对360），不能直接把该差值计为业务失败。没有证据表明已记录、实际启动的请求被 end-only 统计隐藏为超时；记录器外未发出的需求、未被代码记录的失败或其他用户流量不在该结论范围内。

`api_responses.jsonl` 的全部字段只有 timestamp、method、url、status_code、latency_ms、content_length、body_hash，不含响应正文或业务 status/message；`api_attempts.jsonl` 同样没有业务字段。无法从 hash/长度恢复“HTTP200内是否失败”。业务响应记录数（POOL3564、DNS3625）也少于 attempts end，两者不能按行号硬配对。

但加载器源代码提供了更强的间接语义：

- `atomic_queries.py::_preserve` 调用 `preserve_result`，只有 `http_status == 200 and business_code == 1` 才返回 True；business_code 取响应 JSON 的 `status`。
- `test_all_services.py::single_user_loop` 用这个 booked 布尔值累计 successes/failures，异常也计入 failures。
- 两 Case 最终 load_status 的 successes/failures 分别为354/0、360/0。

**若这份归档实现与相应 Case 实际执行一致，successes 就支持这些完成的 booking 被应用业务码接受，而非仅传输层成功。** 不过这是整个 Case 汇总，不能定位每个注入窗；没有逐请求业务字段、支付/订单持久化后置核验，且源码版本对应有上述 manifest 限制，不能升级为完整业务金标或把所有服务标正常。

## 已确认与尚待证实

|等级|事项|
|---|---|
|已确认的配置/路径事实|POOL是网络延迟代理机制；MySQL leader EndpointSlice 指向已选中的同一数据库 Pod；order/user存在对应 SQL Exit。|
|已确认的调用事实|DNS注入区间 preserve 仍调用逻辑名称 ts-order-service；没有记录到来自 preserve 的 order-other Exit。|
|已确认的记录完整性|已记录 begin/end 全配对；注入期开始的请求无已记录传输异常，均HTTP200。|
|有版本限制的业务证据|归档加载器将status==1视为booking success，Case级success汇总无失败。|
|尚未确认的影响|真实网络延迟命中、连接池耗尽、DNS解析失败及其具体业务后果。|
|不能判定的原因|缓存、连接复用、地址发现/替换、代理规则路径、无记录的业务错误。|

## 后续采集应验证什么

1. POOL 同时保存应用 socket 远端 IP/端口、Pod 对应关系、tc 规则和命中计数，并用相同应用网络命名空间内的依赖探针验证延迟；另采 Hikari active/idle/pending、获取连接等待。仅通过修改连接池参数时，应将机制与 NetworkChaos 区分记录。
2. DNS 保存匹配查询及响应、应用解析后地址、JVM DNS 缓存策略和连接复用信息；对同一业务路径验证“新解析”和“复用连接”两种条件。Kubernetes Service 存在或 controller AllInjected 不是该验证的替代品。
3. 客户端以稳定请求ID保存脱敏业务status/message、请求开始/结束/取消及订单接受后的独立结果，避免只有hash和总success。保留响应缺失与Unknown，不将其填为健康。
4. 当前两 Case 继续保留并报告原有评价边界；本审计不授权修改标签、删除困难事件、降低阈值或按测试故障调参。

## 已对齐的后续两条路线

现有数据的只读核查暂告一段落，不重复开展相同扫描。以下为后续探索记录，不表示已经创建 worktree、实施采集或启动实验。

|路线|工作范围|可验收交付|建议分支|
|---|---|---|---|
|当前模型与论文推进（优先）|保留原先验 GRU 与可选 BERT；开展服务先验、时序和评分分支必要消融；保留 POOL/DNS 漏报及标签限制|同协议的消融配置、逐样本结果、可复现摘要和有证据边界的论文结论|`exp/service-prior-ablation`|
|POOL/DNS 受控复现|先验证网络规则命中和 DNS 查询路径，再补采连接池等待、解析响应、连接复用及业务响应证据；确认实际影响后再研究检测特征|机制验证记录、数据 manifest、独立影响证据及版本化标签；若形成新特征，给模型与经典基线相同输入|`exp/pool-dns-impact`|

待 PR #37 合入后，两条路线从同一个合入 commit 分别建立独立 worktree 和分支，再按各自计划启动。使用不同 run_id、输出及跟踪路径；原始数据只读，新采集另建数据版本，不覆盖既有 prepared、checkpoint 或分数。不以两个 Case 的测试效果选择超参数或阈值；已见测试集后的探索须明确披露。BERT 保留为可选消融，目前不扩大其参数搜索。
