# TT A标签基础设施：实现与生产验收

2026-10-20，Task 1–5 已实现并在真实 selected17 上生成独立侧表。未训练、选择阈值、重建特征或改写原始资产；旧 preparation 全 Unknown gate 保持不变。

## 生产结果

50,575 = 26,275负类 + 194正类 + 24,106 Unknown。fit负类10,600，calibration负类2,975，test负类12,700；test正类194。fit另有1,300 Unknown。全网格资源有效41,501，正类资源有效171；这是资源资格，不是三模态共同有效数。资源有效负类fit8,730、calibration2,461。

**生产证据更正**：设计探查的故障前10,600应更正为9,725。TRANSACTION的probe暂停策略不能确认，boundary.reasons=[unknown_pause_policy]，first_intervention_requested_ms=null；不能由缺暂停记录推断不要求暂停。该Case全2,975行Unknown，原参考875个故障前负类撤销。因此原27,150负类/23,231 Unknown/test负13,575均为探查参考，不能继续作为生产计数。规则未因计数调整。

最早Normal保留54条传输超时事件、合并31个区间，覆盖52个15秒时间窗×25服务=1,300行负类排除；事件数不是窗口数。例外清单还保留故障期间失败请求，它们不倒推故障前异常。多个Case的load_failures_unlocalized仅作质量警示；最后Normal两次失败仍不能由HTTP200解释为业务健康。完整警示、逐服务/原因/模态覆盖见本地coverage/evidence，缺执行/日志_valid列不等于这些模态全部有效。

|Case ID|负类|正类|Unknown|
|---|---:|---:|---:|
|Lv_D_CONNECTION_POOL_exhaustion_20261017T102058Z_7b28ab23f0694e7f_em|1150|0|1825|
|Lv_D_TRANSACTION_timeout_20261017T042021Z_3fe5039544954db2_em|0|0|2975|
|Lv_D_cachelimit_20261018T121247Z_7037814c5af743d4_em|950|0|2025|
|Lv_P_CPU_preserve_20261017T154808Z_021eaa51bbcb4b60_em|1175|39|1761|
|Lv_P_DISKIO_preserve_20261018T013644Z_707c577ddacf4703_em|500|38|2437|
|Lv_P_NETLOSS_preserve_20261017T145542Z_8a5e5b4c67c64664_em|1050|39|1886|
|Lv_S_DNSFAIL_preserve_no_order_20261018T140118Z_81f642e966444fe0_em|1250|39|1686|
|Lv_S_HTTPABORT_preserve_20261017T072058Z_daa34643ee434ee4_em|650|39|2286|
|Lv_S_KILLPOD_gateway_20261017T093533Z_94814e0f78844917_em|1350|0|1625|
|Lv_S_KILLPOD_order_20261018T144636Z_74933cf4bb0d48cc_em|550|0|2425|
|Lv_S_KILLPOD_preserve_20261018T022220Z_aa928eae7af646e3_em|1100|0|1875|
|Normal_20261017T020706Z_95064d21e8204018|1675|0|1300|
|Normal_20261017T110632Z_4c14d1c71d664ca9|2975|0|0|
|Normal_20261017T115148Z_cc9a6da5e99a4263|2975|0|0|
|Normal_20261018T030819Z_9adec05d73684fb6|2975|0|0|
|Normal_20261018T035358Z_4496c85fde8e4f6e|2975|0|0|
|Normal_20261018T153211Z_e448a74ed7104ebd|2975|0|0|

正类仅五个preserve事件：CPU39、NETLOSS39、DISKIO38、DNSFAIL39、HTTPABORT39。三个KILL（gateway/order/preserve）是瞬时UID删除，不推断持续不可用区间；三个数据库事件（CONNECTION_POOL/TRANSACTION/cachelimit）目标在应用候选集外，不传播标签。六事件全部保留覆盖，不作检出判断。194窗不是194次独立重复，不支持跨服务检测性能结论。

## 可重跑入口与实际验收

使用conda interface执行以下命令，均退出0；指定85项测试通过（5.13秒）。build两次均使用新目录，旧准备产物保持只读。check重新校验来源hash与逐行规则，未发现输入hash不符。diff确认标签/划分语义变化0。

```bash
conda run -n interface python -m pytest tests/test_service_labeling_evidence.py tests/test_service_labeling_rules.py tests/test_service_labeling_bundle.py tests/test_service_labeling_acceptance.py tests/test_service_label_splits.py tests/test_service_preparation_acceptance.py -q
conda run -n interface python scripts/label_service_data.py build --tt-root data/tt_service_20261017_004546 --preparation outputs/service/preparation-exploratory-20261020-task4 --protocol configs/service/tt_labels_a_v1.json --output outputs/service/tt-labels-a-v1-20261020
conda run -n interface python scripts/label_service_data.py check --bundle outputs/service/tt-labels-a-v1-20261020/labels --preparation outputs/service/preparation-exploratory-20261020-task4
conda run -n interface python scripts/label_service_data.py build --tt-root data/tt_service_20261017_004546 --preparation outputs/service/preparation-exploratory-20261020-task4 --protocol configs/service/tt_labels_a_v1.json --output outputs/service/tt-labels-a-v1-20261020-replay
conda run -n interface python scripts/label_service_data.py diff --old outputs/service/tt-labels-a-v1-20261020/labels --new outputs/service/tt-labels-a-v1-20261020-replay/labels --output outputs/service/tt-labels-a-v1-20261020-diff
```

## 绑定与交接

本地包路径为`outputs/service/tt-labels-a-v1-20261020/labels/`；含protocol/evidence/exceptions/coverage/manifest JSON和labels/splits Parquet。resolved配置即protocol.json；manifest绑定输入、文件、语义hash、代码及环境。大产物不提交，远端读者不能假定拥有这些本地资产。

生成代码commit：`c6e07e3a1e12882ee4ed3192fcf2cfebc5564d84`，dirty=true（含用户预存笔记及未跟踪文档）。环境：`{"numpy": "2.1.2", "pandas": "2.2.3", "pyarrow": "17.0.0", "python": "3.12.13", "scikit-learn": "1.5.2"}`。

- label_version：`label:8eea54d927f53e18a6abc31c27ab472f762f8aab9630e3fd3b7f17e675899094`
- split_version：`tt_normal_case_4_1_1_v1:fdd92e3889854493eff321ae8cdc6584cac5ce8a9b1f8586fce89f5c9852fcd3`
- preparation_manifest_sha256：`88da792dceb2f5a5f46486459ec387b8858c8e46d3f7124911fa48c0a755be43`
- protocol_sha256：`4554297ff856ef1a113259d4e46730c8d7b1ce304652ad1235fe8ac92f83a09b`

语义hash：

```json
{
  "evidence": "8045b68c37e2afb5283dd80fb270cfed4546f91fc640ab61f4186911c2aa7342",
  "grid": "c9cd72f02970a7257ae39d6e3829053987658b2cdc38f32df8ece77d86462587",
  "labels": "d9ece2e825e23ffeb6c5eff5b2ae3d62718268b630914cd3e909da048ea0972f",
  "protocol": "569310d0fa5958a4e189b9ded02a04ac1c40854b162b4e7ccaebde907250313a",
  "rules": "d490ad4eeb457c4504547f873e4360a55d0c2fe4ec29056056ef0096fbeb5fa8",
  "splits": "f7800d334b2e56f1bbf32c999fc0a71ab9a91a5cbaf026ceec70e15414640485"
}
```

后续消费须绑定版本、排除Unknown并显式使用模态mask。阈值与模型均未确定；KILL持续证据、数据库传播及B规则仍待单独授权。最终审查可补公共loader合成fixture与完整六ID划分断言，当前不据此扩展Task5代码范围。
