# 数据准备实现与已执行检查

## 2026-10-20 本轮完成状态

最新独立验收入口为 `outputs/service/preparation-exploratory-20261020-task4-acceptance-final/quality_report.json`（退出0）。最后修复只更新验收器的生产源码hash继承顺序：优先 `production_source_sha256`，兼容旧 `source_sha256`，再回退生产summary的 `implementation_sha256`；三处均缺失时明确拒绝，不使用validation hash冒充生产hash。新格式首次验收后再复验、旧格式及summary回退的小型链路回归先红2项/通过1项，修复后相关验收与划分测试23项通过（1.69s）。未重扫raw或重跑生产。

最终validator SHA256为 `813043209cce742260874e6d3f0209a1b35bcaeb1d6e1f9ccdccc42256883f36`；生产validator仍为 `2bbb24b2082934457a1d42bd66e86ede4af71daf0dc7b7d04b7b0fd3c02ce2dc`，fix1 validator `ba7d0b985bb8889e0ccffa4866d050d66802103543275e372e01c4d10bc332bb`保留为历史。最终quality_report SHA256为 `be1de68a105bfaa6fc2ec555f8cb28dc35292f146b2bef02a4e62f42b71628be`，与fix1相同，结论和数量未变。最终manifest SHA256为 `fd6b61b088de1eac8c261191734943690b1e6ecfbaf8799230ac30067d756357`。修复快照、差异、命令与输出见 `.superpowers/sdd/training-preparation-plan/final-fix-report.md`；旧证据保持原样。

本轮仅完成输入准备与验收，没有训练、编码、模板挖掘或归一化拟合。生产目录为 `outputs/service/preparation-exploratory-20261020-task4/`，首次执行退出0，未覆盖首版输出或修改raw。TT只含selected17；SN只含固定七运行、12Service及四个精确封装。

|数据集|完整Service窗|标签Unknown|fit参考池（模型mask前）|cal参考池（模型mask前）|
|---|---:|---:|---:|---:|
|TT|50,575|50,575|8,925|2,975|
|SN exploratory|7,908|7,908|1,140|1,140|

TT资源有效41,501行、无效9,074行；执行有观测37,306行、无观测13,269行；日志有事件32,010行、无事件18,564行，另1行采集区间仅部分覆盖。SN资源全7,908行有raw观测（单位/物理语义unknown），执行7,096行有Span且错误字段部分已知、812行无观测；日志2,495行有事件、5,413行未观察到事件。无观测保留Unknown/NaN及质量状态，不补健康0。上述观测覆盖不是健康覆盖，也不是最终多模态训练有效数。

TT输入准备验收通过，后续模型必须显式选择模态mask。SN仅在用户批准的exploratory定义下完成输入准备：日志/Span固定减28,800秒是公开数值对齐假设，不是时区已证；资源raw序列及数学均值不冒称CPU率/物理量；执行统计对象是Span，不是请求。三个无声明注入运行提供参考分布，不是健康金标。严格profile未知语义阻断仍保留。

全部58,483个评价标签仍Unknown，独立S1 AUROC/AP不可计算。Fault只test；TT整Case fit/cal/holdout及SN固定整运行划分保持隔离。参考池许可与质量mask分开，没有实施模型Loader或默认多模态有效性规则。

## 2026-10-20 验收修复与只读复验（fix1）

最终审查发现旧验收器未独立核对运行角色：若Fault整运行错误标fit，原检查可被绕过；实际生产表经审查未发现污染。现从已绑定inventory独立推导TT六Normal固定角色，并按原resolved配置hash绑定的冻结SN profile推导SN角色，逐运行精确核对run_role、reference_eligible、training_eligible和calibration_eligible。六表schema全部必需，不能缺失后静默跳过。

新增污染fixture先失败后修复；最终仅复查相关acceptance与labels/splits测试20项通过（0.96s），black/isort及diff检查通过。没有重扫raw、重跑生产parser、改变已有表或原32文件/证据。修复后的只读复验退出0，数量、质量、Unknown和参考池结论不变；新目录 `outputs/service/preparation-exploratory-20261020-task4-acceptance-fix1/` 的quality_report为更新gate后的验收依据，仍只是输入准备就绪，模型mask与真实状态语义未证明。

```bash
conda run -n interface python -m src.service.preparation_acceptance \
  outputs/service/preparation-exploratory-20261020-task4 \
  --manual-evidence outputs/service/preparation-exploratory-20261020-task4/manual_checks.json \
  --report-directory outputs/service/preparation-exploratory-20261020-task4-acceptance-fix1
```

复验目录拒绝覆盖。新manifest明确保存旧production_source_sha256、新validation_source_sha256、原12个schema hash、原manifest hash和复验报告hash。旧validator SHA256为 `2bbb24b2082934457a1d42bd66e86ede4af71daf0dc7b7d04b7b0fd3c02ce2dc`，本轮validator为 `ba7d0b985bb8889e0ccffa4866d050d66802103543275e372e01c4d10bc332bb`；旧manifest是旧源码验收的历史事实，不再声称其旧hash等于当前源码。所有生产parser源码未改变，原32产物及12schema的hash重新核对一致。

## 当前运行方法

必须使用不存在的新目录；默认strict保留来源语义阻断。复现本轮探索输入定义：

```bash
conda run --no-capture-output -n interface python scripts/prepare_service_data.py \
  --tt-root data/tt_service_20261017_004546 \
  --sn-archive /tmp/SN-Dataset.zip \
  --sn-inventory outputs/service/preparation-v1/inventory.json \
  --sn-inventory-sha256 6321ba07f1ac1fcd202bb06e5de9b4fa7295c8ce337a5ba9d8cacd962c42e4a0 \
  --sn-profile exploratory \
  --sn-profile-config configs/service/sn_exploratory_v1.json \
  --output outputs/service/<new-run-id>
```

入口先绑定selected17迁移、资源质量与历史源码hash，并核对资源完整窗与metadata范围及expected数量；TT资源原值不重算。TT原merged Trace按对象流式解析、Entry按结束时间归半开窗，JSONL逐注册流解析、合并多行并保留非replay参与行引用；replay仅计诊断，不进入消息频次。SN封装流式访问、Span逐对象解析，单个日志/CSV成员缓冲；不解压到raw、不读取重复外层data。生产前可用内存约20GiB、磁盘约511GiB；抽查进程RSS约1.3–3.4GiB，输出约408MiB。

## 产物与验收入口

`tt/`、`sn/`分别有 `resource_windows.parquet`、`execution_windows.parquet`、`log_windows.parquet`、`labels.parquet`、`splits.parquet`及每表schema；另有 `eligibility_masks.parquet/schema`、`exposures.json`。共同四元主键、window_end及sample_id保留全网格；特征字段顺序、单位、质量与预测边界见schema。来源/role/注入/profile等不得输入模型。

根目录保存 `inventory.json`、`windows.parquet`、`modality_sources.json`、`summary.json`。`summary.json`是生产writer的验收前快照，其pending状态保留；最终验收以追加的 `quality_report.json` 为准。后者从inventory独立重建每运行完整网格，并核验六表身份、稳定ID、schema字段顺序、整运行隔离、Fault池排除及Unknown标签；报告每运行窗数、模态质量/覆盖、标签原因及参考池排除原因。

`manual_checks.json`保存12项小样本回查：TT/SN各Normal/Fault原始资源、日志、Entry/Span与边界、缺模态记录；`manual_missing_checks.json`另对两数据集各一个无日志窗口回查精确源流/成员，确认没有窗内时间戳事件。小样本不冒充全量人工审计或独立状态真值。

`artifact_manifest.json`保存全部验收前/验收报告产物hash、源码hash、Python和依赖版本、Git commit/dirty及resolved配置；输入精确hash和封装/member引用见inventory和modality_sources。后补缺日志回查由 `supplement_manifest.json` 独立绑定，并绑定原manifest hash，未改写既有产物。

新run需先完成可核查原始小样本记录，然后执行（拒绝覆盖已有验收文件）：

```bash
conda run -n interface python -m src.service.preparation_acceptance \
  outputs/service/<new-run-id> \
  --manual-evidence outputs/service/<new-run-id>/manual_checks.json
```

本轮定向自动检查93项通过（1.79s），包含Service旧协议相关接口；black/isort及git diff --check通过。完整命令与日志、先红后绿记录、源码快照位于 `.superpowers/sdd/training-preparation-plan/task-4-report.md`及同目录日志。没有运行全仓测试、旧DVC管线或重训。

历史来源审计中的 `source_spans` 是原始出现次数，`cropped_spans` 实际是裁窗后唯一Span键数，`cropped_traces` 是唯一Trace键数；不能用后两者与原始次数直接算保留比例。原结果为0且保持不变，本轮生产采用明确去重/accepted诊断，未把旧零裁窗结果改写为物理语义已证。

## 历史首版记录（preparation-v1，保留）

以下描述首版骨架及当时限制，不代表本轮仍未实现，也不改写当时结果。

## 用户已确认范围

共同架构、允许数据集字段不同。TT保留原R3及其质量状态，SN保留自己的资源字段；不要求取字段交集。现阶段仅数据预处理、证据与质量基础设施，不训练基线或神经模型、不运行文本编码器。

## 已实现入口

```bash
conda run -n interface python scripts/prepare_service_data.py \
  --tt-root data/tt_service_20261017_004546 \
  --sn-archive /tmp/SN-Dataset.zip \
  --output outputs/service/<new-run-id>
```

输出路径必须不存在且不得位于TT原始数据根内。TT只读selected17，并核验metadata/features绑定SHA256；SN核验正式包MD5、流式读取嵌套归档，不向磁盘解压原始内容。输出包括：

- `inventory.json`：各运行、模态字段、指标有效性/采样间隔、Trace覆盖、标签字段存在情况及Normal包交集。
- `tt_resources.parquet`：保留TT已有资源列、状态及稳定Service窗身份，不重算历史Counter。
- `windows.parquet`：两数据集完整候选窗、稳定样本ID、划分角色和待准备状态。所有主标签仍为Unknown，`training_ready=false`。
- `exposures.csv`：SN36条注入暴露记录及目标精确映射结果，未匹配名称保留UNRESOLVED，不模糊匹配。
- `summary.json`：产物数量及后续版本的执行环境/Git状态。

TT划分沿用已执行协议前三Normal fit、第四cal、第五六holdout；SN三Normal只写候选角色，不自动承认为严格健康训练数据。

## 实际执行

首版输出：`outputs/service/preparation-v1/`。58,483个候选窗，其中TT50,575、SN7,908；SN36条暴露记录，未执行训练。SN按epoch对齐15秒完整窗口，TT沿用已有资源网格；二者均为15秒，但不假定起点算法相同。

定向测试：`tests/test_service_preparation.py`与`tests/test_service_protocol.py`，合计49项通过。覆盖路径越界、hash、流式JSON截断/非法输入、大对象跨chunk、嵌套tar读取和Normal交集裁窗，及旧Service协议兼容。

## 新发现与先前结论修正

Normal归档的Trace ID交集分别为167、0、77，不能再写成“没有重复Trace”。按每个运行metadata时间边界检查，共享ID中同时位于两运行时间窗内的数量均为0。这与历史记录被查询回收相容，但ID相同不等于span内容完全相同，尚不据此声称数据独立。

三份无注入运行均含大量`error=true`/HTTP500 span：分别17,374、18,466、16,236条。当前计数覆盖归档全部span，尚未逐运行裁窗去重，不能解释为服务级错误率或独立异常数；它们足以说明“无注入”不可直接等于“健康金标”。

日志时间字符串无明确时区，目录时间与epoch也不能直接互换。SN资源列的单位及累计/瞬时语义尚需采集实现证据。本实现保留这些问题，不猜测偏移、不再差分CPU、不把缺error tag记为false。

## 尚未完成的适配

这批实现完成了入口、完整性核验、资源保留和窗口/证据骨架，**尚不是训练就绪的三模态预处理器**：

1. SN日志时区须由采集代码或可核查关联事件确定；当前日志窗口化未执行。
2. SN执行数据须按运行时间裁剪、定义Entry/服务根span及去重，不能直接聚合所有span当请求。
3. SNCPU/内存/网络单位与类型需核实后，才能冻结资源聚合。
4. TT日志与Trace生产adapter尚未接入；旧/tmp聚合表只可在核验来源与hash后复用。
5. 健康/异常标签证据须按评价范围逐项登记；未建立时保持Unknown。无注入场景中的常态业务错误是否在研究正常定义内，需要独立说明，不能为提升分数删除。

当前没有下载模型、拟合归一化、校准阈值、运行IF或训练网络。不能以候选窗数量作为有效训练样本数量。
