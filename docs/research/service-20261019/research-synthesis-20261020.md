# Service模型研发阶段性研究整理

范围更新：用户随后确认采用共同架构、允许TT/SN字段不同，不以特征交集限制主实验；本文第4节交集方案保留为历史候选。当前仅执行数据准备，不运行本文提及的基线、模型和消融；进度以[实现记录](preparation-implementation.md)为准。

更新时间：2026-10-20

## 1. 当前研究问题

主任务是对 `Service×固定时间窗` 输出正常/异常分数及二分类结果。模型使用独立 Normal Case 拟合，独立 Normal Case 校准阈值，异常 Case 只用于测试；Unknown 不填为正常，训练、校准和测试按整 Case 隔离。

当前模型假设为：

```text
资源统计 ── 数值编码器 ── 资源 token ─┐
执行统计 ── 数值编码器 ── 执行 token ─┼─ 同窗 Transformer ── 融合表示 ── Deep SVDD
日志窗口 ── 冻结文本编码器 ── 日志 token ─┘
```

Transformer只交互同一个时间窗内的模态token，不承担跨窗口长时序预测。Deep SVDD是统一的正常表示目标；BERT/文本编码器、数值编码器和融合层是否全部微调，需由正常数据稳定性和后续实验决定。

“BERT＋Transformer＋SVDD”作为宽泛组件组合已有强近邻，不能直接宣称无人做过。模型贡献应落在明确的模态交互机制、输入缺失处理或正常表示约束，并由直接拼接＋Deep SVDD、单模态模型、IF/OCSVM及消融实验支持。

## 2. 为什么需要跨数据集验证

创新点落在模型，而不是数据集。公开数据集承担外部验证作用：相同的模型架构、训练目标、评分方向和校准协议，分别在自有数据与公开数据上训练和测试，以检验模型是否可迁移。

应区分两种实验：

1. **同架构跨数据集复验**：每个数据集用自己的 Normal 数据训练和校准，再在自己的故障数据上测试。这是主要可行方案。
2. **冻结权重跨域测试**：在自有数据训练后直接在公开数据推理，或反向进行。这是更强但更困难的域迁移实验，需统一特征语义和归一化边界，不能与第一种混写。

自有数据上的绝对分数高于公开数据可以如实报告，但不能单独证明自有数据更优，也不能把不同故障难度造成的分数差归因于模型。模型贡献应看每个数据集内相对IF、OCSVM、直接拼接Deep SVDD等基线的增益，以及跨数据集是否保持方向一致。

## 3. Eadro SocialNetwork核验结果

- Zenodo版本：`10.5281/zenodo.7615394`。
- `SN Dataset.zip`：`78,753,809` bytes，MD5 `4c523faf64770b1e9a709cdd610e07c4`。
- `SN-all.tgz`含4个故障运行；`no fault/`含3个不同日期/时间段的Normal归档。
- 每个完整运行目录含 `spans.json`、`logs.json` 和12个服务Metrics CSV。此前的“14个服务CSV”是目录文件总数误读，已纠正。
- Metrics字段为CPU system/total/user、memory usage/working set、rx/tx bytes和epoch秒时间戳，约1秒采样。
- Logs按Service组织为原始日志行；当前不假设每条日志都有TraceID。
- Traces为Jaeger风格span，包含trace/span标识、操作名、起止时间、耗时和引用关系。
- 故障metadata含目标容器、fault type、开始时间和duration。

3个Normal归档内容哈希不同、时间不重叠，未发现归档级重复；这只能支持“可按运行段隔离的候选”，不能证明统计独立。正式实验需记录工作负载连续性，并确认fit/calibration/holdout的选择不依赖故障结果。

## 4. 跨数据集共同契约

为避免把两套数据硬凑成同一输入，建议定义语义共同特征：

- 资源：CPU使用量、内存使用量、网络收发变化；关闭process_count和memory_ratio等外部没有可靠对应物的字段。
- 执行：闭窗span数量、耗时统计、可确认的错误统计，并附覆盖/有效性mask；缺span不填零请求。
- 日志：按Service×Window收集消息，使用相同清洗、冻结文本编码器和窗口聚合；注入信息、故障名和目标字段不进入输入。
- 样本键：`(dataset_id, run_id, canonical_service_id, window_start_ms)`。
- 所有归一化参数只从各自训练Normal拟合；不能用测试故障或公开数据标签调参。

如果某数据集缺少一个模态，只能作为明确的模态消融或部分验证，不能写成完整三模态结果。不能按测试成绩临时关闭某一模态。

## 5. SocialNetwork还需要的处理

在开始训练前，仍需完成以下小型数据适配审计：

1. 解析4个fault metadata，将容器名映射到canonical Service，并保留目标容器、故障类型、暴露时间和证据范围。
2. 对3个Normal运行分别统计服务集合、时间范围、窗口数、日志条数、span覆盖和指标缺口；确认fit/calibration/holdout的运行段选择。
3. 检查Metrics字段是瞬时值还是累计量，确定CPU/网络是否需要差分；内存保留原始usage语义，不伪造limit比例。
4. 统一Logs时间解析和时区表示，确认窗口内日志是否有服务身份、是否存在多行堆栈或重复消息。
5. 从Trace tags中确认error字段是否稳定可用；若不稳定，执行分支必须使用覆盖mask而非填零。
6. 生成Service×15s适配表和标签表，逐行记录`label`、`label_source`、`scope`、`evidence`，Unknown排除。
7. 先跑共同输入的统计基线和经典基线，再接入神经模型；不因数据包可读取就直接声称主标签完整。

## 6. 近期材料索引

临时材料中值得保留的研究结论如下：

- `/tmp/service-cross-dataset-validation/CANDIDATE_AUDIT.md`：公开候选数据矩阵、共同特征和标签准入闸门。
- `/tmp/service-cross-dataset-validation/SN_VALIDATION_REPORT.md`：SocialNetwork文件、模态和故障证据核验。
- `/tmp/service-cross-dataset-validation/detailed/AUDIT.json`：外层成员哈希、Metrics字段、行数和时间范围机器记录；原始包仍只在`/tmp`。
- `/tmp/service-reproduction-selection-20261020/REPRODUCTION_COMPARISON.md`：AnoFusion、ART、USAD及复现边界。
- `/tmp/service-single-window-selection/SELECTION.md`：单窗候选和Deep SVDD/DAGMM/GOAD定位。
- `/tmp/cape-followup-qrw8359c/REPORT.md`：CAPE早期探针的限制与日志语义风险。
- `/tmp/service-idea-restart-20261020/`：尚未验证的模型发散候选，只作后续设计素材。

过程脚本、重复失败日志和未完成外部复核不进入正式研究结论。

## 7. 阶段性判断

“私有数据集上提出并改进模型，公开数据集上用同一架构做跨数据集验证”是可行的硕士论文路线，前提是：

- 模型贡献不写成数据集贡献；
- 公共数据集采用可追溯的共同输入和独立Normal协议；
- 自有集与公开集的差异只作为场景分析，不直接等同方法增益；
- 每个数据集内都保留相同基线和消融；
- 标签覆盖不足时报告Unknown和证据范围，不制造完整金标。

当前最合理的顺序是：完成SocialNetwork适配表和标签审计 → 固定共同特征契约 → 在两个数据集上建立相同基线 → 再实现BERT–同窗Transformer–Deep SVDD及消融。
