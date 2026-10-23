# 服务先验 GRU 查新：硕士论文定位

检查截止：2026-10-23；近六个月重点窗口：2026-04-23—2026-10-23。此为有范围限制的文献查新与选题判断，不是“无人做过”的证明，也不能代替导师及院系判断。代码、数据划分和性能事实由主代理提供；本报告不另行运行实验。

## 结论

**PROCEED WITH CAUTION，5/10。** 已检索文献中未定位到同时包含“每服务 Normal-fit 可用性频率的固定 log-odds 输出偏置、共享 GRU 的动态残差、数值预测与可用性预测联合异常评分”的同一结果，足以继续做硕士论文原型及必要消融。薄弱点很具体：Beta-Bernoulli 平滑、输出 offset、GRU 与缺失指示都是成熟手段；当前最窄差异可能只是合适的参数化与归纳偏置，必须证明其作用超出静态先验报警或服务身份记忆。不能据此承诺创新性评审通过。

推荐可核验表述：**针对服务间正常遥测可用性差异，以仅由正常训练窗统计的服务级固定先验为基准，让共享时序网络学习可用性动态偏离，并将其与可观测数值预测误差共同用于异常检测。**

不推荐表述：“首次用 GRU 做微服务异常检测”“首次利用缺失信息”“首次使用多模态/BERT”“提出新的贝叶斯 GRU”“解决完整真实健康真值”。本方法没有对神经权重做贝叶斯后验，也没有层次先验共享或新服务适配机制。

## 方法及 4 个核心主张

1. **缺失信息有预测价值**：历史 4 个 15 秒窗含 12 数值、12 观测掩码、3 模态可用标志、历史存在标志；GRU 预测当前值。GRU-D 已建立 mask/time-gap 的 informative missingness 思路，此点不能独立主张原创。
2. **服务级静态先验与共享动态预测分离**：每服务只在 Normal fit 统计 8 种可用模式；各模态边缘使用 Beta(1,1) 后验预测概率 `(present+1)/(N+2)`，其 log-odds 加到预测头。8 类计数是统计存储，当前预测头不建模三模态联合分布；没有学习层次先验或服务 embedding。
3. **值与可用性共同评分**：当前值不输入编码器，loss/score 为观测字段 masked MSE 加三模态 BCE。数值缺失不伪装健康 0；但遥测不可用也不能直接等同业务故障。差异主要在把可用性作为显式预测目标及异常证据。
4. **历史日志语义补充**：固定 BERT 的 256 维表示经可训练 64 维投影进入历史上下文，目标和评分不变。此为增量消融，不自动构成创新。已有三 seed 结果未改善：原模型 mean AUROC/AP 0.8239703/0.6800606，BERT 0.81323547/0.66702006，约下降 0.01073/0.01304。结果来自同一已被探索的测试集，不是新盲测。

## 最接近工作

| 原始来源 | 已核验年代/载体 | 重叠 | 本方法可核验差异及保留意见 |
|---|---|---|---|
| Che et al., *Recurrent Neural Networks for Multivariate Time Series with Missing Values*, DOI 10.1038/s41598-018-24271-9；arXiv:1606.01865 | Scientific Reports，2018-04-17 | GRU、mask、时间间隔、缺失有信息；经验均值来自训练集 | GRU-D 是临床时序分类及可学习衰减，不是当前每服务可用性输出先验+BCE异常分数。不得把 informative missingness 当新概念。 |
| Zhao et al., *Robust Multimodal Failure Detection for Microservice Systems* (AnoFusion), arXiv:2305.18985 | 初版 2023-05-30 | 无监督微服务故障检测，metrics/logs/traces，GTN+GAT+GRU | 证明“三模态+GRU”已有；当前窄差异是固定服务可用性先验、显式可用性预测，而非图结构。 |
| Sun et al., *ART: A Unified Unsupervised Framework for Incident Management in Microservice Systems*, DOI 10.1145/3691620.3695495 | ASE 2024；出版页 2024-10-18 | Transformer 通道关系、GRU 时序、GraphSAGE、无监督自回归 | ART 足以否定“用时序预测替换静态融合就是新方法”；本方法收窄到结构性缺失及服务先验。 |
| Qian et al., *Missing-Aware Multimodal Fusion for Unified Microservice Incident Management* (ARMOR), arXiv:2603.25538；ASE DOI 10.1145/3832783.3834338 | v1 2026-03-26；截止前 v3 2026-08-02 元数据/摘要已核实；ASE 页显示 2026-10-07 | **最关键近邻**：缺失感知多模态、自回归、mask-guided reconstruction、动态 bias compensation | 已读 v1/v2 §4.3 的偏置作用于缺失模态融合 gate；§4.4 为观测/人工 dropout 的两项 MSE。当前为按服务 Normal-fit 固定频率作用于 availability 预测输出，并以 BCE 评分。v3 全文和发表全文覆盖不足，不能保证最新版本没有其他更接近机制，也不能将早期版本实验限制直接套用最新版本。 |
| Zhang et al., *ADmM: Anomaly Detection for Microservice Systems with Incomplete Metrics*, DOI 10.1145/3801729 | ACM TWeb 20(2), Article 21，2026-04，35 页；作者全文 | 多模态语义/模板、缺失指标补全、正常生成模式、图依赖 | 作者 §4.8 明确未把 informative missingness 显式作为异常信号；当前不对不可见真值做补全监督，而预测观测可用性。但“缺失感知微服务”问题本身已有。 |
| Menon et al., *Long-Tail Learning via Logit Adjustment*, arXiv:2007.07314 | ICLR 2021 | 基于频率/先验的输出 logit 调整 | 其为长尾分类与 balanced error，不是正常服务可用性预测；但足以说明 offset 不是新数学算子，当前应解释它的领域作用。 |
| Peng et al., *S4M: S4 for Multivariate Time Series Forecasting with Missing Values*；Jang et al., *Towards Robust Real-World Multivariate Time Series Forecasting: A Unified Framework for Dependency, Asynchrony, and Missingness* | ICLR 2025 / ICLR 2026 官方 proceedings | mask/缺失模式与预测联合建模 | 模型和任务有差别，但不能把“缺失处理融入预测”本身主张原创。 |
| Wang et al., *Confidence-Guided Cross-Modal Knowledge Transfer for Multimodal Anomaly Detection in Microservice Systems*, arXiv:2609.25856 | 2026-09 预印本；机构页标记 TSE accepted 2026-10-01，不当作已正式出版 | 近期 logs/metrics 可靠性差异、负迁移、跨模态补充 | 置信引导深聚类/跨模态对齐，未在已读摘要中见当前固定服务先验机制；全文覆盖有限，不作排除证明。 |
| Luan et al., *Wearable Missingness as a Behavioral Signal: Detecting Wear Pattern Anomalies Around Cardiovascular Visits* | ICML 2026 **SD4H workshop**；作者实验室 2026-07-09 公告、7 月 11 日展示，首次 OpenReview 日期未知 | 原始官方摘要说明按个人及 day-type 建模佩戴缺失模式、检测偏离 | 不是微服务，也未在已核验摘要中包含共享 GRU/固定输出 offset；说明“正常缺失习惯偏离是异常信号”也不能宽泛称首创。全文机制覆盖有限，不能把 workshop 当主会论文。 |

原文入口（稳定标识便于复核）：

- GRU-D：`https://www.nature.com/articles/s41598-018-24271-9`
- AnoFusion：`https://arxiv.org/abs/2305.18985`
- ART 作者全文：`https://nkcs.iops.ai/wp-content/uploads/2024/09/ART24_to_ASE.pdf`
- ARMOR 已读版本：`https://arxiv.org/html/2603.25538v1`、`https://arxiv.org/html/2603.25538v2`；截止前最新版本元数据入口 `https://arxiv.org/abs/2603.25538v3`；发表标识 `10.1145/3832783.3834338`
- ADmM 作者全文：`https://zecoo.github.io/hugo/files/ADmM%20An%20Anomaly%20Detection%20Framework%20for%20Microservice%20Systems%20with%20Incomplete%20Metrics.pdf`
- Logit adjustment：`https://arxiv.org/html/2007.07314v2`
- CMT-AD：`https://arxiv.org/abs/2609.25856`
- Wearable missingness 官方摘要：`https://icml.cc/virtual/2026/71778`；作者实验室日期证据：`https://dunn.pratt.duke.edu/news/leeor-hershkovich-attending-icml-in-seoul-south-korea/`
- S4M 官方：`https://proceedings.iclr.cc/paper_files/paper/2025/hash/7b2f0758334389b8ad0665a9bd165463-Abstract-Conference.html`
- ChannelTokenFormer 官方：`https://proceedings.iclr.cc/paper_files/paper/2026/hash/f924dab7a06d0463fe238904678acaee-Abstract-Conference.html`

## 硕士论文要补的最小证据

| 问题 | 预先固定的消融/对照 | 能回答什么 |
|---|---|---|
| 增益是不是全部来自静态服务先验？ | prior-only Bernoulli NLL；现有 8 模式 availability；当前联合模型 | 区分固定服务习惯与动态时序贡献。 |
| 服务条件化是否必要？ | 无 prior、全局 Normal prior、服务 prior；相同 seed/预算/划分 | 比较服务异质性与一般 bias 初始化效果。 |
| 固定统计是否优于可训练服务 bias？ | 参数量/预算可比的自由 service bias 或 embedding 对照 | 证明固定先验的约束价值，而非服务身份信息优势。 |
| 动态历史真的有用？ | 无历史/短历史与正常连续历史；必要时诊断性打乱历史 | 不将先验或分布差异误写为 GRU 时序建模优势。 |
| 哪个评分分支有效？ | value-only、availability-only、联合；固定阈值校准 | 解释 masked MSE 与 BCE 的增益与失效。 |
| 观测掉线会不会误报业务故障？ | 健康业务下的采集器掉线/采样变化负对照，训练内未见服务可用性比例变化诊断 | 区分业务异常、正常低活动、结构性无模态和采集故障。不能把所有缺失标异常。 |

所有对照沿用 Normal-only fit、独立 Normal 校准、整 Case 隔离、完整主样本集合；不依据 Fault 选权重、阈值、checkpoint 或 seed。已观察的同一测试集只能报告探索发现；新增独立事件复测比重复窗口置信区间更有说服力。跨系统并非启动硕士原型的硬门槛，但当前单批次、少独立事件和 CPU/DISKIO/POOL/DNS 弱项必须披露。

## 检索范围与审核边界

采用 4 核心主张的同义检索，覆盖 GRU/missingness、服务条件化/先验/logit、联合 availability/数值异常评分、BERT 多模态；每类至少 3 种表述，加入 2024—2026、arXiv 与近六个月限定。检索 ICLR 2025/2026、NeurIPS 2025、ICML 2025/2026 相关页面；有命中不等于完整会议集合逐篇排查。

本代理调用 `web.run` 多次未返回内容，使用实际可用 Exa 网页检索获取出版社、arXiv、官方 proceedings 与作者全文的原文内容；Google Scholar/Semantic Scholar 仅间接索引检索，不能声称运行了其完整 API/全文数据库。没有发现精确命中不等于不存在；标题同名与否不是裁决标准。搜索中非 primary 来源只用作线索，不据其独立作方法结论。检索原始结果保存在忽略的 `.aris/traces/novelty-check/2026-10-23_run01/search-results.json`。

fresh reviewer：默认路由请求 Codex native `gpt-6-astra/xhigh`，独立上下文。完成意见同为 **PROCEED WITH CAUTION，5/10**；仅宿主宣称的 Codex/GPT-6 家族可见，具体实际 model ID/effort 未暴露，均记 unknown，不用请求值冒充实测身份。原始回复与访问范围更正见本次 trace；最新/发表全文访问不足，scope 未标完整。属于同模型家族的 provisional review，不能称独立跨家族认可。
