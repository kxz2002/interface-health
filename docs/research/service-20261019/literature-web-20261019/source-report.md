# 微服务可观测性、Service/API 异常检测与时间序列异常方法文献调研报告

**检索截止日期：2026-10-19（Asia/Taipei）**

本报告聚焦于“文献到底研究什么任务、用了什么数据、模型具体怎样工作、代码与数据到底开放到什么程度”，不据此为后续论文预设新模型，也不把根因定位、异常服务排序、API 检测与 Service×Window 异常检测混为同一任务。本轮仅进行了论文网页、出版页、数据页和代码仓库的文档核查；**没有 clone 仓库、安装依赖、下载完整数据或实际运行任何项目**。

同时已导出三张便于后续执行 Agent 使用的 CSV：

- [下载文献总表 CSV](sandbox:/mnt/data/microservice_literature.csv)
- [下载数据集专题表 CSV](sandbox:/mnt/data/microservice_datasets.csv)
- [下载开源项目执行交接表 CSV](sandbox:/mnt/data/microservice_open_source_projects.csv)

## 调研摘要与范围

本轮最终重点核查了约二十篇直接相关文献，并扩展检查了通用多变量时间序列异常检测、微服务根因定位和 API/Operation 数据集方向。最直接与“运行时观测 → 异常分数/二分类”相邻的代表包括 **AnoFusion、MSTGAD、ServiceAnomaly、DeepTraLog、TraceAnomaly、TraceGra、Eadro**；其中真正输出的对象并不一致：AnoFusion 是 **service instance × time** 的故障检测，ServiceAnomaly、DeepTraLog、TraceAnomaly、TraceGra 更接近 **trace/execution 粒度**，Eadro 则把系统异常检测与根因服务定位联合起来。citeturn13view0turn15view0turn17search13turn21search2

一个很重要的事实是：微服务论文里所谓 “unsupervised” 并不总等价于同一种训练协议。AnoFusion 明确说明 offline training 不用故障标签，但同时明确假设训练样本具有正常模式；它把多模态信号预测误差作为在线 failure score，再用 EVT 自动给阈值，因此与“normal-only 拟合、独立校准”的思想相近，但它的训练/阈值协议不能简单等同于你们当前 Case 隔离协议。citeturn14view0turn14view2turn14view3 BARO 则基本是统计变化点检测加根因打分，而不是神经网络 normal-only 表示学习；ART 虽称 unified unsupervised framework，其核心先通过自监督预测建立共享表示，再为 AD、failure triage 和 RCL 分别设计无监督后处理。citeturn21search9turn20view1

多模态方向有两类明显不同的数据前提。Eadro、AnoFusion、MSTGAD、ART 等通常要求把 metrics/logs/traces 对齐为统一时间表示，并且部分方法进一步依赖调用图或实例图；ServiceAnomaly 和 DeepTraLog 则以完整 trace/trace-derived graph 为主要组织单元。citeturn7view2turn14view1turn15view0turn17search13 因而“论文支持 metrics+logs+traces”并不意味着只要三种原始文件存在就能直接套用：对齐时钟、稳定身份、调用边、trace 完整性、log template、标签粒度都是模型输入契约的一部分。

目前网页核查中，**源码与数据同时较清楚**的项目主要是 AnoFusion、MSTGAD、BARO、ServiceAnomaly；Eadro 有官方代码和数据 DOI，但数据 DOI/Zenodo record 的具体对应关系值得后续执行 Agent 再核一次；DeepTraLog 有官方仓库和项目数据页，但 README 相比 AnoFusion/BARO 简略；Nezha 有公开实现且仓库明确显示 MIT License，但其任务是细粒度 RCA，而不是直接的 Service×Window 检测。citeturn14view5turn14view6turn21search0turn17search0turn13view3turn19search4

数据方面，尤其值得后续实跑分析的是三类资源。第一类是 **GAIA/MicroSS**：AnoFusion 在其中使用 10 个实例的 metrics/logs/traces，是较典型的 instance failure detection 公共数据。第二类是 **MSDS**：OpenStack 多源数据，明确提供 logs、metrics、traces、fault/workload scripts，并特别警告 logs/metrics 与 traces 存在 CEST/UTC 时区差异。第三类是 **BARO artifact**：Online Boutique、Sock Shop、Train-Ticket 三套 metric 数据，作者明确说明每种 fault×target 组合重复五次，每个 benchmark 得到 100 个 failure cases，这种“独立故障运行次数”口径比仅给窗口数的论文更清楚。citeturn14view4turn15view1turn21search33

API/Endpoint 方面，本轮发现的最直接资源是 2025 年 PROMISE 的 **LO2: Microservice API Anomaly Dataset of Logs and Metrics**。它使用 Light-OAuth2，通过 Locust 动态测试不同 API，并按触发错误给测试调用贴标签。值得特别注意的是：作者建立了 logs/metrics/traces 三模态采集管线，但实际 Light-OAuth2 实验中只获得了足量 logs 和 metrics，trace 数据被明确作为采集困难/局限来讨论。因此 LO2 不能简单记录为“三模态完整 API 数据集”。citeturn19search2turn19academia28

中文检索得到的高相关原始资料包括《软件学报》的“基于调用链控制流分析的大型微服务系统性能建模与异常定位”、清华大学陈文潇博士论文《基于深度学习的微服务异常检测与根因定位》，以及北京大学团队 2024 年“基于服务依赖图的微服务系统故障根因定位方法”。后两者尤其有助于区分**异常检测**和**根因定位**：清华论文分别研究 BUZZ 异常检测、DCGM/MetaRCA 根因定位；北大工作则属于已发生故障后的服务依赖图 RCA，并不是直接异常二分类器。citeturn22search5turn22search6turn22search16

## 检索记录与证据方法

本轮实际进行了中英文多轮扩展检索，而不是只搜索 “service anomaly detection”。初始组合包括：

|阶段|实际使用的代表性查询|
|---|---|
|初始广搜|`microservice anomaly detection service anomaly detection metrics logs traces root cause analysis paper arXiv 2021 2022 2023 2024 2025`|
|中文广搜|`微服务 异常检测 指标 日志 链路追踪 根因定位 论文`|
|API/Operation|`API anomaly detection endpoint performance anomaly distributed tracing paper microservices`|
|Train-Ticket|`Train-Ticket microservices anomaly detection dataset fault injection traces metrics logs paper`|
|已知线索展开|`ServiceAnomaly microservice anomaly detection paper`；`Eadro microservices multi-source data ICSE 2023`；`AnoFusion microservice anomaly detection`；`ART microservice anomaly detection traces metrics logs`|
|多模态方法展开|`UAC-AD microservice github arxiv`；`Triple Trace-Metric-Log microservice github`；`TraLogAnomaly microservice github`；`MADMM microservice anomaly multimodal data source code`|
|RCA 分流|`BARO Multivariate Bayesian Online Change Point Detection github`；`MicroRCA root cause localization microservices github DOI`；`Unsupervised Detection of Microservice Trace Anomalies through Service-Level Deep Bayesian Networks github`|
|API 数据集|`LO2: Microservice API Anomaly Dataset of Logs and Metrics arxiv dataset github`|
|中文二次扩展|`微服务 异常检测 多模态 指标 日志 链路追踪 学位论文`；`微服务系统 异常检测 服务级 时间序列 期刊 论文`；`微服务 根因定位 链路追踪 指标 日志 中文 学位论文`；`微服务 API 异常检测 接口 性能 论文 中文`|

扩展过程不是只按标题关键词扩张。AnoFusion 的参考文献进一步引出了 TraceAnomaly、JumpStarter/时间序列检测、Practical Root Cause Localization；BARO 和 Nezha 又把检索扩到 metric-only RCA、fine-grained multimodal RCA；LO2 则引出了 API 测试调用、数据采集完整性与 trace instrumentation 的问题。AnoFusion 本身的参考文献清楚列出了 TraceAnomaly、OmniAnomaly、USAD、DeepLog、Practical Root Cause Localization 等前序方法，因此这些方法被作为回溯线索，而不是把所有 “anomaly” 论文都纳入核心表。citeturn13view0

实际能够可靠访问并用于事实核查的来源包括 arXiv HTML/abstract、ACM Digital Library、IEEE Computer Society/Xplore 可检索元数据、ScienceDirect、Springer 搜索结果、作者主页、GitHub 和 Zenodo；Semantic Scholar 至少用于 Nezha 等论文的身份交叉确认。citeturn19search7 Google Scholar 在当前环境下更多返回作者 profile/索引片段，因此没有把 Scholar snippet 当作模型细节的主要证据。中文方面直接访问了《软件学报》、清华大学学位论文系统和北大机构知识库。citeturn22search5turn22search6turn22search16

**CNKI 本轮未获得足够稳定、可复查的直接全文入口**，所以本报告不声称已经系统检索 CNKI 全库。类似地，数据库中只有标题/摘要、而没有可访问全文的论文，其架构细节均保留“未核实”，没有根据二手综述自动补全。

证据优先级采用：**正式论文/作者预印本全文 > 出版页 > 官方数据页面 > 作者官方代码仓库 > 作者主页 > 学术索引 > 二手介绍**。例如 AnoFusion 可以直接从 arXiv HTML 核对 serialization、GTN/GAT/GRU、loss、failure score 和 EVT，而 Triple/UAC-AD 当前可直接获得的全文细节明显更少，所以后两篇的记录有更多 Unknown。citeturn14view0turn14view2turn14view3turn18search1turn18search0

纳入标准是：论文必须实质研究运行时异常检测、failure detection、trace anomaly、API anomaly、微服务 RCA，或是高度相关的通用时间序列方法/数据集。仅因为标题含 cloud、microservice、AIOps 而实际任务不同的工作没有进入核心表。正式版和预印本按同一工作计数，例如 ServiceAnomaly 的 SSRN 版和 JSS 正式版不重复计。正式 JSS 版本为 Volume 209、article 111917、DOI `10.1016/j.jss.2023.111917`。citeturn17search7turn17search13

## 文献总表与数据集专题

下面的表先回答“这些论文到底是不是同一个任务”。“AD”指直接异常/故障检测，“RCL/RCA”指已观察到问题后定位根因；两者同时出现并不意味着其监督条件、评价单位相同。

|文献|实际任务与输出粒度|主要数据|核心模型|训练含义|代码状态|
|---|---|---|---|---|---|
|Eadro, ICSE 2023|AD + 根因服务排序；窗口/服务图|Train-Ticket、SocialNetwork；M/L/T|Drain/Hawkes、DCC+attention、融合、GAT、双任务头|监督多任务|官方仓库+数据线索 citeturn7view0turn7view2turn6view1|
|AnoFusion, KDD 2023|**instance failure detection**，按时刻给 failure score/判定|GAIA；银行内部数据；M/L/T|GTN→GAT→GRU预测；EVT阈值|offline 不用标签；假设训练为正常模式|官方代码；GAIA公开，D2不公开 citeturn13view0turn14view4turn14view5|
|MSTGAD, ASE 2023|实例/系统图异常检测|MSDS + 私有数据；M/L/T|MST graph + Transformer 空间/时间注意力|论文称 semi-supervised|官方代码；MSDS公开 citeturn15view0turn14view6|
|ServiceAnomaly, JSS 2024|**trace anomaly detection**；可解释异常位置|TeaStore、Train-Ticket|CPG/DAG + 六个 profiling metrics 的关系模型|正常行为 baseline|作者代码+作者数据 citeturn17search13turn17search0|
|DeepTraLog, ICSE 2022|**trace-level binary anomaly**|公开 trace-log reproduction data|Trace Event Graph→GGNN→Deep SVDD|one-class/unsupervised-style|官方仓库+项目数据页 citeturn13view3|
|TraceAnomaly, ISSRE 2020|trace anomaly；service-level probabilistic modeling|微服务 trace|service-level Deep Bayesian Networks|无监督，学习正常 trace pattern|官方 NetManAIOps repo citeturn21search2turn21search5|
|TraceGra, Computer Communications 2023|trace anomaly|公开数据 + 本地 Train-Ticket 类实验|Trace Performance Graph + GNN + LSTM encoder-decoder|无监督|本轮未确认官方代码|
|Meng et al., FGCS 2021|execution-trace 结构/性能异常|Bench4Q、Social Network|tree-edit distance + PCA|正常历史 baseline|未确认官方代码 citeturn16search23|
|Deep Attentive AD, ICWS 2022|微服务 multimodal TS AD|Train-Ticket 故障实验|deep attentive multimodal TS|全文训练细节本轮未充分核实|未确认官方代码 citeturn18search12|
|Triple, IWQoS 2023|T/M/L 多模态异常检测+解释|待完整核实|统一图/时空学习方向|待核实|未确认官方代码 citeturn18search1turn18search9|
|UAC-AD, TSC 2024|多模态异常检测|2套开放模拟数据+1套工业数据|adversarial contrastive learning|无监督|官方代码本轮未确认 citeturn18search0turn18search19|
|MADMM, NCA 2024|metrics+logs 滑窗异常检测|待核实|metric-GCN + log-GAT + CMA-GRU|待全文核实|未确认官方代码 citeturn18search11turn16search12|
|TraLogAnomaly, SCP 2025|trace+log hybrid-sequence AD|待完整核实|HVS + Transformer + TF-IDF weighting|待全文核实|未确认官方代码 citeturn18search18turn18search14|
|ART, ASE 2024|**AD + failure triage + RCL**|两个 benchmark 数据集|Transformer→GRU→GraphSAGE→deviation|self-supervised shared representation + unsupervised downstream|官方代码本轮未确定 citeturn19search3turn20view0turn20view1|
|BARO, FSE 2024|AD + RCA；主目的为 root-cause ranking|OB、Sock Shop、Train-Ticket metrics|Multivariate BOCPD + RobustScorer|统计式，无需异常监督训练|官方代码、PyPI、Zenodo artifact citeturn21search0turn21search33|
|Nezha, FSE 2023|**fine-grained RCA**，不是直接 AD|2 public + 1 constructed dataset|多模态、可解释 RCA|待全文精核|官方 basic implementation，MIT citeturn19search4turn19search10|
|MicroRCA, NOMS 2020|**performance RCA**|微服务 topology+metrics|metric-enriched graph + personalized random walk|非监督图排序|官方代码未确认 citeturn21search4turn21search7|
|Practical RCL/TraceRCA, IWQoS 2021|RCA；内部含 invocation anomaly detection|traces|invocation anomaly + association-based localization|多阶段|官方 TraceRCA 仓库有实现线索；需实跑复核 citeturn13view0|
|LO2, PROMISE 2025|**API anomaly dataset**；非新核心 detector|Light-OAuth2 logs/metrics|数据集+初步 PCA/检测分析|N/A|数据/replication package公开 citeturn19search2turn19academia28|
|陈文潇博士论文, 2022|BUZZ AD；DCGM/MetaRCA RCA|Alibaba、eBay 等工业数据|生成模型/VAE、因果模型、GAT|不同子任务分别无监督/监督|公开代码状态未确认 citeturn22search6|

这里最容易造成误判的是“输出粒度”。例如 ServiceAnomaly 明确将一个 span 定义在服务粒度，但最终检测的是**一条测试 distributed trace 是否偏离 normal CPG**，因此“span 是 service”不能改写成“Service×固定时间窗 detector”。citeturn17search13 同理，TraceAnomaly 的模型名称包含 Service-Level Deep Bayesian Networks，但其论文任务仍然是 **microservice trace anomalies**。citeturn21search5

**按名称/版本去重后的数据集专题表如下。**

|数据集|公开性质|模态/格式线索|规模与故障单位|标签与划分要点|当前可核实获取方式|
|---|---|---|---|---|---|
|GAIA / MicroSS|公开|metrics、logs、traces；AnoFusion README 示例为 CSV/trace table|AnoFusion D1：5 microservices、10 instances；734,165 metric records、87,974,577 logs、28,681,438 traces；约两周|论文报告平均 failure-window ratio 4.908%；注入现实型故障|CloudWise GAIA GitHub citeturn14view4turn14view5|
|AnoFusion D2|**NDA，不公开**|M/L/T|14 microservices、28 instances；3,122,168 metrics、14,894,069 logs、9,473,763 traces|人工注入 CPU/memory/disk、packet loss/latency、application/VM failure|不可下载 citeturn14view4|
|MSDS v1.1|Zenodo Open|OpenStack traces、logs、metrics、workload/fault scripts；aggregated logs 有 CSV|concurrent 335.4 MB；sequential 315.1 MB|fault scripts/Rally report 可作为实验 ground truth|Zenodo DOI `10.5281/zenodo.3549604` citeturn15view1|
|MSTGAD 第二数据|不可公开|M/L/T|论文数据|作者仓库明确说无权限发布|不可获取 citeturn14view6|
|Eadro Train-Ticket|作者公开数据线索|M/L/T|41 microservices、27 business；normal 7h；162 injections；48,296 traces|CPU exhaustion、network delay/jam、packet loss；60/40 不同时段划分|Eadro repo/Zenodo；**DOI record 需再核对**|
|Eadro SocialNetwork|同上|M/L/T|21 microservices、14 business；normal 1.2h；72 injections；126,384 traces|同类 fault injection；60/40|作者数据包|
|ServiceAnomaly TeaStore/Train-Ticket|作者 GitHub 表示包含“used dataset”|distributed traces + six profiling metrics|论文公开页面未充分暴露完整 run/count 统计|normal CPG 与 fault traces 对比|作者 GitHub `data/` citeturn17search0turn17search13|
|DeepTraLog reproduction data|公开项目资料|raw trace/log + processed graph data|项目页按 normal/fault 数据组织；本轮未下载统计|fault grouping/normal data|FudanSELab 项目仓库/项目页 citeturn13view3|
|BARO artifact v0.0.1|Zenodo Open|metrics，压缩 ZIP；模型接口为 multivariate DataFrame|Online Boutique、Sock Shop、Train-Ticket；**每 benchmark 100 failure cases**；fault×target 重复5次|fault injection/root-cause metadata|Zenodo DOI `10.5281/zenodo.11046533` citeturn21search33|
|LO2|公开 Zenodo|主要是 logs + metrics；485 unique metrics|约657,000个 log files、超过20亿行日志、超过4,500万个 metric files|正确/错误 API 测试调用，标签来自触发的 error|Zenodo dataset DOI `10.5281/zenodo.14257989` citeturn19academia28|
|LO2 replication package|公开|Light-OAuth2、Locust、Prometheus、Jaeger 相关复现管线|不是单独 benchmark count|用于重建采集/测试流程|Zenodo DOI `10.5281/zenodo.14229369` citeturn19search2|

MSDS 的一个特别有复现意义的细节是：数据页明确警告 **logs 与 metrics 使用 CEST，而 traces 使用 UTC，相差 2 小时，需要自行同步**。这是论文“有三模态”之外非常关键的实际数据契约，后续 Agent 应优先检查而不是直接 concat。citeturn15view1

BARO 则展示了为什么必须区分“窗口数”和“独立故障重复”。官方 Zenodo 明确说对每一个 fault type × targeted service 的组合进行五次重复，从而每个 benchmark 得到 100 个 failure cases；这一数字具有 Case/run 层面的意义，而不是把每个时间点当成独立故障样本。citeturn21search33

LO2 也值得作为“系统公开≠完整观测数据公开”的正面例子：其贡献文字明确说明采集架构尝试三模态，但 Light-OAuth2 最终只有足够规模的 logs 和 metrics，trace instrumentation 是已报告的限制。citeturn19search2

## 核心文献逐篇记录

**Eadro — Lee et al., “Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data”, ICSE 2023，DOI `10.1109/ICSE48619.2023.00150`.**
**A｜任务。** 它并非单纯异常检测器，而是将 **anomaly detection 与 root cause localization 联合训练**；给定 observation window，输入每个 microservice 的 logs、KPI time series、trace records，同时输出系统是否异常以及 culprit microservice 概率/排序。citeturn7view0
**B｜数据。** Train-Ticket 41 个 microservices、其中 27 个 business services；DeathStarBench SocialNetwork 21/14。监控栈包括 Jaeger traces、cAdvisor+Prometheus KPI，日志通过 Elasticsearch/Fluentd/Kibana；KPI 包括 CPU system/total/user、memory、working set、network rx/tx。故障由 Chaosblade 注入 CPU exhaustion、network delay/jam、packet loss。Train-Ticket 正常采集约 7 小时，SocialNetwork 约 1.2 小时；各故障对每个 microservice 注入一次，总操作数分别 162 和 72。citeturn7view3
**C｜模型。** 日志先 Drain 模板解析，再用 Hawkes-process event occurrence intensity 表示；KPI 进入 1-D dilated causal convolution + self-attention；trace 按时间 slot 计算 callee latency，缺调用 slot 做论文规定的 padding，再经相似时序编码；历史 trace 建 service dependency graph。多模态中间表示经 concat/FC/GLU 融合后送入 GAT，最后由 detector 与 localizer 两个 head 联合优化。citeturn7view1turn7view2turn7view3
**D｜实验。** 论文 AD 基线包括 TraceAnomaly 等，RCA 基线包括 TBAC、NetMedic、MonitorRank、CloudRanger、DyCause 等；论文报告 Eadro 在其协议下 Train-Ticket 和 SocialNetwork 的 AD F1 接近 0.99，但这是其**监督联合训练、其故障注入协议和其窗口定义**下的数字，不宜与 normal-only 方法跨协议直接比较。citeturn6view0 作者明确讨论了无法检测不反映在可观测信号中的 silent/logical bug，并指出监督标签需求。citeturn6view0
**E｜开源。** 官方仓库提供 `codes/main.py`、`base.py`、`model.py`、`preprocess/align.py` 等，README 给出 `python main.py --data <dataset>` 入口；本轮未实际运行。citeturn6view1 数据 DOI 在仓库和搜索到的 Zenodo record 之间存在 record-id 表面差异，建议执行 Agent 保存最终重定向后的 concept/version DOI，不要在未核验前认为是两个不同数据集。

**AnoFusion — Zhao et al., “Robust Multimodal Failure Detection for Microservice Systems”, KDD 2023，DOI `10.1145/3580305.3599902`.**
**A｜任务。** 明确是 **service-instance failure detection**，不是 RCA。论文把一个实例内部 metrics、logs、traces 的多模态关联用于按时间检测失败。citeturn13view0
**B｜数据。** D1 为 GAIA/MicroSS，10 instances；D2 为大型商业银行 28 instances。D1 约 73.4 万 metric records、8797 万 logs、2868 万 traces；D2 约 312 万、1489 万、947 万。D2 因 NDA 不公开。citeturn14view4
**C｜输入与架构。** Metrics 先归一化；日志用 `bert-base-uncased` embedding 后 DBSCAN 聚类模板，并在滑窗中统计各 cluster 计数及日志总数；trace 在窗口内提取 invocation response time 的 mean、median、range、standard deviation，若有 status code 可作为额外 channel。所有通道按分钟同步。每一时刻建立 heterogeneous graph，三种 modality node type 形成六种 edge type；GTN 学异质跨模态关系，GAT 做 feature filtering，然后 GRU 预测下一时刻/图。citeturn14view1turn14view2
**训练目标。** MSE 预测损失，论文明确称 offline training 不使用标签，并进一步写明模型假设训练数据具有 normal patterns。citeturn14view0turn14view2
**异常分数。** 对各 channel 预测误差取绝对值，使用 median/IQR 标准化后取最大 channel 误差，最终以 EVT 动态阈值判故障。citeturn14view3
**E｜代码。** README 固定了较旧依赖：NumPy 1.19.2、Pandas 1.1.5、SciPy 1.5.4、scikit-learn 0.24.2、PyTorch 1.6.0 等；给出 `utils/generate_channels.py`、`config.py`、`main.py --mode train/eval` 和 `train.sh`。这使 AnoFusion 成为下一轮较适合优先验证“论文方法是否真正落到代码”的项目。citeturn14view5

**MSTGAD — Huang et al., “Twin Graph-based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System”, ASE 2023.**
**A｜任务。** 论文称其为 semi-supervised graph-based anomaly detection；节点对应 service instance，边对应实例间 scheduling relationship。citeturn15view0
**B｜数据。** 官方代码明确支持公开 MSDS，另一个实验数据因无发布许可而不可提供。MSDS 自身是 OpenStack 产生的 traces/logs/metrics 数据，并附 workload/fault scripts。citeturn14view6turn15view1
**C｜模型。** 三模态先特征提取、归一化，组成 Microservice System Twin graph；随后 Transformer-based network 同时采用 spatial attention 和 temporal attention，分别建模实例/模态相关性和时间依赖。citeturn15view0 当前公开 abstract 不足以可靠填写全部层数、hidden dimension、标签如何进入每一项 loss，因此这些字段保留待代码核查，而不是凭“semi-supervised”自行推断。
**E｜代码。** 官方 Alipay 仓库明确写出 Ubuntu 18.04、Python 3.8、PyTorch 1.12.0、PyG 2.2.0；预处理 `python util/pre_MSDS.py`，训练 `python main.py`。citeturn14view6

**ServiceAnomaly — Panahandeh et al., JSS 2024，DOI `10.1016/j.jss.2023.111917`.**
**A｜任务。** 这是**异常 distributed trace 识别**。论文把 span 定义在 service granularity，但检测对象仍是一条运行 trace 是否偏离正常执行，而不是每个 service 每个固定时间窗的二分类。citeturn17search13
**B｜数据。** TeaStore 和 Train-Ticket 两个开源 benchmark system，但这里必须区分“系统开源”和“论文实验数据”：作者另外在 GitHub reproduction package 中声明提供论文 used dataset 和 scripts。citeturn17search0turn17search13
**C｜模型。** 先把正常 distributed traces 的 context propagation trees 合并成 CPG/DAG，再用六个 profiling metrics 的线性和非线性关系对图进行 annotation。测试时将新 trace/CPG 与 baseline 匹配，结构差异、缺失/新节点、异常 metric relationship 等构成异常证据。它不是 Transformer/GNN detector。citeturn17search6turn17search13
**D｜结果。** 正式页面报告 TeaStore trace anomaly F1 约 85%、Train-Ticket 约 86%。这仍是 trace-level protocol，不可直接当作 Service×Window F1。citeturn17search13
**E｜实现。** 作者仓库 `M-panahandeh/ServiceAnomaly` 当前可访问，含 `ServiceAnomaly/` 与 `data/`，但网页首页没有像 AnoFusion 那样给出完整环境和统一命令，因此“有 reproduction scripts”与“可一键复现”应保持区分。citeturn17search0

**DeepTraLog — Zhang et al., “DeepTraLog: Trace-Log Combined Microservice Anomaly Detection through Graph-based Deep Learning”, ICSE 2022，DOI `10.1145/3510003.3510180`.**
**A｜任务。** 直接检测异常 trace，不是事后 service RCA。
**C｜模型。** 核心思想是将跨服务 trace 事件与服务内 log event 统一为 Trace Event Graph，再由 GGNN 学图表示，用 Deep SVDD 约束正常表示靠近 hypersphere center，以距离形成异常依据。也就是说，它属于“图表示 + one-class objective”，而不是预测未来时间窗。
**B/E｜数据与实现。** FudanSELab 官方仓库包含 `GraphData/`、`TraceLogData/`、`model/`、`results/`，项目资料还提供 raw/processed data 线索；但 GitHub README 首页本身非常简短，当前页面没有暴露清晰的依赖版本和一条完整复现命令。citeturn13view3 因而下一轮最需要核实的是 normal/fault split、Deep SVDD center 初始化、threshold/evaluation 实现以及 raw→TEG 的真正入口。

**TraceAnomaly — Liu et al., “Unsupervised Detection of Microservice Trace Anomalies through Service-Level Deep Bayesian Networks”, ISSRE 2020，DOI `10.1109/ISSRE5003.2020.00014`.**
**A｜任务。** 检测 anomalous traces；论文动机明确同时考虑 response-time 与 invocation-path 依赖。citeturn21search5
**C｜模型。** 论文以 service-level probabilistic/deep Bayesian representation 学习正常 trace 的总体规律，因此“service-level”描述的是内部建模层级，不是最终输出标签变成 service-window。
**训练。** 论文标题及作者资料将其定义为 unsupervised，摘要说明周期性 offline training 学 overall normal patterns。citeturn21search12
**E｜代码。** NetManAIOps 官方仓库可访问，当前为 `master`，网页显示 48 commits，但没有 release；本轮没有读取/运行历史依赖。citeturn21search2 对 2020 年项目应特别保存 Python/TensorFlow/PyTorch 的真实版本和失败日志。

**TraceGra — Chen et al., “TraceGra: A trace-based anomaly detection for microservice using graph deep learning”, Computer Communications 204 (2023), 109–117，DOI `10.1016/j.comcom.2023.03.028`.**
**A｜任务。** trace-level anomaly detection。
**C｜模型。** 将 trace topology 与 container performance metric 组织成 Trace Performance Graph，GNN 建模拓扑、LSTM 建模时间特征，采用 encoder-decoder/reconstruction deviation 构造 anomaly score。
**D｜数据。** 出版页说明同时使用 open-source data 与本地集群实验；本轮未可靠定位到可确认的官方 repo，因此不能把第三方相似仓库当作作者实现。
**E｜状态。** “未找到官方代码”仅表示本轮检索没有可靠确认，不等于作者声明不公开。

**Meng et al. — “Detecting anomalies in microservices with execution trace comparison”, FGCS 116 (2021), 291–301，DOI `10.1016/j.future.2020.10.040`.**
这是一个很有价值的非深度学习对照。它从历史 trace 建正常基线，用 **tree edit distance** 检测结构/path anomaly，并对 component execution time 使用 PCA 识别响应时间异常及相关 component。论文在 Bench4Q 与 DeathStarBench Social Network 上评价 CPU/network/memory/service 等故障。citeturn16search23 它说明微服务异常检测并非必须从 Transformer/GNN 出发，也提醒“trace structural anomaly”和“Service×Window resource anomaly”是两个不同问题。

**Deep Attentive Anomaly Detection for Microservice Systems with Multimodal Time-Series Data — Chen et al., ICWS 2022，DOI `10.1109/ICWS55610.2022.00062`.**
论文身份和 Train-Ticket 多模态异常检测方向可核实，并在后续综述/论文中持续作为微服务时序异常基线引用。citeturn18search12 本轮能够直接访问的原始全文信息不足以稳妥恢复每层结构、输入维度、完整 loss、threshold protocol；因此这些字段记为**未核实**。这篇适合下一轮执行 Agent 在拿到 PDF/实现后补齐，而不宜本轮仅根据“Deep Attentive”名称推断架构。

**Triple — Ren et al., “Triple: The Interpretable Deep Learning Anomaly Detection Framework based on Trace-Metric-Log of Microservice”, IWQoS 2023，DOI `10.1109/IWQoS57198.2023.10188773`.**
可确认其明确面向 Trace-Metric-Log 三模态微服务异常检测，并强调 interpretability。citeturn18search1turn18search9 搜索线索指向统一图/时空建模和 Deep-SVDD 类目标，但本轮没有获得足够的作者全文 HTML/官方实现来逐项核对，因此**数据集名称、训练 split、隐藏维度、阈值和官方代码均保留待核实**。本报告不把二手描述当成已完全验证的论文架构。

**UAC-AD — Liu et al., “Unsupervised Adversarial Contrastive Learning for Anomaly Detection on Multi-Modal Data in Microservice Systems”, IEEE TSC 2024，DOI `10.1109/TSC.2024.3411481`.**
**A｜任务。** 多模态异常检测。
**C｜核心问题。** 论文重点不是 forecasting，而是 unsupervised contrastive representation，并针对“正常但难分类的 hard samples”加入 adversarial/weighting 思路，避免这些样本被误当异常。citeturn18search19
**B｜数据。** IEEE 页面摘要说明评价包含 **两套 open-source simulated datasets 和一套大型通信公司的真实工业数据集**。citeturn18search0 本轮没有足够原始页面信息确定两套公开数据的准确版本和完整 split，所以不在此猜名称。
**E｜代码。** 本轮搜索到论文条目和第三方论文列表，但未确认作者官方 repo；状态因此是“官方实现未找到/未核实”，不是“作者明确不开放”。

**MADMM — Wang et al., “Microservice system anomaly detection via multi-modal data and multi-feature extraction”, Neural Computing and Applications 36 (2024), DOI `10.1007/s00521-024-09918-1`.**
它使用 metrics+logs 而不是完整 T/M/L。论文摘要描述每个时间窗分别构造 metric feature graph 和 log graph，metrics 用 GCN 建 spatial correlation，logs 用 GAT 建 log-event context，随后由 Cross-Modal Attention GRU 建模时间依赖与跨模态交互。citeturn18search11turn16search12 当前未确认官方代码和可复现实验数据，所以模型描述仅止于原文摘要公开的信息。

**TraLogAnomaly — Wei et al., “TraLogAnomaly: A microservice system anomaly detection approach based on hybrid event sequences”, Science of Computer Programming 245 (2025), article 103303.**
该工作明确试图解决“只看 inter-service traces 或只看 intra-service logs 会遗漏二者耦合异常”的问题，将二者组织成 Hybrid Event Vector Sequences，再利用 Transformer 建模式并使用 TF-IDF 对 hybrid-sequence feature 加权。citeturn18search18turn18search14 本轮未确认作者官方 code/data，故具体训练标签、threshold 与实验数据不作推测。

**ART — Sun et al., “ART: A Unified Unsupervised Framework for Incident Management in Microservice Systems”, ASE 2024，DOI `10.1145/3691620.3695495`.**
**A｜任务边界非常重要。** ART 不是单一 detector，而是把 **anomaly detection、failure triage、root cause localization** 放进共享 representation framework。论文表 5 分别报告 AD precision/recall/F1、FT precision/recall/F1 和 RCL Top-1/Top-3/AVG@5，说明三类输出在评价上也明确分开。citeturn20view2
**C｜模型。** 它依次抽取 channel dependency、temporal dependency 和 call dependency，分别使用 Transformer Encoder、GRU、GraphSAGE；论文还特别做了依赖抽取顺序的消融。citeturn20view0turn20view2 预测值和 observation 之间的 squared error 得到 `N×K` deviation matrix，先形成 instance-level deviations，再聚合为 system-level deviation。citeturn20view1
**AD。** 用正常时段 system-level deviation 的 L1 norm 初始化 EVT threshold；论文还增加 delay window，避免一次瞬态波动直接触发 system failure。citeturn20view1
**FT/RCL。** FT 使用 cut-tree based clustering；RCL 利用 instance deviation 做定位。换言之，它的共享模型可以同时服务不同诊断任务，但并不把三者变成同一种标签问题。
**E｜代码。** 本轮确认论文和作者站材料，但没有可靠确认一个与 ART 论文一一对应且文档完整的官方仓库，因此执行交接中不把第三方代码误标成官方。

**BARO — Pham, Ha, Zhang, FSE 2024，DOI `10.1145/3660805`.**
**A｜任务。** 官方 README 明确把 BARO 称为 end-to-end anomaly detection + root cause analysis 方法，但其研究目标重点是 RCA robustness。citeturn21search9
**C｜模型。** 前级是 **Multivariate Bayesian Online Change Point Detection**，利用多变量 metric 之间的相关结构检测 change/anomaly；随后 RobustScorer 用统计检验式方法对 root-cause metrics 排序。citeturn21search3turn21search9
**B｜数据。** 官方 artifact 包含 Online Boutique、Sock Shop、Train-Ticket。每个 fault type × target service 组合重复五次、每个 benchmark 100 failure cases；这对于后续研究独立 case/repetition 很有参考价值。citeturn21search33
**输入接口。** 官方库要求 Pandas DataFrame，`time` 一列，其他列按 `<service>_<metric>` 命名，例如 `cart_cpu`。citeturn21search9
**E｜实现。** 可 `pip install fse-baro`，也可源码安装；README 暴露 `baro.anomaly_detection.bocpd` 和 `baro.root_cause_analysis.robust_scorer`，并给出 `download_data()` 的最小运行例。citeturn21search0turn21search9 它是本轮最清楚的“网页文档看起来具备最小运行路径”项目之一，但仍没有在本轮实跑。

**Nezha — Yu et al., “Nezha: Interpretable Fine-Grained Root Causes Analysis for Microservices on Multi-modal Observability Data”, ESEC/FSE 2023，DOI `10.1145/3611643.3616249`.**
这篇必须标作 **fine-grained RCA** 而非 anomaly detector。论文实验包含两套 public datasets 和一套 constructed dataset。citeturn19search10 官方/作者仓库 `IntelligentDDS/Nezha` 明确自称 “basic implementation” 并显示 MIT license。citeturn19search4 因此后续 Agent 需要特别判断“basic implementation”究竟覆盖了论文多少预处理、训练和 evaluation，而不能看到 MIT 仓库就默认完整 reproduction package。

**MicroRCA — Wu et al., “Root Cause Localization of Performance Issues in Microservices”, NOMS 2020，DOI `10.1109/NOMS47738.2020.9110353`.**
任务是**性能问题根因服务定位**，并非主动异常二分类。其经典做法是给 microservice topology graph 注入 service/server metrics，再使用 personalized random walk 产生 root-cause ranking。citeturn21search4turn21search7 因此，它要求的“拓扑+metric”数据条件和只做每 Service 独立异常评分显著不同。

**Practical Root Cause Localization for Microservice Systems via Trace Analysis — Li et al., IWQoS 2021，DOI `10.1109/IWQOS52092.2021.9521340`.**
这也是 RCA，不应因仓库中有 anomaly detection script 就归为普通异常检测论文。AnoFusion 正式参考文献可核实该论文身份与 DOI。citeturn13view0 官方 TraceRCA 代码线索包含 feature selection、`run_anomaly_detection_invo.py`、训练文件准备以及 association-based localization，因此很适合作为后续 Agent 检查“异常 invocation → RCA”如何衔接的案例，而不是拿最终 root-cause accuracy 和 Service×Window F1 对比。

**LO2 — Bakhtin et al., “LO2: Microservice API Anomaly Dataset of Logs and Metrics”, PROMISE 2025，DOI `10.1145/3727582.3728682`.**
**A｜任务。** 这是数据集论文，特别针对 **microservice API** 测试与异常研究，而不是提出主要的新 detector。citeturn19search5turn19academia28
**B｜系统与标签。** Light-OAuth2；Locust 测不同 API，每次执行同时收集 correct/erroneous calls，并按测试触发的 error 产生标签。约有 657,000 个 individual log files、超过 20 亿日志行、超过 4,500 万 metric files、485 个 unique metrics。citeturn19academia28
**Trace 限制。** 作者搭建了 Jaeger trace 管线，但在 Light-OAuth2 上没有得到足量 trace，因此发布贡献实际上以 logs/metrics 为主。citeturn19search2 这一点与“原始系统存在 Trace JSON”完全不同：模型可用性取决于**有效覆盖和契约**，不是有没有某种文件名。
**E｜开放。** 数据 DOI `10.5281/zenodo.14257989`，replication DOI `10.5281/zenodo.14229369`，另有 community GitHub 用于提交/审查扩展数据。citeturn19search2

**陈文潇，《基于深度学习的微服务异常检测与根因定位》，清华大学博士论文，2022。**
这是中文资料里任务划分最清楚的一项。论文摘要明确区分三个算法：BUZZ 面向复杂 KPI time series 无监督异常检测；DCGM 面向长期部署微服务的 RCA，使用监督训练的 causal parameter table 和 VAE likelihood；MetaRCA 面向短期部署、利用 event metadata、causal attention 与 diffusion aggregation GAT 快速定位根因。其 BUZZ 用 Alibaba monitoring data，DCGM 用 eBay microservice data。citeturn22search6 工业原始数据能否公开下载、本论文三个算法是否有统一官方 repo，本轮未确认，所以不可把“论文公开”推导为“数据公开”。

**于庆洋等，《基于调用链控制流分析的大型微服务系统性能建模与异常定位》，《软件学报》，2022。**
中文摘要直接以大型微服务调用链、性能建模和异常定位为任务背景，强调故障传播造成大量服务异常，因此与“哪个服务是异常症状”和“哪个服务是根因”之间的区别相关。citeturn22search5 当前网页摘要不足以完整填写模型每个超参数、数据格式和代码状态，故没有人为补充。

**张齐勋等，《基于服务依赖图的微服务系统故障根因定位方法》，2024。**
该工作根据配置信息和日志动态构建服务依赖图，并在故障发生后结合异常事件推断因果链，属于 **RCA** 而不是 normal-only AD。北大机构库可核实标题、作者、2024 年发表信息与方法摘要。citeturn22search16

## 方法分类与任务边界

从这些文献可以观察到，比“Transformer/GNN/传统机器学习”更有用的分类轴其实是**模型预测什么、异常分数在哪里产生、模型需要什么数据组织结构**。

|类别|代表工作|真正输入/目标|典型异常分数或输出|常见数据前提|
|---|---|---|---|---|
|统计/经典 ML|Meng et al., BARO；通用 IF/OCSVM/PCA|metric TS、component time、trace comparison|change point、distance、PCA deviation、outlier score|可不需要完整 topology；通常依赖稳定数值特征|
|预测型|AnoFusion、ART|历史窗口→下一时刻/当前观测预测|prediction residual→聚合→threshold|连续时间序列、固定对齐；上下文不能随意跨 run|
|重建型|TraceGra、通用 AE/USAD 类|窗口/图→自身重建|reconstruction error|通常假设训练主要反映 normal pattern|
|概率/one-class 表示|TraceAnomaly、DeepTraLog|trace/path/TEG|likelihood 或到 normal center 的距离|需要足够正常 trace、稳定 event encoding|
|显式调用/依赖图|Eadro、MSTGAD、ART、MicroRCA|service/instance nodes + call/scheduling edges|node score、graph representation、ranking|要求边语义明确；有些必须有较完整 dependency graph|
|多模态融合|Eadro、AnoFusion、MSTGAD、MADMM、UAC-AD|metrics/logs/traces 中两种或三种|feature-level/representation-level score|模态时间对齐、身份对齐、missing semantics 很关键|
|Trace/API 相关|ServiceAnomaly、DeepTraLog、TraceAnomaly、LO2|trace、span、API test execution|trace anomaly 或 API-call label|请求级身份和 trace/log/API 关联通常比固定窗更重要|
|RCA/排序|MicroRCA、Nezha、BARO、TraceRCA|异常上下文 + topology/metrics/traces|Top-k root-cause service/metric|通常假设“有需要诊断的事件”，评价是 Top-k 而非 point F1|

AnoFusion 是“prediction-based multimodal graph”最完整的例子：不是先分别跑 metric/log/trace detector 后投票，而是先把各 modality 变成 channel，再构建异质图共同预测，因此属于表示级/结构级融合。论文还专门建立了 JLT 基线，即把 JumpStarter、LogAnomaly、TraceAnomaly 的结果多数投票；JLT 的表现用于说明简单 score-level voting 不等于学习跨模态关系。citeturn14view0turn14view2

Eadro 的融合又不同。它先分别学习 log event intensity、KPI temporal representation、trace latency representation，再使用中间融合和 service dependency graph/GAT，并同时训练 detector/localizer。因此它既使用了 feature/representation fusion，也利用了监督根因标签。citeturn7view2turn7view3 这与 normal-only anomaly scoring 是根本不同的学习协议。

ART 则进一步说明“统一框架”也不意味着评价指标统一：AD 是 precision/recall/F1，FT 又是另一组分类指标，RCL 是 Top-1、Top-3、AVG@5。论文表 5 把它们分列报告。citeturn20view2 所以今后整理结果时，应避免将根因 Top-3 与 event-level recall 放到一个“检测性能”列中直接排序。

通用多变量时间序列方法仍有价值，但应作为**方法类别/基线来源**而非微服务任务的直接证据。AnoFusion 的正式论文参考文献中明确包含 OmniAnomaly（stochastic recurrent neural network）和 USAD 等 multivariate time-series anomaly methods，说明这些成熟方法在微服务研究中本身就是常见参照系。citeturn13view0 这类方法一般不原生理解“service、trace、fault target、调用边”，需要与任务粒度分开描述。

在阈值方面也不能用“unsupervised”一词掩盖差异。AnoFusion 用 EVT 从误差分布形成动态 threshold；ART AD 也用 normal system-level deviation 初始化 EVT；Deep SVDD 类方法形成到 hypersphere center 的距离；ServiceAnomaly 则用图与关系匹配的偏离规则。citeturn14view3turn20view1 后续公平比较时需要记录 threshold 是否通过训练正常集、单独 calibration、validation label 或 test label 确定，而不能只记录 backbone 名称。

缺测处理是目前论文证据中明显不统一的一项。AnoFusion 在 trace serialization 中将“有请求产生的 RT 统计”转成固定通道；Eadro 论文对无 invocation slot 采用特定 padding；MSDS 明确要求时区同步。citeturn14view2turn7view2turn15view1 这些操作都带有原始数据语义，不能泛化成“任何 missing 都补 0”。尤其在你们的数据中“Trace 缺 Entry ≠ 无请求/健康”“资源质量无效 ≠ 健康 0”，这一点与文献中针对特定模态设计 missing semantics 的事实是一致的问题域差异，而不是简单工程细节。

## 开源项目执行交接

以下排序只表示**下一轮网页端执行 Agent 的检查收益**，不是模型优选，也不是预计性能排名。

|优先级|论文/项目|仓库|版本定位|数据/条件|环境与入口|预计阻碍|网页核查状态|
|---|---|---|---|---|---|---|---|
|高|AnoFusion|`https://github.com/AIOps-Lab-NKU/AnoFusion`|`main`；网页显示20 commits；SHA待保存|GAIA可公开；银行D2不可得|PyTorch 1.6；`generate_channels.py`→`config.py`→`main.py`|老依赖；GAIA目录契约；D2不可复现|README 和全文均较完整，未运行 citeturn14view5|
|高|MSTGAD|`https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection`|`main`；10 commits；SHA待保存|MSDS公开；第二数据不可得|Python3.8/PyTorch1.12/PyG2.2；`pre_MSDS.py`→`main.py`|semi-supervised label语义、时区对齐|README较清晰，未运行 citeturn14view6|
|高|BARO|`https://github.com/phamquiluan/baro`|`main` + PyPI；SHA待保存|Zenodo 三 benchmark 公开|`pip install fse-baro`; `bocpd`; `robust_scorer`|完整论文实验与library demo需区分|文档最好之一，未运行 citeturn21search9turn21search33|
|高|ServiceAnomaly|`https://github.com/M-panahandeh/ServiceAnomaly`|`main`；39 commits；SHA待保存|仓库含作者 used data|具体入口待 clone 后找|依赖文档不如前三者清楚|paper+repo已确认，未运行 citeturn17search0|
|高/中|Eadro|`https://github.com/BEbillionaireUSD/Eadro`|默认 branch；commit待保存|Zenodo/public data；DOI重定向需核实|`codes/main.py --data <dataset>`|数据 record、预处理对齐、许可证|代码结构和命令已确认，未运行 citeturn6view1|
|中|DeepTraLog|`https://github.com/FudanSELab/DeepTraLog`|默认 branch；无 release 线索|raw/processed data公开线索|`GraphData/`、`TraceLogData/`、`model/`|README简短；dependency和entry不清|官方repo确认，未运行 citeturn13view3|
|中|TraceAnomaly|`https://github.com/NetManAIOps/TraceAnomaly`|`master`; 48 commits|数据入口需检查|代码树待 clone|2020年代依赖兼容性风险|官方repo确认，未运行 citeturn21search2|
|中|TraceRCA|`https://github.com/NetManAIOps/TraceRCA`|commit待保存|数据链接需验证|feature selection、invocation AD、localization 多脚本|脚本顺序、旧依赖、数据链接|有作者实验室实现线索，未运行|
|中|Nezha|`https://github.com/IntelligentDDS/Nezha`|default；SHA待保存|数据需进一步检查|README/代码待执行|README 自称 basic implementation|MIT repo已确认，未运行 citeturn19search4|
|数据专项|LO2|`https://github.com/M3SOulu/LO2-Dataset-Community` + Zenodo replication|版本/commit均需保存|dataset+replication公开|Docker/Locust/Prometheus/Jaeger体系|数据巨大；trace 本身是已知缺口|论文和数据 DOI 已确认，未运行 citeturn19search2|

后续执行 Agent 对这些项目不应只测试“`python main.py` 是否退出码为 0”，而应优先留下能判断**论文实现真实性**的证据。

**AnoFusion** 首先检查 `serialize/`、`utils/generate_channels.py`、`model/`、`config.py`、训练/评价分支，确认代码是否真的实现 BERT/DBSCAN log serialization、trace RT statistics、六种 heterogeneous edge type、GTN、GAT、GRU、max normalized error 和 EVT，而不是只实现论文的一部分。官方最小路径是准备 GAIA `mobservice2` 文件 → `generate_channels.py` → 编辑配置 → `main.py --mode train` → `main.py --mode eval`。citeturn14view2turn14view3turn14view5 必须运行后才能判断 GAIA 当前目录结构是否仍与脚本匹配，以及 evaluation 是否只在最后用 labels，还是意外参与 normalization/threshold。

**MSTGAD** 应先比较 `Architecture.jpg`、`src/` 与论文 abstract 中的 MST graph、spatial/temporal attention，随后追踪 `util/pre_MSDS.py` 到实际模型 input tensor。官方最小命令只有两步，因而任何隐含文件路径或预计算文件都容易暴露。citeturn14view6 特别需要保存 MSDS 时间转换步骤，因为 Zenodo 明确提示 traces 与 logs/metrics 存在时区偏移。citeturn15view1

**BARO** 最适合作为“先验证环境和数据流水线”的项目。先跑 README 的 `download_data()` 最小例，保存 anomaly timestep 与 Top-5 root-causes，再转向 artifact 中三个 benchmark 的完整实验配置。官方 README 已明确给出示例输出。citeturn21search9 必须分开记录“library sample 能跑”与“FSE 全部表格可复现”这两个结论。

**ServiceAnomaly** 首先找六个 profiling metrics 的具体采集/处理代码、CPG construction、关系拟合和 graph matching 文件；然后核对 `data/` 到底包含原始 traces/metrics、预处理后中间件，还是二者兼有。论文正式版明确说明 reproduction package 提供论文所用 scripts/data，所以如果仓库缺关键数据，应记录为“当前仓库状态与论文声明不一致”，而不是自动用别的 Train-Ticket 数据替换。citeturn17search12turn17search0

**Eadro** 应从 `preprocess/align.py` 开始，因为它决定三模态时间窗和服务身份如何对齐；再核 `model.py` 是否包含 log Hawkes、KPI/trace DCC-attention、fusion、GAT、detector/localizer。论文的 supervised multi-task loss 必须在代码中找到对应实现后才能称“完整复现”。citeturn7view2turn7view3 数据方面不要用“Train-Ticket 是开源系统”替代作者实验数据。

**DeepTraLog** 的首要问题反而不是 GPU，而是仓库结构与数据契约。应找 raw trace/log → TEG 的 parser、GGNN、SVDD center/loss、threshold 和 test script，明确 `GraphData/` 是否是不可逆的作者预处理产物。如果只跑 processed graph，报告中要写“复现模型训练/评价”，而不能写“复现完整数据预处理”。

**TraceAnomaly/TraceRCA** 都是较老代码。执行 Agent 要保存 `requirements`、Python interpreter、TensorFlow/PyTorch 实际兼容性、数据链接 HTTP 状态和任何 patch；不要把为了适配现代库而修改的代码结果当成原仓库原样成功。TraceAnomaly 官方 repo 确实存在，但没有 GitHub release 可直接作为论文固定版本。citeturn21search2

**LO2** 的执行目标与模型 repo 不一样：先验证数据下载、文件数/压缩包大小和 label/schema，再核 replication environment。由于论文自己说明 trace coverage 不足，下一轮不应把“Jaeger 被配置”写成“完整 trace modality 已提供”。citeturn19search2

所有项目建议统一保存：**完整 commit SHA、branch/tag、git status、Python/OS/CUDA、lock/requirements、数据 DOI 与 checksum、原始命令、stdout/stderr、wall-clock、峰值内存、随机种子、配置快照、实际输出文件、评价脚本版本和所有本地 patch diff**。这不是复现结果本身，而是防止之后“README 运行”“改过后运行”“论文数字复现”三种状态混在一起。

## 与本地数据的客观对照、待核实问题与参考来源

根据本轮任务中提供的本地 Train-Ticket 数据说明，你们当前正式问题是 **25 个应用 Service × 15 秒闭窗 → anomaly score + normal/anomaly 二分类，随后再聚合观察 Case/event 效果**。这与文献中的四类粒度都不完全等同：

|对照方面|你们当前条件|文献常见条件|客观差异|
|---|---|---|---|
|预测对象|Service×15s Window|instance×time、trace、system window、root-cause ranking|AnoFusion 最接近 instance-time；ServiceAnomaly/DeepTraLog/TraceAnomaly 主要是 trace|
|独立实验|17 Cases；6 Normal/11 fault|部分论文只报告 windows；BARO明确有独立 fault repeats|你们明确不能把 50,575 行当成 50,575 次独立实验|
|训练|3 Normal Cases fit、1 Normal calibration、2 Normal holdout；fault仅test|AnoFusion normal-pattern training；Eadro supervised；MSTGAD semi-supervised；ART self-supervised|监督条件差异很大|
|资源特征|`cpu_cores`,`memory_usage_ratio`,`process_count` 三维，且质量状态显式|AnoFusion/Eadro 常有更多 CPU/mem/network/user metrics|特征维数更少，但有更严格的 Counter/reset/limit/quality 语义|
|缺测|9,074 行资源不可评分，不可等同健康0|一些方法通过 padding/serialization 得到固定张量|不能直接照搬论文的 0-padding 语义|
|Logs|存在但连续覆盖未完全证明|多模态论文往往假定可稳定模板化、按窗计数|“缺日志”不能自动转成0-error|
|Traces|Entry/Exit/Local 可区分；缺Entry不代表无请求|trace方法常把一条完整 trace 作为样本|需要先证明请求/trace coverage 才能获得同等输入契约|
|拓扑|观测调用边不保证完整 dependency graph|Eadro/MSTGAD/ART/MicroRCA 直接依赖 graph|原始边存在不等于拥有论文要求的稳定图|
|标签|注入时间/target 是暴露证据，不是所有 Service×Window 完整真值|多项注入 benchmark 直接将故障窗/target作为 label|评价语义比许多公开实验更保守|
|API级|独立 endpoint/API 研究方向|LO2 是 API-call/error 标签；ServiceAnomaly 是 trace/span|API任务应与 Service window detector 分开评价|

因此，从“数据前提是否完全匹配”看，当前最值得注意的不是模型复杂度，而是各文献的**单位定义**。AnoFusion 的目标是单实例 failure，因此概念上比 trace-level 工作更接近 Service×Window，但 AnoFusion 的输入包含大量多模态 channel，且使用一分钟对齐，而你们目前正式资源表只有三维核心 resource feature。citeturn14view2 这只是事实差异，并不意味着应选或不选 AnoFusion。

BARO 使用 `<service>_<metric>` 式全系统 multivariate matrix，因而其统计依赖跨 service 共同建模；你们目前数据自然可以表示成 `Service×Window` 长表，但这不自动等价于 BARO 所需的一行一个 timestamp、列为服务×指标的 wide matrix。citeturn21search9 更重要的是 BARO 的论文核心评价是 root-cause ranking，因此即便其前级 BOCPD 能给异常时间，也不能把整篇 BARO 当作纯 Service detector 基线。

Eadro 对当前数据的主要差异是：它要求三模态和 dependency graph，并且异常/root-cause 标签用于监督联合训练；你们当前故障 case 只用于测试，并且故障 target/time 不能视为完整 Service×Window 真值。因此两者首先是**学习问题定义不同**，不是仅仅“feature 少一些”。citeturn7view0turn7view3

ServiceAnomaly 的差异则主要在样本单位：它构建 normal CPG 后，对单条 trace 与 baseline graph/metric relationship 做比较。你们虽然保有 Trace JSON，但正式研究的 Service detector 样本是固定 15 秒窗口。把 ServiceAnomaly 迁到 Service×Window 会改变原论文问题，不能记为“原模型直接支持该粒度”。citeturn17search13

DeepTraLog/TraceAnomaly 同样更适合作为**独立 Trace/API 异常研究的参考**：前者的 TEG 本身围绕一次 trace 组织日志事件，后者对 trace invocation path/response time 建正常概率模型。它们与资源三维 Service detector 可以属于同一论文的不同观测层，但评价样本不能混在一起。citeturn13view3turn21search5

LO2 与你们独立 Endpoint/API 方向的对应关系最直接，但标签性质仍不同。LO2 的 API tests 主动触发正确/错误调用，可以在 call/test execution 级产生相对明确标签；你们当前强调客户端失败、trace error、log ERROR 三者语义不完全一致，因此不能把 LO2 的标签策略原封不动视为你们的 ground-truth 定义。citeturn19academia28

目前最值得下一轮执行 Agent 补齐的事实缺口有以下几组：**Triple、UAC-AD、Deep Attentive AD、TraLogAnomaly、MADMM** 的官方代码身份/数据版本尚未充分核实；**ART** 的正式作者代码入口未确认；**TraceGra** 的官方代码未找到；**ServiceAnomaly** 的依赖与一键运行路径需要实际 clone 后查看；**Eadro** 的 Zenodo version/concept DOI 应通过真实下载页最终固定；**MSDS** 的许可证字段及具体 raw schema 需要解包；**Nezha** 仓库所称 “basic implementation” 到底覆盖多少论文步骤必须实测。相关论文身份本身已经能够由出版页或作者材料确认。citeturn18search1turn18search0turn18search11turn19search3turn19search4

另一个需要专门核实的是评价协议中的 **point adjustment、容忍窗口与 test-label thresholding**。AnoFusion 的公开全文清楚给出了 EVT threshold，因此这一点相对透明。citeturn14view3 但部分其他论文当前网页摘要不足以证明阈值完全不看测试标签；后续执行 Agent 应直接追 evaluation script，而不是因为论文写“unsupervised”就默认 test label 只用于最后算 F1。

最后，关于本轮证据边界：可以较高置信度确认 AnoFusion、Eadro、MSTGAD、ServiceAnomaly、BARO、ART、LO2 的任务定义及上述主要数据/架构事实；DeepTraLog、TraceAnomaly 的论文身份与官方仓库也较清楚；而 Triple、UAC-AD、MADMM、TraLogAnomaly、Deep Attentive 等仍有若干实现和数据字段只获得摘要级证据。报告和 CSV 中已经对这些项目保留 `Unknown/未核实`，没有以相邻论文、综述或仓库名称自行补齐。citeturn13view0turn15view0turn17search13turn21search9turn19search2

本轮最重要的原始入口包括：AnoFusion 的 KDD/arXiv 全文与官方仓库；MSTGAD arXiv、Alipay 仓库和 MSDS Zenodo；ServiceAnomaly ScienceDirect 与作者 reproduction repo；ART ASE 正式论文；BARO 官方 repo 与 Zenodo artifact；Nezha FSE/官方仓库；LO2 arXiv/ACM/Zenodo；以及中文的清华学位论文系统、《软件学报》和北大机构库。citeturn13view0turn14view5turn14view6turn15view1turn17search13turn20view2turn21search33turn19search4turn19search2turn22search6turn22search5turn22search16
