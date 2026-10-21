# 网页文献调研提炼：文献、数据集与模型机制

接收于2026-10-19，供后续Service模型brainstorming参考。完整原件见[source-report.md](source-report.md)，来源、hash和附件缺失说明见[source-provenance.md](source-provenance.md)。

**以下文献事实均转述外部网页报告，本轮未独立核验或运行。** “有代码/公开数据”是报告记录的状态，不是当前实测可用性；Unknown保持未知。原报告的会话引用不可本地解析，正式引用须使用后续核验过的原文。下文的“可参考”不等于确定采用，也不意味着已有性能增益。

## 后续外部复核补充

2026-10-19新增[复核增量与修正](followup-findings.md)，附[原始报告合集](followup-source-collection.md)。首轮台账保留原貌；涉及代码完整性、数据身份、阈值与实际运行状态时请并读增量，不能仅根据下表早期“官方/公开”描述判断可直接复用。两轮均非本仓库独立复现。

## 1. 本次调研对我们的价值

- 有直接相邻的instance×time多模态检测工作，也有trace级、系统级检测和RCA；不能因名称带Service就当成Service×Window模型。
- 成熟模型上的适度改进符合本项目硕士论文尺度。文献用于学习机制、选择可比较的参照，不构成“先证明简单方法无效才能研发”的门槛。
- 可参考的机制覆盖正常时序预测、重建、单类表示、概率模型、跨模态注意力、图相关性、自监督/对比学习和统计变化检测；这些是设计素材，不是当前选型或新颖性结论。
- 数据公开、系统开源、代码开源是三件事；“unsupervised”也不自动等于我们的Normal-only、整Case隔离和独立校准协议。

## 2. 核心文献与机制台账

以下以原报告的名称/年份定位；完整标题、作者与更详细转述见原件“核心文献逐篇记录”。没有从原报告获得的字段不自行补全。

|工作|实际任务/粒度|报告所述模型与机制|训练与数据条件|值得保留的信息/未知项|
|---|---|---|---|---|
|AnoFusion，KDD 2023|Service instance×time故障检测|指标归一化；日志BERT+DBSCAN后按窗统计；Trace耗时统计；异质模态图→GTN→GAT→GRU预测；绝对预测误差median/IQR归一化后取max；EVT阈值|训练假设正常模式，不用故障标签；GAIA与银行私有数据；分钟对齐|最接近实例时间窗任务的参照之一；图主要建实例内部模态关系，不能与跨服务调用图混同；协议不等同独立Case校准|
|MSTGAD，ASE 2023|实例/系统图异常检测|Microservice System Twin graph；Transformer空间与时间注意力|论文称semi-supervised；MSDS+不可公开第二数据|标签如何进入loss、完整层数与阈值未核实|
|Eadro，ICSE 2023|系统异常检测+根因服务定位|日志Drain/Hawkes；指标与Trace的DCC+attention；concat/FC/GLU融合；服务依赖图GAT；检测/定位双头|监督联合训练；Train-Ticket、SocialNetwork|模态分支、时序编码和融合机制可参考；原版不是Normal-only Service二分类；数据DOI对应关系待核实|
|ART，ASE 2024|AD+故障分流FT+RCL|Transformer抽通道依赖→GRU抽时间依赖→GraphSAGE抽调用依赖；预测偏差矩阵；AD聚合L1/EVT及delay window；FT聚类，RCL用实例偏差|自监督共享表示+不同无监督后处理；要求调用关系|依赖类型/顺序及任务分离值得记录；官方代码入口未确认，不能写为已可复现|
|DeepTraLog，ICSE 2022|Trace级二分类|Trace+Log→Trace Event Graph→GGNN→Deep SVDD；到正常中心的距离|单类目标；需要请求级事件关联|单类图表示机制；不是固定Service窗；原始解析、中心/阈值/划分实现待核实|
|TraceAnomaly，ISSRE 2020|Trace异常|Service-level Deep Bayesian Networks，学习调用路径和响应时间的正常概率规律|正常Trace模式；无监督|名称中的Service-level不代表最终输出Service窗；具体依赖与数据契约未知|
|ServiceAnomaly，JSS 2024|Trace异常及解释|正常Trace树合并CPG/DAG；六类profiling指标的线性/非线性关系；结构及关系匹配偏离|正常基线；TeaStore、Train-Ticket|关系模型与结构对照；不是Transformer/GNN，也不是Service窗检测|
|TraceGra，Computer Communications 2023|Trace异常|Trace Performance Graph；GNN+LSTM encoder-decoder；重建偏差|Trace拓扑+容器性能观测|结构/时间重建机制；代码和完整实验数据未确认|
|Meng等，FGCS 2021|执行Trace结构/性能异常|树编辑距离检测路径偏离，PCA分析组件执行时间|历史正常基线；Bench4Q、SocialNetwork|非深度结构与性能对照；未确认官方代码|
|Deep Attentive AD，ICWS 2022|多模态时序异常检测|报告确认方向，未完整核实架构|Train-Ticket实验|不要凭名称补出层数、loss、阈值或实现|
|Triple，IWQoS 2023|Trace/Metric/Log异常检测与解释|报告有统一图/时空、Deep-SVDD类目标线索，但不足以确认|准确数据、split与代码未知|保留论文身份线索；二手架构描述不能当确认事实|
|UAC-AD，TSC 2024|多模态异常检测|对抗式对比表示学习，关注正常hard samples|无监督；两套开放模拟数据+一套通信工业数据|可参考对比学习/难样本处理问题；公开数据名称版本与官方代码未确认|
|MADMM，NCA 2024|指标+日志滑窗检测|metric-GCN、log-GAT、Cross-Modal Attention GRU|依据摘要级描述|跨模态时序交互；数据、训练loss与代码未知|
|TraLogAnomaly，SCP 2025|Trace+Log混合事件序列检测|Hybrid Event Vector Sequences；Transformer；TF-IDF加权|全文训练/标签与阈值未核实|事件序列组织方式；代码/数据未知|
|BARO，FSE 2024|异常检测+RCA，论文重点RCA|多变量BOCPD变化检测→RobustScorer根因指标排序|统计方法；全系统时间×服务指标宽表|统计变化检测参照与重复故障数据；根因排序不能直接当Service二分类结果|
|Nezha，FSE 2023|细粒度RCA|多模态可解释定位；报告未给足完整架构|两套公开+一套构建数据，名称需核实|仓库自称basic implementation；不能当完整复现或直接AD基线|
|MicroRCA，NOMS 2020|性能RCA|指标增强拓扑图+personalized random walk|需要拓扑与指标|保留图排序背景，不与窗口检测性能混比|
|Practical RCL / TraceRCA，IWQoS 2021|RCA，内部有invocation异常检测|调用异常→association-based localization|Trace及多阶段处理|区分内部AD模块与最终定位任务|
|LO2，PROMISE 2025|API异常数据集论文|Light-OAuth2+Locust调用测试与采集，非主要新detector|API测试触发正确/错误调用标签|与独立Endpoint方向相关；不是三模态完整数据集|

### 中文资料与通用方法线索

- 陈文潇博士论文《基于深度学习的微服务异常检测与根因定位》（清华，2022）：报告区分BUZZ的KPI无监督检测、DCGM的监督因果/VAE定位、MetaRCA的元数据/因果注意力/GAT定位。BUZZ用Alibaba监控数据，DCGM用eBay数据；工业数据公开性与统一代码未确认。
- 于庆洋等《基于调用链控制流分析的大型微服务系统性能建模与异常定位》（软件学报，2022）：调用链性能建模与定位背景，细节/代码未知。
- 张齐勋等《基于服务依赖图的微服务系统故障根因定位方法》（2024）：配置/日志建依赖图并推断故障因果链，属于RCA。
- OmniAnomaly、USAD、PCA、IF、OCSVM作为通用时序/经典方法线索。外部报告没有对OmniAnomaly、USAD做同等深度记录，不能称本轮完成逐篇分析。

## 3. 数据集台账

**数值均为外部报告转述，未下载统计。** 格式栏的“线索”不代表所有原始文件格式已核实。不同论文使用同一系统，不等于共享相同数据集。

|数据/版本|公开性质（报告所述）|模态/格式线索|规模与实验单位|标签、划分与注意事项|
|---|---|---|---|---|
|GAIA / MicroSS，AnoFusion D1|公开|M/L/T；README有CSV与trace table线索|5服务、10实例；约两周；734,165 metric records、87,974,577 logs、28,681,438 traces|故障窗口占比平均4.908%；三个模态record口径不同，不直接横比；原始格式/版本仍需固定|
|AnoFusion D2|银行NDA，不公开|M/L/T，具体格式未知|14服务、28实例；约312万metric、1489万log、947万trace records|多种资源/网络/应用/VM故障；不能要求完整公开复现|
|MSDS v1.1|Zenodo公开|OpenStack M/L/T、工作负载与故障脚本；aggregated logs有CSV|concurrent约335.4MB，sequential约315.1MB|logs/metrics为CEST、traces为UTC，报告称需校正2小时；准确schema与许可证未知|
|MSTGAD第二数据|不可公开|M/L/T|未列充分统计|作者无发布许可|
|Eadro Train-Ticket|作者数据公开线索|M/L/T，具体落盘格式未明确|41服务/27业务；Normal约7h；162注入；48,296 traces|60/40时段划分；监督标签；DOI record尚未消歧|
|Eadro SocialNetwork|同上|M/L/T，格式未明确|21服务/14业务；Normal约1.2h；72注入；126,384 traces|同上；不要把注入数当作已确认独立完整Case数|
|ServiceAnomaly TeaStore/Train-Ticket|仓库声明含used dataset|Trace+六类profiling指标；格式未知|完整run/Case统计不足|正常CPG与故障Trace；原始/中间产物占比未知|
|DeepTraLog reproduction data|项目数据公开线索|Trace/Log及处理后图；具体schema未知|normal/fault分组，规模未知|不能将只使用处理图称为完整原始预处理复现|
|BARO artifact v0.0.1|Zenodo公开|Metrics ZIP；接口Pandas宽表，time及service_metric列|Online Boutique、Sock Shop、Train-Ticket；报告称每benchmark100 failure cases、fault×target重复5次|故障/根因metadata；最有价值的线索之一是独立重复口径|
|LO2数据|Zenodo公开|主要Logs+Metrics；485种metrics；原始格式需核实|报告称约657,000 log files、20亿以上日志行、4500万以上metric files|大规模数字尤其需要后续原件/实际包核实；API测试错误标签；Trace不足|
|LO2 replication package|公开，区别于数据包|Light-OAuth2/Locust/Prometheus/Jaeger等采集管线|不是另一份独立检测benchmark|存在Jaeger配置不代表有完整有效Trace数据|

### 原报告明确给出的数据定位符

- MSDS v1.1：`10.5281/zenodo.3549604`。
- BARO artifact：`10.5281/zenodo.11046533`。
- LO2 dataset：`10.5281/zenodo.14257989`；replication：`10.5281/zenodo.14229369`。
- Eadro DOI对应关系未确定，暂不填。

## 4. 开源项目定位台账（未运行，不是执行计划）

本表仅保存外部报告的定位信息，便于接收用户后续复核结果。commit、当前可访问性、完整性、许可证和可运行性未在本轮检查；没有精确SHA的条目不能作为论文固定实现版本。

|项目|报告给出的仓库URL|实现/输入线索|证据边界|
|---|---|---|---|
|AnoFusion|https://github.com/AIOps-Lab-NKU/AnoFusion|generate_channels.py、config.py、main.py；旧PyTorch1.6环境|D2不可得，公开数据到模型的完整路径未运行|
|MSTGAD|https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection|util/pre_MSDS.py、main.py；Python3.8/PyTorch1.12/PyG2.2|训练标签与图输入细节未核实|
|BARO|https://github.com/phamquiluan/baro|fse-baro；bocpd、robust_scorer、download_data|库示例、完整论文实验、Service窗比较是不同结论|
|ServiceAnomaly|https://github.com/M-panahandeh/ServiceAnomaly|ServiceAnomaly/、data/|缺清楚统一环境与入口线索|
|Eadro|https://github.com/BEbillionaireUSD/Eadro|codes/main.py、model.py、preprocess/align.py|监督双任务；数据DOI与许可证未知|
|DeepTraLog|https://github.com/FudanSELab/DeepTraLog|GraphData/、TraceLogData/、model/|README较简略，预处理与SVDD实现未实查|
|TraceAnomaly|https://github.com/NetManAIOps/TraceAnomaly|报告称master，旧依赖|没有固定论文SHA或运行记录|
|TraceRCA|https://github.com/NetManAIOps/TraceRCA|invocation检测与定位脚本线索|最终任务RCA，不能直接当二分类模型|
|Nezha|https://github.com/IntelligentDDS/Nezha|报告称MIT、basic implementation|覆盖论文多少步骤未知，许可证未在本轮复查|
|LO2 community|https://github.com/M3SOulu/LO2-Dataset-Community|数据扩展/采集相关，另有Zenodo包|不是现成Service detector；本地历史lo2-sample已丢失，与外部公开数据的可获取性分开|

原报告没有确认ART、Triple、UAC-AD、MADMM、TraLogAnomaly、TraceGra、Deep Attentive AD的官方完整实现。保留为文献机制线索，不写成“确定无代码”，也不以此否定其研究参考价值。

## 5. 可带入brainstorming的机制素材

本节是对报告的机制归类，**不提出新模型、实施计划或已成立的创新点**。

|机制|文献线索|后续讨论时可问什么|
|---|---|---|
|正常时序预测与残差|AnoFusion、ART|历史上下文、分通道误差与聚合方式分别起什么作用？|
|重建式检测|TraceGra；USAD类别线索|重建对象是窗口、Trace还是图？迁移后任务是否变化？|
|单类/概率表示|DeepTraLog、TraceAnomaly|正常中心距离或likelihood如何形成分数？依赖哪些身份/事件组织？|
|多模态分支与融合|Eadro、AnoFusion、MADMM|特征、表示、分数层融合如何区分？额外模态与机制增量如何分别比较？|
|多维依赖建模|ART、MSTGAD|通道、时间、跨服务依赖分别需要什么输入？模态图不等于调用图。|
|对比学习与hard samples|UAC-AD|正负对如何构造、是否需要异常标签？原报告尚不足回答。|
|结构与指标关系基线|ServiceAnomaly、Meng等|正常关系偏离如何评分？是否局限请求级Trace？|
|变化检测与稳健统计|BARO|前级变化检测和后级根因排序能否分别理解、评价？|
|日志/Trace混合事件序列|TraLogAnomaly、DeepTraLog|是否能建立请求级关联？与固定Service窗表示差在哪里？|

本项目允许成熟模型适度改进；这些素材不需要先通过旧报告的高新颖性门槛。是否采用、怎样简化、能否优于IF/OCSVM与强规则，留给后续设计和公平实验。

## 6. 与当前数据及既有研究的衔接

1. 当前输出仍是Service×15s分数/二分类，不因某论文使用Trace、system score或RCA而更改任务。
2. 当前Normal-only、独立Normal校准、整Case划分保持。Eadro监督目标、MSTGAD未明标签语义、各论文EVT/阈值不能直接合并成相同训练协议。
3. 资源表已有R3；Log/Trace原始数据存在，但正式特征契约、模态对齐与质量语义尚需设计。公开论文的padding不能自动变成我们的Unknown补0。
4. 当前调用边不完备；引用图方法只代表有机制可学习，不代表已具备完整图输入或必须先补齐图才能做模型。
5. 此报告不新增本项目实验结果，不覆盖已完成的资源基线，也不改变旧研究中资源/执行互补、IF失败和Normal参考变化的事实。
6. 完整标签并非轻量原型启动门槛；后续评价按实际证据范围说明，不能把曝光代理和全服务真实状态混同。

## 7. 当前保留的未知与后续结果接收方式

- 网页报告的引用映射、三份CSV附件未提供；正式引文和完整结构化原表仍不足。
- 部分架构仅有摘要或二手线索，尤其Triple、UAC-AD、MADMM、TraLogAnomaly、Deep Attentive AD；不能将推测升级成事实。
- 数据准确版本、原始schema、许可证、Normal/异常划分、阈值是否看测试标签、point adjustment、容忍窗口等很多字段未明。
- 未有任何项目的本轮clone、依赖安装、运行或复现证据。用户另行开展的复核可能不完整，不作为本次归档或后续讨论的硬门槛。
- 接收后续复核时可单独增量记录“原报告主张→实际证据→确认/修正/未知”，引用具体仓库SHA/文件或原文位置；本次原件保持不改。
