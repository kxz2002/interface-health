# 外部Agent复核：增量结论与首轮修正

接收日期2026-10-19。来源为用户提供的[完整合集](followup-source-collection.md)，hash与未随附证据见[source-provenance.md](source-provenance.md)。本轮只阅读/归档/提炼，不联网复查、clone或执行合集T01–T24。

**证据层级：以下“源码发现”“实际运行”“下载校验”都是外部Agent报告的工作，本仓库没有重做。** 合集中未内嵌的stdout、源码快照、patch、manifest、CSV等未随附，无法在本仓库逐条复核。固定SHA只界定报告审计的版本，不推及作者本地实现、其他分支或论文结果本身。

## 1. 最值得保留的变化

1. 首轮“有官方代码/有数据/README有入口”不能等同完整可运行链。AnoFusion、Eadro、DeepTraLog、ServiceAnomaly均有数据预处理或入口/评价缺口，机制仍可参考，无需把重跑其原仓库作为我们的研发前置条件。
2. “unsupervised”需细分。报告确认DeepTraLog的正常样本单类目标；MSTGAD分类头实际使用已知异常标签；Eadro是监督双任务。不能将三个原版模型直接并入同一Normal-only比赛。
3. 论文评分、公开实现评分、阈值与评价是独立环节。AnoFusion和TraceAnomaly可见代码的二值决策链不完整；借鉴架构时需要明确我们自己的独立Normal校准，不能声称原版完整复现。
4. BARO是外部报告中最明确的成功执行案例，但大规模结果主要是预存变化点的RQ2/RQ1 replay。它提供可参考数据与统计机制，不提供我们的Service窗性能证明。
5. MSDS原始时间字段和源码转换与“统一减2小时”的早期描述冲突；不能直接沿用首轮时区结论。LO2的host指标/API负向测试语义也得到更清楚区分。
6. 这些事实用于减少不必要的移植成本；当前硕士论文尺度、成熟模型适度改进与公平超越基线的目标保持。

## 2. 项目、固定版本与外部执行范围

完整细节见原合集相应T编号；表中SHA为外部报告记录，不是本轮git查询结果。

|工作|报告固定的代码版本|外部报告的实际执行范围|关键新增发现|
|---|---|---|---|
|AnoFusion（T01）|`ae3c4ea98509008844a3eecf7c0bd1529faaba67`|clone/样例审查/入口尝试；未训练|README预处理命令退出0但不调用生成函数；样例仅19行且缺第二段metrics；目录单复数不一致；Trace统计写入疑点；硬编码checkpoint/GPU；公开仓库与论文所链旧仓库身份仍未闭合|
|MSTGAD（T02/T09）|`e78201a1890d1988fcb17c7f94e464815e567c7b`|真实MSDS短片段预处理原样失败，两行兼容补丁后成功；458窗；未训练|半监督分类/重建标签加权；label.pkl另行预打包，不由raw脚本生成；代码70/30与论文60/10/30不同；train-F1选checkpoint，不应误写test-F1选模；硬CUDA|
|BARO（T03）|tag0.2.0：`582d4ead3d3301a34fea4e395bfa3f750fc9a40d`|library例成功；三系统RQ2共300 Case replay；RQ1缓存replay|RQ2也读naive_bocpd.json再计算排序；不是fresh BOCPD；论文与后期artifact的RQ1数字不同，需分版|
|ServiceAnomaly（T04/T11）|`6acc9dfd7ec50d3007d65e61a188f40d72b422b6`|代码/数据审计和失败入口尝试；未完整运行|GUI/Windows/在线Jaeger-Prometheus/人工输入依赖；缺统一自动P/R/F1；baseline训练残差RMSE作cutoff；公开raw baseline不足论文报告数量|
|Eadro（T05/T11）|`82ff9c9a7a58a9cd2b713aa78754a73c543a15db`|两ZIP下载校验/源码审计；入口缺tick/torch；未训练|raw→parsed转换缺失；监督检测+定位；默认重叠窗随机切分、全run标准化、test HR@1选checkpoint；评价TN计入TP及多个静态代码问题|
|DeepTraLog（T06）|`04376dcb99d4b0fd0c50850255c2c4f9ec9503c1`|固定SHA源码及一个raw ZIP检查；未训练|raw→Drain/TF-IDF/GloVe/TEG生成未找到；GGNN读取edge_index却不使用保存的edge_attr；normal-only SVDD实现可追；未固定split seed|
|TraceAnomaly（T07）|`946e44791b7c19eddd61b0a7fa98e54100e1c03a`|发布向量ZIP统计与入口尝试；未训练|模型为VAE+posterior flow，输出likelihood；不生成raw→890维STV；论文KDE/p-value阈值未在公开代码实现|
|TraceRCA（T07）|`b9087c72d201fbb17d724114cd389de4854815ad`|静态多阶段流程/外链/失败入口检查|invocation统计AD、监督trace RF/MLP、RCA是不同分支；特征选择用test batch无标签分布；缺数据/辅助脚本|
|Nezha（T08）|`d8140101fdb4e7dfb60d3ef9f64706f382b68470`|固定源码/数据tree/schema与失败入口尝试；未完整RCA|basic implementation含主要事件整合、Drain、模式挖掘/排序模块；两套增强系统，construct_data不是第三系统；alarm描述见下方内部差异|
|LO2（T10）|replication关联commit：`4f0c304942dba0797348c4b52119762cca8ebbcd`|小包下载校验/sample ZIP范围读取；未全量raw/容器运行|原版是host-level metrics、API负向测试标签；sample无Trace，full多header-only；reduce_logs浮点slice错误；新增LO2v2单独记版|
|Triple（T12）|候选仓库`56df327974c1a02d808262381df5180b8d33ad09`|摘要与候选树核查；无可运行实现|摘要支持STGCN+Deep SVDD；仓库只有README/图片PDF，官方归属未确认；Normal-only等正文细节仍未知|

### BARO结果只作为外部执行记录

- 外部报告RQ2的OB/SS/TT Avg@5为0.86/0.95/0.81，合计300个故障Case；任务是根因排序。
- RQ1 `--saved` 对OB/SS/TT的P/R/F1为.76/1/.87、.79/1/.88、.83/1/.90。报告解释其对应后期artifact版本，不是原论文Table2，也不是fresh检测算法重算。
- fresh全量BOCPD、RQ3/RQ4未执行。不能将这些数字加入本项目Service×Window模型结果表。

## 3. 数据集身份、格式与首轮修正

|数据|新增定位/格式信息（均为外部报告）|与首轮相比需要保留的修正|
|---|---|---|
|GAIA/MicroSS|报告固定GAIA tree `238a9e24ae0b69e7e05dff112652a3fa62db88fd`；Git pack曾下载4.62GiB后删除，未解包/训练|“下载过Git对象”不等于读过完整三模态；AnoFusion样例不能代表完整数据；数据具体许可仍需核实|
|MSDS v1.1|version DOI `10.5281/zenodo.3549604`，concept `.3484800`，CC BY4.0；五主机CSV指标、aggregated log CSV、嵌套Trace JSON；concurrent ZIP335,390,038B、解压约4.72GB|日志实含+01:00，metric标CEST，源码−1h且依赖本机TZ；与网页“−2h”矛盾。31个Rally hook可与log/metric连接，但不是全量逐pod状态真值；Trace/终止时间仍未闭合|
|Eadro v1|concept `.7615393`、version `10.5281/zenodo.7615394`；SN/TT ZIP MD5分别`4c523faf64770b1e9a709cdd610e07c4`/`2d77861e4a025781e598d51866fd6f4e`|DOI差异已由外部报告解释为concept/version，不是两个不同数据；open download但许可未明。raw spans/log JSON、7指标CSV与parsed中间文件不同；SN36/TT81 fault metadata rows不能直接等同论文72/162 injection operations|
|ServiceAnomaly作者数据|1718文件、42,331,839B；Jaeger raw/dependency、Prometheus JSON及关系文本，部分JSON双重编码|TeaStore baseline只有100 raw traces，论文称3000；TT公开baseline是三个metric snapshots和依赖图，非论文5000 raw traces；测试目录数相近但缺失/空值/损坏，不能称完整论文数据|
|DeepTraLog|raw Span CSV有Service/TraceId/SpanId/ParentSpan/IsError等；日志有SkyWalking SW_CTX；公开raw、processed graph、results分开|仅一个F05.zip小包实际检查，未核实全部论文记录数；请求级日志-Span关联是其输入前提，不能因我们有日志就默认同等契约|
|TraceAnomaly向量数据|3ZIP分别100000 normal train、30355 normal test、5079 abnormal test；每行trace_id+890维文本向量|确认的是model-ready向量，不含已核实raw-span编码器；缺失映射−1是其数据特定规则，不能迁为健康填值|
|BARO v0.0.1|data DOI `10.5281/zenodo.11046533`，CC BY4.0；OB/SS/TT各100Case，20 target×fault组合各5次；data.csv宽表及inject_time/cached changepoints|外部报告实查了独立重复与三包checksum；可作为外部数据线索，不自动授权我们下载/跑实验|
|Nezha|两个增强系统：Augmented-OnlineBoutique/TrainTicket；construct_data正常建模阶段、rca_data故障阶段；CSV明确含TraceID/SpanID关联|纠正首轮“两套public+一套constructed”可能被理解为三个独立系统；实际两系统两阶段。TT总45故障，公开标签21资源+24代码与论文20+25不同，原因未知|
|LO2原始系列|concept `.14257989`、报告所查v3 `.14938118`，CC BY4.0；replication version `.14229370`，Apache2.0|原版Metrics是host级node_exporter。485 metric names≠1124 series features≠约4550万raw文件；sample CSV1127列=3索引列+1124特征、172行/54test；按API/test标签，不是服务健康标签|
|LO2v2|另一concept `.16982695`，报告version `10.5281/zenodo.18937117`（2026-03-10），115run×54test、cAdvisor container metrics|与原论文1740-run系列区分，不能把新数据/新随机顺序和容器指标算作旧论文复现；本轮未查其当前最新状态|

Eadro包raw schema更清楚：日志是service→raw string列表，span是Jaeger list，指标CSV含timestamp与cpu system/total/user、memory usage/working set、rx/tx七列。该格式信息可保留用于理解文献，不意味着要把它移入当前selected17。

## 4. 架构、目标函数与阈值的有用细节

- **DeepTraLog的日志表示比首轮更明确**：论文Drain事件→清理/分词→TF-IDF加权300维GloVe；Trace Event Graph→GGNN+soft-attention readout→soft-boundary SVDD。正常中心初次forward均值、半径按训练距离分位数更新，score=`||h-c||²−R²`。这提供了单类学习与可学习聚合的参照，不等于选择完整Trace图。
- **DeepTraLog代码差异**：报告称四类关系edge_attr未用于GGNN，训练正则与论文λ设定不同。借鉴机制时以明确的新实现/协议命名，不能静默改造后称原论文完整复现。
- **AnoFusion代码差异**：报告称5层GTN、GAT/GRU预测下一时刻MSE；可见归一化是减median后除P25，而首轮论文描述为median/IQR。阈值EVT在公开入口未找到。不能把这两种分数定义混作同一实现。
- **MSTGAD标签使用**：known/unknown标签与分类头、标签加权重建有关，不能因有重建项就归为Normal-only。458窗smoke不是458个独立Case。
- **Eadro融合机制仍可借鉴**：分模态编码、GLU及依赖图是机制素材；原监督双头/测试反馈/评价实现不应照搬。
- **TraceAnomaly**：源码为2×500 dense encoder/decoder、Gaussian latent、posterior flow likelihood；论文阈值normal KDE p=.001未在源码闭合。公开CSV中的label是真值而非prediction。
- **Triple**：摘要可提升为STGCN+Deep SVDD线索；具体normal-only协议、中心/半径/阈值/图粒度未确认，继续Unknown。

## 5. 需要保留的外部报告内部差异

本合集不是逐项无误的权威结论，以下不能自行裁决：

1. **Nezha alarm**：T08模块表写成功率/P90对正常统计的3σ触发；T20进一步称有效determine_alarm用CPU/Memory>80及其他指标>200/300等硬阈值，历史3σ比较被注释。更细源码指向可作为线索，但本轮未取得源码，记录为“论文/高层叙述与有效路径存在差异，待独立确认”。两种描述不混写。
2. **AnoFusion下载**：报告既写实际下载4.62GiB Git pack，也写未下载完整数据；解释为未checkout/解包/使用全量语料，不能概括成“未发生大下载”。
3. **LO2体积**：同v3材料450GB与540GB uncompressed口径冲突，保留；sample整包MD5未本地重算，Range片段不等于整包校验。
4. **审计完成与复现完成**：T04–T08等勾选完成是审计范围完成，不能说模型都跑通；缺torch的入口失败本身不证明模型不可运行，其他源码/数据缺口是独立观察。
5. T13–T19与全局T24仍未完成；ART/UAC-AD/MADMM/TraLogAnomaly/TraceGra等不能因为合集回来就升级为完整验证。

## 6. 对当前brainstorming的含义（设计推论，不是文献实验证明）

- 用户已选择资源+执行+日志，采用本地文本编码器，三个日志层次（计数/TF-IDF、语义简单池化、改进聚合）共同实现；不把长序列建模当主干前提。训练设备独立，最低RTX2060，显存未核实。
- 可用正常窗口学习重建或单类表征，再用独立Normal校准，而不是使用异常标签训练普通分类头。具体目标函数尚未选定。
- 本地日志画像（仅/tmp，尚未写入仓库产物）显示拟合期抽样片段窗口内消息冗余约74.74%，且六服务无拟合期Spring事件，支持讨论频次保留的聚合与模态可用性；它不证明注意力、语义或融合增益。
- 两个主要改进应独立开关：改进日志聚合×改进融合的2×2消融，另保留模态消融、日志表示对照与IF/OCSVM等同输入对照。并行实现三层不意味着按测试故障反复挑最好模块。
- 正常数据上学习的attention/gate权重不自动等于异常证据强度；若log在Normal期未见或异常信号只在单模态，融合需要明确评分/不可用策略，不能把无日志填为健康，也不能保证门控天然解决这些问题。
- 研发目标仍是可信硕士论文原型与公平增益，不要求逐篇原仓库跑通或达到顶会新颖性后才开始。下一步继续在对话中明确检测目标函数、模块输入输出与消融，然后形成可审阅设计。

## 7. 如何引用本补充

对外论文引用应回到原论文/固定版本代码；本文件是内部知识台账。可以写“外部Agent报告在SHA X观察到Y”，不能写“我们已验证Y/已复现论文”。后续若补充原始日志/patch/源码，可逐条升级证据状态并保留此次归档原件。
