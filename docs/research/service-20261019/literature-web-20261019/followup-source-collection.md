# 微服务可观测性文献调研：已产出报告合集版

**整理日期：2026-10-19（Asia/Taipei）**
**用途：** 将工作区中已产出的主要审计报告与当前主任务清单合并为单一 Markdown 文件，便于一次打开、搜索、复制与归档。
**范围说明：** 下文包含主任务清单及 T01、T02/T09、T03–T08、T10–T12、T20、T23 的完整报告正文。原报告内容按来源保留，仅为合集目录将标题层级整体下调一级；未补写未验证事实，也未把部分审计表述成完整复现。

## 快速状态总览

|任务|状态|报告/说明|
|---|---|---|
| T01 AnoFusion | 部分完成 | 公开 GAIA 审计；未完整下载/解包并运行模型。 |
| T02 MSTGAD/MSDS | 部分完成 | 子集预处理与 458 窗口 smoke；未训练。 |
| T03 BARO | 公开 artifact replay 已完成（范围有限） | RQ2 300 cases 与缓存 RQ1 replay；fresh BOCPD、RQ3/RQ4 未运行。 |
| T04 ServiceAnomaly | 公开审计完成，论文复现受阻 | 入口/数据/评价核验；未完整运行。 |
| T05 Eadro | 公开审计完成，训练未运行 | 数据/代码/协议检查；依赖和代码问题阻塞。 |
| T06 DeepTraLog | 公开审计完成，训练未运行 | 数据契约、one-class 协议和代码检查。 |
| T07 TraceAnomaly/TraceRCA | 公开审计完成，完整流程未运行 | 入口依赖/数据准备阻塞。 |
| T08 Nezha | 公开审计完成，完整 RCA 未运行 | 数据/标签/代码审计。 |
| T09 MSDS data/labels | 部分完成 | Rally hook 与日志/指标时间窗部分关联；完整 trace 和逐行/pod 标签未验证。 |
| T10 LO2 | 公开身份/schema 审计完成 | 未下载全量 raw 或运行完整容器栈。 |
| T11 ServiceAnomaly/Eadro identity | 公开数据身份审计完成 | 不代表完整数据或模型复现。 |
| T12 Triple | 部分完成 | 摘要与候选仓库核验；正文级未知字段保留 Unknown。 |
| T13–T19 | 待执行/待补证 | 文献全文、实现及协议待逐项核验。 |
| T20 threshold/evaluation | 部分完成 | 已覆盖 T01–T08、T10；T12–T19 尚未完整覆盖。 |
| T21–T22 | 未完成；私有数据部分暂缓 | 遵守不分析用户私有数据的要求。 |
| T23 API/trace/service-window | 公开部分完成，本地关系暂缓 | 未检查本地私有 client/trace/log 信号。 |
| T24 reproducibility ledger | 部分完成 | 全局归档完整性仍待补齐。 |

## 目录

[主任务清单与状态](#tracker)
[T01 AnoFusion](#t01)
[T02 / T09 MSTGAD 与 MSDS](#t02-t09)
[T03 BARO](#t03)
[T04 ServiceAnomaly](#t04)
[T05 Eadro](#t05)
[T06 DeepTraLog](#t06)
[T07 TraceAnomaly / TraceRCA](#t07)
[T08 Nezha](#t08)
[T10 LO2](#t10)
[T11 ServiceAnomaly / Eadro 数据身份](#t11)
[T12 Triple](#t12)
[T20 分数、阈值、评价矩阵](#t20)
[T23 Endpoint/API 任务分轨](#t23)

## 报告正文合集

<a id="tracker"></a>
## 主任务清单与逐项状态（完整原文）

> 来源文件：`microservice_literature_followup_tasks.md`。以下为完整正文；标题层级为合集显示下调一级。

## 微服务可观测性文献调研：后续复核与执行任务清单

> **用途**：把前一轮报告中明确标注为“待核实、需复核、未运行、需进一步排查”的事项拆成可逐项执行的任务。审阅完成后，可按编号逐项展开。
>
> **当前状态**：2026-10-19 起逐项执行。T01（AnoFusion × 公开 GAIA）仍部分完成，见 `anofusion_open_gaia_initial_audit.md`；GAIA Git pack 曾下载 4.62 GiB，未解包/运行，随后已删除。T02（MSTGAD × MSDS）和 T09（MSDS 数据契约）仍部分完成，详见 `mstgad_msds_execution.md`，包含单样本预处理 smoke test、窗口生成及缺 PyTorch/PyG 等阻碍。T03（BARO）公开 artifact 核验、三 benchmark 的完整 RQ2 artifact replay 及缓存 RQ1 replay 已完成，执行记录见 `baro_execution.md`，原始 stdout/运行记录见 `baro_execution_logs/`；BOCPD 耗时重算、RQ3/RQ4 未运行。T04（ServiceAnomaly）仓库/数据/评价协议审计及入口失败核验完成；论文端到端复现仍受依赖、GUI/OS 服务、数据缺口和无自动 evaluation 阻塞，详见 `serviceanomaly_execution_audit.md`。T05（Eadro）已核对 concept/version DOI、实际下载 MD5、raw/parsed schema、监督流程与多处代码/评估阻碍；入口分别因 `tick`、`torch` 缺失停住，未训练，详见 `eadro_execution_audit.md`。T06（DeepTraLog）已核验论文/项目页、固定 SHA 的关键源码、公开 raw 样本、数据目录大小及 GraphDataset 契约；入口因缺 PyTorch 停止，未重建 TEG 或训练，详见 `deeptralog_execution_audit.md`。T07（TraceAnomaly/TraceRCA）和 T08（Nezha）已完成代码/数据/论文覆盖与运行入口审计；旧依赖、外部数据或当前环境阻止完整运行，未声称论文结果复现，详见 `traceanomaly_tracerca_execution_audit.md`、`nezha_execution_audit.md`。T10（LO2）已解析 concept/version DOI、校验 replication/scripts ZIP，借助 HTTP byte-range 实检 sample metrics CSV/log、485 metric names、test labels 与 trace coverage 缺口；未下载 46 GB full raw 或运行 Docker 全栈，详见 `lo2_execution_audit.md`。T11（ServiceAnomaly/Eadro）已完成公开数据 identity、版本/校验值/许可状态与论文实验范围对照，详见 `serviceanomaly_eadro_data_identity.md`；未声称完整论文数据或模型复现。T23 已完成公开 API/trace/service-window/RCA 单位分轨；本地信号关联因私有数据约束暂缓，详见 `endpoint_vs_servicewindow_taxonomy.md`。T20 已对 T01–T08、T10 完成 score/threshold/labels/evaluator 的逐项矩阵，T12 仅完成摘要/候选仓库核验，T13–T19 尚未逐项审计，整体仍部分完成，详见 `anomaly_score_threshold_evaluation_matrix.md`。T12 记录在 `triple_iwqos2023_evidence_audit.md`。不要把“网页可访问/README 有命令”记成“已复现”。
>
> **范围约定（用户要求）**：暂不评估或分析需要用户私有数据集的事项；只使用可公开获取的数据集。T21、T22 中涉及用户本地数据的部分暂缓，不下载、索取或推断私有数据。论文/公开数据层面的任务继续执行。
>
> **优先级**：P0 = 直接影响优先项目能否运行或数据能否正确解释；P1 = 核心证据/实现缺口；P2 = 扩展文献核查；P3 = 本地任务映射与补充检索。

### 一、优先复核公开实现与运行路径

#### [~] T01｜AnoFusion：源码实现审计与 GAIA 最小运行（P0；已部分执行，模型复现未完成）
- **本轮执行记录**：初步源码/数据审计已写入 `anofusion_open_gaia_initial_audit.md`；已固定源码 SHA 并尝试 README 命令。公开样例缺第二段 metrics，预处理命令无实际生成逻辑，训练入口缺 PyTorch；GAIA 仓库 pack 下载后发现达 4.62 GiB，未解包/运行并已删除大体积克隆。后续剩余项：核实论文链接仓库身份、评估是否继续获取/整理完整公开 GAIA、在必要时用隔离环境完成修复前后复核。
- 固定官方仓库的 branch、完整 commit SHA、工作区状态，并检查 README、依赖文件及数据入口。
- 对照论文核实是否实现：metrics/logs/traces 序列化、BERT/DBSCAN 日志处理、trace 响应时间统计、异质图边类型、GTN、GAT、GRU、预测误差归一化/聚合及 EVT 阈值。
- 追踪训练、评估与阈值计算代码，确认标签只用于评估，且 normalization、通道构建或阈值校准没有读取测试标签。
- 使用公开 GAIA 数据验证 README 路径；记录缺文件、格式不匹配、兼容性问题及必要 patch。银行 D2 不公开，不把它列为可复现数据。
- **完成证据**：源码位置/论文组件对应表；实际命令及 stdout/stderr；环境、配置、输出和是否成功的明确结论。

#### [~] T02｜MSTGAD：MSDS 预处理、监督语义与最小运行（P0；公开子集 smoke test 完成，全量/训练未完成）
- **本轮执行记录**：详见 `mstgad_msds_execution.md`。MSDS concurrent ZIP 下载并核对 MD5/CC BY 4.0；25 条 trace JSON + 短时间片通过两行临时兼容补丁完成预处理，原始脚本在 trace timestamp split 报错；458 个窗口构造成功。训练因无 PyTorch/PyG 未运行。已发现 repo 仅按 70/30 切分而论文写 60/10/30，`label.pkl` 也不是预处理脚本生成。
- 固定仓库 SHA，检查 `util/pre_MSDS.py` 到模型输入的完整数据流，并与论文中的 twin graph、spatial attention、temporal attention 对照。
- 查明“semi-supervised”具体如何使用标签：哪些标签进入训练、损失、验证或阈值选择；不要只凭论文摘要推断。
- 核对 MSDS 原始 schema、许可证、可下载文件及预处理后的样本/标签格式。
- 实测 logs/metrics 的 CEST 与 traces 的 UTC 时差处理是否落实，检查转换后时间戳对齐。
- 尝试官方预处理与训练入口；第二套未获发布许可的数据单独记为不可复现限制。
- **完成证据**：数据字段/时间处理说明、标签使用追踪、运行日志与真实输出；如失败，记录最小复现阻碍。

#### [x] T03｜BARO：区分库级 smoke test 与论文 artifact 复现（P0；已完成公开数据核验和 RQ2 全量 artifact replay）
- **执行记录**：详见 `baro_execution.md`；固定 Zenodo software v0.2.0 / Git tag `0.2.0`, commit `582d4ead3d3301a34fea4e395bfa3f750fc9a40d`。完成 README library smoke test，sample shape `(721, 48)`、anomaly indices 与 Top-5 和 README 一致。
- 从 Zenodo dataset DOI `10.5281/zenodo.11046533` 下载并核验三 ZIP 大小及官方 MD5；各归档实查到 100 case、20 target×fault 组合、每组合 run 1–5。所有 case 临时解包并运行 RQ2：Online Boutique/Sock Shop/Train Ticket Avg@5=`0.86/0.95/0.81`，与论文相同；Online Boutique CPU/MEM/DELAY/LOSS=`0.91/0.96/0.95/0.62`，与 README 一致。
- `--saved` RQ1 使用打包的 `naive_bocpd.json`，复算出 OB/SS/TT P/R/F1=`.76/1.00/.87`、`.79/1.00/.88`、`.83/1.00/.90`。与论文 Table 2 不同，但 tag README 明确记载后期 artifact 的输出略优于论文 Table 2；此项是缓存 change-point replay，不是重新计算 BOCPD。未执行耗时版 fresh RQ1、RQ3 或 RQ4，因此不声称完成全文实验复现。
- **完成证据**：版本、DOI、CC BY 4.0、大小/MD5、运行环境/退出码/耗时/采样 RSS、所有 RQ2/RQ1 replay 数字、限制和原始日志均在 `baro_execution.md` 与 `baro_execution_logs/`。

#### [x] T04｜ServiceAnomaly：仓库内容、论文数据对应关系与运行入口（P0；审计完成，端到端复现受阻）
- **执行记录**：`serviceanomaly_execution_audit.md`。官方 GitHub `main` commit `6acc9dfd7ec50d3007d65e61a188f40d72b422b6`，工作区 clean；无 requirements/environment lock/运行手册或 LICENSE 文件。已对照论文追踪 CPG、Prometheus metrics、线性/非线性拟合、threshold、test relation matching 及样本级评价路径。
- 数据树共 1,718 文件 / 42,331,839 B。TeaStore baseline raw trace 只有 100 条 vs 论文描述 3,000；TrainTicket baseline 是 3 个 metrics snapshots，附 dependencies graph 而非论文声称的 5,000 raw traces。测试目录数量表面对应 15+15 / 32+32，但部分 raw traces 缺失、metric result 为空，且一条 TrainTicket fault relation list 语法损坏。TeaStore 当前实现路径只使用 5 个 metric，throughput 文件全空并被排除；两个项目无自动 trace-level P/R/F1 evaluator。
- 入口实测：`python main.py` 与 `mainTeastore.py` 在缺 `netgraph` import 时退出 1；`CPG-generation2.py` 因无显示器退出 1。另发现后续 `Tools.scripts.dutree` 也缺失，主流程依赖 GUI、Windows `ctypes.windll`、Live Jaeger/Prometheus 和人工输入/路径选择。
- **判定**：代码/数据/协议审计完成；原样入口未能启动，公开数据有覆盖/格式限制，论文结果不可从当前仓库直接自动复算。未修补、未安装依赖、未声称复现。

#### [x] T05｜Eadro：数据 DOI 身份与监督式端到端复现条件（P0；公开审计完成，训练受阻）
- **执行记录**：详见 `eadro_execution_audit.md`。GitHub `main` commit `82ff9c9a7a58a9cd2b713aa78754a73c543a15db`。实际下载 Zenodo SN/TT archives 并核对官方 MD5：`4c523faf64770b1e9a709cdd610e07c4` / `2d77861e4a025781e598d51866fd6f4e`；concept DOI `.7615393` 与 version `v1` DOI `.7615394` 的关联已确认。Zenodo `access_right=open`，metadata 未给 license，故不推断再分发许可。
- 核查 raw→parsed 断点：Zenodo raw `spans.json`/`logs.json`/metrics CSV/fault JSON 与 `align.py` 期待的 `records*.json`、`logs*.csv`、`templates.json`、预聚合 `traces*.json` 不同；没有相应 Drain/raw-span/parser 文件。源码实现的 log Hawkes、metric/trace DCC、GATv2、fusion 和 detector/localizer 是有监督双任务；但存在未定义 dropout、head 名称不匹配、缺 NumPy import、dataset 缺 `edges` 参数等静态错误。
- `align.py` 默认 10-step、步长 1 的重叠窗，默认随机 70/30 split；论文写不同时段 60/40。全数据先标准化再拆窗，且训练期间按 test HR@1 选 checkpoint。`BaseModel.evaluate()` 的 detection TP 计数含正常 TN，指标实现不等价常规 anomaly P/R/F1。
- 实际入口：`python align.py --name A_TT` 因无 `tick` 退出 1；README 形式 `python main.py --data A_TT` 因无 `torch` 退出 1。完整训练/论文结果均未运行。
- **判定**：DOI、数据结构/公开可用性、源码方法/监督及协议审计完成；原样端到端训练不可复现（缺 raw parser/环境，且源码和 evaluator 有硬错误）。不声称复现。

#### [x] T06｜DeepTraLog：从原始 trace/log 到 TEG 的完整链路（P1；源码/数据契约审计完成，训练未运行）
- **执行记录**：`deeptralog_execution_audit.md`。官方 `FudanSELab/DeepTraLog` `main` HEAD `04376dcb99d4b0fd0c50850255c2c4f9ec9503c1`，无 tags/license。未全量 clone（GitHub API size 1,057,012 KB）；以固定 SHA 获取源码快照 `/home/user/deeptralog_repo/model/`，并检查小型 `F05.zip` raw sample。
- 作者页明确 `TraceLogData` 是 raw、`GraphData` 是 processed graph、`results` 是实验结果。GraphDataset 只读取已有 graph JSON/embedding CSV 并存 `processed.pt`；未找到 Drain、raw span→span-event、TF-IDF/GloVe、TEG graph-builder 脚本。训练入口硬编码 `../gae/data/`，但仓库目录/README 未给出对应路径准备说明。
- 确认 one-class/normal-only protocol：60% normal train、10% normal validation、其余 30% normal + 全部 anomalies test；center 初始化、soft-boundary radius、score `dist-R²`，`>0` 判异常与论文大体对应。评估入口实测因缺 `torch` 退出 1。
- 实现差异/限制：GGNN 不使用 dataset 保存的 `edge_attr`；SVDD loss 的正则实现/数值与论文 lambda 描述不同；`np.float` 与当前 NumPy 2.3.5 不兼容；requirements 的 PyTorch Geometric 版本和论文环境不同且依赖不完整。未训练、未重算论文指标。
- **判定**：数据/代码/监督与划分审计完成，raw-to-TEG及端到端复现受阻；不声称论文结果已复现。

#### [x] T07｜TraceAnomaly 与 TraceRCA：旧仓库版本、依赖和脚本顺序（P1；公开实现/数据与运行条件审计完成，完整训练受阻）
- **执行记录**：详见 `traceanomaly_tracerca_execution_audit.md`；TraceAnomaly HEAD `946e44791b7c19eddd61b0a7fa98e54100e1c03a`，TraceRCA HEAD `b9087c72d201fbb17d724114cd389de4854815ad`。TraceAnomaly 3 个 TrainTicket 向量 ZIP 核实 SHA-256/MD5、行数（100,000 train / 30,355 normal test / 5,079 abnormal test）及 890 维；代码输出 likelihood score，但未实现 paper KDE/p-value threshold 或 root-cause stage。TraceRCA 记录了外部数据链接状态、Makefile 手工数据与缺脚本、feature-selection → invocation AD / supervised trace classifier → association RCA 的分支顺序；清华数据链接返回不存在，OneDrive 当前自动化访问 403（不可判定失效）。
- **实跑**：TraceAnomaly `python -m traceanomaly.main --help` 因缺 TensorFlow 退出；TraceRCA `python run_selecting_features.py --help` 因缺 `loguru` 退出，`make -n target` 因预期路径/数据/多个未跟踪辅助脚本失败。stdout/stderr：`traceanomaly_smoke.log`、`tracerca_smoke.log`。
- 未安装旧依赖、未修改源码、无 patch diff；不声称训练或论文结果复现。完整审计范围满足本任务完成证据要求。
- 分别固定仓库 SHA、Python/框架版本、requirements、数据链接可用性；记录旧依赖在当前环境中的兼容性。
- TraceAnomaly：确认正常 trace 建模、分数/阈值实现与论文描述的对应关系，定位可执行训练/测试入口。
- TraceRCA：梳理 feature selection、invocation anomaly detection、训练数据准备和 association-based localization 的先后关系，区分异常检测与根因定位输出。
- 任何为现代环境所做的修改均保存 patch diff；原仓库运行与修改后运行分别报告。
- **完成证据**：版本/环境/数据状态表、真实命令与日志、失败原因和 patch diff。

#### [x] T08｜Nezha：“basic implementation”覆盖范围核实（P1；代码/公开数据身份审计完成，完整 RCA run 未执行）
- **执行记录**：详见 `nezha_execution_audit.md`。固定 Nezha GitHub `main` SHA `d8140101fdb4e7dfb60d3ef9f64706f382b68470`、Zenodo concept/version `.8276374`/`.8276375`（v0.1，metadata MD5 `31a7408955c77112b5117655adb9b07f`）及 repo MIT license；另锁定 Augmented-OnlineBoutique `94a01ba4c7c79e0c0b2c9da073ba081fcedc4bf1`、Augmented-TrainTicket `32abe642659fc4e863e8348d699574aa30b3b2e2`（两者 Apache-2.0）。指出 `construct_data` 是两系统 fault-free phase，`rca_data` 是 fault-suffering phase，不是第三个独立系统。API tree manifest/inventory 见 `nezha_audit/`。
- `alarm.py`、`data_integrate.py`、`log_parsing.py`、`pattern_miner.py`、`pattern_ranker.py` 与论文阶段有对应；实测公开数据目录计数/原始 schema；发现 TT fault JSON 为 21 resource +24 code，与论文写 20+25 不一致；README 有 TrainTicket 日期 typo。
- `python main.py --help` 在 Python 3.13.16 因缺 `more_itertools`、解析参数前退出；未装依赖、未下载完整 Zenodo/GitHub数据、未修改代码或复算 paper tables。日志 `nezha_audit/nezha_smoke.log`；不声称 full RCA run。
- **判定**：本任务要求的 coverage、来源/版本、依赖/入口和缺口审计完成；完整复现未完成。

### 二、数据集与数据契约复核

#### [~] T09｜MSDS：许可证、raw schema、时间对齐与 fault ground truth（P1；fault taxonomy/hook 已映射，逐行时序标签仍待裁定）
- **本轮执行记录**：详见 `mstgad_msds_execution.md`；Zenodo version DOI `10.5281/zenodo.3549604`，CC BY 4.0，downloaded archive MD5 已匹配。使用 ZIP byte-range 抽取 `IMPORTANT` 与 3 个 Rally JSON，映射 Nova/Glance/Neutron workload、hook schedule 与实际调用脚本；31 个 hook result 的 epoch/UTC 起止、status 和 shell return code 已逐条导出到 `msds_audit/rally_hook_windows.csv`，可作为 event windows。发现 report 结果数少于计划 trigger 数、YAML 与论文描述的 Nova 目标不一致；`output_boot` 结束于 19:36 而脚本在 `TZ=UTC` 的 cut-off 约 18:18，`IMPORTANT` end date 又早于 start。新增对 ZIP 内 1 个全日志 CSV+5 个指标 CSV 的 range-only 解析与 31 个 event windows join（脚本 `msds_audit/range_time_coverage.py`、结果 `msds_audit/rally_telemetry_overlap.csv`）：log 169,230 rows 规范化 UTC 14:00–19:56，metrics 4,302,048 rows 覆盖 Nov25 14:57 至 Nov29 09:53；原始日志 `+01:00`、metrics `CEST` 与代码 `−1h` 在 `TZ=UTC` 下相符。未读 11,003 traces，也未生成已验证的全量逐行/逐pod标签；文档时区说法、唯一实验终止时间及跨模态标签仍待裁定。
- 下载/解包前确认许可和使用限制；记录文件版本、大小与 checksum。
- 解包后列出 raw 文件格式、字段、时间戳时区、实例/服务 ID、fault/workload script 与 Rally report 的关联方式。
- 实测并记录 CEST/UTC 转换和跨模态对齐，确认不是仅根据 README 写下的“减两小时”。
- **完成证据**：字段字典、时区转换规则、故障标签映射和可复查数据清单。

#### [x] T10｜LO2：验证实际发布模态、数据 schema 与标签粒度（P1；DOI/样本/复现包 audit 完成，full raw 未下载）
- **执行记录**：详见 `lo2_execution_audit.md`。Dataset concept DOI `.14257989` 最新 v3 record `.14938118`，CC BY 4.0；replication concept DOI `.14229369` 版本 `.14229370` v1.0.0，Apache 2.0。Replication ZIP 19,757,322 B、MD5 `72eb73073d4947da810169633fbf4f1c`；`lo2-scripts.zip` 17,195 B、MD5 `f579536be380e74b70db49d9597f99f9`，均实际下载核验。Full raw API checksum/size 已记，但 46.5 GB dataset 未下载；sample checksum仅 API metadata，没对全包重算。
- 通过 HTTP Range 而非全量下载 1.07 GB `lo2-sample.zip`：central directory 43,402 entries、37,800 service logs、100 metric CSV、无 trace 路径；定点提取一个 1,029,680 B metrics CSV 与 847,488 B log。实测 1,127 CSV columns（1124 metrics features）、172 rows/54 test names，与论文 172 timestamps/1124 feature vectors 和 1 correct+53 error tasks一致；replication `prometheus_metrics.txt` 为 485 unique metric names。Label 单位为 API error/test-run，而非性能 fault/service root cause。
- 查验 paper/data appendix：full dataset 的 trace files 多数 header-only（58 B），作者报告每 trace single span 且未分析 traces；Jaeger/OTel 配置存在不等于有效覆盖充足。Replication package 还设 Jaeger API limit=20。
- Docker/Compose、MySQL、Locust 在当前环境均不存在，未运行 full replication。发现 helper `reduce_logs.py` 使用 float list-slice index，会 `TypeError`；独立小列表烟测验证。
- 已记录 2026-03 发布 LO2v2 `.18937117` supersede 关系与大小/许可证；与原始 1,740-run paper dataset 分开，不作为同一版结果复现。
- **判定**：满足本任务公开数据身份、schema、labels、trace coverage 与 replication path 审计；full raw 下载和 full system replay 未执行，不声称完整复现。

#### [x] T11｜ServiceAnomaly 与 Eadro 数据：来源、版本和论文实验数据一致性（P1；公开数据身份审计完成，模型复现不在本任务范围）
- **执行记录**：详见 `serviceanomaly_eadro_data_identity.md`。ServiceAnomaly 固定 GitHub HEAD `6acc9dfd7ec50d3007d65e61a188f40d72b422b6`，data/ 共 1,718 files / 42,331,839 B，SHA-256 per-file manifest 为 `serviceanomaly_data_sha256.txt`；无独立 data DOI/license。实测 baseline trace counts 远低于 paper 描述，test case directory count 近似匹配但存在缺失/损坏。Eadro 固定 GitHub SHA `82ff9c9a7a58a9cd2b713aa78754a73c543a15db`，Zenodo concept/version `.7615393`/`.7615394`；SN/TT ZIP MD5 与 T05 报告一致，但记录无 license。与论文 benchmark identity 对应，不等于已核验论文 trace 总量或可重算结果；fault metadata rows 与论文 injection-operation 数量口径仍未解释。
- **判定**：完成源头、版本、公开包校验值/许可状态及论文数据 identity 对照；不声称 full data identity equivalence、模型训练或论文结果复现。
- 对 ServiceAnomaly 仓库 `data/` 和 Eadro 发布包分别记录来源、版本、文件清单、许可证和校验值。
- 核实这些文件是否对应论文报告的实验 run，而非仅为同名开源系统的另一次采集或通用示例数据。
- 对 Eadro Train-Ticket/SocialNetwork 公开包和仓库/Zenodo DOI 的关联做最终确认。
- **完成证据**：每份数据“论文实验数据 / 系统代码 / 作者公开复现数据”的身份区分与证据链接。

### 三、论文/代码/数据证据尚不完整的工作

以下项目的目标不是先猜出架构，而是优先找正式全文、作者页面、官方仓库和数据页；找不到时保留“未确认”，记录检索证据。

#### [~] T12｜Triple（IWQoS 2023）（P2；摘要/候选仓库核验完成，正式全文细节待核）
- **执行记录**：`triple_iwqos2023_evidence_audit.md`。固定论文身份/DOI `10.1109/IWQoS57198.2023.10188773`；IEEE 摘要支持 trace dependency graph、log/metric node features、STGCN+Deep SVDD 及解释器。正式正文抓取未得内容，图粒度、输入维度、精确 loss/labels/threshold、数据版本与 split/指标均保留 Unknown。
- 搜索到 `TripleResearcher/Triple` 候选仓库（main 两 commits，只有 README 和 3 个 figure PDFs，无可运行代码/数据/依赖/LICENSE）；论文页未链接它，故作者官方归属未确认。
- **判定**：已完成可访问公开证据核验，尚未完成全文方法/协议审计；不使用二手摘要填补。

#### [ ] T13｜UAC-AD（IEEE TSC 2024）（P2）
- 核实两套公开模拟数据的准确名称/版本、真实工业数据边界、训练协议、adversarial/contrastive 目标及 threshold/evaluation。
- 确认官方代码和数据是否开放；若未找到，记录检索范围，不将“未找到”写成“作者不开放”。
- **完成证据**：数据、模型、评价、代码状态的原始来源记录。

#### [ ] T14｜Deep Attentive AD（ICWS 2022）（P2）
- 获取可复查的作者全文或实现，补核架构、输入维度、训练目标、故障/正常划分、阈值与评价协议。
- 核实 Train-Ticket 数据版本及官方代码状态。
- **完成证据**：以原始论文/官方实现支持的字段表；没有证据的细节继续标 Unknown。

#### [ ] T15｜TraLogAnomaly（2025）（P2）
- 核实 hybrid event sequence 的精确定义、Transformer/TF-IDF 的实现位置、训练标签、阈值及评价粒度。
- 查清论文使用的数据集、split、作者代码与数据开放状态。
- **完成证据**：论文—数据—实现证据表；明确 trace/log 样本单位。

#### [ ] T16｜MADMM（Neural Computing and Applications 2024）（P2）
- 从正式全文核实 metric graph、log graph、GCN/GAT、Cross-Modal Attention GRU 的输入输出、loss 和阈值。
- 确认实验数据版本、监督条件、split、评价方式及官方代码/数据状态。
- **完成证据**：原文或作者资料支持的架构/数据/协议字段；摘要级证据与全文证据分开标记。

#### [ ] T17｜ART（ASE 2024）（P2）
- 确认是否存在与论文一一对应的作者官方仓库；若存在，固定 SHA 并检查其是否覆盖 AD、failure triage、RCL 三类任务。
- 核实数据集、切分、预测偏差矩阵、EVT/delay window、cut-tree 和 RCL 实现及评价脚本。
- 保持 AD、FT、RCL 任务和指标分别记录，不合并成一个检测成绩。
- **完成证据**：官方代码身份结论、三任务实现覆盖表和阈值/评价来源。

#### [ ] T18｜TraceGra（2023）（P2）
- 继续查找作者/机构官方实现；核实 Trace Performance Graph、GNN、LSTM encoder-decoder 与异常分数是否有代码对应。
- 确认论文所称公开数据、本地集群实验数据的身份、划分和评价脚本。
- **完成证据**：官方代码检索结果和数据/方法证据；未找到时保留“未确认”，避免用相似第三方仓库替代。

#### [ ] T19｜中文文献检索缺口（P3，可选）
- 若需要较完整的中文文献覆盖，补做可复查的 CNKI 检索；同时核实《软件学报》论文、清华博士论文、北大机构库论文的全文/数据/实现细节。
- 保存检索式、日期、可访问记录及无法获取全文的限制；不要声称已系统检索未访问的数据库。
- **完成证据**：中文检索日志及新增/排除文献理由。

### 四、评价协议与本地任务映射

> **当前范围调整**：T21、T22 中需要用户私有数据才能完成的部分按用户要求暂缓；本轮不会索取或分析私有数据。仅基于公开论文/开源数据的任务仍可继续。

#### [~] T20｜统一审计异常分数、阈值和评价实现（P1；已审计公开项目，其他论文待覆盖）
- **执行记录**：`anomaly_score_threshold_evaluation_matrix.md` 已逐项整理 T01–T08、T10 的输出单位、score/decision、阈值来源、label 使用时机、evaluator 与执行状态。包括 AnoFusion/TraceAnomaly 论文阈值机制与可见代码缺口、MSTGAD 半监督标签与 train-F1 选模、Eadro test-HR@1 checkpoint 选择及指标/切分风险、DeepTraLog normal-score>0、ServiceAnomaly baseline-RMSE cutoff，以及 BARO/TraceRCA/Nezha 的 RCA 与 AD 单位区别。
- **覆盖边界**：逐项目 matrix 已列明可核实代码位置和未发现的 point-adjustment/容忍窗；T12 仅有摘要级核验；T13–T19 尚未逐一完成。T20 整体保持部分完成，不外推为所有论文均无阈值泄漏。

#### [ ] T21｜文献粒度与本地 Service×15 秒任务的兼容映射（P1）
- 建立逐论文任务单位表，至少区分 instance×time、service/system window、trace、API/test call、root-cause ranking；不以“service-level”字样代替真实输出单位。
- 对照本地 25 个 Service × 15 秒窗口、17 Cases（6 normal/11 fault）、normal-only 训练/校准/holdout 与故障仅测试的协议，标记每篇论文是直接同任务、可改造参考还是任务不同。
- 保持故障 Case/run 为独立实验单位，不把约 50,575 行窗口视为独立实验；根因 Top-k 与异常检测 F1 分开报告。
- **完成证据**：兼容性矩阵，明确“不等价”的原因，不据此预设新模型或排名。

#### [ ] T22｜本地多模态缺测、质量状态与数据覆盖审计（P1）
- 对本地数据验证资源无效/不可评分的 9,074 行、quality 状态、counter reset、limit 语义；不可评分值不能静默补成健康 0。
- 分别审计 logs 的连续覆盖、trace 的 Entry/Exit/Local 完整性、时间戳/服务身份对齐及观测调用边覆盖。
- 明确 Trace 缺 Entry 不等于无请求/健康；原始调用边不自动等于完整稳定 dependency graph；文件存在不等于模态覆盖足够。
- 与文献逐项比较 padding/serialization 语义，只记录各自数据契约，不直接照搬 0-padding。
- **完成证据**：覆盖率和缺测语义报告、可评分性规则及模态对齐问题清单。

#### [~] T23｜Endpoint/API 任务与 Service-window 任务分轨（P2；公开文献单位表完成，本地信号关系按隐私要求暂缓）
- **执行记录**：公开部分见 `endpoint_vs_servicewindow_taxonomy.md`。区分 LO2 API-test outcome、ServiceAnomaly/TraceAnomaly trace-level detection、TraceRCA/Nezha case-level root-cause ranking 与 MSTGAD/MSDS windows；记录每类 labels/metrics，明确不同单位不可直接互相复制。
- **暂缓项（用户范围约束）**：未读取私有本地 client failure、trace error、log ERROR 或质量状态，因此未验证这些信号彼此的一致性/因果关联，也未把公开任务强行映射为用户本地服务窗标签。此类本地联结需之后授权后执行。
- **判定**：公开任务分轨完成；整项仍标部分完成，不声称本地 API/trace ground truth 已核实。

### 五、统一复核记录规范

#### [~] T24｜建立项目级复现台账并为每次执行归档（已为 T01、T02、T03 建立初始/逐项记录；全任务台账仍未完成）
- **本轮新增**：T03 的版本、环境、Zenodo DOI/许可证/官方 MD5、命令、退出码、wall time、采样 RSS、结果与未执行项见 `baro_execution.md`；10 条命令的 stdout 和运行摘要见 `baro_execution_logs/`。T04 仓库/数据/协议审计及三个失败入口的 stdout 见 `serviceanomaly_execution_audit.md` 与 `serviceanomaly_*.stdout.log`；T05 Eadro 的 DOI、hash、schema、运行 stdout 与限制见 `eadro_execution_audit.md` 和 `eadro_*.stdout.log`；T06 DeepTraLog 的固定 SHA 源码快照、raw sample、failed smoke 与链路限制见 `deeptralog_execution_audit.md`、`deeptralog_repo/model/` 和 `deeptralog_main.stdout.log`；T10 LO2 的 DOI/version、byte-range sample、metrics/log schema、trace coverage 与无容器环境限制见 `lo2_execution_audit.md`、`lo2_audit/`。T09/MSDS 时间与标签映射补充见 `mstgad_msds_execution.md`、`msds_audit/` 和 `msds_observability_repo/`（固定 SHA `ea6ae992d5ab6442ad1a419a3e0a0bd8a29dca51`）；T02 原始子集 smoke 与 patch 细节仍见同一报告。T11 两项目数据身份审计见 `serviceanomaly_eadro_data_identity.md`，ServiceAnomaly per-file hash manifest 为 `serviceanomaly_data_sha256.txt`。T07 TraceAnomaly/TraceRCA 版本、模型/协议与运行障碍见 `traceanomaly_tracerca_execution_audit.md`，执行日志在 `traceanomaly_smoke.log`、`tracerca_smoke.log`；T08 Nezha 代码/数据/标签审计见 `nezha_execution_audit.md`，tree manifest、inventory 和 smoke log 在 `nezha_audit/`。T23 公开数据单位分轨见 `endpoint_vs_servicewindow_taxonomy.md`（本地私有信号核验暂缓）；T20 已审计项目协议矩阵见 `anomaly_score_threshold_evaluation_matrix.md`（T12 仅摘要级，T13–T19 待覆盖）；T12 Triple 的来源和候选仓库记录见 `triple_iwqos2023_evidence_audit.md`。T01 见 `anofusion_open_gaia_initial_audit.md`。
- **限制**：T24 需随每项继续补齐，尚非全局完成；例如各早期任务原始日志、随机种子、逐命令运行资源等并非都已归档。不得因已有数份报告而勾选完成。

每个仓库/数据集至少保存：
- 仓库 URL、branch/tag、完整 commit SHA、执行前后 `git status`；
- OS、Python、CUDA/驱动、依赖锁定文件/实际安装版本；
- 数据 DOI（区分 concept/version）、来源、许可证、文件大小与 checksum；
- 原始命令、配置/随机种子、stdout/stderr、退出码、wall-clock、峰值内存；
- 实际产物、评价脚本版本、结果和论文结果对照；
- 所有本地 patch diff，并将“原样运行”“修改后运行”“论文数值复现”分别标注。

**完成标准**：每一个已执行任务都能由另一个 Agent 根据台账追溯到确切代码、数据、环境、命令与结论；没有执行的事项保持未勾选。

---

### 建议的逐项推进顺序

1. 先做 T24 的台账模板，然后依次执行 T01–T05（最接近 Service×Window/多模态或有清楚数据和运行入口的项目）。
2. 再做 T09–T11 数据核查，并完成 T20–T22 的评价与本地兼容性审计。
3. 然后执行 T06–T08 的 trace/RCA 代码项目。
4. 最后逐项处理 T12–T19 的论文证据缺口及 T23 的 API 任务分轨。

> 顺序是执行便利性建议，不代表模型优劣、最终选型或预设论文方法。每项完成后先记录结论与证据，再开始下一项。


---

<a id="t01"></a>
## T01｜AnoFusion × 公开 GAIA

> 来源文件：`anofusion_open_gaia_initial_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T01 执行记录：AnoFusion × 公开 GAIA（初步源码/数据/运行核查）

**执行日期**：2026-10-19（Asia/Taipei）
**范围限制**：只检查可公开获取的 GAIA/MicroSS D1；未获取、未请求、未检查任何私有数据。论文所述银行 D2 因 NDA 明确排除。
**结论状态**：**部分完成，未复现模型**。本次已 clone 公开代码、检查公开数据仓库和本地样例、执行 README 预处理命令及训练入口尝试；GAIA Git pack 实际下载了 4.62 GiB，但未 checkout、解压或用于运行。由于样例数据缺件、代码入口/路径问题和当前环境缺少 PyTorch，无法训练或得到模型异常分数。

### 1. 代码与数据版本

|对象|核验结果|
|---|---|
|AnoFusion 代码|`https://github.com/AIOps-Lab-NKU/AnoFusion`，branch `main`，HEAD `ae3c4ea98509008844a3eecf7c0bd1529faaba67`（2023-08-23）；浅克隆。工作树执行前干净。|
|论文中代码链接|论文 PDF 搜索结果指向 `https://github.com/zcyyc/AnoFusion`；对该 URL 执行 `git ls-remote` 返回 GitHub 认证提示，无法作为匿名公开仓库访问。与当前可公开访问的 AIOps-Lab-NKU 仓库之间的身份/迁移关系仍待核实。|
|GAIA 数据仓库|`https://github.com/CloudWise-OpenSource/GAIA-DataSet`，HEAD `238a9e24ae0b69e7e05dff112652a3fa62db88fd`。最初以 `--filter=blob:none --no-checkout` 获取仓库；之后运行递归 `git ls-tree -l` 后发现 `.git` pack 已达 **4.62 GiB**，说明大型数据对象实际已被下载到 Git 对象库（但未 checkout/解压）。为避免在工作区留下数 GB 中间数据，记录完成后已删除该克隆。|
|GAIA 发布可得性|公开仓库可下载。Git tree 中 MicroSS 分卷对象大小合计约 5.39 GB（business 2.198 GiB、metric 1.666 GiB、trace 1.500 GiB，另有 run 与说明文件；Git pack 为 4.62 GiB）。这些是数据仓库对象/分卷，不代表已解压语料大小。本次实际下载过 Git pack，但未 checkout、解压或用完整 GAIA 跑模型。仓库根目录含 GPL-2.0 文本；数据文件适用许可仍需谨慎按数据页面/具体条款确认，不能仅凭根目录文件作法律结论。|
|本次运行环境|Python 3.13.16；NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.1、scikit-learn 1.6.1、tqdm 4.70.1、PyYAML 6.0.3；未安装 PyTorch/torchvision。未安装旧版依赖，也未修改仓库代码。|

### 2. 公开仓库现有样例并非 README 所列完整输入

AnoFusion 浅克隆的 `data/` 实际文件：

|文件|实测记录数（不含表头）|要点|
|---|---:|---|
|`mobservice2_2021-07-01_2021-07-15.csv`|19|metrics 样例，160 列；时间戳只覆盖约 1,080 秒。|
|`mobservice2_stru.csv`|19|log 结构样例。|
|`mobservice2_temp.csv`|19|log template 样例。|
|`trace_table_mobservice2_2021-csv 07.csv`|19|trace 样例。|
|`mobservice2_2021-07-15_2021-07-31.csv`|**缺失**|README 文件树列出的第二段 metric 文件，`write()` 无条件读取。|
|`labeled_service/mobservice2.csv`|2,678,400|标签时间序列，timestamp 为 2021-07-01 至 2021-07-31；label=1 共 123,479 行。标签完整并不代表三模态原始样例完整。|

来源：在克隆的 AnoFusion 仓库中对文件逐项 `pandas.read_csv` 统计；数据未外传。

### 3. 实际执行结果

|命令/检查|结果|
|---|---|
|`python -m compileall -q .`|通过（语法编译通过不等于依赖、输入或训练通过）。|
|在 `utils/` 下执行 README 的 `python3 generate_channels.py`|退出码 0，但**没有生成序列化/对齐数据文件**；只因模块导入而建立了空的 `../serilize_data/{metrics,log,trace}` 目录。源码中没有 `__main__` 调用，也没有脚本级调用 `get_channels()`。|
|直接调用 `get_channels('mobservice2','train',0.6)`|失败：`FileNotFoundError: ../data/mobservice2_2021-07-15_2021-07-31.csv`。这是模型训练前的确定性阻断。|
|`python3 main.py --mode train --service_s mobservice2`|失败于 import：`ModuleNotFoundError: No module named 'torch'`。因此没有安装依赖或执行任何训练。|
|完整数据/模型运行|未执行。GAIA Git 对象曾实际下载（4.62 GiB pack），但未 checkout/解压；当前 AnoFusion 克隆样例仍缺必需 metric 文件，且环境无 PyTorch，源码审计另发现预处理/评估阻断。未用该大型数据包开展预处理或模型运行。|

### 4. 代码与论文步骤的初步对照

已确认的实现线索：
- `utils/main.py` 中将 metrics、logs、traces 拼成 channel，按六种模态内/模态间组合计算互信息矩阵；模型 `model/AnoFusion.py` 包含 5 层 GTN，后接 GAT/GRU 模块。主训练损失为对下一时刻 channel 的 MSE。
- `model/MyDataset.py` 的 `read_label()` 只取 timestamp；`__getitem__()` 返回的是下一时刻的 channel 作为预测目标。因此从代码路径看，`label` 值没有用于训练 target，训练目标是预测序列；评价时再按 timestamp 取标签。
- 因输入缺失，尚不能证明该实现可从公开 GAIA raw data 端到端构建出论文所需 channel，也未验证完整图结构/张量形状。

值得注意的实现问题（需要作者代码修复或额外数据/说明后才能继续实跑）：

1. **README 预处理命令看起来不执行预处理。** `utils/generate_channels.py` 建立函数并创建目录，但没有 `if __name__ == '__main__'` 或其他脚本级 `get_channels()` 调用。实测命令退出码 0，却没有生成文件。
2. **公开代码克隆内缺 README 列出的第二段 metric 文件。** `get_channels()` 调用的 `write()` 在 `utils/generate_channels.py:176–178` 无条件读取两段文件，故当前样例立即 `FileNotFoundError`。
3. **目录名不一致。** 初始化创建 `serilize_data/log`、`serilize_data/trace`（单数），后续分别写/读 `serilize_data/logs`、`serilize_data/traces`（复数）；另有 `update_linear_interpolation_data` 下写入 metrics/logs/traces 前未见相应目录初始化。即使补齐 metric，仍可能在序列化阶段失败。
4. **Trace 统计向量疑似未被写入。** `serialize/trace_to_sequence.py:30–37` 计算 `values=[span,mean,ptp,std,p25,p75]`，却将 `span_data`（随每秒 span 数量变化、长度不定的 duration/status 列表）写进 `trace_series`。下游将其数组直接组成 DataFrame，可能遇到变长数组；也与预期固定通道特征不吻合。尚未在完整 trace 数据上实测这一错误。
5. **评估异常分数与论文阈值环节不一致/至少未在当前源码中看到。** `utils/main.py:130–143` 对残差用 `abs(error)`、减去 median 后除以第 25 百分位，再取最大值并输出 CSV；当前仓库检索未发现 EVT/POT 阈值拟合或二分类输出。因此公开代码目前可见的是残差分数输出，并未展示论文描述的 EVT 自动阈值判定。需进一步核对论文 PDF 版本、可能遗漏的分支或旧仓库，不能据此替论文方法下最终结论。
6. **评估 checkpoint 硬编码。** `utils/main.py:245` 固定加载 `model_save_v2/checkpoint_mobservice2_0_0.2996_model.pkl`，而训练逻辑按第一个可用目录创建 `model_save_v0/v1/...`。默认训练产物和默认评估 checkpoint 不保证一致。
7. **GPU 设备写死。** 主流程 `cuda_device=1` 并在训练/评估中显式 `.cuda(1)`；当前没有 PyTorch，也没有验证无 GPU 环境下可运行。
8. **依赖文件与 README 不一致。** README 列出较旧的 PyTorch 1.6.0、pandas 1.1.5 等，但仓库没有发现 requirements/lock 文件；当前 Python 3.13 环境不能直接视为兼容旧栈。

以上均为对公开克隆当前 HEAD 的源码观察，不等同于对论文作者本地版本或完整数据实验的最终判定。

### 5. 下一步建议（仍只使用公开数据）

1. 不要直接对当前不完整 `data/` 运行长时间序列化或安装旧依赖；先确认论文链接仓库与 AIOps-Lab-NKU 公共仓库的关系、以及是否存在可单独获取的 AnoFusion 预处理数据。
2. 若继续 AnoFusion，需要先从 GAIA/MicroSS 官方发布获取与 `mobservice2` 对应的完整输入，并确认下载量、许可、文件清单/校验值；本次仅确认公开仓库可访问，未下载完整数据。
3. 在不修改原仓库的副本中逐步补齐数据、跑最小 preprocessing；每项修复保留 patch diff，并分开报告“原样运行失败”与“修改后可运行”。
4. 只有依赖与数据路径确认后，才值得花资源装兼容 PyTorch 环境并尝试训练/评估；若源码确无 EVT 阈值逻辑，记录为论文复现差异，不静默补写实现。

### 6. 可复查链接

- AnoFusion 公共仓库：[AIOps-Lab-NKU/AnoFusion](https://github.com/AIOps-Lab-NKU/AnoFusion)
- GAIA/MicroSS 数据仓库：[CloudWise-OpenSource/GAIA-DataSet](https://github.com/CloudWise-OpenSource/GAIA-DataSet)
- AnoFusion 论文预印本：[arXiv:2305.18985](https://arxiv.org/abs/2305.18985)


---

<a id="t02-t09"></a>
## T02 / T09｜MSTGAD × MSDS 执行与数据审计

> 来源文件：`mstgad_msds_execution.md`。以下为完整正文；标题层级为合集显示下调一级。

## T02 / T09 执行记录：MSTGAD × 公开 MSDS v1.1

**执行日期**：2026-10-19（Asia/Taipei）
**数据范围**：仅使用公开 MSDS concurrent 数据（CC BY 4.0），未访问任何用户私有数据。为避免在工作区保留大型数据，仅下载并核验完整 ZIP 后，用其中一个短时间片和 25 条真实 trace JSON 做预处理 smoke test；全量解包/训练未执行。
**结论**：源码与数据契约审计完成；**公开数据下载、完整性验证、子集预处理与窗口数据转换成功（需 2 行兼容性修补）**；官方原样预处理失败；训练无法运行（未安装 Torch/PyG，当前环境 Python 3.13；README 推荐 Python 3.8/Torch 1.12/PyG 2.2）。

### 1. 版本与来源

|项目|核验结果|
|---|---|
|MSTGAD 仓库|`https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection`；branch `main`；HEAD `e78201a1890d1988fcb17c7f94e464815e567c7b`；clone 后源码工作树已恢复干净。仓库未见 LICENSE 文件。|
|MSDS DOI|Zenodo version DOI `10.5281/zenodo.3549604`，version 1.1；concept DOI `10.5281/zenodo.3484800`。开放下载，record 标注 CC BY 4.0。|
|concurrent archive|下载地址来自 Zenodo record；实际大小 335,390,038 bytes；下载后 MD5 `be2ba2f17f774e7f3f045dcf5f406ff0`，与 Zenodo 文件清单一致。下载文件 SHA-256 `3e1e7415c55a5852857dbe5a0177768ce92044e5f6819b0dc504f567e7c2beab`。|
|压缩包结构|ZIP 清单 11,026 entries；记录的总 uncompressed bytes 为 4,723,134,938（约 4.40 GiB），压缩 payload 约 332 MB。trace 子目录约 4.02 GiB；全量解压到本环境 `/tmp` 会超过 992 MiB tmpfs，因此未全量解压。ZIP 与临时解包子集均已清理。|

公开来源：[MSDS Zenodo record](https://zenodo.org/records/3549604)、[MSTGAD 作者仓库](https://github.com/alipay/microservice_system_twin_graph_based_anomaly_detection)、[MSTGAD arXiv HTML](https://arxiv.org/html/2310.04701)。

### 2. 数据 schema、许可与时间信息

- **metrics**：五个主机/节点 `wally113/117/122/123/124`；每节点 CSV 提供 `now,cpu.user,mem.used,load.cpucore,load.min1,load.min5,load.min15`。仓库预处理最终对每节点保留 5 个指标（`cpu.user, mem.used, load.min1, load.min5, load.min15`），并 min-max normalize。
- **logs**：压缩包含 `logs_aggregated_concurrent.csv`（原始条目另有 tar.gz）。示例字段包括 Hostname、`@timestamp`、log_level、Payload 等；作者预处理用 Drain3 对 Payload 做模板聚类并生成 `templateid`。
- **traces**：目录以 11k 量级 JSON 文件组织，记录嵌套 `children`、host、name、trace/parent IDs 和带微秒的小数时间戳；预处理按 pod/host 和 span 类型聚合 duration，并生成 5×5 的无向调用/调度关系图。
- **ground truth**：Zenodo 说明含 workload/fault scripts 与 Rally report。后续通过 ZIP byte-range 抽取了三份 concurrent Rally reports 和脚本：已将 `output_boot/image/network.json` 对到 Nova/Glance/Neutron tasks、fault hooks 与脚本动作，并核验 report 里的部分 Unix-time hook windows。`IMPORTANT` end date、硬编码预处理终点与 Nova report 实际结束不一致；全量时序对齐、逐行/逐pod label 尚未闭合，详见本报告第 5 节。
- **许可证**：Zenodo record 明确 CC BY 4.0。归档文件及复用应按署名要求引用数据和论文。
- **时间戳的观察**：Zenodo 页面称 logs/metrics 为 CEST、traces 为 UTC、相差 2 小时，并要求同步。原始 aggregated log 样例时间含明确 `+01:00`；metric 字符串却带 `CEST` 后缀，而数据日期是 2019-11-25。作者预处理对 logs 和 metrics 均减 3600 秒、trace 不减；在本轮选取的短 slice、以 `TZ=UTC` 执行后，metric 与 log 输出 epoch 区间相互对齐（metric 1574694733–1574695199、log 1574694734–1574695199）。这支持本数据导出实际使用 +01:00 的转换，但与网页描述的“2 小时”表述并不完全一致。trace 时间为无时区后缀的 naive 字符串，脚本用 `time.mktime`，结果依赖运行机器本地时区；全量跨模态 trace 对齐仍须在实际全量数据上复核。
- **时间范围不一致**：压缩包 `IMPORTANT_experiment_start_end_data.txt` 中 start 为 `2019-11-25 15:12:13`，end 却写成 `2019-11-20 19:45:00`（早于 start）；`pre_MSDS.py` 则硬编码 `2019-11-25 15:12:13` 到 `2019-11-25 18:18:13`。复现必须明确采用哪个数据时间范围，不能默默把文档 end 当正确值。

### 3. 预处理/窗口构造实测

#### 3.1 官方原样命令

以本次公开数据 ZIP 的真实子集（五个节点的 8 分钟 metric slice、2,496 条 16:12–16:20 log 行、25 个真实 trace JSON）构造预处理输入，在 UTC 时区运行官方 `python util/pre_MSDS.py`：

- metrics 解析和日志 Drain3 模板聚类成功；
- 进入 trace 阶段后，原始代码在 `util/pre_MSDS.py:179` 报错：`TypeError: 'StringMethods' object is not iterable`。
- 出错语句把 `Series.str.split(...).str` 直接解包到两个列；同一错误也存在于 end_time 处理行。

#### 3.2 最小临时兼容补丁

仅将这两行改成 `Series.str.rsplit('.', n=1, expand=True)` 后，官方预处理在该子集上成功生成：

|输出|实测数据量|
|---|---:|
|`metric.csv`|467 rows × 26 columns（含 now）|
|`log.csv`|2,496 rows × 5 columns|
|`trace.csv`|5,278 span rows × 11 columns|
|`trace_path.pkl`|5×5 关系矩阵（350 bytes）|

补丁文件：`mstgad_preprocess_compat.patch`，SHA-256 `6231274af4f4290f2846065c510eaa1ed098c977b2dd04d60aafec6c0d7d5c5c`。补丁只用于 smoke test，已还原仓库原文件；不能把补丁后的成功说成原仓库原样可运行。

接着调用官方 `util.data_MSDS.Process`（window=10, step=1, 五节点，训练默认特征尺寸）成功构造 **458 个窗口**；首个样本张量尺寸为：`data_node (10,5,3)`、`data_log (10,5,256)`、`groundtruth_cls (5,3)`、`groundtruth_real (5,2)`、`data_edge (10,5,5,7)`。这是一个数据切片 smoke test，不是论文全量预处理或结果复现。

### 4. 代码审计与监督/评价协议

#### 数据和标签来源

- 官方仓库 clone 自带 `data/MSDS-pre/label.pkl`（446,602 bytes；SHA-256 `fe4e375a8b5b436a4cd3eda158904e5b5ef173b9d370cb2c064cb802a4a31047`）。`pre_MSDS.py` 本身**没有生成 `label.pkl`**；该文件是另行预打包的输入。因此从 Zenodo raw data 运行该预处理脚本，不会独立产生模型运行所需的完整 `MSDS-pre` 目录。
- `data_MSDS.py` 将 `label.pkl` 转为两类 one-hot 标签，并用 `label_percent`（默认 0.5）对部分标签标记为 unknown；重建损失对正常/异常/unknown 使用不同权重，`src/model.py` 的分类头对已知标签训练。故这确实是半监督学习，不是 normal-only 无监督训练。
- `main.py` 实际按序列顺序切成前 70% train、后 30% test。论文 arXiv HTML 的实验设置写 60% train、10% validation、30% test，并称半监督方法从训练段随机抽 50% unlabeled。当前仓库未实现明确的 60/10/30 split；这是重要的代码—论文协议差异。
- `train.py` 每轮会调用一次 `evaluate(test_loader)`，但该返回值未用于选模；保存 `f1` checkpoint 的比较值来自 `evaluate(train_loader)`。最终对测试集分别评估 loss/f1 checkpoint。虽源码有每轮测试集探测调用，不能仅由此断言选模直接使用 test F1；不过评估边界和论文 60/10/30 协议仍不相同。
- `util.calc_index` 计算标准 sklearn precision/recall/AUC/AP/F1；没有 point-adjustment、事件容忍窗或自动阈值搜索，二分类预测用 argmax。异常标签与服务节点 one-hot 是直接评价对象。

#### 资源、依赖与运行阻碍

- README 推荐 Ubuntu 18.04、Python 3.8、PyTorch 1.12.0、PyG 2.2.0；requirements.txt 只列 AdaBelief、Drain3、jsonpickle、matplotlib、NumPy、pandas、scikit-learn、tqdm，没有 Torch/PyG。
- 目前沙箱 Python 3.13.16 没有 Torch、torch_geometric。官方 `python main.py --epochs 1 --gpu false` 在 import 处以 `ModuleNotFoundError: No module named 'torch'` 失败；没有安装 GPU 栈，也没有训练。
- 代码有硬 CUDA 调用：`src/model.py` 构造图时直接 `.cuda()`，`util/train.py` 中 `label_weight` 和空分类损失也直接 `.cuda()`；因此 README/parser 的 `--gpu false` 并不足以提供完整 CPU 训练路径。
- `data_MSDS.py` 在指标存在缺秒时调用 `DataFrame.append`，该 API 在 pandas 2.x 已删除；当前 smoke slice 连续，因此没有触发。应在全量数据缺口场景中单独测，不能据此宣称完全兼容 pandas 2.2。
- 模型数据处理按 `num` 位置索引标签与时间窗，不是逐条使用 timestamp join；要求 label、metric、log、trace 序列长度与起点严格一致。全量标签/数据时间轴的准确对齐尚未通过全量实验验证。

### 5. T09 补充：Rally ground truth 的公开文件级映射

原 concurrent ZIP 已在早前核验 MD5 后清理。本次未重下 335 MB 全包，而是对 Zenodo ZIP 使用 HTTP byte-range 读取 central directory，再按 ZIP local-header range 取回 `IMPORTANT_experiment_start_end_data.txt` 和 3 个小型 Rally JSON reports。Central directory 是 11,026 entries、4,723,134,938 uncompressed bytes；reports 中 `output_boot.json`、`output_image.json`、`output_network.json` 与论文三类并发 Rally workload 对上。抽取内容保存在 `/home/user/msds_audit/`。

|报告|Rally workload / label|配置的 fault hook|report 记录的 hook 结果|
|---|---|---|---|
|`output_boot.json`|`NovaServers.boot_and_delete_server`; `times=2000`, concurrency 2|每 250 iterations 到 7000：`restart_nova_container.sh`|YAML 有 27 个 iteration hook entries（250–7000）；报告记录 7 个 hook results，均 `status=success` 且 shell `RetCode: 0`。Rally task 15:12:13–19:36:19 UTC；hook 实际开始/结束总包络为 16:08:22.875–19:36:11.131 UTC。|
|`output_image.json`|`GlanceImages.create_and_delete_image`; `times=3000`, concurrency 5|每 250 iterations 到 3000：`restart_glance_container.sh`|YAML 有 12 个 iteration hook entries（250–3000）；报告 12 个 hook results，均 `status=success` 且 shell `RetCode: 0`。task 15:12:28–17:51:08 UTC；hook 总包络 15:28:37.729–17:50:45.106 UTC。|
|`output_network.json`|`NeutronNetworks.create_and_delete_networks`; `times=6000`, concurrency 1|每 500 iterations 到 7000：`restart_neutron_container.sh`|YAML 有 14 个 iteration hook entries（500–7000）；报告 12 个 hook results，均 `status=success` 且 shell `RetCode: 0`。task 15:12:53–18:18:40 UTC；hook 总包络 15:29:16.571–18:18:34.007 UTC。|

每个 `results` 含 epoch `started_at`/`finished_at`。已把 31 个 hook event 全部展开为 `/home/user/msds_audit/rally_hook_windows.csv`，逐条保留 report、workload、服务故障、事件序号、status、epoch/UTC 起止时间和 shell return code；可据此构造脚本执行窗口并在统一时区后关联三模态数据。注意窗口只代表 Rally hook 脚本执行时段，不证明故障效应只局限于该时段。

#### 按 ZIP 成员 range 抽取的日志/指标时间轴对照

为减少下载，没有取完整 ZIP 或 4.7 GB 解压树；另用 HTTP Range 只抽取 `logs_aggregated_concurrent.csv` 与五个 host `*_metrics_concurrent.csv` 压缩成员，流式解析原始记录，再与 31 个 Rally hook windows 做精确 UTC 起止区间查询。可复算脚本为 `msds_audit/range_time_coverage.py`；成员级范围统计为 `msds_audit/range_time_coverage_summary.csv`；逐 event × source 的前 60s / hook 内 / 后 60s 行数为 `msds_audit/rally_telemetry_overlap.csv`（186 行）。

- Logs：169,230 行，`@timestamp` 含明确 +01:00 offset；规范化 UTC 覆盖 `2019-11-25 14:00:17`–`19:56:40`。Metrics：五个 CSV 合计 4,302,048 行，字段 `now` 标注 CEST；规范化 UTC 从 `2019-11-25 14:57:09` 延伸至 `2019-11-29 09:53:42`（wally117），远晚于 Rally workload 日期/预处理的单日 cutoff。此发现支持 logs/metrics 的实际数据时标为 UTC+01，而不是笼统按论文 prose 直接减两小时；MSTGAD 原脚本去掉 offset/CEST 后减 3600 秒，在 `TZ=UTC` 运行时与此转换相符，但其 `time.mktime` 对系统本地时区敏感。
- 对每个 hook 区间直接数行：日志窗口内 Nova/Glance/Neutron 各 hook 的逐事件计数可在 overlap CSV 复核；五主机指标在 hook 执行窗内累计计数分别为 Nova 1,415、Glance 1,281、Neutron 5,161（窗口内区间是脚本执行窗口，不是持续故障影响标签；窗口相交时这些逐事件加总也可能重复计数）。这说明 hook UTC 时刻确实可与 raw log/metric 时间轴连接，但还不等于确认故障效应持续多久或服务/pod row-level labels 正确。
- 本轮仍未全量 range-read 11,003 个 trace JSON；未把 trace spans 与 hook windows 逐一 join，也未写/生成被验证的完整三模态 label file。Trace 字段的时区/`time.mktime` 处理仍留待抽查与全量裁定。

实际 code 与 paper/report 描述有几处值得保留：

- `boot-and-delete.yaml` 引用的是 `restart_nova_container.sh`，脚本将 `commands-to-exec-nova.sh` 发送到 controller 并执行 `docker restart nova_api`；shell loop 实为 2 次，尽管注释写 5 次。仓库另有 `restart_nova_compute_nodes.sh`，会 stop/start 多台机器上的 `nova_compute`，但当前 Rally YAML 没引用它。这与论文文字所说“compute fault / API container on compute nodes”不一致；本轮应以当前 YAML+report 的实际调用链为证据，不替换成未被该报告调用的脚本。
- Glance Rally hook 指向 `restart_glance_container.sh` → `docker restart glance_api`；脚本 loop 实为 1 次，注释写 10 次。
- Neutron 脚本 stop/start metadata、L3、DHCP、Open vSwitch agent 与 Neutron server；Open vSwitch 命令在脚本中重复一次，sleep 是 `0.1` 秒而不是注释所称的 25 秒。
- Rally report 有计划 trigger 列表，但结果记录数少于部分计划 trigger 数（Nova 7/27、Neutron 12/14）；不能只按 YAML 迭代列表假定每个 trigger 都成功发生。

#### 尚未闭合的时间标注冲突

- Report 中 `output_boot` 记录到 19:36:19，且最后 hook result 在约 19:36 UTC；`pre_MSDS.py`/既有 smoke 取数终点却硬编码 18:18:13。Output_image 结束 17:51，output_network 结束 18:18；不能假设 boot 数据在相同终点停止。
- `IMPORTANT_experiment_start_end_data.txt` 的 `END OF EXPERIMENTS` 是 2019-11-20（早于 start 2019-11-25），其所谓 log/metric range 也用了 2019-11-19 日期；与 Rally reports 及代码时间段矛盾。Zenodo prose 的“logs/metrics CEST、trace UTC −2h”与真实字段/解析脚本转换也并非完全一致。
- 因此可以准确指出**实验级 label taxonomy 与 hook schedule、部分实际 hook event timestamps**；但不能把唯一终止时间、全部 injections、每个 service/pod 的逐行 fault label 当作已核实。要生成全量 row-level label，需先裁定时间范围并在完整 data timeline 上合并各 Rally event windows；本轮未对 4.7 GB uncompressed raw 全量解包/扫描。

### 6. T02/T09 完成状态

- **T02：部分完成**——已固定 SHA、检查预处理和输入模型契约、完成公开数据 ZIP 的 checksum/许可/结构核验、在真实公开子集上原样失败并在最小补丁后完成预处理/窗口 smoke test；全量 preprocess 未运行，训练未运行，论文结果未复现。
- **T09：部分完成**——license、raw schema、数据大小/checksum、官方 fault scripts 与 3 份 Rally reports 的映射已核；31 个 hook windows 已对六个 logs/metrics CSV 成员做 event-window row-count join。尚未确定唯一权威实验终止时间，未逐一读取 11,003 trace JSON、未生成经过验证的完整 row-/pod-level multimodal label file。
- 临时下载与子集已清理；MSTGAD 源码 clone 保留在 `mstgad_repo/` 以便后续审阅，未对官方仓库留下修改。


---

<a id="t03"></a>
## T03｜BARO 执行审计

> 来源文件：`baro_execution.md`。以下为完整正文；标题层级为合集显示下调一级。

## T03 执行记录：BARO × Zenodo artifact（全量 RQ2 + 缓存结果 RQ1）

**执行日期**：2026-10-19（Asia/Taipei）
**范围**：只使用公开 BARO 样例与 Zenodo 数据；没有接触用户数据。
**结论**：BARO library 示例、RQ2 三 benchmark case artifact 复核及 RQ2 全部 300 cases 均成功；缓存结果 RQ1 在三 benchmark 上可快速复算并与当前 artifact README 的“比论文表格略好”输出一致。**未重跑 BOCPD 的耗时版 RQ1、RQ3、RQ4 全套实验**；原仓库 README 估计 RQ1 Online Boutique 全量重算约两小时。本报告将缓存 replay 与算法重算区分开。原始 stdout 与运行记录保存在 `/home/user/baro_execution_logs/`（每条命令一个 `.stdout.log` 和 `.run.txt`）。

### 1. 固定软件与公开数据版本

|对象|执行结果|
|---|---|
|论文版本软件 artifact|Zenodo `10.5281/zenodo.11094092`，version `0.2.0`；Git tag `0.2.0` peeled commit `582d4ead3d3301a34fea4e395bfa3f750fc9a40d`。这比单纯使用滚动 `main` 更适合做论文 artifact 锚点。|
|GitHub 当前滚动 main|clone 当时 HEAD `e35f4ec1095e5cac891d52de9ad18a5b32a37ec8`（2026-02-02）；这不是上述 0.2.0 论文版 tag。本轮 RQ 与 demo 以 tag 0.2.0 执行。|
|数据 artifact|Zenodo `10.5281/zenodo.11046533`，version 0.0.1，开放下载，CC BY 4.0。文件：Online Boutique `fse-ob.zip` 150,740,576 B / MD5 `4c81245a90b02094566111ddac41bdd8`；Sock Shop `fse-ss.zip` 126,610,585 B / MD5 `5603b151ee7b9ebe079d27bc75210763`；Train Ticket `fse-tt.zip` 286,198,459 B / MD5 `9c2f5be0b0c22612031337774ae502f4`。三份实际下载文件的 MD5 都与 Zenodo record 一致。|
|运行环境|Linux `6.1.158+-x86_64-with-glibc2.41`，Python 3.13.16 (GCC 14.2.0)，NumPy 2.3.5、pandas 2.2.3、SciPy 1.17.1、scikit-learn 1.6.1；CPU。BARO checkout 为 detached HEAD (`0.2.0`)，执行前/后 `git status --porcelain` 均为空。未另外安装 BARO 或改代码，直接在 tag 0.2.0 源码中运行。无专门设置随机种子或修改配置。|

来源：[BARO source](https://github.com/phamquiluan/baro)、[BARO software artifact](https://zenodo.org/records/11094092)、[BARO dataset artifact](https://zenodo.org/records/11046533)。

### 2. Artifact 目录与独立故障重复实查

对三个完整 ZIP 的 central directory 逐项核验（没有把文件数只抄自 Zenodo 描述）：

|benchmark|ZIP entries|`data.csv` case 文件数|target×fault 组合|每组合 run 编号|压缩/解压总量|
|---|---:|---:|---:|---|---|
|Online Boutique|1,321|100|20|1–5，全部各 5 次|150,740,576 B / 495,785,998 B|
|Sock Shop|1,321|100|20|1–5，全部各 5 次|126,610,585 B / 480,996,581 B|
|Train Ticket|1,321|100|20|1–5，全部各 5 次|286,198,459 B / 1,586,227,163 B|

每个 case 目录可见 `data.csv`、`simple_data.csv`、`inject_time.txt`、`naive_bocpd.json` 及基线结果/日志/绘图文件。20 个目录组合为每 benchmark 5 个目标 service × 4 种 fault（cpu/mem/delay/loss），每组合 5 次；因此 100 是独立注入运行数，不是窗口数。样例原始 metric 表头是 `time` 加各 `<service>_<metric>` 宽表列；OB/SS `data.csv` 样例 721 个数据行，Train Ticket 样例 961 行。

Zenodo record 标明每 benchmark 100 failure cases；解包元数据检查与该口径一致。临时下载的三份压缩包和解包数据均已在记录后清理，避免在工作区保留约 563 MB 下载和约 2.56 GB 展开数据。

### 3. 最小 library smoke test（不是论文全表复现）

tag `0.2.0` 上执行 README 的基本用法：

```python
from baro.utility import download_data, read_data
from baro.anomaly_detection import bocpd
from baro.root_cause_analysis import robust_scorer

download_data()
data = read_data("data.csv")
anomalies = bocpd(data)
ranks = robust_scorer(data, anomalies=anomalies)["ranks"]
```

输出：样例数据 `(721,48)`；9 个 anomaly indices，首个为 `243`；Top-5 为 `checkoutservice_latency, cartservice_mem, cartservice_latency, cartservice_cpu, main_mem`，与 0.2.0 README 示例一致。它只验证安装/函数链路和示例样本，不代表三套 benchmark 论文结果。

### 4. RQ2：实际读取三份完整 artifact 并运行 300 cases

在 tag `0.2.0` 下，将完整三个 Zenodo archive 临时解压到代码默认 `data/` 路径，使用仓库 RQ2 实现运行：

```bash
python main.py --dataset OnlineBoutique
python main.py --dataset SockShop
python main.py --dataset TrainTicket
```

每个 benchmark 运行 100 个 case（合计 300）。输出与论文 RQ2 coarse-grained Avg@5 一致：

|benchmark|实际输出|论文报告|
|---|---:|---:|
|Online Boutique|0.86|0.86|
|Sock Shop|0.95|0.95|
|Train Ticket|0.81|0.81|

此外，Online Boutique fault-specific RQ2 实跑：CPU 0.91、MEM 0.96、DELAY 0.95、LOSS 0.62、ALL 0.86，和 artifact README 所列一致。

**解释限制**：tag 0.2.0 的 `reproduce_baro()` 从每 case 的 `naive_bocpd.json` 读取已存 anomaly time，再计算 RobustScorer 的 root-cause ranking；因此这次是完整的 RQ2 artifact replay（全部 300 cases），不是从原始 metrics 重新拟合/重算 BOCPD 的全流水线。RQ2 正好是任务目标中的 RCA 结果，不能改写成 service-window anomaly detector 的 F1。

为归档 stdout 和资源信息，执行命令另做一次批量复跑；所有退出码均为 0。每个命令耗时 2.74–17.98 秒，轮询采样到的进程峰值 RSS 约 174–186 MiB（20 ms 采样，非 OS 精确 peak）。各项原始 stdout 与 `.run.txt`（command、exit code、wall time、sampled RSS）见 `/home/user/baro_execution_logs/`。

### 5. RQ1 缓存结果 replay 与全量算法重算的区别

在 `--saved` 下，分别用三个 benchmark 的 100 case saved change-point 文件复算：

|benchmark|Precision|Recall|F1|
|---|---:|---:|---:|
|Online Boutique|0.76|1.00|0.87|
|Sock Shop|0.79|1.00|0.88|
|Train Ticket|0.83|1.00|0.90|

这和论文 Table 2（OB 0.69/1.00/0.82；SS 0.60/1.00/0.75；TT 0.68/1.00/0.81）不同；**并非未发现的结果差异**：tag 0.2.0 README 明确写该耗时重算输出“比 paper Table 2 略好”，并给 Online Boutique 0.76/1.00/0.87；本次三 benchmark replay 与 README 相符。故论文出版表与后期 artifact README 的数值不能混为一版。

`--saved` 是使用保存的 `naive_bocpd.json` changepoint 结果的 replay；没有运行 multivariate BOCPD 计算。README 说明对 Online Boutique 不加 `--saved` 的 Table 2 全量 anomaly detection 要约两小时，本轮不声称重跑该计算。RQ3、RQ4 也未执行。

### 5.1 本次归档重跑耗时与资源（观测值）

|命令类别 / 数据|Wall time|轮询采样峰值 RSS|
|---|---:|---:|
|RQ2 `OnlineBoutique` 100 cases|6.20 s|175.7 MiB|
|RQ2 `SockShop` 100 cases|6.61 s|175.8 MiB|
|RQ2 `TrainTicket` 100 cases|17.98 s|186.1 MiB|
|OB fault CPU / MEM / DELAY / LOSS（各 25 cases）|2.84 / 3.00 / 2.80 / 2.74 s|175.0 / 174.1 / 175.4 / 174.9 MiB|
|RQ1 `--saved` OB / SS / TT（各 100 cases）|2.77 / 2.69 / 5.02 s|174.4 / 174.3 / 182.7 MiB|

RSS 由 `psutil` 每 20 ms 采样，作为估计，不等同于内核精确峰值。完整每条命令退出码和采样记录已保存为 `/home/user/baro_execution_logs/*.run.txt`，console output 在同目录 `.stdout.log`。

### 6. T03 完成状态

- Artifact DOI/version/license、文件名/大小/官方 MD5、实际下载一致性：**完成**。
- 每 benchmark 100 cases、20 target×fault 组合、每组合 5 repeats 的实际 archive 复核：**完成**。
- README library sample：**实际运行成功**。
- RQ2 三系统全 case replay + Online Boutique 分 fault replay：**完成，匹配所报告值**。
- Full BOCPD 从 metric 原始数据重新计算、RQ3/RQ4、全论文全表复现：**未执行**；与上述 cached RQ2/RQ1 replay 分开记录。


---

<a id="t04"></a>
## T04｜ServiceAnomaly 执行审计

> 来源文件：`serviceanomaly_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T04 执行审计：ServiceAnomaly 源码、公开数据与运行入口

**审计日期**：2026-10-19（Asia/Taipei）
**结论**：已完成官方公开仓库/数据树的静态核对及入口 smoke attempt。**仓库公开了部分作者采集的 TeaStore、TrainTicket trace/metric 样例和脚本；不等于论文实验的完整原始训练数据、可直接运行的复现包或可复算的评价套件。**源码缺少依赖/运行说明，实际入口绑定 GUI、Windows API、Live Jaeger/Prometheus 或人工文件选择/输入，且没有自动 P/R/F1 evaluator。本轮未安装、修改或 patch 源码，未声称复现论文结果。

### 1. 固定源码、文档、开放性

|项目|实查结果|
|---|---|
|论文|Panahandeh et al., *ServiceAnomaly: An anomaly detection approach in microservices using distributed traces and profiling metrics*, Journal of Systems and Software 209 (2024), 111917, DOI `10.1016/j.jss.2023.111917`。|
|官方仓库|`https://github.com/M-panahandeh/ServiceAnomaly`；clone branch `main`，HEAD `6acc9dfd7ec50d3007d65e61a188f40d72b422b6`；clone 时最新提交日期 2023-03-09；执行前/后 `git status --porcelain` 为空。|
|文档与依赖|根 README 仅 99 bytes（说明这里有脚本和用于评估的数据）；`ServiceAnomaly/readme` 仅 1 byte。无 `requirements.txt`、`pyproject.toml`、environment lock、数据/运行指南或仓库根 LICENSE 文件。README 所称“data and scripts”不提供复现步骤/版本约束。|
|仓库数据|tracked 总数 1,718 个文件、约 42,331,839 B：TeaStore 18,474,526 B；TrainTicket 23,857,313 B。内容为作者采集的两套 benchmark 特定 traces、dependency/edge 关系及 Prometheus query JSON，不是可直接运行的 TeaStore/TrainTicket 服务环境，也不是同名的其他公开 benchmark artifact。|
|运行环境|Linux `6.1.158+-x86_64-with-glibc2.41`，Python 3.13.16；未安装新依赖/更改源码。基础科学栈可导入，但 `netgraph` 和 `Tools.scripts.dutree` 不存在；无 `$DISPLAY`。|

论文官方预览页说明方法结合 distributed traces 与 profiling metrics、构造 baseline annotated CPG，并报告 TeaStore/TrainTicket F1 最高 0.85/0.86：[ScienceDirect article preview](https://www.sciencedirect.com/science/article/abs/pii/S0164121223003126)。官方仓库：[GitHub](https://github.com/M-panahandeh/ServiceAnomaly)。

### 2. 论文流程与仓库代码路径对照

|论文步骤/模块|仓库实际代码与差异|
|---|---|
|从 normal distributed traces 建 Context Propagation Graph (CPG)：节点为 service、边为 invocation；聚合 request call trees。|`TraceCollection.py` 查询 `http://localhost:16686/jaeger/api/dependencies?endTs=1639432124762&lookback=864000000`，把依赖 API JSON 另包成 JSON string 写入 `traces.json`。这是固定时间窗的 Live Jaeger dependencies 查询，不是从仓库 baseline raw spans 批量构建。两个 `main*.py` 均读取 cwd 下的 `traces.json`；仓库未在脚本所需位置提供该入口文件。`DAG.py` 仅取 `parent` / `child` 建 NetworkX graph，并忽略 `callCount`。`CPG-generation*.py` 是 Tk 文件对话框驱动的单条 trace 可视化/转 DAG 辅助脚本，不是批量训练入口。|
|metric relationship annotation：六项 profiling metrics、linear/nonlinear relationships。|`MetericsProfiling.py` / `MetericsProfilingTeastore.py` 读 Prometheus API range response 的 `data.result[].values`，按 timestamp 聚合平均，再将 timestamp 转成本地秒级字符串，横向拼接并 `dropna()`；未做时间容忍 join，秒级格式可能合并不同采样点。指标名为 request_duration、request_byte、response_byte、queue_size、latency、throughput。TeaStore 特化版只生成 5×5 NMI matrix、5 列 metric frame，并在 concat 中排除 throughput；对应 TeaStore baseline 中 throughput 查询全为空，故此路径实际不是六指标。|
|linear/nonlinear relationship fitting。|`MetricLearning*.py`：absolute Pearson correlation 阈值 0.6；线性回归在同一 baseline 数据上拟合并保存训练 RMSE。nonlinear 路径直接将连续 metric 值交给 `normalized_mutual_info_score`（没有看到离散化），筛选 NMI 阈值与低相关组合，RBF SVR (`C=100,gamma=0.1,epsilon=0.1`) 同样在完整 baseline 上拟合并保存训练 RMSE。`main.py` 的 NMI threshold 为 0.47；`mainTeastore.py` 为 0.6。训练内残差 RMSE 后续直接被用作 test deviation cutoff；没有独立 validation/calibration split。|
|CPG matching + metric anomaly decision。|`Test*.py` 对 test graph 调 `GraphMatcher(...).subgraph_is_isomorphic()` 并只打印布尔值；主程序不保存/结合 graph match 预测标签。对每条 test relation，逐项比较单值 metric 与 baseline regression/SVR prediction，若绝对残差大于已存训练 RMSE 就计 violation。`Test*.test_metric_gathering()` 通过 Tk 目录选择器读取 JSON，并只取每个 metric `values` 的第一个样本值；缺值处理主要是宽泛 `except: pass`。不存在将 violation/graph match 映射到 trace-level label 的自动评估脚本。|
|论文 P/R/F1 与 Table 4 多种错误度量。|源码虽计算 R²、MAE、MAPE、MSE、RMSE、MedAE、max error，但实际记录/判断主要使用 RMSE；没有 `precision/recall/F1`、TP/FP/FN 统计或批量 evaluation 脚本。无法从仓库入口复算论文表格。|

#### 入口具体阻碍

- `ServiceAnomaly/main.py`：先 `tk.Tk()`；读 `traces.json`；随后又硬编码 TrainTicket topology (`main.py` 第 88 行附近，`dag.edges` 被写死)，metric 输入路径默认为 cwd；训练关系结束后调用 `ctypes.windll.user32.MessageBoxW`（Windows-only），再进入无限 `input()` 循环并 `eval()` 用户输入，test metrics 又每条边打开 GUI folder picker。
- `mainTeastore.py`：相同结构、硬编码 TeaStore topology；它调用 Teastore metric grouping，并有相同 Windows-only message box / Tk / 人工输入。
- metric collection helper 要求本机 Prometheus `localhost:9090`；trace dependency helper 要求 Jaeger `localhost:16686`，且 endTs/lookback 硬编码。源码未提供 Docker Compose/Kubernetes manifest 或复现实验服务启动说明。
- `DAG.py` 无条件 import `netgraph`（实际未用）及 `Tools.scripts.dutree.display`（未用，当前 Python 3.13 无 `Tools` 模块）；这两个 import 可在业务逻辑运行前阻塞。

### 3. 仓库数据与论文声称数据对应情况

论文 Data collection 描述：baseline CPG 使用 TeaStore 3,000 traces / 9,160 metric rows、TrainTicket 5,000 traces / 5,410 metric rows；test 分别 15 normal + 15 anomaly traces、32 normal + 32 anomaly traces。测试数据 case 数与仓库顶层目录数大致对应，但 baseline raw trace 覆盖、case 文件完整性、metric 质量不完全一致。

|数据项|仓库实查|论文/解读|
|---|---|---|
|TeaStore baseline raw trace|`data/TeaStore/baseline/Traces.json` 只有 100 个 Jaeger trace records、合计 204 spans。|论文称 3,000 baseline traces。仓库文件不能代表论文所述 3,000 个训练 traces。|
|TeaStore baseline metrics|`baseline/metrics/` 有 72 个 JSON（12 source→destination 组合 × 6 名义指标）；每组 throughput 文件均为空 `data.result=[]`（12/72）。|训练实现 `MetricLearningTeastore.py` 实际只建 5 指标矩阵；empty throughput 不可当成健康 0。其余指标每文件约 8,000–9,300 Prometheus sample values，表明 metric rows 大体在论文声称规模附近，但不能补足缺少的 raw baseline traces。|
|TeaStore test|15 normal case dirs + 15 `fault-injected` case dirs，均有 parseable parent/child relation-list 文件。有效 Jaeger raw trace JSON：normal 12/15，fault 15/15。normal metric files 229（38 empty result）；fault metric files 180（55 empty result）。|case 数与论文 15+15 一致，但正常 trace 原始 JSON 缺 3 份；逐服务 metric coverage 不齐。目录名隐含 stress/latency/timeout/hangup fault 类别，无独立标签表。|
|TrainTicket baseline|`baseline/001`, `002`, `003` 三组 metric snapshot，共分别 120、126、126 个 JSON metric 文件（合计 372）；各组各有 3 个 empty result。`003/trace/traces.json` 是双重 JSON 编码的 Jaeger **dependencies** response（17 条 parent-child links），不是 5,000 个 raw Jaeger traces。|论文称 5,000 baseline traces / 5,410 metric rows；仓库并未公开该批 raw training traces。三个 metric snapshot 的含义/划分也没有 README 解释。|
|TrainTicket test|32 normal + 32 `injected faults` case dirs。全部 32 normal relation files 可由 Python literal parser 读取；fault cases 中 31/32 可读取，`timeout1-consign-24/trace` 语法损坏。有效 raw Jaeger trace JSON：normal 19/32，fault 31/32。metric files：normal 270（3 empty）；fault 408（130 empty result）。|顶层 case 数与论文 32+32 相符，但一个 fault graph list malformed；raw spans 并非每 case 都在，metric empty/missing coverage 较明显。用文件夹数量不能推断完整论文数据已发布。|

**字段/格式**：raw trace 文件一般为 Jaeger API response `data[] → traceID, spans[]`，span 带 `spanID`, `references`, `startTime`, `duration`, `tags`, `processID`；不同场景有些只有 edge-list relation text，TrainTicket 有的 relation-list 文件被命名为 `trace` 或 `.json` 但不是 JSON。metric 文件是**JSON 编码的 JSON string**，第二次 `json.loads()` 后结构为 Prometheus `data.result[]`，每个 series 有 label 与 `values=[[timestamp,value], ...]`；77-byte metric payload 在这些 case 中表示 `result=[]`，不是数值 0。Fault identity 主要由目录名提供，未发现统一 label/split manifest。数据没有 logs 模态。

数据文件均位于官方仓库固定 commit；没有在源仓库找到 LICENSE 文件/数据独立许可证，故不能据此推定仓库中全部采集数据有明确再分发许可。

### 4. 实际入口 smoke attempt

从 `/home/user/serviceanomaly_repo/ServiceAnomaly` 在当前环境实际执行，stdout/stderr 保存在对应文件：

|命令|退出码/实际结果|
|---|---|
|`python main.py`|1；`ModuleNotFoundError: No module named 'netgraph'`，在 import 阶段停止。|
|`python mainTeastore.py`|1；同样先因缺 `netgraph` 停止。|
|`python CPG-generation2.py`|1；`_tkinter.TclError: no display name and no $DISPLAY environment variable`。|

记录文件：`serviceanomaly_main.stdout.log`、`serviceanomaly_mainTeastore.stdout.log`、`serviceanomaly_cpg_generation2.stdout.log`。静态检查确认即使补上 `netgraph`，当前环境仍找不到 `Tools.scripts.dutree`，且后续流程仍受无 GUI、`ctypes.windll`, Jaeger/Prometheus、人工文件选择/输入、缺入口数据与路径配置所阻。未为得到“能运行”的表象而装包、改源码、复制/重命名数据或假造标签；不执行完整训练/评价。

### 5. T04 结论与状态

- **论文方法—源代码**：部分可追踪，relationship fitting 与逐 test edge violation 代码存在；未发现论文表格所需的自动化 trace-level evaluator 或可复用的完整 end-to-end runbook。
- **数据身份**：同名 TeaStore/TrainTicket benchmark 的作者采集数据；test case directory counts 大体呼应论文，但公开树的 raw baseline trace 数/有效 test trace、relation files 和 metrics completeness 存在缺口。**不得拿其他 TrainTicket dataset 或公开运行环境替代为“已复现作者数据”。**
- **评价协议**：paper-level positive/negative unit 是 trace；repo 测试脚本目前是人工单 case graph/metric 交互，不可自动计算 P/R/F1。阈值来自 baseline 同数据 fit 的 RMSE，未见 validation/calibration；不能称为严格独立测试校准。
- **执行判定**：审计与运行障碍核验完成；官方原样入口未能启动，论文表格复算未完成。T04 归档状态为“审计完成；端到端运行受阻/数据复现不完整”，不声称复现。


---

<a id="t05"></a>
## T05｜Eadro 执行审计

> 来源文件：`eadro_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T05 执行审计：Eadro × Zenodo 数据、监督协议与端到端运行条件

**审计日期**：2026-10-19（Asia/Taipei）
**判定**：公开仓库、数据 DOI 身份、archive checksum/schema、预处理契约及训练/评价源码已核对；README 命令实际尝试但缺 PyTorch，预处理命令缺 `tick`。**未安装老版本依赖、未自写缺失的 raw→parsed 转换器、未训练模型、未复算表格。** 开放数据在原始层可下载，但代码所需中间文件未包含，也没有完整 parser/runbook；当前仓库不能原样端到端复现论文结果。

### 1. Repository 与 Zenodo DOI 身份

|对象|核验结果|
|---|---|
|论文|Lee et al., *Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data*, ICSE 2023, DOI `10.1109/ICSE48619.2023.00150`；公开预印本 [arXiv:2302.05092](https://arxiv.org/abs/2302.05092)。|
|官方源码|[BEbillionaireUSD/Eadro](https://github.com/BEbillionaireUSD/Eadro)，`main` HEAD `82ff9c9a7a58a9cd2b713aa78754a73c543a15db`；无 tag，repo 工作区 clean。GitHub 最新 commit 为 2023-02-07。README 显示 version 0.1。|
|DOI 记录关系|GitHub README 链接 `10.5281/zenodo.7615393`，这是 Zenodo **concept DOI**（conceptrecid 7615393）；它重定向到唯一发布版本 `v1` 的 version DOI `10.5281/zenodo.7615394`（record id 7615394）。不是两份不同数据或错误 ID：引用具体文件/版本用 `.7615394`，泛指数据集用 `.7615393`。|
|Record access / license|Zenodo API 标记 `access_right: open`、version `v1`；record metadata 未返回 `license` 字段，页面也未列出可确认的 license。故只能确认可公开下载，**不能把 open access 误写成 CC BY/MIT 或推定再分发许可**。GitHub checkout 也未见 LICENSE 文件。|
|ZIP files|`SN Dataset.zip`: 78,753,809 B, 官方 MD5 `4c523faf64770b1e9a709cdd610e07c4`；`TT Dataset.zip`: 48,923,113 B, 官方 MD5 `2d77861e4a025781e598d51866fd6f4e`。两文件实际下载后计算 MD5，均匹配 Zenodo。|
|运行环境|Linux `6.1.158+-x86_64-with-glibc2.41`，Python 3.13.16；环境无 `torch`、`tick`。requirements 固定 `dgl==0.9.1`, `numpy==1.21.5`, `pandas==1.4.2`, `scikit_learn==1.1.2`, `tick==0.7.0.1`, `torch==1.12.1`；没有 CUDA/OS/完整安装锁定说明。|

来源：[Zenodo version page](https://zenodo.org/records/7615394)、[Zenodo API record](https://zenodo.org/api/records/7615394)、[GitHub](https://github.com/BEbillionaireUSD/Eadro)、[paper/arXiv](https://arxiv.org/abs/2302.05092)。

### 2. Artifact 下载与结构核验

两个 Zenodo zip 已实际下载、MD5 匹配，并检查 ZIP central directory；本次未全部展开（避免重复写入约 1.29 GB 解压数据）。checksum/schema 核验后已删除临时下载 zip，保留来源、大小、hash 与中央目录审计结果。

|benchmark|压缩文件|ZIP entries / regular files|entry file size 合计|归档内容概况|
|---|---:|---:|---:|---|
|SocialNetwork (SN)|78,753,809 B|158 / 147|669,505,265 B|4 个 fault 运行批次（每批 `spans.json`、`logs.json`、每 service metrics CSV、fault metadata）；另有 3 个 normal `tar.xz`；重复封装的 `SN-all.tgz` 包含约 618 MB 记录。|
|TrainTicket (TT)|48,923,113 B|598 / 577|617,624,509 B|9 个 fault 运行批次；另有 3 个 no-fault archive（含 `TT-no-fault.txz`）；重复封装的 `TT-all.txz` 包含约 597 MB 记录。ZIP 有 AppleDouble/`.DS_Store` 元数据。|

抽查到的原始契约：

- Fault metadata JSON 顶层是 `start`, `faults`, `end`；每条 fault 含容器 `name`、fault type、开始时间和 duration。每个 fault-run JSON 中的 fault records 计数是 9：SN 4 份共 36，TT 9 份共 81。论文描述总 injection operations 为 SN 72、TT 162；artifact record 数与论文 operation 数相差约 2×，具体计数单位未在 README/Zenodo 说明，不能把两者直接等同。
- 每个 raw run 包含 `spans.json`（Jaeger 风格 trace list，带 `traceID`, `spans`, `startTime`, `duration`, `references`, `tags` 等）、`logs.json`（以 service 为 key 的原始 log-line 列表）和 `metrics/{service}.csv`。抽查 CSV 头为 `timestamp` 加 7 列：`cpu_usage_system`, `cpu_usage_total`, `cpu_usage_user`, `memory_usage`, `memory_working_set`, `rx_bytes`, `tx_bytes`；数据是一秒时间戳的服务级 metric series。
- 日志时间是各自格式的无时区字符串（SN 与 TT 格式不同）；metric 是 epoch 秒；span `startTime` 是微秒级 epoch。运行器需要统一 service ID、时间单位/时区、事件模板和 window 边界；仓库没有把这些 raw files 转换为模型所需中间 schema 的解析器。

论文对照：预印本报告 TT/SN 共 48,296 / 126,384 traces，并以不同时段划分训练约 60%、测试约 40%；artifact metadata/每批运行数与该总量不是一一对应的，README 没有给出批次到 train/test 的映射。

### 3. Raw schema 到代码输入的断点

Zenodo raw data **并非** `codes/preprocess/align.py` 的直接输入。代码路径实际要求：

|代码期望|Zenodo 提供|缺少的转换/说明|
|---|---|---|
|`parsed_data/<name>/records<i>.json`：`start`, `end`, `faults[{s,e,service}]`|原始 `*.fault-*.json`：`start`, `end`, `faults[{name,fault,start,duration}]`|缺少从容器名到 `Info.service2nid` service name 的正规化；`start/duration` 到 `s/e` 的转换和多fault 标签规则没有 converter。|
|`logs<i>.csv`：至少含 `timestamp`, `service`, `event`；另需 `templates.json`|`logs.json` 是 service→raw strings|代码仓库无 Drain parser、template 文件生成器或 raw log→CSV 命令。论文提到 Drain event parsing，但此步未在发布源码覆盖。|
|`traces<i>.json`：整数秒 key 到 service-node/span latency list 的预聚合映射|`spans.json` 是 raw trace list/Jaeger span 对象，startTime 微秒级|`single_process.py` 假设 trace 已聚合成以秒为 key 的字典；没有 raw span parent/child、service name、duration → 该 dict 的转换器。|
|`metrics<i>/<service>.csv`：timestamp index + `Info.metric_names` 的 7 个数值列|Zenodo metrics CSV 的列名和值大体可直接映射|CSV 列契约最接近代码预期，但仍需放到无文档的 `parsed_data` 目录结构，并保证每个 service 每秒行数恰为 chunk length。|

GitHub 当前只有 `align.py`, `single_process.py`, `util.py` 三个 preprocessing 文件；未找到 parse/raw import 代码。README 只写 `cd codes & python main.py --data <dataset>`，没有 raw dataset extraction、`parsed_data` 建立或 preprocess 命令顺序。`align.py` 中 `get_all_chunks()` 依据 `name[0] == "A"` 选择 TrainTicket，否则选 SocialNetwork；README/Zenodo 没说明该内部 dataset name 命名约定，archive 目录却为 `TT`/`SN`。

### 4. 模型与监督协议核查

论文方法包含 modality-specific encoders、multi-source gated fusion、依赖图 GAT、联合 detection/localization 两任务；代码高层组件大致可对应，但当前发布源码里有多个原样运行阻碍：

- `single_process.py` 用 `tick.hawkes.HawkesADM4` 从 parsed log events 估计 Hawkes baseline intensities；`model.py` 用 log FC embedding、dilated causal ConvNet (trace/KPI)、GATv2、GlobalAttentionPooling、fusion + GLU，以及 binary detector / service localizer 双头。`align.py` 标签 `-1` 表正常、fault service node ID 表异常根因；`MainModel` 目标是 `alpha * detect_loss + (1-alpha) * locate_loss`。这是**有监督**检测+定位训练，不是 normal-only 训练。
- 代码静态硬错误：`model.py` 中 `trace_dropout`、`metric_dropout` 未定义；`MainModel.__init__` 定义 `detecter` / `localizer` / `detecter_criterion` / `localizer_criterion`，但 `forward()` 调 `self.detector`, `self.locator`, `self.decoder_criterion`；`base.py` 在 `evaluate()` 调 `np.zeros` 但没有 import NumPy；`chunkDataset` 要求 `edges` 参数，但 `main.py` 构造时未传 `edges`。即便补齐 torch/DGL/tick，也需要 debug/修改才可能走通模型。
- `BaseModel.evaluate()` 的 detection confusion accounting 把正确预测的 normal（TN）和正确检测的 anomaly 一起累计在 `TP`；因此该函数算出的 `Pre/Rec/F1` 不是常规以 anomaly 为 positive class 的 P/R/F1。其 `pos=TP+FN` 又包含 TN，HR@k 分母也可能不是仅 fault traces。需要先修正指标实现，再与论文数值比较。
- `BaseModel.fit()` 在训练期间反复对 `test_loader` 计算 HR@1，并用 test HR@1 选择/保存 best checkpoint；这意味着 test set 参与模型选择，不是只在最终评测用一次。没有 validation split 的独立使用证据。
- `align.py` 以 `chunk_lenth=10`、fault-overlap threshold 1 默认生成步长为 1 的重叠 windows；`test_ratio` CLI 默认 0.3，`np.random.shuffle` 随机拆 window，和论文所述按不同时段 60/40 的描述不一致（用户可改 test_ratio，但没有 chronological split implementation）。相邻高度重叠窗口可能被随机分到 train 与 test 两边，产生泄漏。metrics normalization 在整个 run 上先算均值/标准差，trace latency 也在全 run 上标准化，然后才拆分，属于使用 test 时段分布的 transductive preprocessing。

这些是源码可静态定位的问题，不是通过运行后推测；未应用 patch，也未伪称运行成功。

### 5. 实际运行记录

README 推荐在 `codes/` 运行 `python main.py --data <dataset>`。另尝试从 `codes/preprocess/` 调用 `python align.py --name A_TT`：

|命令|结果|
|---|---|
|`python align.py --name A_TT`（cwd=`codes/preprocess`）|退出 1：import `single_process.py` 时 `ModuleNotFoundError: No module named 'tick'`。即使装好 tick，代码仍会依赖不在 GitHub/Zenodo 的 `parsed_data/A_TT/records0.json` 等中间文件。|
|`python main.py --data A_TT`（cwd=`codes`，README 的入口形式）|退出 1：`ModuleNotFoundError: No module named 'torch'`，在 import 阶段停止。|

stdout/stderr：`/home/user/eadro_align.stdout.log`、`/home/user/eadro_main.stdout.log`。未安装 requirements（钉死 PyTorch 1.12.1 / DGL 0.9.1 / tick 0.7.0.1，与当前 Python 3.13 不匹配风险高），没有继续做无依据的修复。完整训练、fresh P/R/F1 与 HR/NDCG 复算均未执行。

### 6. T05 结论

- **数据 DOI 表面差异**：GitHub 指向 concept DOI `.7615393`；实际文件 version DOI 是 `.7615394`，Zenodo metadata 明确两者关系。version `v1`、zip filename/大小/MD5 已核验，实际下载 checksum 匹配。
- **监督与指标**：论文/代码均是基于故障服务标签的 supervised multitask AD+RCA；窗口 culprit 标签直接训练 binary detector 和 service localizer。代码声称的 P/R/F1 与其当前 counter implementation 存在不一致，HR denominator 也需审计。
- **可复现结论**：数据 raw archives 可获取；代码发布没有 raw→parsed converter，要求的中间 `parsed_data` 未随 GitHub/Zenodo 提供；README 也没有预处理命令/运行顺序。当前环境的入口分别因 `tick`、`torch` 缺失而停住，源码另有静态硬错误和评价协议问题。故 T05 是“公开数据和源码审计完成；端到端论文复现受阻”，而非结果复现完成。


---

<a id="t06"></a>
## T06｜DeepTraLog 执行审计

> 来源文件：`deeptralog_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T06 执行审计：DeepTraLog 数据链路、Deep SVDD 训练协议与代码实现

**审计日期**：2026-10-19（Asia/Taipei）
**结论**：公开论文、作者项目页、官方 GitHub 数据目录及固定版本源文件已核对；下载并检查一个小型 raw fault sample；模型入口在当前环境因缺 PyTorch 停止。**没有克隆/下载全部约 1 GB repository data、没有从 raw 重建 TEG、没有训练或重算论文指标。** 项目提供 raw traces/logs 与 processed graph data，但 raw→TEG/log parsing/embedding 的代码并未随仓库发布；目前不能声称完整端到端复现。

### 1. 论文、作者项目页、仓库版本

- 论文：Zhang et al., *DeepTraLog: Trace-Log Combined Microservice Anomaly Detection through Graph-based Deep Learning*, ICSE 2022, DOI [`10.1145/3510003.3510180`](https://dl.acm.org/doi/10.1145/3510003.3510180)。作者 PDF：[icse22-DeepTraLog.pdf](https://cspengxin.github.io/publications/icse22-DeepTraLog.pdf)。
- 作者项目页：[FudanSELab DeepTraLog](https://fudanselab.github.io/DeepTraLog/)，分别将 `TraceLogData` 说明为 Raw Dataset、`GraphData` 说明为 Processed Graph Data、`results` 说明为实验结果。
- 当前可访问的官方仓库：[FudanSELab/DeepTraLog](https://github.com/FudanSELab/DeepTraLog)，`main` HEAD 经 `git ls-remote` 核实为 `04376dcb99d4b0fd0c50850255c2c4f9ec9503c1`（2026-01-05）；无 tags。GitHub API repository size `1,057,012 KB`，license 元数据为 null，根目录未见 LICENSE。
- 作者项目页的 raw-data 链接仍指向 `github.com/DeepTraLog/DeepTraLog`，截至审计日该 URL/API 返回 404；当前仍可访问的数据/代码仓库是上述 `FudanSELab/DeepTraLog`。当前根目录只有 `.gitattributes`（空）、`GraphData/`、`TraceLogData/`、`model/`、`results/` 和 109-byte 的 README（仅标题/论文名），没有安装、预处理或训练说明。
- 因 GitHub 标示仓库约 1.06 GB，本次未全量 clone；以固定 HEAD SHA 通过 GitHub API/raw 获取了小型源码文件供静态核查。`/home/user/deeptralog_repo/model/` 保存了这些 SHA 对应的 `main.py`、`dataset.py`、`model.py`、`utils.py` 和 `requirements` 文件；这是源码快照，不是完整 git clone。

### 2. 数据目录与 raw sample 实检

GitHub API recursive tree 文件大小（非压缩后大小）如下：

|目录|文件数|API 报告大小|内容|
|---|---:|---:|---|
|`TraceLogData/`|29|546,866,246 B|14 个 fault case ZIP（`F01.zip`–`F14.zip`）、normal ZIP、5 段 `alllog.zip.001`–`.005`。|
|`GraphData/`|8|181,965,367 B|`graph_data.z01`–`.z07` 每片 25,600,000 B，加 `graph_data.zip` 2,765,367 B；项目页说明它是基于 raw data 的 processed graph data。|
|`results/`|10|253,047,502 B|DeepTraLog 及 TraceAnomaly、DeepLog、LogAnomaly、MultimodalTrace、GRU-based Deep SVDD 等结果 ZIP/分卷。|

没有官方 checksum manifest。为看 raw schema，仅下载 `TraceLogData/F05.zip`（572,551 B），SHA-256 `c4c72a200c73e337da7198ba8c0319bcc9bc5a67b7f0849b5a6a509396573338`；GitHub blob SHA 与 API 文件记录 `40a5875c48a305621c7bcba92124faf2b20a3e1a` 一致。未下载其余数百 MB 数据。

`F05.zip` 有 F05-01 至 F05-05 五组；每组为一个 `ERROR_*SpanData*.csv` 和 `*raw_log*.log`。抽样 Span CSV header：

`StartTime,EndTime,URL,SpanType,Service,SpanId,TraceId,Peer,ParentSpan,Component,IsError`

抽样 log 记录含 SkyWalking `SW_CTX` 上下文（service、instance、trace/span ID 等）及应用日志。故 raw 数据确实包含可关联的 spans 与带 trace/span context 的 logs；原始格式不是 PyG graph dataset。

论文称 TrainTicket V0.2.0，45 个服务；全数据 132,485 traces、7,705,050 log messages；其中 23,334 条异常 trace（17.6%），源自 14 类 fault cases 中的 73 个 faults。GitHub raw folder 名与这 14 个 case 相符。未完整下载数据逐条核对论文总数，故这些总数是论文报告，不是本轮重新计数。

### 3. 论文方法与发布代码之间的数据链路

#### 论文规定的流程

1. Drain 将原始 logs 解析为 log events，并保留 trace ID / span ID。
2. 按 trace/span 解析 span 类型、operation、parent/child 及时间，生成 client/server request/response、producer/consumer 等 span events。
3. 文本清理、停用词/复合词处理；对 log event 与 span event 分别算 TF-IDF，使用 300-d GloVe word vector 得到 event embedding。
4. 为每个 trace 构造 TEG：节点是 log/span events；四种边是 sequence、synchronous request、synchronous response、asynchronous request。
5. GGNN message passing + soft-attention readout 得 trace embedding；用 soft-boundary Deep SVDD 训练 one-class model。中心 `c` 由训练 normal traces 初次 forward pass 的均值初始化；soft-boundary radius 以 `(1-ν)` quantile 更新；score 为 `||h-c||²-R²`，score > 0 判异常。

#### 当前仓库代码实际覆盖

- `model/dataset.py` 不是 raw log/span parser：`GraphDataset.process()` 读已生成的 graph JSON lines（包含 `node_info`, `edge_index`, `edge_attr`, `trace_bool`, `trace_id`, `error_trace_type`）和预先生成的 `embedding_id_url+type.csv` 向量表，形成 PyTorch Geometric `Data` 并写 `processed.pt`。它没有生成 log templates、Drain event、span event、TF-IDF、GloVe vectors、TEG JSON 的脚本。
- 作者项目页确认 `GraphData` 是已处理的 graph data；当前源码需 `root/raw/` 中匹配 `process` 的图文件，以及 `id_url+type.csv`、`embedding_id_url+type.csv`。但训练入口 `model/main.py` 默认路径是 `../gae/data/`；当前仓库根目录没有 `gae/data/`，也没有解释如何从 `GraphData` archive 解出/搬到此路径或如何生成所需 CSV。
- 训练模型部分确实实现 GGNN、全图 GlobalAttention readout、center 初始化、soft-boundary Deep SVDD、`ν=0.05`、100 epochs、batch 32、hidden size 300/3 layers。类别映射为 `trace_bool=True → y=0 (normal)`、否则 `y=1 (anomaly)`。
- 划分逻辑：无固定 seed 地分别 shuffle normal/anomaly 索引；60% normal 进 train、接着 10% normal 做 validation、最后 30% normal 加全部 anomalies 做 test。train anomalies 为 0，符合 one-class/normal-only 训练；论文的 60%/10%/其余 normal+全部 anomalies test 协议在比例上吻合。`val_loader` 只含 normal；代码计算 validation scores 但不用于选阈值，测试决策阈值仍是 SVDD score `> 0`。数据顺序和 random seeds 未固定，逐次 split 不完全可复现。
- `model/model.py` 的 `GGNN.forward()` 只调用 `self.ggnn(data.x, data.edge_index)`；虽 Dataset 解析并保存 `edge_attr`，模型并不把 edge type 喂入 GatedGraphConv。按当前代码，TEG 的四种关系都进入同一个无类型 edge_index 通道，无法从模型层面区分四类边，和论文强调的 relation semantics 有实现落差。
- 论文 Eq. (6) 的权重正则项 λ=0.001；当前 `main.py` 实际 SVDD loss 只写 soft-boundary 项，正则交给 `AdamW(weight_decay=0.0001)`，并非论文报告数值/形式。应视为实现与实验设置差异，需对照作者 intended code 再确认。
- `dataset.py` 使用 `np.float`；当前环境 NumPy 2.3.5 中该别名不存在，raw graph dataset processing 会报错。requirements 文件只有两行：`pytorch==1.8.0`、`torch_geometric==1.6.1`；未锁 numpy、scikit-learn 或完整 PyG 扩展依赖。论文实验环境则报告 Python 3.8.8、PyTorch 1.8.0、PyG 1.7.2，和 requirements 的 PyG 版本不同。主入口 import `sklearn.metrics`，requirements 未列 scikit-learn。

### 4. 实际运行和限制

|项目|结果|
|---|---|
|`python main.py`（cwd=`/home/user/deeptralog_repo/model`）|退出 1：首行 import `torch` 报 `ModuleNotFoundError: No module named 'torch'`。stdout/stderr 保存于 `/home/user/deeptralog_main.stdout.log`。|
|当前环境|Python 3.13.16、NumPy 2.3.5；无 PyTorch/PyG。未安装旧依赖；也未尝试直接全量下载约 1 GB 仓库/约 547 MB raw data。|
|完整运行前置|需要兼容的 Python 3.8/PyTorch/PyG 栈；补齐 scikit-learn/NumPy 等依赖；明确 graph archive 解包与 `../gae/data/raw` 目录契约；解决 NumPy `np.float`；获取未在仓库找到的 raw→Drain/TF-IDF/GloVe/TEG 生成流程。|

### 5. T06 判定

- **已核实**：官方 HEAD、论文数据与方法、作者页对 raw/processed/results 目录的定义、GitHub data file count/size、一个 raw fault ZIP 的实际 schema、源码数据输入契约、one-class split、center/radius/score 及实现差异。
- **未完成/不可声称**：未全部下载/解压 1 GB 数据，未复算 trace/log 总数，未重建 TEG，未跑训练/测试，未复算论文 `P=0.930, R=0.978, F1=0.954`。
- **复现判定**：数据本身公开可访问，但当前发布包缺 raw→graph parser 和可操作 runbook；训练入口需要的已处理文件目录没有明确部署步骤，当前环境也缺 PyTorch。T06 的仓库/方法/数据契约审计完成，论文端到端结果复现受阻。


---

<a id="t07"></a>
## T07｜TraceAnomaly / TraceRCA 执行审计

> 来源文件：`traceanomaly_tracerca_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T07 执行审计：TraceAnomaly 与 TraceRCA

**日期**：2026-10-19（Asia/Taipei）
**范围**：官方公开仓库的固定版本、数据发布入口、源码执行/脚本顺序；未运行论文完整训练/评估，也未安装旧依赖或修改代码。
**环境**：Python 3.13.16；未安装 TensorFlow、`loguru`、`diskcache`、`imblearn`、`mlxtend`、`zhusuan`、`tfsnippet`。没有 patch，未建立旧版环境。命令记录：[`traceanomaly_smoke.log`](traceanomaly_smoke.log)、[`tracerca_smoke.log`](tracerca_smoke.log)。

### 摘要判定

|项目|官方仓库固定版本|数据|可执行性/当前结论|
|---|---|---|---|
|TraceAnomaly|`NetManAIOps/TraceAnomaly`, HEAD `946e44791b7c19eddd61b0a7fa98e54100e1c03a`（2020-12-02）|仓库内 TrainTicket text vectors；3 个 ZIP 有明确 train/normal-test/abnormal-test roles，sha256/row count 可实测。|主入口 `run.sh` → `python -m traceanomaly.main`；当前因缺 TensorFlow 在 import 时退出。README 要 Python 3.6/TensorFlow 1.5，当前 Python 3.13 不匹配；未安装/patch。该源码实现 normal-only VAE+posterior flow likelihood scoring，但没有实现论文所述 KDE/p-value threshold 或 root-cause localization，也不含自动评价器。|
|TraceRCA|`NetManAIOps/TraceRCA`, HEAD `b9087c72d201fbb17d724114cd389de4854815ad`（2023-04-03）|README 指向 OneDrive 与清华云盘；Git repo 本身没有 `data/` / processed `.pkl`。清华 share 于审计日返回“Link does not exist”；OneDrive HEAD request 返回 403（无法由本环境确认链接是否失效）。|Makefile 依赖绝对路径 `/TraceAnalysis/tracerca-exp` 与手工准备数据；没有 requirements/environment 文件。`python run_selecting_features.py --help` 因缺 `loguru` 退出；`make -n target` 在默认路径因数据及多个引用脚本缺失而失败。执行顺序可静态重建，但不声称完整跑通。|

### 1. TraceAnomaly

#### 1.1 版本、依赖与数据身份

- 官方仓库：[NetManAIOps/TraceAnomaly](https://github.com/NetManAIOps/TraceAnomaly)，浅克隆 branch `master`，HEAD `946e44791b7c19eddd61b0a7fa98e54100e1c03a`（2020-12-02），无 LICENSE 文件；未推断代码/数据再分发许可证。
- README 指定 Python `==3.6`，`requirements.txt` 包括 `tensorflow==1.5`、`zhusuan` Git HEAD（未固定 commit）、`tfsnippet@v0.2.0-alpha1`，及 NumPy/Pandas/Scikit-learn 等。该栈高度依赖 TF1 兼容时代，当前 Python 3.13 环境没有 TensorFlow。README 另给 Docker image `silence1990/docker_for_traceanomaly:latest`，本轮没有 pull/运行。
- 三份已提交 ZIP 的 SHA-256、MD5、实际文本行数与向量维度：

|文件|压缩字节|MD5|SHA-256|ZIP 内未压缩 member|实际记录|向量维度|
|---|---:|---|---|---|---:|---:|
|`train_ticket/train.zip`|7,875,674|`d37010e22fa0f7bcb8f2c1437decc347`|`01068a7603d821a7cdce65d5147d04aff750b466f2eeab8e46803dd52be62f2e`|`train`, 187,671,891 B|100,000|890|
|`train_ticket/test_normal.zip`|1,764,831|`45a9fa8ade942794821119ccdb7a2fbc`|`847ed23cbffb563474862967f96b85dcfcf61390d3e07741384c2327a2c12737`|`test_normal`, 56,151,973 B|30,355|890|
|`train_ticket/test_abnormal.zip`|1,224,372|`9789787feae10df15e0a4525ceea8cf6`|`b036bac0e2dc9f31b4ae943c753c8c5076ecc1b2c2a64f3732e3bc8973312ee5`|`test_abnormal`, 10,836,080 B|5,079|890|

  每条文本行是 `trace_id:comma-separated vector`，3 个文件每条记录有 890 个 float tokens，IDs 在各自文件内唯一；ZIP member 内容以只读流检查，未执行 pickle。README 将它们直接标成 training set、normal test traces、anomalous test traces；属于作者发布的 TrainTicket 输入文件，而非本轮新发现的第三方替代数据。仓库另含 `idx.pkl`，本轮不反序列化（pickle 不可信）；未发现 preprocessing/raw-span → 890-d vector 生成器，故只核实了 model-ready vector release。
- README 写 `./run.sh`，而 `run.sh` 传入 `./train_ticket/train`、`test_normal`、`test_abnormal`，仓库实际仅有 `.zip`，未有这三份解压后的无扩展名文本文件。必须先手工解压；README 未给 unzip 命令。输出写入 `webankdata/*.csv` 和 model path；仓库无 `webankdata/`，`main.py` 没看到创建该目录的代码，需由用户预先准备输出目录。

#### 1.2 论文机制与当前源码

- 论文：Liu et al., *Unsupervised Detection of Microservice Trace Anomalies through Service-Level Deep Bayesian Networks*, ISSRE 2020, DOI `10.1109/ISSRE5003.2020.00014`；官方 PDF：[netman.aiops.org](https://netman.aiops.org/wp-content/uploads/2020/09/%E5%88%98%E5%B9%B3issre.pdf)。论文将 trace 编为 service trace vector (STV)，normal-only offline 训练 posterior-flow variational Bayesian model，按 `log pθ(x)` 为 trace anomaly score（低 likelihood 为异常）。论文 threshold 不是手工固定分值：在 normal traces 的 log-likelihood 分布上做 KDE，并以 p-value `0.001` 判异。
- `traceanomaly/readdata.py` 并不生成 STV：它读取已经形成的 890 维文本 vector，按训练数据保留有正值的列，以训练集均值/std 标准化，缺失/近零值映射为 `−1`；训练 vectors 被随机打乱，源码未设置随机种子。`main.py` 再从训练集随机留 10% 作 validation（同样未固定 seed）。
- `traceanomaly/main.py` 实现 2×500 dense encoder/decoder、Gaussian latent variable、SGVB objective；`run.sh` 配 `flow_type=rnvp`，默认 10 组 ActNorm/coupling/invertible-dense posterior flow；训练默认 max 2000 epochs、early stopping，按 validation NLL 保存模型。
- 测试路径使用 `test_chain.vi.evaluation.is_loglikelihood()` 并按 `x_dim` 归一化，将 `id,label,score` 写到 CSV；验证分数另存一列。**代码未对 normal calibration scores 做 KDE/p-value，也未根据 0.001 得出 binary prediction；输出 label 是数据集真值，不是预测。**因此不能把 score CSV 视为含评估结果的二值检测输出；源码没有 P/R/F1/ROC 自动 evaluator。
- `grep`/入口检查未发现 STV dimension-to-service/call-path localization 的完整 root-cause stage；论文提及的 localization contribution 在该公开 prototype 中没有对应可执行的定位 pipeline。本 repo 只能据现有脚本支持 trace score/model prototype，不能复算论文所有 detection/localization results。
- 实跑 `python -m traceanomaly.main --help`：exit 1，`ModuleNotFoundError: tensorflow`（发生在 CLI 解析之前）。未建 TF1.5 / Python3.6 环境、未 unzip 整份 256 MB raw text member、未训练，未声称模型复现。

### 2. TraceRCA

#### 2.1 版本、依赖、数据位置

- 官方仓库：[NetManAIOps/TraceRCA](https://github.com/NetManAIOps/TraceRCA)，branch `master`，HEAD `b9087c72d201fbb17d724114cd389de4854815ad`（2023-04-03）。Git tree 仅含 README、Makefile、Python scripts/config；无数据目录、`requirements.txt`、lock/environment 文件或 LICENSE。
- README 提供 [OneDrive share](https://1drv.ms/u/s!Ao2DxaN2zku_bAUszKmCUiodw94?e=7ThI47) 与 [Tsinghua Cloud share](https://cloud.tsinghua.edu.cn/d/8371855eddd64a8db23b/)。审计当日清华云盘页面返回 “Link does not exist”；OneDrive HEAD 为 HTTP 403，仅表示当前自动化访问受拒，不能据此断言资源已删除。没有取到数据压缩包，未计算其 checksum。
- Makefile 写死 `ROOT := /TraceAnalysis`，并期待脚本安装在 `/TraceAnalysis/tracerca-exp`、数据在 `/TraceAnalysis/tracerca-exp/data/{test,normal,train}/*.pkl`，root-cause labels 在 `data/root_causes/`。README 没说明克隆目录应放在该绝对路径或如何改 ROOT。
- Makefile 输入是已经预处理为 Python pickle 的 trace dictionaries；编码器所需键包括 `trace_id`, `s_t`, `timestamp`, `endtime`, `latency`, `http_status`, `label`，可选 `root_cause`/`fault_type` 及其他 resource metrics。没有 raw Jaeger spans → pkl parser 或数据 schema guide。`run_invo_encoding.py` 还 import `data.trainticket.download.simple_name`，但此 `data/` package 不在此 Git commit 内，故除权重/依赖外仍少本地辅助模块。
- Makefile 引用的 association path `run_localization_association_rule_mining.py` 不存在于仓库；唯一相近文件为 `run_localization_association_rule_mining_20210516.py`。它也引用未跟踪的 `run_localization_pagerank.py`、`run_localization_RCSF.py`、`run_localization_MEPFL.py`、`run_localization_microscope.py`、`run_concatenate.py`、`run_effect_of_trace_localization_collect.py`、`plot/*` 等。因此 README 宣称的 `make` workflow 并非当前 Git tree 中闭合的 executable pipeline，即使数据路径补齐也会在依赖阶段缺脚本。
- 当前环境在 CLI import smoke 中首先因缺 `loguru` 退出；源码 import 还依赖 `diskcache`、`imblearn`、`mlxtend` 等，且没有官方安装清单。与 NumPy 2 环境兼容性还需注意：`run_trace_encoding.py` 使用已删除别名 `np.bool`，可选 PRFL 分支用 `np.float`；未 patch。

#### 2.2 静态 pipeline：数据准备、AD 与 RCA 分开

1. **数据准备（README 没有提供完整步骤）**：Makefile 预设 test/normal/train 下的 pickle files。`prepare_train_file_tmp.py` 不在 Makefile；若运行，它按现有文件顺序取每个 test pickle 前 20% 写入 train、再原地覆盖 test 为余下 80%，无 shuffle/stratification，并改变原 test 文件。公开 Git tree 未提供把原始数据得到这些 pickle 的步骤。
2. **表示生成**：`run_invo_encoding.py` 把每条 trace 转为 invocation-level rows（source,target,start/end time, trace id/label, latency/http_status 等）；`run_trace_encoding.py` 产生 trace-level `data/labels/masks/trace_ids` arrays。当前默认 config 只选 `latency,http_status` 两项；完整 9 metrics 为关闭的 `ENABLE_ALL_FEATURES=True` 分支。
3. **历史训练输入**：Makefile 由 `normal/*.pkl` 聚合 `trainticket_historical_normal.pkl`，用于 invocation reference；另将 normal 与 abnormal train 合并成 historical-all trace pkl（带 root-cause 标签），经 encoding 提供 trace classifiers。两套训练假设不同，不能统称为“全模型无监督”。
4. **Invocation anomaly detection**：每个 test 文件先由 `run_selecting_features.py` 对 test-vs-normal-history 的当前 source→target empirical feature distribution 做 shift/dispersion criteria（Makefile `FISHER=3`；实际 `stderr_criteria`，非传统 Fisher test），选 useful features；`run_anomaly_detection_invo.py` 再以 normal reference mean/std 和 1σ 默认阈值把 invocation 标 0/1（另算 no-selection 和 IsolationForest 对照）。Feature selection 用的是 test batch 的无标签特征分布，不看 trace labels，但它是按 test batch transductive 选择，不是单条在线 scoring。
5. **Trace anomaly detection（独立分支）**：`run_anomaly_detection_prepare_model.py` 对 historical-all traces 的 `labels` 拟合 RandomForest/MLP classifiers；`run_anomaly_detection_trace.py` 对测试 trace 输出 RF/MLP predictions。该分支使用带 label 的 historical fault traces，是 supervised trace-level AD baseline，不是上一步的 unsupervised invocation detector。
6. **RCA**：Makefile 的 association path 使用 invocation predictions；`ItemsetHandler` 将每条 trace 的异常 invocation 汇总为 trace abnormal 与服务节点集，在异常 traces 上挖 frequent itemsets；默认 support rate 0.1、top-K=100。`TraceRCA` 按 pattern support/confidence 的 Jaccard-like 合成分数和异常 invocation 的 service in/out 差异排序 service 候选。此输出是 root-cause service ranking，不是 AD 的 trace labels 或 P/R/F1；`run_localization_collect.py` 另用 `root_causes` reference 对定位结果汇总。可选 PRFL 代码不是 Makefile 默认（association script 调用 `enable_PRFL=False`）。

#### 2.3 真实 smoke attempts / 完成范围

- `python run_selecting_features.py --help`：exit 1，`ModuleNotFoundError: loguru`。
- `make -n target`：exit 2；默认 `/TraceAnalysis/tracerca-exp` 下无 data files，且 Makefile 指定的 localization / auxiliary scripts 不在 repo tree，因此无法解析完整 target。
- 未安装依赖、未重定向/移动仓库、未取回外链数据或 pickle、没有 patch diff；无结果可用于比较论文表格。命令 stdout/stderr 归档至 `tracerca_smoke.log`。

### 3. 最终 T07 判定

T07 的版本、依赖声明/缺失、公开数据链接、代码所实现的 scoring/threshold 与执行路径、TraceRCA 的 AD/RCA 分支顺序均完成核验，且有真实入口失败日志；**可复现审计任务完成，原样训练/论文结果复现不完成**。在无可用数据包、旧依赖、当前必要本地子模块缺失时，没有用 modernized patch 伪装成原版复现。


---

<a id="t08"></a>
## T08｜Nezha 执行审计

> 来源文件：`nezha_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T08 执行审计：Nezha “basic implementation”、实验数据与运行路径

**审计日期**：2026-10-19（Asia/Taipei）
**范围**：固定作者仓库/Zenodo artifact 身份，核对 FSE’23 方法与代码、公开 augmented systems / 两阶段观测数据、文件清单与标签、运行入口和依赖；未全量下载 337 MB Zenodo ZIP 或 3.04 GB GitHub 解压数据，未改码/安装依赖/跑完整 RCA。命令日志：[`nezha_smoke.log`](nezha_audit/nezha_smoke.log)；GitHub tree manifest 与汇总：[`nezha_audit/github_tree_manifest.csv`](nezha_audit/github_tree_manifest.csv)、[`nezha_audit/github_tree_inventory.md`](nezha_audit/github_tree_inventory.md)。

### 1. 代码、artifact 与 benchmark 来源

- 论文：Yu et al., *Nezha: Interpretable Fine-Grained Root Causes Analysis for Microservices on Multi-modal Observability Data*, ESEC/FSE 2023, DOI `10.1145/3611643.3616249`; [ACM paper/PDF](https://dl.acm.org/doi/10.1145/3611643.3616249)。
- 官方实现/数据：[IntelligentDDS/Nezha](https://github.com/IntelligentDDS/Nezha), `main` HEAD `d8140101fdb4e7dfb60d3ef9f64706f382b68470` (2025-05-20); GitHub API repository size 342,975 KB; recursive tree 1,297 entries / 1,252 blob files, declared raw file sizes total 3,042,817,082 B. Current root LICENSE is MIT (repo-level; no separate data license manifest).
- Immutable publication artifact: Zenodo concept DOI `10.5281/zenodo.8276374`, version record DOI `10.5281/zenodo.8276375`, v0.1, Open, file `yuxiaoba/Nezha-v0.1.zip` 337,137,421 B, metadata MD5 `31a7408955c77112b5117655adb9b07f`. The file itself was not downloaded/checksummed locally. API license identifier is `other-open`; it does not identify a conventional SPDX license. GitHub current `main` has later code edits than the 2023 artifact and GitHub shows no tags; use Zenodo v0.1 as the fixed paper artifact, or record SHA if auditing current `main`. `STATUS.md` says authors “would like to apply” for artifact badges; this is not evidence badges were awarded.
- Paper’s two application benchmarks are the **augmented** public systems [Augmented-OnlineBoutique](https://github.com/IntelligentDDS/Augmented-OnlineBoutique), `master` SHA `94a01ba4c7c79e0c0b2c9da073ba081fcedc4bf1`, Apache-2.0; and [Augmented-TrainTicket](https://github.com/IntelligentDDS/Augmented-TrainTicket), `master` SHA `32abe642659fc4e863e8348d699574aa30b3b2e2`, Apache-2.0. The paper instrumentation adds OpenTelemetry trace/span identifiers to logs and complete traces; they are not unchanged stock benchmark telemetry.

#### Dataset phrase clarification

The Nezha paper evaluates **two applications**, not three independent application datasets. In the Nezha data repository, `construct_data/` is the fault-free construction-phase baseline for those same two applications, while `rca_data/` is their fault-suffering evaluation data. Thus a “constructed dataset” exists as the construction-phase collection, but it is not a third independent benchmark. For both apps, the paper and README describe comparing fault-free patterns with fault-suffering patterns.

### 2. Open-data inventory and schema

The GitHub API tree at the pinned commit was used to produce a path/size/Git blob-SHA inventory without cloning/downloading the multi-gigabyte data tree.

|Data tree|files|raw bytes|interpretation|
|---|---:|---:|---|
|`construct_data/`|136|72,765,520|fault-free/construction-phase logs, metrics, traces, traceid data and inner-service root-cause dictionaries.|
|`rca_data/`|1,035|2,965,396,038|fault-suffering logs, metrics, traces, traceid files, and four day-specific service-level fault lists.|

|System/date|Construction-phase data files|Fault-phase files (logs / metrics / traces / traceid)|Fault-list records|
|---|---|---|---:|
|OnlineBoutique `2022-08-22`|1 log + 14 metric + 1 trace + 1 traceid|72 / 14 / 72 / 72|24|
|OnlineBoutique `2022-08-23`|1 log + 14 metric + 1 trace + 1 traceid|96 / 14 / 96 / 96|32|
|TrainTicket `2023-01-29`|1 log + 47 metric + 1 trace + 1 traceid|84 / 47 / 84 / 84|28|
|TrainTicket `2023-01-30`|1 log + 47 metric + 1 trace + 1 traceid|51 / 47 / 51 / 51|17|

Schema samples were inspected with HTTP byte ranges, without retrieving entire data files:

- Trace CSV columns: `TraceID, SpanID, ParentID, PodName, OperationName, StartTimeUnixNano, EndTimeUnixNano, Duration`.
- Log CSV columns: `Timestamp, TimeUnixNano, Node, PodName, Container, TraceID, SpanID, Log`; log text embeds trace/span identifiers, the join keys used by the implementation.
- OnlineBoutique front-end metric sample: `Time, TimeStamp, ServiceName, SuccessRate(%), LatencyP50(s), LatencyP90(s), LatencyP95(s), LatencyP99(s)`; other metric files include per-service/system observations. The TrainTicket metric set has 47 files per date.
- Fault-list records contain `inject_time`, Unix `inject_timestamp`, `inject_pod`, `inject_type`; `construct_data/root_cause_hipster.json` and `root_cause_ts.json` map services/fault types to code-region or resource-type labels.
- The separate augmented app repositories and the Nezha repository’s telemetry/labels together let a researcher identify source systems, data phases, and root-cause labels; the README gives no independent dataset version DOI or per-file external checksum. The GitHub commit plus saved tree/blob-SHA manifest is the versioned inventory.

#### Paper-vs-fault-list count discrepancy

- OnlineBoutique: public fault-list JSON contains 24+32 = 56 records: `cpu_contention=16`, `cpu_consumed=10`, `network_delay=16` (42 resource) and `return=7`, `exception=7` (14 code). This matches paper’s 56 (42 resource +14 code).
- TrainTicket: JSON contains 28+17 = 45 records: `cpu_contention=7`, `network_delay=14` (21 resource) and `return=11`, `exception=13` (24 code). Paper §5.1 reports 45 faults but states 20 resource +25 code. The total matches; the public label-type split differs by one case in each category. No source evidence found to reconcile it; do not silently alter labels or claim identical breakdown.
- README contains date typos in its TrainTicket fault-list description (`2022-01-29`/`2022-01-30`); actual files, `main.py`, and paper refer to `2023-01-29`/`2023-01-30`.

### 3. Paper modules vs. repository implementation

|Paper stage|Repository implementation / scope finding|
|---|---|
|Application anomaly detector triggers RCA|`alarm.py` computes success ratio and P90 latency against fault-free construction-phase statistics with default `k=3`, one-minute interval; the paper describes the same. It monitors front-end application-level metrics. This is a trigger, not the inner-service root-cause ranker.|
|Metric, log, trace unification|`data_integrate.py` loads metrics/logs/traces, represents span start/end, parsed log templates, and metric alerts as events; correlates trace and log with TraceID/SpanID; constructs per-request event graphs and uses parent span IDs across services. `log_parsing.py` uses Drain3; `log_template/` stores config/state. This corresponds to the paper’s event representation and event graph modules.|
|Pattern extraction|`pattern_miner.py` computes event-pattern supports in fault-free and fault-suffering event graphs.|
|Expected/actual ranking & aggregation|`pattern_ranker.py` ranks construction-only/expected patterns and fault-only/actual patterns, filters redundant paths and correlates pattern pairs. Default minimum score is 0.67 (paper’s selected setting). The source contains the expected ranker, actual ranker and pattern aggregator, not only a high-level placeholder.|
|Evaluation|`main.py` has four dataset/target-level paths: `--ns hipster|ts`, `--level service|inner`. README/INSTALL shows Top-k AS/AIS outputs. These are root-cause rankings, not trace anomaly classification metrics.|

Paper-reported protocol: 3-minute fault duration; 56 OnlineBoutique and 45 TrainTicket fault cases; top-k accuracy at service level (`AS@k`) and inner-service level (`AIS@k`). Paper reports overall mean Top-1 accuracy 89.77%. README example values match its tables: OnlineBoutique AS/AIS@1 92.857%, TrainTicket AS/AIS@1 86.667%, with @3/@5 as printed. These are **author-reported README results**, not results rerun by this audit.

### 4. Entry path and execution attempt

- README says “basic implementation”, recommends Python 3.6 for anomaly detection (otherwise Python 3 generally), and calls for `pip install -r requirements.txt`. `requirements.txt` pins `drain3==0.9.10`, `numpy==1.15.4`, `pandas==0.23.4`, `matplotlib==3.3.4`, `more_itertools==8.12.0`, `psutil==5.9.0`, `PyYAML==6.0.1`. Source additionally imports `tqdm`, which is not listed. The pins predate current Python 3.13 / NumPy 2 and are not a modern lock.
- README/INSTALL commands are `python3.6 ./main.py --ns hipster --level service|inner` and `python3.6 ./main.py --ns ts --level service|inner`; `main.py` contains fixed fault-free time windows and fault-list paths for the four day runs. Static offline telemetry exists in the repository, so those evaluation commands do not require re-deploying the apps, but the original *data collection* still depends on the instrumented systems/toolchain described in the paper.
- Source subset downloaded only from the pinned commit to `/home/user/nezha_audit/source/`; no raw telemetry clone. Attempted `python main.py --help` under Python 3.13.16; exit 1 before argument parsing: `ModuleNotFoundError: No module named 'more_itertools'`. No dependency installation or local patch was attempted; modern NumPy/pandas and data volume remain additional untested issues.
- This establishes that the public repository contains the main paper-stage modules and public phase data, but its current source is not a turnkey modern environment. Full data ingestion, all four RCA runs, and paper table reproduction remain unexecuted.

### 5. T08 conclusion

The Nezha artifact is more than a README-only “basic implementation”: its tracked source includes alarm, multi-modal integration/event graph, Drain log parsing, pattern mining, expected/actual ranking, aggregation, labels, and evaluation entry points. The two benchmarks are the augmented OnlineBoutique and TrainTicket; the `construct_data` tree is their fault-free construction phase, not a third system. Public artifact identity, licenses/versions, data schema/counts, paper coverage and discrepancies, and a real blocked smoke attempt have been documented. **T08 audit is complete; full run and result replication are not claimed.**


---

<a id="t10"></a>
## T10｜LO2 数据与复现审计

> 来源文件：`lo2_execution_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T10 执行审计：LO2 的 DOI/版本、数据契约、标注单位与 trace 覆盖

**审计日期**：2026-10-19（Asia/Taipei）
**判定**：用户任务指定的原始 LO2 dataset/software DOI 已核实为 concept DOI；当前 dataset 版本、Zenodo 清单/校验值、复现包脚本、Locust/Prometheus/Jaeger 采集路径已核验。已实际下载并 checksum 验证小型 `lo2-scripts.zip` 与 19.8 MB replication ZIP；未下载最大数据包、1.07 GB sample 或运行 Docker/Locust 全栈。本数据是 **API negative-test/test-run 标签**，不是常规性能故障注入，也不是覆盖充分的 trace 数据集。

### 1. DOI 身份、当前版本、许可证与文件

#### 原始任务中的 dataset DOI

- `10.5281/zenodo.14257989` 是 dataset **concept DOI**，当前版本 record 为 `10.5281/zenodo.14938118`（Zenodo version history index 2；页面描述称 Version v3）。该 v3 API 记录标记 `access_right=open`、license `CC BY 4.0`。
- v3 文件清单（大小/MD5 来自 Zenodo API）：

|文件|bytes|Zenodo MD5|本轮是否实际下载校验|
|---|---:|---|---|
|`lo2-data.zip`|46,483,584,345|`9a8e962c56bcfd7115429d5e6489a9b5`|否（46.5 GB）|
|`lo2-sample.zip`|1,070,011,492|`65ef33ca28627026a3feb8ec6d0a9c7e`|否（1.07 GB）|
|`lo2-scripts.zip`|17,195|`f579536be380e74b70db49d9597f99f9`|**是**，实际 MD5 匹配；已展开脚本检查|
|`data-appendix.pdf`|1,618,706|`ec4f8b978bd369ae8622c2c88a19e296`|通过 Zenodo PDF 内容查看，未保存整份文件|
|`README.md`|2,143|`6535599da5ad4dd5306e161ffe757411`|通过 Zenodo API 查看|

- v1 原论文和 API 记录称原始数据约 **540 GB uncompressed**；同一 v3 Zenodo README 现写 **450 GB**；record 描述仍写 540 GB。该差异未在 v3 metadata 中解释，报告时保留为作者发布材料口径冲突，不替作者选定一个数字。
- `lo2-sample.zip` 只含 first 100 runs 的 service logs 与较重要的 metrics，metric 已合并成便于分析的 CSV；不是全量 raw，也**不含 trace**。未下载整个 1.07 GB sample，而是通过 HTTP byte-range 读取 ZIP central directory（约 7.6 MB）并按 local header 定点取回一个 run-level metrics CSV（1,029,680 B）和一个 service log（847,488 B）；Zenodo 提供的 sample MD5 未对整包本地重算。

#### 原始任务中的 replication package DOI

- `10.5281/zenodo.14229369` 是 replication/software **concept DOI**，当前（且唯一）version record 是 `10.5281/zenodo.14229370`，版本文件名 `light-oauth2-repl_v1.0.0.zip`；API metadata 的 license 为 Apache 2.0，related GitHub source commit `4f0c304942dba0797348c4b52119762cca8ebbcd`。
- replication ZIP 实际下载为 19,757,322 B，MD5 `72eb73073d4947da810169633fbf4e1c` 与 Zenodo 匹配，包含 2,080 entries：Light-OAuth2 服务源码、Locust tests、Docker Compose、Prometheus 配置/指标列表、Jaeger/OpenTelemetry Java agent、`data_run.sh`、`fetch_data.py`。
- 当前原始数据 concept DOI `.14257989` 的 Zenodo 描述注明它已被 **LO2v2** supersede。LO2v2 是另一概念 DOI `.16982695`，当前 version DOI `10.5281/zenodo.18937117`（2026-03-10）；record 与原 DOI `.14257989` 和软件 DOI `.14229369` 均有 `isNewVersionOf` 关系。LO2v2 有 115 runs × 54 tests，约 62 GB ZIP/640 GB uncompressed；含 1.317 GB metrics export、2.904 GB logs archive、31 MB index，dataset CC BY 4.0、代码 Apache 2.0。它调整为 cAdvisor container metrics、随机测试顺序并加入 run log，**不可与原论文所用 1,740-run v1 当成同一数据集或表格复现结果**。

来源：[LO2 paper/arXiv](https://arxiv.org/html/2504.12067v1)、[原 dataset concept DOI](https://zenodo.org/records/14257989)、[原 dataset 当前 version](https://zenodo.org/records/14938118)、[replication concept DOI](https://zenodo.org/records/14229369)、[replication version](https://zenodo.org/records/14229370)、[LO2v2 current record](https://zenodo.org/records/18937117)。

### 2. 数据实际语义与 label unit

- 系统为 Light-OAuth2（7 个微服务 + MySQL）。每个 run 部署系统后：执行 60 秒 `correct` Locust test；再分别对每个 error task 执行 10 秒“全部 correct tasks + 一个指定 error task”的 test。每 run 54 个 tests（1 correct + 53 error tests）；作者报告 1,740 runs，采集约 657,000 份服务日志、20 亿+ log lines。
- error tasks 是特定 OAuth2/CRUD API 的**负向测试**（如缺参数、错误账号、资源不存在等），每个任务对应明确 error label。测试目录/Locust log 记录 error task 和发生时间，可将整个 run-test/service 文件标为 correct 或特定 API error；这不是 Chaos/fault injection（作者明确未注入网络延迟、硬件异常或慢磁盘），也不是“全部相关服务的根因 service”标签。
- 原论文初步分析只抽 100 runs：每 run 抽 correct + 一个随机 error test；stage 1 得 700 个 service-log sequences，stage 2 得到 53×7 error×service 分析矩阵。分析用随机 50/50 train/test、正常/异常均衡；无 unsupervised model 时还用带 labels 的 optimizer 选择最佳 F1 threshold。因此这些是预备性分类实验，不是提供了一个固定、通用的 train/validation/test split 或无标签部署阈值。
- 适用性：适合研究 **API negative-test 的 log / host-metric 行为**；不等价于 performance anomaly、服务级 15 秒窗标签、或 RCA 根因排序任务。指标和 label unit 必须保持 test/API 级。

### 3. 三种监控数据是否真正可用

#### Logs

- 原论文记载 DEBUG-level 容器日志，按 run×test×service 分开；服务日志约 657k files，Locust log 另外提供请求任务/时间 ground truth。
- sample ZIP central directory 实数 43,402 entries：37,800 个 `.log`（恰为 100 runs×54 tests×7 services）、100 个 metrics `.csv`，其余 5,502 项为目录/无扩展名项。按 range 取回的一份 client-service log 为 847,488 B，确有原始 service log 内容；sample 中没有 Locust log 文件，因此样本内 ground truth 可从 `test_name`/目录取得，Locust 请求级时间日志要回到 full raw 或 replication package 采集。
- `reduce_logs.py` 在 `lo2-scripts.zip` 中说明有些初始化行会泄露 correct/error；代码首先删去前 100 行，随后对 `correct` 目录只保留余下行约 1/7.05，以减小与 10 秒 error-test 的时长不平衡。
- 静态烟测发现该脚本的 `lines[:len(lines) // 7.05]` 在 Python 3 产生 float slice index，直接执行会 `TypeError`；用 120 行内存列表验证确实报错。要复用样本分析需把切片索引显式转成 int，并记录 patch，不能默认为 release script 原样可运行。

#### Metrics

- v1 来自 Prometheus + node_exporter 的**host-level**指标（Docker Compose 配置只有 `job_name: node`，scrape/query step 均 5 s；没有 cAdvisor per-container metrics）。
- 复现 ZIP 中 `prometheus_metrics.txt` 实查 **485 行、485 个 unique metric names**，与论文“485 unique metrics”一致。原始数据按每个 run/test/metric 保存 Prometheus JSON；1740×54×485 ≈ 45.52 million，与论文“45.5 million metric files”数量级吻合。
- 通过 sample ZIP 的 HTTP range 取得一个实际 run-level CSV：header 共 1,127 列（`timestamp`, `run_end`, `test_name` + 1,124 metric features），172 rows、54 个 test names（1 correct + 53 error），对应数据时间戳 800 秒范围。逐字段与论文报告的 PCA 输入“172 timestamps / 1,124 feature vectors”一致；文件结构也直接证明 label 粒度为 test name。它不是 485 列：每个 metric name 按不同 Prometheus 标签产生多个 time-series features。不能将 485 metric names、1,124 time series、45.5m 个 raw metric JSON files 混称为同一统计单位。
- 作者 `lo2-scripts.zip` 提供 `csv_generator.py` 与两个 merge 脚本。数据 README 说明跨 run 全局 merge 很耗内存；无全量下载/聚合运行。

#### Traces

- 复现包确实有 Jaeger all-in-one、OpenTelemetry Java agent（应用 service 以 `JAVA_TOOL_OPTIONS` 注入 agent 并把 trace 发往 Jaeger），`fetch_data.py` 对 8 个 service 名调用 Jaeger API，再写 trace CSV（`Trace ID, Span ID, Operation Name, Start Time, End Time, Tags`）。因此不能说发布包“没有 tracing 配置”。
- 但原论文明确：当前每条 trace 只有 **one span**；Light-OAuth2 trace agent 主要观察直接 MySQL 调用，未覆盖 service 内的 API 子操作；许多服务没有 DB operation，CSV 为空，作者因此**没有分析 trace**。这说明存在 trace 文件/Jaeger 配置不等于有可融合的完整分布式调用链或足够 coverage。
- 完整 data appendix 基于 full dataset：每类 trace CSV 记录 93,575 个 run-test files；理论组合为 1,740×54=93,960，文件数少 385（约 0.41%）。该 appendix 中 `traces_jaeger-all-in-one.csv` median=58 B（header-only），多个 service trace 文件 median 也为 58 B；有些文件均值也仅几十/数百 bytes。**58 B 是空 CSV header**；因此以“trace CSV 文件数”计算 coverage 会严重高估有效 trace 覆盖。对 sample ZIP central directory 的 43,402 个 entries 也未发现任何包含 `trace` 的路径；sample 本来就不含 traces，不能用来证明 trace coverage。
- `fetch_data.py` 按 service query Jaeger 的 `limit=20`；这是 collector 的潜在 per-service/query cap，需要结合请求速率审计是否截断，不能据源文件存在就保证完整 trace retrieval。没有下载全量数据逐一重数非空 span。

### 4. 复现入口与实际运行条件

Replication README 给出了明确的一次采集流程：安装 Docker/Compose、MySQL `mysqladmin`、Locust、Python `requests`；`data_run.sh` deploy Docker Compose、初始化数据库、跑 correct/error tags，抓取 Docker logs，再通过 Prometheus query-range 和 Jaeger query 保存 metrics/traces，最后 `docker compose down -v`。Locust config 固定 headless、7 users、默认 10 s；正确场景命令覆盖为 1 min。原始 run 数据按 test 子目录保存。

源码配置核查：

- image pins：Light-OAuth2 services `2.0.28`、MySQL `5.7.16`；Jaeger、Prometheus、node_exporter 用 `latest`；OpenTelemetry agent 随 ZIP 附带但未在 Compose 中按 checksum/version pin，影响长期 bitwise reproducibility。
- Prometheus 由 `prometheus_metrics.txt` 明确要求采集 485 metric names；`fetch_data.py` 每 5 秒 Query Range。Jaeger collector 返回失败时仅打印错误并继续，缺少完整重试/覆盖校验。
- 当前执行环境无 `docker`、`docker compose`、`locust` 或 `mysqladmin`，未运行 replication package。全量 1,740 runs 也远大于一个入口 smoke test；本轮不伪称重新采集了公开数据。

### 5. T10 判定

- **DOI/身份/许可**：原始任务 DOI `.14257989`（dataset）与 `.14229369`（software）均为 concept DOI；具体 dataset 最新版本为 `.14938118` v3，replication version 为 `.14229370` v1.0.0。Dataset CC BY 4.0；replication package Apache 2.0。另有新 dataset LO2v2 `.18937117`，已明确 supersede 原始 DOI 概念。
- **数据契约**：v1 是 test/API error labels + 7-service logs + host-level Prometheus metrics；485 个 unique metric names 经 replication 文件实数验证；raw metrics 为 JSON、sample 中聚合 CSV；trace CSV 很稀疏且每 trace 单 span。
- **复现边界**：小型 scripts/replication archives checksum 匹配；full raw 数据未下，数据 API checksum 未本地重算，未全量检查 raw trace/log coverage、未执行 full Docker/Locust replication。已有代码显示 `reduce_logs.py` 切片类型 bug，完整多模态复现未完成。
- **任务兼容性**：应将其列为 API/test-level negative-test dataset（logs + host metrics 可用、traces coverage 弱），不能当作一般 service-window / performance-fault / RCA ground truth。


---

<a id="t11"></a>
## T11｜ServiceAnomaly / Eadro 数据身份审计

> 来源文件：`serviceanomaly_eadro_data_identity.md`。以下为完整正文；标题层级为合集显示下调一级。

## T11 数据身份交叉核验：ServiceAnomaly 与 Eadro

**审计日期**：2026-10-19（Asia/Taipei）
**范围**：仅公开作者仓库、Zenodo records 与可下载数据；未访问任何私有数据。
**结论**：两项目都有作者公开内容，但“仓库含有同名数据”不能等于论文完整实验数据。ServiceAnomaly 仓库树明显小于论文 baseline 原始 trace 声称规模，且无正式 data DOI/license；Eadro 有具体 Zenodo data version，package 名称/内容对应论文中的 TrainTicket、SocialNetwork，但 raw→parsed 转换缺失、fault record vs injection-operation 计数口径未解释、Zenodo 未标明确认 license。

### 1. ServiceAnomaly

#### 来源与版本

- 论文：Panahandeh et al., *ServiceAnomaly: An anomaly detection approach in microservices using distributed traces and profiling metrics*, JSS 209 (2024), DOI `10.1016/j.jss.2023.111917`。
- 作者仓库：[M-panahandeh/ServiceAnomaly](https://github.com/M-panahandeh/ServiceAnomaly)，branch `main`，HEAD `6acc9dfd7ec50d3007d65e61a188f40d72b422b6`，无 separate data DOI，数据在 GitHub `data/`。根 README 仅 99 B，代码 readme 仅 1 B；没有 LICENSE 文件，数据许可/再分发权不能推定。
- 在该 commit 上对 `data/` 做 SHA-256 manifest：1,718 files / 42,331,839 bytes；manifest 保存于 [`serviceanomaly_data_sha256.txt`](serviceanomaly_data_sha256.txt)。按 manifest 的排序路径+size+每文件 SHA-256 计算目录 canonical hash：整体 `92efbb99f755b893c05c59d0d20335a7ae40dac47045aff8721f41505fba31bb`；TeaStore `ff47f096b4e100b33a7e1028f4c21a7da6e3ee38bd7309ede46e07f69b90ad5d`；TrainTicket `8eb5dd46a97f700ec75894b702e93fb775deb723c7b34e36ba117bd0a1c83f8b`。这不是 Git blob hash，也不是外部 Zenodo checksum；它是对本报告明确算法下的稳定 directory hash。

#### 仓库数据与论文实验规模对照

|系统|repo 中数据|论文描述|身份判定|
|---|---|---|---|
|TeaStore baseline|539 files / 18,474,526 B；原始 `Traces.json` 只有 100 Jaeger traces / 204 spans；72 metric JSON 中 12 个 throughput 文件为空。|3,000 baseline traces / 9,160 metric rows。|同名系统/作者采集内容，能作 example/partial data；无法充当论文所称的 3,000-trace baseline。|
|TeaStore test|15 normal + 15 fault case directories；raw Jaeger traces 有缺失，normal 12/15、fault 15/15；metric files 空值多。|15 normal + 15 anomaly cases。|目录数呼应论文，但数据覆盖不完整；目录名给出 stress/latency/timeout/hangup 等情形，无单独统一标签 manifest。|
|TrainTicket baseline|1,179 files / 23,857,313 B；3 个 metrics snapshots（120/126/126 JSON metric files），`003/trace/traces.json` 是 dependencies graph（17 parent-child links），不是 baseline raw trace set。|5,000 baseline traces / 5,410 metric rows。|不能视为论文完整的 5,000-trace baseline。|
|TrainTicket test|32 normal + 32 injected-fault directories；有效 raw trace 缺失、一个 relation list 损坏；metric result 空文件显著。|32 normal + 32 anomaly traces/cases。|Case 目录数量大致对应，但单目录 raw coverage/quality 不完全；文件夹数不等于 64 个完整 trace samples。|

详细字段、missing/empty counts、模型代码与失败入口见 [`serviceanomaly_execution_audit.md`](serviceanomaly_execution_audit.md)。T04 entry `main.py` / `mainTeastore.py` 因缺 `netgraph` 退出；没有自动 trace-level P/R/F1 evaluator。因仓库没有能通过 checksum/metadata 证明数据与论文完整 run 一一对应的 manifest，身份结论限定为“同项目作者数据的部分公开快照”，不能写成论文完整原始数据。

### 2. Eadro

#### DOI、版本和文件 identity

- 论文：Lee et al., *Eadro: An End-to-End Troubleshooting Framework for Microservices on Multi-source Data*, ICSE 2023, DOI `10.1109/ICSE48619.2023.00150`；公开预印本 [arXiv:2302.05092](https://arxiv.org/abs/2302.05092)。
- 作者代码：[BEbillionaireUSD/Eadro](https://github.com/BEbillionaireUSD/Eadro)，`main` SHA `82ff9c9a7a58a9cd2b713aa78754a73c543a15db`，README 指向 Zenodo concept DOI `10.5281/zenodo.7615393`。当前 version DOI 为 `10.5281/zenodo.7615394`（v1）；这是同一 concept 的版本关系，不是冲突发布。
- Zenodo API 标记 `access_right=open`，但 record metadata 未列 license；GitHub checkout 也无 LICENSE。故不能把公开下载推断成 CC BY/MIT。
- Version v1 数据文件实际下载并 checksum 验证：`SN Dataset.zip` 78,753,809 B，MD5 `4c523faf64770b1e9a709cdd610e07c4`；`TT Dataset.zip` 48,923,113 B，MD5 `2d77861e4a025781e598d51866fd6f4e`。ZIP 经 central-directory 和样例文件检查后已删除，以免 workspace 保留约 122 MB 压缩件；file list/checksum/schema 在 [`eadro_execution_audit.md`](eadro_execution_audit.md)。

#### Zenodo raw data 与论文实验的关系

- Archive 名称明确标识 `SN`（SocialNetwork）与 `TT`（TrainTicket）；ZIP 中见 SN 4 个 fault run、TT 9 个 fault run，每个 fault JSON 有 9 条 metadata records；另有每个 benchmark 的 normal archives。这与论文采用这两个 benchmark 的身份一致，作者 README 也将 concept DOI 作为 dataset 链接。
- 论文统计 126,384 SN traces / 48,296 TT traces，Zenodo ZIP 是按运行批次打包的 raw JSON/logs/metrics；本轮未全量展开、未逐条重算 trace 数，故该文中 traces 总量仍标注为**论文报告值**，不能声称由本轮实测验证。
- 每份 fault JSON 包含 9 fault metadata rows；本轮观察总数 SN 36 / TT 81，论文提到 injection operations SN 72 / TT 162（正好约 2×）。字段为 `name/fault/start/duration`；record 与 operation 的统计单位尚未解释，不能直接称为数据错误或把两种计数当成相同单位。
- 公开 raw 是可下载的实际数据，不是直接的 model-ready directory。代码期待 `parsed_data` 下的 `records*.json`, `logs*.csv`, `templates.json`, `traces*.json` 和每-service metric series；没有找到 raw logs/Jaeger spans → Drain templates/预聚合 trace input 的独立 converter。论文数据的 benchmark 身份对应，不等于本文结果可从原包直接重算。

详见 [`eadro_execution_audit.md`](eadro_execution_audit.md)：当前代码预处理默认窗口/切分和论文描述不一致，存在随机重叠窗口泄漏风险、静态运行错误及 evaluation counter 问题；两入口分别因缺 `tick` 和 `torch` 退出，完整训练/指标复算未执行。

### 3. Cross-project 判定

|问题|ServiceAnomaly|Eadro|
|---|---|---|
|公开 source identity|GitHub commit 固定|GitHub commit + Zenodo concept/version DOI 固定|
|公开数据 identity|仓库 `data/`，无独立 DOI/license；通过 Git commit 与 SHA manifest 定位|Zenodo v1 `.7615394` 具体 data version、SN/TT ZIP、MD5 核验；license 未声明|
|与论文 benchmark 对应|名字和部分 test case 数吻合，但 baseline trace/数据完整度与论文报告显著不同|benchmark names 与 raw package 对应，论文报总量尚未逐条重算；fault record/injection counts 口径未闭合|
|raw→model input|数据/模型入口不完整；无完整 automatic evaluator|raw→`parsed_data` converter 缺失；模型包名/字段不直接匹配|
|是否可声称论文结果复现|否|否|

**T11 判定**：公开数据身份、release/version/checksum（可获得者）与论文实验对应关系审计完成；尚不能证明任一仓库公开树等同论文完整运行数据，也未重跑模型或重算论文表格。该结论不是对其开源数据的质量/许可评价，而是可追溯的 artifact-scope 判定。


---

<a id="t12"></a>
## T12｜Triple（IWQoS 2023）公开证据核验

> 来源文件：`triple_iwqos2023_evidence_audit.md`。以下为完整正文；标题层级为合集显示下调一级。

## T12：Triple（IWQoS 2023）公开证据核验

**日期**：2026-10-19（Asia/Taipei）
**状态**：部分完成。可公开核验到官方论文元数据/摘要及一个与论文同名的 GitHub 候选仓库；正式全文、完整算法协议与可运行实现均未取得/确认。未据摘要补写未知字段，也未把候选仓库直接断言为作者官方完整代码。

### 1. 论文身份与目前可直接核对的内容

- Rui Ren, Yang Wang, Fengrui Liu, Zhenyu Li, Gaogang Xie, “Triple: The Interpretable Deep Learning Anomaly Detection Framework based on Trace-Metric-Log of Microservice,” 2023 IEEE/ACM 31st International Symposium on Quality of Service (IWQoS), DOI `10.1109/IWQoS57198.2023.10188773`。
- IEEE Xplore 官方条目可读摘要说明：以 trace 表示复杂服务依赖关系，将 logs 与 metrics 嵌入图节点特征；在图表示上用 Spatial-Temporal Graph Convolutional Network (STGCN) 加 Deep SVDD 学习异常边界；解释器输出与 logs/metrics/traces 有关的信息。摘要称实验基于真实系统数据，并报告相对比较方法 11%–65% improvement，但未提供足够细节以复核数字含义或实验协议。
- IEEE Xplore full-text/stamp URL 在本次抓取中没有返回正文（仅标题/元数据）；以下需要表格和算法段落支撑的项目均不从摘要外推。

### 2. 字段证据表

|字段|目前可核实内容|尚缺/结论|
|---|---|---|
|样本/输入表示|trace 形成图依赖关系；logs 与 metrics 作为节点特征整合。|图是 span、service 还是 instance 粒度；日志模板/事件序列化方法；metrics aggregation/window length、特征维度与缺测处理均 Unknown。|
|图结构与时空建模|摘要明确 STGCN，结合 spatial-temporal graph convolution。|图的边定义、时间步、卷积核/层数、空间/时间先后结构及动态图细节 Unknown。|
|模型与 loss|摘要说明 Deep SVDD 产生决策边界。|网络损失、中心初始化、ν/半径更新、优化器、训练轮次、是否仅 normal 训练及标签参与方式 Unknown；不得仅因 Deep SVDD 就断定其完整实验是 normal-only。|
|输出与阈值|摘要称有 binary anomaly result，并有解释器。|异常分数定义、cutoff/校准集/阈值来源、是否使用验证或测试标签 Unknown。|
|数据与系统|摘要仅称 real-world system dataset。|精确 benchmark 名、版本/运行次数、公开/私有数据边界、可下载链接、样本数量 Unknown。|
|split 与评价|摘要称有一系列实验并报告 11%–65% improvements。|train/validation/test split、指标定义、point adjustment、事件容忍窗口、实验随机种子与阈值调参协议 Unknown。|
|官方代码/数据|发现 GitHub `TripleResearcher/Triple`：当前 `main` 仅两个提交；树列出 README 与 Figure3/4/7 PDF，README 仅有项目标题/一句简介；无代码、数据、依赖、LICENSE 或 release。|IEEE 论文条目/摘要没有指向该仓库；GitHub 页面没有足以核实与论文作者身份关系的组织说明。因此只能记作“高度相关候选仓库，官方归属未确认”，且当前内容不是可运行实现。数据仓库未确认。|

### 3. 检索与可复查链接

- IEEE Xplore 官方论文页：`https://ieeexplore.ieee.org/document/10188773/`（摘要/元数据）。
- IEEE stamp/full-text 入口：`https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10188773`（本次抓取未获得论文正文）。
- DOI：`https://doi.org/10.1109/IWQoS57198.2023.10188773`。
- GitHub 候选仓库：`https://github.com/TripleResearcher/Triple`；当前页面显示 main、一条 branch、无 tags，最新 commit `56df327974c1a02d808262381df5180b8d33ad09`，仅 README 和 3 个 figure PDF。
- 补充发现（非官方证据）：搜索结果中的 2023 中文推荐文章列出上述 GitHub 项目链接；它能证明该 repo 曾被关联到 Triple，但不足以独立证明仓库作者身份或代码完整性。
- 检索范围：按完整论文标题、DOI、`Triple IWQoS microservice GitHub` 搜索；未将搜索未命中写成作者绝无其他私有/未索引代码。

### 4. 判定与下一步

- **已完成**：论文身份、摘要级方法线索、候选 GitHub 仓库内容/当前可运行性状态核查。
- **仍待全文**：输入维度、精确图/时空构造、loss、训练标签、数据版本/划分、阈值、评价脚本。
- **结论**：T12 保持部分完成。要完成论文方法和评价细节核查，需要取得可复查的 IEEE 正文或作者版；在此之前所有正文级字段维持 Unknown，不用相似项目或二手综述代填。


---

<a id="t20"></a>
## T20｜异常分数、阈值与评价矩阵

> 来源文件：`anomaly_score_threshold_evaluation_matrix.md`。以下为完整正文；标题层级为合集显示下调一级。

## T20：异常分数、阈值、标签时机与评价实现矩阵（已审计公开项目）

**日期**：2026-10-19。**状态**：部分完成；覆盖已审计的 T01–T08、T10 公开代码（非所有清单论文），基于当前固定版本静态源码及既有 smoke/replay；没有为补齐此表训练模型。任何“未发现”都仅指本次核查的仓库版本/入口，不推断作者私有实现。

### 项目矩阵

|项目 / 实际输出单位|分数与有效决策规则|阈值从哪里来|标签何时/如何使用|评价实现、泄漏/协议注意点|实跑状态|
|---|---|---|---|---|---|
|**T01 AnoFusion**：按服务/对齐时间戳的多模态序列点，输出 anomaly score|`utils/main.py::eval` 按预测与实际 channel 的绝对误差归一化（减误差中位数、除以误差第25百分位），跨 channel 取最大值，输出 `distance`；没有在可见入口产出 binary prediction。|论文描述 normal residual 的 EVT/POT 类 thresholding；本次固定 HEAD 的运行代码未见 EVT/POT 拟合、阈值或 p-value 分类。因此论文阈值协议与可见实现未闭合，不能声称已复现论文判定。|`MyDataset.read_label()` 只取时间戳；训练 target 是下一时刻 channel，不是 label。评价阶段按 timestamp 查真值并随 score CSV 输出。|仓库未见 point adjustment、容忍窗、PR/F1/ROC 自动计算；需要单独评估器和阈值步骤。来源：`AnoFusion/utils/main.py::eval/get_data`、`model/MyDataset.py::read_label/__getitem__`。|预处理因样例缺第二段 metrics 失败；训练因缺 PyTorch 阻塞。未训练。|
|**T02 MSTGAD/MSDS**：窗口末端的 service/pod 节点分类|分类头输出 logits；`util.calc_index` 用 argmax 形成二分类预测。重构误差参与训练损失，但评价的类别结果不是经独立校准的连续 anomaly-score cutoff。|代码没有单独校准阈值；argmax 是硬分类规则。|`label.pkl` 被转为 known/unknown 标签，`label_percent` 默认 0.5；分类损失仅对已标注节点监督，重构项另用标签状态加权。|代码前70%/后30%拆分，论文描述60/10/30不同。每轮会评估 test_loader（结果未用于选模）；`f1` checkpoint 比较值来自 `evaluate(train_loader)`，即训练集 F1。`calc_index` 直接用 argmax、无 point-adjustment/事件容忍窗。见 `util/train.py::MY.fit/evaluate`、`util/util.py::calc_index`。|公开子集兼容性补丁后生成458窗；训练未运行，full label alignment 未验证。|
|**T03 BARO**：故障 case 的 root-cause ranking，不是 Service×window anomaly classification|RQ2 用每 case 的预存 `naive_bocpd.json` change-point 时间，交给 RobustScorer 生成根因排序；Avg@5/Top-k 按 case ground truth 计分。RQ1 `--saved` 也读取已存 change points。|本次实际 RQ2 replay 没有从 raw metrics 生成/拟合 BOCPD cutoff；因此不能在本矩阵中声称已核实 BOCPD threshold。|fault/target labels 用于排名评价，不是训练 anomaly classifier 的 row-level label。|不能把论文的 Avg@5 与 P/R/F1、或 cached change point replay 与新算 AD 混为一谈；无 event point-adjustment 证据。见 `baro_execution.md`。|RQ2 300 cases 与作者报告一致；fresh BOCPD、RQ3/RQ4 未跑。|
|**T04 ServiceAnomaly**：单个 test trace/case 的关系图 + metric violations|关系图用 `GraphMatcher(...).subgraph_is_isomorphic()` 打印布尔值；metric anomaly 为 test 单值相对 baseline regression/SVR 预测的绝对残差 `>` 已保存 RMSE。|RMSE 是同一 baseline 数据拟合所得训练残差；没有独立 validation/calibration split。|测试 normal/fault 主要由目录名标识；代码不把 graph/violation 自动映射为完整 trace label。|未发现批量 TP/FP/FN、P/R/F1 evaluator；论文 trace-level F1 无法由入口直接复算。RMSE cutoff 不是独立校准。见 `serviceanomaly_execution_audit.md`。|未完整执行；入口含 GUI、Windows API、人工交互等阻断。|
|**T05 Eadro**：重叠 time chunk 的检测与 culprit service localization|模型 detection head 为两类 logits，`codes/model.py` 取 `argmax`；localizer 按 service 预测。|无独立概率校准/阈值搜索；argmax 分类。|fault overlap-derived window labels 用于监督 detection/localization 训练与测试。|`codes/base.py::fit` 每轮对 `test_loader` 评估，并按 test HR@1 选择/保存 best checkpoint，属 test-set model selection。`evaluate()` 把正確正常预测 TN 也累计到 TP，报告的 detection P/R/F1 不符合 anomaly-positive 常规定义；HR 分母也可疑。窗口随机打乱拆分且相邻 stride-1 重叠窗可跨 split；标准化先在完整 run 上做。**有测试标签/时间信息参与模型选择与预处理泄漏风险。**|未训练；依赖缺失及源码硬错误。|
|**T06 DeepTraLog**：trace-level one-class detection|soft-boundary Deep SVDD score `||h-c||²-R²`；`score > 0` 判异常。|center 从训练 normal traces 初始化；radius 按训练目标的 `(1-ν)` quantile 更新。validation 仅 normal，计算分数但不用于选阈值；决策仍是零。|训练仅 normal traces；test 含剩余 normal 与全部 anomalies，labels 用于最终 P/R/F1。|无点级/事件容忍 adjustment 证据；索引 shuffle 未固定 seed，split 不完全可复现。见 `deeptralog_execution_audit.md`。|未训练/测试，论文 P/R/F1 未复算。|
|**T07 TraceAnomaly**：trace-level likelihood score|模型输出 `log pθ(x)`（低似然异常），测试代码可归一化并写 `id,label,score`。|论文描述在 normal likelihood 分布上 KDE，并以 p-value `0.001` 判异常；当前仓库代码没有 KDE/p-value 二值化。|normal/anomalous test labels 随发布数据；代码把 ground-truth label 写入 score CSV，不代表预测。|未见 P/R/F1/ROC 自动 evaluator、point adjustment 或预测 label。论文—代码 threshold 差异须保留。见 `traceanomaly_tracerca_execution_audit.md`。|入口缺 TensorFlow，未训练。|
|**T07 TraceRCA**：有 invocation AD、trace AD、service RCA 三个不同输出|默认 invocation AD 先按 test batch 与 normal history 的无标签分布 shift 选 feature（`FISHER=3` 实为 `stderr_criteria`），再按 normal reference mean/std 的 1σ 规则给 invocation 标0/1；另有带标签 historical faults 训练的 trace RF/MLP。RCA 按异常 invocation 汇总后输出 service root-cause ranks。|invocation cutoff 来源于 normal reference mean/std + 1σ；feature selection 是 test-batch transductive（无标签）；trace RF/MLP 是监督训练，不可统称无监督。|监督 trace classifier 使用历史 trace labels；RCA evaluation 使用 `root_causes` reference。|RCA Top-k 与 invocation anomaly prediction 是不同评价单位；没有证据表明一个统一窗口级 F1。Makefile 缺预处理数据与辅助 package，协议未完整跑。|未执行完整 pipeline，路径/数据/依赖阻断。|
|**T08 Nezha**：fault case 中 service / inner-service root-cause pattern ranking；metric alarm 是触发信号|有效 `determine_alarm()` 路径使用硬编码 raw-value 条件（CPU/Memory >80；hipster其他指标 >200；TrainTicket >300）；尽管 `generate_threshold()` 会写均值/标准差文件，但当前判警函数中该历史 3σ 比较是注释代码。pattern ranker 默认 `min_score=0.67`。|alarm cutoffs 为硬编码常数；`min_score=0.67` 是代码默认。审计未找到独立 normal calibration 选择 0.67 的证据，也不据此断言测试标签调参。|fault lists/root-cause labels 用于 AS@k/AIS@k case-level Top-k ranking evaluation；不是每窗 anomaly confusion matrix。|根因排名指标与异常 detection F1 不同；未发现 point-adjustment/延迟容忍窗口协议。见 `nezha_audit/source/alarm.py::determine_alarm/generate_threshold`、`pattern_ranker.py`、`nezha_execution_audit.md`。|`--help` 因缺依赖退出；未跑完整算法。|
|**T10 LO2**：数据集/API test execution × run，不是异常检测器|不定义 detector score 或阈值；labels 是 correct/error API-test outcome。|不适用。|API test name/outcome 是 endpoint-level 标签，不是服务异常或根因标签。|LO2 论文/数据的 run/test 统计不能被当作模型 detection P/R/F1；不可无证据将失败标签复制到全部 service windows。见 `lo2_execution_audit.md`、`endpoint_vs_servicewindow_taxonomy.md`。|数据/公开结构审计；未运行方法模型。|

### 横向结论（限本次覆盖）

- **阈值由 normal reference 明确导出的案例**：DeepTraLog 的训练 normal SVDD 半径/score>0；TraceRCA invocation 默认 normal mean/std 1σ；ServiceAnomaly 以 baseline training residual RMSE 为 cutoff（但不是独立校准）。
- **论文与当前公开实现不一致/未闭合**：AnoFusion 论文 EVT/KDE 类阈值在可见仓库评估路径中缺失；TraceAnomaly 论文 KDE/p-value 0.001 在可见代码中缺失。两者当前运行代码只能输出分数/标签文件，不能据此声称得到论文的二值检测结果。
- **强监督或测试反馈风险**：MSTGAD 半监督分类头使用已知训练 labels，最佳 F1 checkpoint 从 train set 选择；Eadro 明确每 epoch 用 test HR@1 选 checkpoint，另有全run标准化与重叠窗跨 split 风险。不能笼统说“无监督所以无标签泄漏”。
- **输出目标不同**：BARO、TraceRCA RCA 分支、Nezha 的主结果为 case-level root-cause rank；不是逐服务窗 anomaly score 的可比 P/R/F1。LO2 是 API-test 数据，不是 detector。
- **事件调整**：本次所列公开代码路径中未找到一个跨项目统一 point-adjustment/事件容忍窗口；逐项目结果和未发现项见表，不能仅凭未发现将所有论文方法下结论。
- **未覆盖**：任务清单 T12–T19 等其他论文的阈值、标签时机和 evaluator 尚未逐项目完成；T21/T22 私有数据评价部分仍按用户要求暂缓。T20 整体保持部分完成。


---

<a id="t23"></a>
## T23｜Endpoint/API 与 Service-window 任务分轨

> 来源文件：`endpoint_vs_servicewindow_taxonomy.md`。以下为完整正文；标题层级为合集显示下调一级。

## T23：Endpoint/API 与 Service-window 任务分轨（公开证据部分）

**日期**：2026-10-19。**限制**：未访问、读取或推断任何用户本地/私有 telemetry。最后一节的本地 client failure / trace error / log ERROR 对照暂缓；其余均由已公开审计报告与官方论文/代码字段支撑。

### 1. 单位与 ground-truth 对照

|项目|实际样本/输出单位|标签来源|评价对象|是否等同 Service×固定 15s window|
|---|---|---|---|---|
|LO2 dataset|API test execution × run；一次正确 API test 60s + 53 个 error API test 各10s；发布1,740 runs，论文统计1,740×54 test conditions。metrics CSV 每5s采样；日志按服务；可见 traces 非有效全覆盖。|API/test name 是否正确/出错（test condition）；是 endpoint-level API outcome，不标哪个 service/root cause fault。|论文数据分析按 API test labels / runs；不是 service anomaly ground-truth 或服务窗事件标签。|否；可按时间窗重采样，但API test 标签不能直接复制给每个服务窗。|
|ServiceAnomaly|一条 distributed trace / fault-case directory；公开 repo 的 case 数为 TS 15 normal+15 fault，TT 32+32；每case配关系表和 metric snapshots。|normal/fault directory name；部分 raw trace/metrics 缺失或空，未发现统一标签 manifest。|论文报告 trace-level P/R/F1；公开代码无自动 trace-level evaluator，入口是人工 case/graph + metric violation。|否；一个 trace/case 可横跨多个服务/时间窗，case label 不等于每个服务每窗标签。|
|TraceAnomaly|一个 STV/trace input row；model 输出每 trace log-likelihood score。|输入文本分为训练、normal test、abnormal test；测试真值随数据提供。|论文为 trace-level anomaly detection；论文用normal score distribution的 KDE/p-value 0.001阈值；当前仓库 prototype 输出分数 CSV，没有该 threshold/evaluator，也未实现 paper root-cause stage。|否；trace 路径/响应时间不等于单个 service×15s状态。|
|TraceRCA|invocation-level predictions → trace-level abnormal inference → service-level root-cause ranking。|历史 training 数据中的 labels 支持 supervised trace classifier；association RCA 对 test invocation predictions 汇总。|trace/invocation AD 与 service ranking 是两类输出；top-k RCA accuracy 不可当 anomaly F1。|否；RCA 是故障案例级排序，不是固定时长服务窗分类。|
|Nezha|由触发异常的时间窗取其请求 traces，整合每 request event graph；输出一对 expected/actual patterns / root-cause ranking。|`construct_data` fault-free reference 与 `rca_data` fault-suffering fault-list；标签粒度 service 与 inner-service/code/resource。|Paper 用 service-level AS@k 与 inner-service AIS@k（案例级Top-k root cause）；不是窗口二分类。|否；窗口是触发/抽取入口，但模型目标是图中的根因pattern，不是每服务窗口 anomaly F1。|
|MSTGAD/MSDS|模型按 `num` 位置构造窗口，metrics/log/trace 聚合与 label 长度/起点需对齐；公开论文按 time windows 做异常 detection。|MSDS fault/workload/Rally pseudo-ground-truth；当前已建立部分 hook windows，但未验证全量 trace join/标签。|窗口异常分数/类别，不等同 trace/API endpoint labels；T02 训练未运行。|可能可映射，但目前 timestamp/fault-effect范围冲突，不能声称一一可转换。|
|Eadro|trace/fault case 及关联 service/metric/log 输入；模型有 supervised detection + localization heads。|原始包 fault metadata、训练/测试切分；数据/代码 raw→parsed pipeline 不完整。|论文异常 detection 与 root-cause rank 分别报告；T05 audit 发现评估实现有差异。|否；trace/case/root-cause label 不自动生成固定服务窗口标签。|

### 2. 不能跨单位复制的规则

1. **Endpoint/API outcome ≠ service-health label。** LO2 的正确/错误 test 名称可表示该 API call outcome，但一个失败请求的因果服务可能多个、某些服务不出现异常指标；因此不把它复制成所有当时活跃 service 的异常标签。
2. **Trace/case label ≠ 每个 span 或 service window label。** 一个异常 trace 跨多个 span/service，trace-level positive 不能自动把整条路径上的每个服务窗标为正例。
3. **RCA Top-k ≠ anomaly detection F1。** Nezha / TraceRCA 输出候选服务/根因模式排名，评价单位按 failure case；不能把 rank@k 当成 window confusion-matrix score。
4. **数据采样区间 ≠ ground-truth effect interval。** Rally fault hook 的开始/结束仅是脚本执行窗口，不直接证明故障对 telemetry 的持续影响；MSDS 仍需要裁定运行终止时刻并完成 trace event joins。
5. **资源无效值/模态缺失不能自动回填为健康。** 此条留给 T22（私有质量状态部分按用户要求暂缓）；本文件不接触本地 telemetry。

### 3. 本地映射与保留项

任务清单对用户本地数据的声明（25 services × 15s windows、17 cases、normal-only train/calibration/holdout、fault-only test）可能需要 private dataset 才能实证。本轮遵守用户要求，不读文件、不校验 50,575 windows、不将私有 labels/trace coverage 反推至任何公开论文。公开文献的单位分类已完成；要标定哪些公开方法可直接对本地 service-window 评分、哪些需聚合/改造，须等用户授权提供可用的 public/local-safe schema 或撤销暂缓。

### 4. 来源报告

- `lo2_execution_audit.md`：LO2 API test labels、5s metrics、trace覆盖。
- `serviceanomaly_execution_audit.md`：ServiceAnomaly trace/case dataset与无自动 evaluator。
- `traceanomaly_tracerca_execution_audit.md`：两类trace模型单位与RCA分轨。
- `nezha_execution_audit.md`：service / inner-service Top-k root-cause output。
- `mstgad_msds_execution.md`：MSDS窗口、Rally hooks和尚未闭合的full label alignment。
- `eadro_execution_audit.md`：Eadro的监督 detection/localization 和数据契约限制。


---

## 未内嵌的支撑性原始文件

以下是原始日志、manifest、数据表或源码快照，而非叙述性报告；合集正文引用它们的结果，但不把大型/机器可读附件逐字展开：

- BARO 原始 stdout / 运行摘要：`baro_execution_logs/`。
- TraceAnomaly / TraceRCA smoke logs：`traceanomaly_smoke.log`、`tracerca_smoke.log`。
- Eadro、ServiceAnomaly、DeepTraLog 入口 stdout/stderr：相应 `eadro_*.stdout.log`、`serviceanomaly_*.stdout.log`、`deeptralog_main.stdout.log`。
- ServiceAnomaly per-file SHA-256 清单：`serviceanomaly_data_sha256.txt`。
- Nezha 源码树、inventory、manifest 与 smoke artifacts：`nezha_audit/`。
- MSDS byte-range / 时间覆盖 / Rally hook 文件：`msds_audit/`；Range-only 脚本为 `msds_audit/range_time_coverage.py`。
- T02 compatibility patch 与窗口 smoke artifacts：以 `mstgad_msds_execution.md` 中引用的文件为准。

**复现声明：** 该合集汇总的是公开文献/代码/开放数据审计及其已记录的有限运行；除明确注明的 smoke test、数据预处理子集或 artifact replay 外，没有运行模型训练，也没有声称复现论文全表结果。
