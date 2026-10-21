# TT 注入关联标签基础设施实施 Plan

> v2 增量已于2026-10-20（America/Los_Angeles）实现并验收，见[主评价范围决策](evaluation-scope-decision-20261021.md)及[v2实施交接](labeling-v2-implementation-20261020.md)：新版评价清单为479正类与2975负类，已完成来源重放、独立check及零差异复验。下文保留已执行 A 版历史计划，新增规则以v2报告为准。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. 本计划已按审阅后的执行方式完成实施；此处保留历史执行说明。

**Goal:** 为已准备的TT窗口生成可追溯、可版本化重标的A协议标签与4/1/1划分包。

**Architecture:** 保持既有准备产物只读，新增TT证据adapter、纯标签判定函数和独立包发布/验收入口。特征有效性与标签分离；单独比较标签版本，不接入训练器。

**Tech Stack:** Python、pandas nullable Int8、pyarrow、JSON、SHA256、argparse、pytest；conda环境interface，不新增依赖。

**Spec:** [labeling-infrastructure-spec.md](labeling-infrastructure-spec.md)

## Global Constraints

- 仅selected17、25应用候选、15秒窗口，全网格50575行；原始数据和旧产物只读。
- 4个Normal fit、1个calibration、1个test，Case ID逐项采用Spec；Fault全部test。
- 正类仅直接应用目标完整注入窗口；瞬时KILL与数据库不生成窗口正类。
- Normal及干预前参考负类应用传输失败例外；Unknown为null，不进入训练/校准/分类指标。
- 既有preparation全Unknown和3/1/2验收保持原样；不调用旧build_label_splits生成新划分。
- 输出新目录，拒绝覆盖；不含模型训练、重采、改写SN或DVC重建。
- 不提交raw/parquet、外部技能symlink及用户已有修改；提交遵循`[Type]: description`。

## Review Focus

- 同一plan item失败重试不能替换selected attempt：Task 1绑定Case/attempt与manifest。
- probe暂停早于注入，确认晚于暂停边界：Task 1校验时钟锚点和暂停政策。
- 跨干预才结束的超时不能倒推故障前异常：Task 2边界与例外测试。
- 中断写包或重放造成半成品/版本漂移：Task 3发布失败和语义hash测试。
- Unknown、单类Case及缺评分被静默删除：Task 4行守恒和评分连接测试。

## 文件与接口边界

新增 `src/service/labeling/`，不重构旧preparation模块：

|文件|职责|
|---|---|
|`__init__.py`|无副作用包入口|
|`evidence.py`|输入身份/hash、TT控制证据、流式load attempts索引|
|`rules.py`|纯区间判定、例外、标签与Case划分|
|`bundle.py`|协议读取、版本hash、发布、差异比较|
|`acceptance.py`|独立验收、覆盖和评分连接校验|
|`scripts/label_service_data.py`|build/check/diff命令|
|`configs/service/tt_labels_a_v1.json`|协议参数、固定Case角色、输入身份|
|`tests/test_service_labeling_*.py`|证据、规则、写包与验收测试|

`evidence.py`定义dataclass `EvidenceIndex`：grid、case_records、exceptions、sources、warnings。
grid沿用原主键；case_records包含Case身份、运行准入、target/backend、实际区间、
故障前上界；exceptions为case_id、start/end、证据ID和全候选服务范围。
`rules.py`定义 `LabelTables`：labels与splits两个DataFrame。

可复用纯函数：preparation.py的checked_path/file_digest，modality_preparation.py的
KEYS/stable_ids/validate_grid；调用前核对签名。不得调用重建数据或强制旧划分的入口。
`.claude/rules/`在本轮检查时不存在，不宣称已读；以AGENTS、当前Spec和实际源码为准。

### Task 1：证据adapter与运行准入

**Files:** 新建labeling/__init__.py、evidence.py、tt_labels_a_v1.json；测试`tests/test_service_labeling_evidence.py`。

**Interfaces:** `load_evidence(tt_root: Path, preparation: Path, protocol: dict) -> EvidenceIndex`。
读取selected_cases、迁移manifest/path map、准备manifest、TT resource网格及各Case小文件。
原绝对路径用现有路径映射解析，不改原hash。Task 2只消费该结构，不自行读取raw。

- [ ] 写失败测试`test_selected_attempt_binding`：篡改attempt/hash必须ValueError，failed attempt不得替换选中Case。
- [ ] 写失败测试`test_admission_is_not_health_gold`：HTTP200/日志ERROR不控制准入；顶层采集错误不得忽略；failures汇总只警示。
- [ ] 写失败测试`test_pause_anchor_precedes_confirmation`：start_ms=100000、start_mono_ns=1000000000、pause=31000000000得到130000ms，不取晚到确认值。
- [ ] 写失败测试`test_optional_pause_vs_missing_required_anchor`：明确无暂停可用apply请求；有暂停政策却缺锚点时故障前资格Unknown。
- [ ] 运行`conda run -n interface python -m pytest tests/test_service_labeling_evidence.py -q`，确认缺实现失败。
- [ ] 实现adapter与配置：逐Case校验身份/hash、整型区间、运行准入；将暂停时钟转换、缺失和冲突处理落实为证据字段，不猜时区。
- [ ] 流式读取load事件：只定位已发出且end有效的exception；保留证据ID/原文件定位，suppressed不作为传输失败。准备阶段文件hash需绑定到其原manifest，控制/请求新增来源保存自身hash。
- [ ] 重跑本任务测试，通过后选择性提交`[Feature]: bind TT labeling evidence`。

### Task 2：纯标签判定与4/1/1划分

**Files:** 新建rules.py；测试`tests/test_service_labeling_rules.py`。

**Interfaces:**
- `classify_windows(grid: pd.DataFrame, evidence: EvidenceIndex, protocol: dict) -> LabelTables`。
- `build_splits(grid: pd.DataFrame, protocol: dict) -> pd.DataFrame`；角色取Spec固定清单，不复制旧split。

- [ ] 写失败测试`test_full_containment`：区间[15000,45000)，[15000,30000)和[30000,45000)标1；部分重叠和非目标不标1。
- [ ] 写失败测试`test_pre_intervention_negative`：window_end等于上界可标0；probe暂停至生效间及恢复段Unknown。
- [ ] 写失败测试`test_instant_and_database`：KILL和数据库无正类，但保留事件和合格故障前负类，不删整Case。
- [ ] 写失败测试`test_timeout_all_services`：失败[20000,31000)排除与其重叠的[15000,30000)、[30000,45000)内所有服务；仅端点相接不排除。
- [ ] 写失败测试`test_cross_intervention_timeout`：请求边界前发起、之后结束，不倒推故障前例外。
- [ ] 写失败测试`test_fixed_split_and_mask`：4/1/1与Fault test；Unknown保持原split，但不在`split==fit & label==0`集合；特征缺失不改变标签。
- [ ] 运行`conda run -n interface python -m pytest tests/test_service_labeling_rules.py -q`，确认失败。
- [ ] 实现Spec判定顺序、nullable Int8、稳定键、证据ID及原因；正负冲突报错，不后写覆盖前写。
- [ ] 重跑本任务测试，通过后提交`[Feature]: implement injection association labeling rules`。

### Task 3：版本包、重标和差异

**Files:** 新建bundle.py；测试`tests/test_service_labeling_bundle.py`。

**Interfaces:**
- `build_bundle(tt_root: Path, preparation: Path, protocol_path: Path, output: Path) -> dict`。
- `diff_bundles(old: Path, new: Path, output: Path) -> dict`。
- 消费Task 1/2；输出run目录下labels包的Spec七个文件，manifest最后发布。

- [ ] 写失败测试`test_replay_semantic_hash`：不同输出目录重放，排序标签/split与语义hash相同。
- [ ] 写失败测试`test_no_overwrite_or_partial_publish`：已存在目录拒绝；模拟写失败不留下成功manifest。
- [ ] 写失败测试`test_version_bindings`：规则/证据变更更新标签版本，split变更更新划分版本，时间戳/目录不进入语义hash。
- [ ] 写失败测试`test_diff_stable_keys`：重排行无假差异；新增/删除键与0/1/null转换均报告，不静默inner join。
- [ ] 运行`conda run -n interface python -m pytest tests/test_service_labeling_bundle.py -q`，确认失败。
- [ ] 实现临时目录、内部校验、发布与diff；保存输入文件hash、代码commit/dirty、依赖版本。diff输出新目录，不读取模型分数或修改输入。
- [ ] 重跑本任务测试，通过后提交`[Feature]: version TT label bundles and relabel diffs`。

### Task 4：独立验收与CLI

**Files:** 新建acceptance.py、scripts/label_service_data.py；测试`tests/test_service_labeling_acceptance.py`。

**Interfaces:**
- `check_bundle(bundle: Path, preparation: Path) -> dict`：核对hash、规则/覆盖，不能只相信coverage汇总；回读manifest中的证据来源以复核判定。
- `join_scores(labels: pd.DataFrame, splits: pd.DataFrame, scores: pd.DataFrame, expected_ids: set[str]) -> pd.DataFrame`：校验连接，不提供训练/模型指标算法。
- CLI：`build --tt-root --preparation --protocol --output`、`check --bundle --preparation`、`diff --old --new --output`。

- [ ] 写失败测试`test_conservation_and_tamper`：缺行/重复、非法标签、文件hash错误、悬空证据拒绝。
- [ ] 写失败测试`test_explicit_score_mask`：expected_ids内缺评分/非有限评分、重复评分拒绝；校验标签包和评分声明的label/split版本一致。
- [ ] 写失败测试`test_unknown_and_single_class`：Unknown不进入已知标签集合；仅单类资格报告标明AUROC/AP不可计算，不伪造0。
- [ ] 写失败测试`test_old_gate_unchanged`：旧验收仍拒绝非Unknown，新验收接受合法A包；不改旧门槛。
- [ ] 写失败测试`test_output_path_safety`：拒绝raw目录、outputs/service根目录和已存在run目录。
- [ ] 运行`conda run -n interface python -m pytest tests/test_service_labeling_acceptance.py -q`，确认失败。
- [ ] 实现独立验收/CLI，错误退出码非零；训练和校准资格明确要求label==0及独立模态mask。
- [ ] 重跑Task 1—4测试，通过后提交`[Feature]: validate TT label bundles independently`。

### Task 5：真实标签包验收与交接

**Files:** 新建`docs/research/service-20261019/labeling-infrastructure-implementation.md`；定点更新研究README，依project-history技能维护history entry/index。

**Interfaces:** 消费Task 4 CLI。批准执行本计划时包括新A包生成与验收，不包括模型训练。

- [ ] 核对git状态和conda环境，保留用户dirty内容；实际输入与核查hash不同则停止发布并报告。
- [ ] 运行相关测试：

```bash
conda run -n interface python -m pytest tests/test_service_labeling_evidence.py tests/test_service_labeling_rules.py tests/test_service_labeling_bundle.py tests/test_service_labeling_acceptance.py tests/test_service_label_splits.py tests/test_service_preparation_acceptance.py -q
```

- [ ] 生成新包；run已存在时换新run_id，不覆盖：

```bash
conda run -n interface python scripts/label_service_data.py build \
  --tt-root data/tt_service_20261017_004546 \
  --preparation outputs/service/preparation-exploratory-20261020-task4 \
  --protocol configs/service/tt_labels_a_v1.json \
  --output outputs/service/tt-labels-a-v1-20261020
```

- [ ] 运行`conda run -n interface python scripts/label_service_data.py check --bundle outputs/service/tt-labels-a-v1-20261020/labels --preparation outputs/service/preparation-exploratory-20261020-task4`。
- [ ] 核对50575总行、27150负、194正、23231Unknown及fit10600/cal2975/test负13575。不符先查来源，不调整规则凑数；这些计数不代替逐行规则验收。
- [ ] 第二次在新run目录重放，用diff确认语义无变化；规则变化的重标用合成样本验证，不根据私有测试效果选版本。
- [ ] 逐Case报告正负/Unknown、5个preserve正类事件、6个无正类事件原因、超时1300行排除与质量警示；资源有效正类171需重核，不称三模态共同有效数。
- [ ] 记录实际命令、结果、环境、版本/manifest和局限；不将计划写为完成。实现与测试应替代/tmp探查成为正式可复现入口。
- [ ] history查重后记录本次决策依据、陷阱与遗留事项；选择性提交实现和文档，不提交parquet或用户笔记。用户要求PR时使用正式PR，不自动push/merge。

## 自查与执行交接

证据/准入落到Task 1，标签/划分与例外落到Task 2，重标/版本落到Task 3，验收/评分连接落到Task 4，生产复核/历史交接落到Task 5。
B、KILL持续区间推断、数据库传播及训练不在本计划内。具体阈值分位数留到模型session确定。
复用开发分支`feature/service-labeling-infrastructure`；执行隔离遵循届时技能与用户工作区约束，不覆盖dirty文件。
本计划中的测试和生产构建已执行；实际结果见上方实施状态与实现报告。

## 实施状态（2026-10-20）

Task 1–5 已实施。生产包、85项指定测试、独立check及新目录语义重放结果见 [实现报告](labeling-infrastructure-implementation.md)。旧探查计数中的TRANSACTION故障前875负类因unknown_pause_policy撤销，不按参考凑数。未运行模型训练。
