# Service 模型研发基础设施实施 Plan

> **For agentic workers:** 实施时按任务逐项执行与验收；使用届时可用的 `superpowers:executing-plans`，或用户明确选择后的 `superpowers:subagent-driven-development`。不自动启动代理。Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在版本绑定的TT特征/标签包上建立一套共享端到端模型、可替换编码器、同集合基线及本地MLflow记录，先交付可测试基础设施，再按独立授权开展真实训练。

**Architecture:** 新增隔离的 `src/service/modeling/`，通过稳定样本键连接现有特征与标签侧表。预处理只在fit拟合，训练器只消费特征和掩码；可配置编码器→同窗融合→固定中心SVDD，评分/校准/评价与训练分开。复用现有纯函数及ExperimentLogger，不迁移旧Endpoint训练框架。

**Tech Stack:** conda interface、Python、pandas/pyarrow、numpy/sklearn、PyTorch、Drain3、MLflow、pytest；文本加载候选为transformers，具体兼容版本在实施预检固定，不在本session安装或下载。

**Spec:** [model-infrastructure-spec.md](model-infrastructure-spec.md)，用户已认可完整设计及技术默认值。

## 执行与授权边界

本Plan于2026-10-21将v2接入整合到各任务正文，正式消费verified包及evaluation侧表。一套标签用于主评价和内部同Case/同Service前后诊断；不重新打标、不SMOTE或削减Normal追求均衡。内部诊断不用于择优选协议、模型或阈值，影响结论的运行混淆须如实报告。

- 本文档最初只用于设计审阅；本次实施session已在最新`master`上创建独立开发分支`feature/service-model-infrastructure`，README研究索引更新已从原共享分支的stash纳入。后续代码实现、定向测试和Git提交须按用户逐步授权执行。
- 当前分支与文档状态以`feature/service-model-infrastructure`为准；开始每个Task前重新核对最新接口、Git状态和输入版本，不回写或覆盖原打标分支，也不自动处理其他stash。
- 下列Task 1—8是未来开发及定向测试计划；Task 9是真实输入预检及fit小批量GPU检查，Task 10是真实训练/测试实验。阅读或认可本Plan不等于本session开始执行。
- 下个实施session先明确获准执行至哪个Task。真实实验未获授权时停在可审阅代码/合成验收，不把Task 9/10当默认附带动作。
- 代码提交仅在后续用户授权提交时执行，按任务选择性暂存，格式 `[Feature]: ...` / `[Test]: ...` 等；不自动push/PR/merge。以下各任务“交付”不隐含立即提交。

## Global Constraints

- 只用selected17，Service×15秒，全网格50,575，固定4 Normal fit / 1 calibration / 1 test；11 Fault仅test。Unknown不填0，训练/校准要求对应split且label=0。
- 原始数据、旧特征及标签包只读，不glob、不调用旧DVC/build_contract/准备重建入口。新输出不存在才可创建，失败不发布成功manifest。
- 一套共享网络，无Service ID输入；逐服务fit数值预处理，部分缺失保留掩码，全模态不可用仅过滤模型输入并保留排除记录。
- 资源R3；执行数量/mean/p95/max/完整已知错误率；日志规则严格遵循Spec第4、11节，不按Fault结果改模板/截断/参数。
- token64，MLP隐藏64，Transformer1层/4头/FF128，SVDD32维；小BERT为google/bert_uncased_L-4_H-256_A-4，全部层联合训练。
- 日志16消息/128token，ERROR/FATAL与未匹配各预留4；模板内/模板间等权，拼接四项全窗统计；固定中心来自fit初始表示，近零分量保护0.1。
- AdamW，BERT lr=1e-5、其他lr=1e-3、weight_decay=1e-4，20轮最后checkpoint；微批量1、累积16，尾批按实际窗口数归一化。
- 原型seed42，正式{1,2,42}，不挑seed/checkpoint。99%主阈值、95/97.5/99/99.5敏感性，linear分位数，严格大于报警。
- 统一样本集合与高分异常方向；总体/逐Case/覆盖报告，单类AUROC/AP为null，缺必需分数报错，不事后取成功评分交集。
- 主测试严格为evaluation_eligible与预定共同有效集合交集，Fault pre只作内部诊断。当前各Case单类，不计算Case内AUROC/AP宏平均；逐Fault报告召回、逐Normal服务报告告警。保存完整内部诊断，不依结果选择性隐去影响结论的证据。
- 本地MLflow和独立产物双记录；数据、标签、划分、预处理、模型及评分版本绑定。失败seed保留，诊断不替代检测证据。

## Review Focus

1. Parquet回读丢失DataFrame.attrs：Task 1/6由已验证manifest显式恢复版本声明，防止合法分数无法连接或伪版本被接受。
2. 仅日志有效、资源质量不明和部分错误tag：Task 2/3独立判掩码与固定集合，不能把缺valid列当全有效。
3. ERROR与未匹配同时命中、空正文、超长堆栈：Task 3去重配额、头尾截断、全窗统计/选取覆盖分开，拒绝空平均。
4. 模态缺失、文本缓存断图和累积尾批：Task 4/5验证可用分支有梯度、无效内容不影响分数、末批不降权/丢失。
5. OOM/追踪失败/中断写包与退化seed：Task 7/8不发布成功状态、不自动改模型重跑，不静默删除失败结果。

## 现状与接口核对依据

本Plan核对时labeling/evidence、rules、bundle、acceptance、evaluation及CLI已存在。`read_bundle(path: Path) -> dict`读取schema 2包时返回manifest/labels/splits/evaluation并重算评价语义；`check_bundle(bundle: Path, preparation: Path) -> dict`重读证据复核，适合独立预检，不在每epoch调用。

`join_evaluation_scores(data: dict, scores: DataFrame, valid_ids: set[str]) -> DataFrame`是主入口，内部重算评价语义并与预定有效ID相交；scores显式声明label/split/evaluation版本，列名为score。底层join_scores不负责主范围，不能用全部test非空标签绕过evaluation；二者不用于fit/cal资格计算。

正式包为`outputs/service/tt-labels-a-v2-20261020-verified/labels`，不使用中间包。交接记录全网格26,275负、479正、23,821 Unknown；fit负10,600/cal负2,975不变。主范围2975负/479正，9725个Fault pre保留标签但退出主评价；resource_valid口径2453负/397正不硬编码为模型共同集合。正类覆盖10个Fault Case、5服务及三种来源，TRANSACTION不可按A版断言整Case全Unknown。以上是他session验收引用，本session未重验。

已有 `fit_normalizer` 仅全局均值/尺度且阈值1e-9，不满足新三级回退，不能直接复用其fit语义；可复用apply_normalizer及IF/OCSVM fit/score、安全指标纯函数。旧 `DeepSVDD` 内置单个MLP，不能冒充现有可接三token的模型。ExperimentLogger已封装但此前无训练调用；transformers未见于当前pyproject依赖，不假定已经安装。

## 文件与窄接口

新增包仅服务本轮模型，不构建通用插件平台；若后续合入代码已有同职责模块，先记录并定点复用，不重构旧管线。

|文件|职责|
|---|---|
|`src/service/modeling/__init__.py`|无副作用包入口|
|`config.py`|配置校验、resolved参数及字段白名单|
|`data.py`|表/版本绑定、稳定清单、特征批次与排除原因|
|`numeric.py`|逐字段变换、fit统计、三级尺度及掩码|
|`logs.py`|fit模板、冻结匹配、消息选择、tokenization及统计|
|`encoders.py`|可替换数值/文本编码器及统一输出|
|`model.py`|三token、融合变体、固定中心SVDD|
|`training.py`|等窗损失、梯度累积、诊断、checkpoint|
|`baselines.py`|数值/冻结文本传统基线|
|`evaluation.py`|评分连接、阈值、指标及覆盖|
|`runs.py`|原子产物、配置/版本manifest、MLflow协调|
|`scripts/service_model.py`|显式子命令，不接入旧Hydra入口|
|`configs/service/tt_model_v1.json`|所有已批准参数、路径和字段顺序|
|`configs/service/tt_model_matrix_v1.json`|固定方法/seed/阶段及比较清单|
|`tests/test_service_model_*.py`|按下述Task的定向检查|

共享类型在所属模块定义：`InputTables`（data.py：grid/resource/execution/logs/labels/splits/evaluation/provenance，含评价版本及内容hash）；`SampleLists`（data.py：fit/calibration/main_test/full_test/fit_diagnostic，以及按Case/Service分组的diagnostic_pre/diagnostic_positive与exclusions）；`NumericState`（numeric.py：列序/服务均值/尺度来源/fit ID hash）；`LogState`（logs.py：逐服务模板快照与fit/input/config hash）；`PreparedData`（data.py：按ID索引的数值、字段/模态mask、日志tokens/组关系/四统计及版本）；`EncodedModality`（encoders.py：token、valid、具名aux_losses）。fit_diagnostic是训练表示诊断，不同于测试前后诊断；不用数组行号充当身份。

所有代码测试默认CPU、小型合成表和本地tiny文本网络，不下载checkpoint、不读真实Fault或raw；生产GPU检查单列Task 9。以下命令仅为未来执行说明，均未在本session运行。

### Task 1：版本绑定的输入读取与配置

**Files:** config.py、data.py、__init__.py、tt_model_v1.json；tests/test_service_model_data.py。

**Interfaces:** `load_config(path: Path) -> dict`；`load_inputs(preparation: Path, label_bundle: Path, config: dict) -> InputTables`。只加载tt/三模态，labels/splits/evaluation来自schema 2 read_bundle。返回完整网格，不按标签inner join丢行。

- [ ] 实施起点先只读核对interface中的Python及本任务依赖版本；缺依赖导致的导入失败属于环境阻断，不当作目标测试红灯。只有后续授权确有该依赖需求时定点安装，不在本轮文档session操作环境。
- [ ] 写失败测试 `test_roundtrip_versions_and_grid`：保存/回读后attrs丢失，由manifest恢复；标签/划分版本、hash、主键及sample_id逐一核对；旧3/1/2侧表不得覆盖新角色。
- [ ] 写失败测试 `test_evaluation_content_binding`：schema 2评价表覆盖同一全网格且语义重算一致；label/split/evaluation_version、表内容hash及主清单绑定，同名评价版本下篡改内容必须拒绝。
- [ ] 写失败测试 `test_identity_and_feature_allowlist`：重复/额外/缺失键、伪版本及冲突结束时间均拒绝；Case/target/phase/probe及label不会出现在模型特征批次。
- [ ] 运行 `conda run -n interface python -m pytest tests/test_service_model_data.py -q`，确认目标测试因缺实现失败，不以环境导入错误代替红灯。
- [ ] 实现输入校验、配置完整展开及hash绑定；read_bundle仅提供标签侧输入，不重建标签。读取resource/execution/log schema校验白名单，不假定execution_valid/log_valid存在。
- [ ] 重跑本任务测试，记录通过结果并交付输入适配器；不生成真实新数据包。

### Task 2：数值fit预处理与固定样本资格

**Files:** numeric.py、data.py；tests/test_service_model_numeric.py、tests/test_service_model_membership.py。

**Interfaces:** `fit_numeric(tables: InputTables, fit_ids: set[str], config: dict) -> NumericState`；`transform_numeric(tables: InputTables, state: NumericState) -> dict[str, DataFrame]`（values/masks/quality均含sample_id）；`build_sample_lists(tables: InputTables, numeric: dict, log_valid: Series, mode: str) -> SampleLists`。

- [ ] 写失败测试 `test_fit_only_and_three_level_scale`：cal/test极端值不影响fit；同服务常量回退全服务fit，再回退1；完全无fit观测保持字段不可用。ddof=0，尺度须>1e-6。

  该fixture固定某未变换字段：服务A的fit值[5,5]、服务B的fit值[3,7]，A应保持均值5且回退全局尺度sqrt(2)；全体fit都为5时尺度为1；test加入1e9不能改变上述参数。
- [ ] 写失败测试 `test_field_quality_and_execution_error`：resource_valid=false/unknown时不得仅凭非空值启用资源；无更细证据则该行R3全屏蔽；执行部分错误未知只屏蔽error_rate，无Entry屏蔽执行；负数非法域/非有限值不经log1p“修复”。
- [ ] 写失败测试 `test_common_and_extended_membership`：共同集合至少一个资源/执行有效字段；main_test=共同有效ID∩evaluation_eligible；仅日志有效只进扩展推理，其分类指标仍受evaluation约束；Unknown不进fit/cal/main_test，全无效记排除。
- [ ] 写失败测试 `test_fault_pre_excluded_from_main_but_kept_for_diagnostics`：Fault pre虽为label0也不进main_test，按Case/Service进入diagnostic_pre，同组已知正类进diagnostic_positive；缺前参考标不可用，不借其他服务/Unknown补充，清单先于出分冻结。
- [ ] 运行本任务两个测试文件，确认失败；实现Spec第11.1节列变换、状态与稳定清单。
- [ ] 先用固定Normal fit资格建立原始数值可用性与fit状态，再冻结共同清单；如资格计算要求重算fit统计，重算后断言清单稳定，否则报循环资格错误，不静默依赖test。保存初选到最终清单的原因。
- [ ] 重跑定向测试；交付fit状态与清单hash，不发布模型。

### Task 3：Drain3、日志预算和可反向传播的文本输入

**Files:** logs.py、data.py；tests/test_service_model_logs.py。

**Interfaces:** `fit_logs(tables: InputTables, fit_ids: set[str], config: dict) -> LogState`；`prepare_logs(tables: InputTables, state: LogState, tokenizer: object, config: dict) -> dict`（按ID的input_ids/attention_mask/special_mask/group_ids、counts、masks、coverage）。`assemble_prepared(tables, numeric, logs, lists) -> PreparedData` 在data.py定义。

- [ ] 写失败测试 `test_frozen_service_templates_and_unknown_groups`：fit不同服务模板隔离，cal/test调用后模板快照hash不变，未知不同文本分开；无fit日志的服务无模板可新增，但其新消息仍可进入未知组/BERT。
- [ ] 写失败测试 `test_priority_overlap_and_round_robin_16`：同时ERROR和未知只选一次；优先4+4后补足，确定排序不受表行重排影响；不超过16条，记录未选覆盖。
- [ ] 写失败测试 `test_head_tail_128_and_empty_body`：扣除真实特殊符号，头ceil(2B/3)/尾剩余；空正文无空平均；ERROR级别解析失败不猜；堆栈和数字仍留给BERT。
- [ ] 写失败测试 `test_full_window_stats_not_selected_stats`：四统计来自全部合格正文消息及count，非16条子集；日志无事件和采集缺失分别记原因；空正文排除次数另报。
- [ ] 运行日志测试确认失败；实现纯消息清洗/组内计数及稳定排序；模板fit按稳定sample_id和文本顺序，重复count不为改变模板状态而反复喂相同消息。
- [ ] 模板学习完成后，fit/cal/test都用最终冻结状态重新匹配，不使用训练过程中会变化的临时cluster ID作最终组；模型列配置保存Drain实际resolved参数。首版使用新建TemplateMinerConfig的默认解析参数、空自定义mask清单并显式保存，不继承旧2000字符截断或旧模板文件；超长消息的Drain处理先以fit预检衡量，若需新增字符预算须作为输入设计变更记录，不偷偷启用旧截断。
- [ ] 实现输入缓存仅包含清洗/tokenization和版本，不缓存用于联合训练的固定BERT embedding；使用假tokenizer运行单测，重跑至通过。

### Task 4：编码器、三种融合及固定中心SVDD

**Files:** encoders.py、model.py；tests/test_service_model_network.py。

**Interfaces:** `NumericEncoder.forward(values, field_mask) -> EncodedModality`；`TextEncoder.forward(log_batch, stats, stats_mask) -> EncodedModality`；`ServiceSVDD.encode(batch: dict) -> dict`（representation、aux_losses）；`ServiceSVDD.forward(batch: dict) -> dict`（另含score，中心未初始化则拒绝评分）；`initialize_center(model, fit_loader, eps: float = 0.1) -> None`（调用encode初始化，不依赖score）；`build_model(config: dict, text_backbone: Module) -> ServiceSVDD`。

- [ ] 写失败测试 `test_all_available_branches_receive_gradients`：合成三模态训练反传后BERT替身、数值MLP、投影与Transformer均有有限梯度；模板解析无梯度；不得将预训练参数requires_grad=false。
- [ ] 写失败测试 `test_equal_message_template_pooling`：排除padding/特殊符号，模板内及模板间等权，改变重复count不改变文本均值但改变独立统计。
- [ ] 写失败测试 `test_missing_token_does_not_influence_output`：eval下将被mask的占位值改为极端值或NaN仍不改变有限score，不能用NaN乘0假装屏蔽；三模态全无效在入网前拒绝；mean模式不把缺失输出纳入分母，cls不能单独构成观测。
- [ ] 写失败测试 `test_center_fixed_and_reload_score`：中心仅fit初始化，近零分量保护0.1，无梯度、state_dict回读保持，eval分数相同。
- [ ] 运行network测试确认失败；实现Spec第11.3结构及三种配置：transformer_cls、transformer_mean、concat_svdd；concat固定三token+三模态mask为195维输入，经64→32无bias投影。
- [ ] 实现必要的modality type embedding，缺失concat token在加类型表示后重新置0并传mask；主比较不使用service/time embedding。token64、SVDD32严格形状断言。
- [ ] 重跑network测试；tiny本地backbone证明接口即可，此任务不下载真BERT、不声称显存已通过。

### Task 5：等窗训练、尾批、checkpoint与fit诊断

**Files:** training.py；tests/test_service_model_training.py。

**Interfaces:** `train_model(model, prepared: PreparedData, lists: SampleLists, config: dict, output: Path) -> dict`；`score_model(model, prepared, ids: set[str]) -> DataFrame`；`probe_fit(model, prepared, lists: SampleLists, config: dict) -> dict`（仅取fit清单首16窗前向/反向，返回显存/耗时/梯度及状态，不输出正式模型）。训练返回checkpoint路径、fit诊断、配置/清单hash及状态。分数列固定为score，带稳定键与版本。

- [ ] 写失败测试 `test_accumulation_tail_matches_window_mean`：17窗以16+1更新，尾批不丢/不除以16降权；不同消息数不改变窗口主损失权重。
- [ ] 写失败测试 `test_partition_inaccessible_and_last_checkpoint`：训练loader只包含fit label0，中心只读fit；校准/test评分函数不参与optimizer；保存最后一轮，不按test或损失最小回选。
- [ ] 写失败测试 `test_diagnostics_eval_and_failed_seed_retained`：诊断固定至多256窗，eval模式不靠dropout制造方差；非有限值失败，低方差登记警示且保留seed，不自动训练新seed。
- [ ] 运行training测试确认失败；复用set_seed，num_workers=0作为首版，按窗口等权shuffle；一个AdamW两个lr组、20轮、无scheduler、无额外辅助loss。不要调用旧训练main。
- [ ] 实现每epoch诊断、checkpoint含模型/中心/config/preprocessor引用与hash；本轮仅CPU合成测试可缩短epoch，不把测试epoch写成真实实验20轮已完成。
- [ ] 重跑测试；不引入自动OOM降模/冻结BERT/回选checkpoint。继续训练恢复不属于首版，失败后新run显式重启并记录关联。

### Task 6：基线、独立阈值与评分连接

**Files:** baselines.py、evaluation.py；tests/test_service_model_baselines.py、tests/test_service_model_evaluation.py。

**Interfaces:** `fit_baselines(prepared: PreparedData, lists: SampleLists, config: dict) -> dict`；`score_baselines(bundle: dict, prepared, ids: set[str]) -> dict[str, DataFrame]`；`calibrate(scores: DataFrame, expected_ids: set[str], quantiles: list[float]) -> dict`；`evaluate_run(tables: InputTables, lists: SampleLists, scores: DataFrame, thresholds: dict) -> dict`。

evaluate_run接收主测试、前后诊断及可选扩展ID所需分数的并集，分别校验和输出main、within_case_diagnostic、extended、coverage结果，不能在主连接过滤后丢掉诊断分数。主范围必须通过join_evaluation_scores，诊断按自身清单校验；任一预定诊断缺必需分数显式标失败，不用成功子集冒充完整诊断。

- [ ] 写失败测试 `test_rules_masks_and_frozen_text_baseline`：L2/Max不计mask位，全部数值无效不伪造0分；文本冻结、四统计/选取一致，embedding仅fit全局逐维标准化，缺失占位不污染fit。
- [ ] 写失败测试 `test_params_and_score_orientation`：IF100/n_jobs1与seed，OCSVM RBF/gamma scale/nu0.1；正常方向反转由固定score函数完成，不允许按测试标签翻方向。OCSVM本身无随机性，相同输入无需伪造三次独立重复。
- [ ] 写失败测试 `test_quantiles_and_missing_calibration`：linear四分位点、相等不报警、无校准失败、校准expected_ids缺一个分数也失败，不借test。

  例如cal分数[0,1,2,3,4]的99百分位为3.96；score=3.96不报警、score=4报警。删去预定cal ID对应一条分数必须失败，即使剩余分数仍足以计算分位数。
- [ ] 写失败测试 `test_roundtrip_attrs_and_main_evaluation_join`：恢复label/split及scores评价声明后调用join_evaluation_scores，传入预定valid_ids，结果严格等于main_test；重复/冲突版本/非有限及缺分拒绝，Unknown/Fault pre不得混入主指标。
- [ ] 写失败测试 `test_single_class_main_cases_and_zero_metrics`：Fault主集合仅正、Normal仅负，逐Case AUROC/AP及其宏平均null并说明；整体有两类时可计算。无报警precision null，有正无检出recall/F1=0；Case/服务/来源正类分母固定，失败不缩分母。
- [ ] 写失败测试 `test_diagnostic_scope_does_not_change_main_or_threshold`：同Case/Service前后诊断复用同模型分数与cal阈值，独立输出分布/轨迹、前参考告警率/正类召回；不改变main_test或fit/cal，缺前参考标不可用，不重新训练/调参。
- [ ] 运行两个测试文件确认失败；复用benchmark_downstream的纯fit/score/safe函数，调用前显式校验输入，防safe函数吞异常后误作单类。新规则和阈值不调用旧Service协议入口。
- [ ] 实现主报告与内部诊断：整体AUROC/AP、四档分类指标、逐Fault/来源/服务召回及Normal告警，不算空Case内AUC宏平均。默认同模型/阈值扩展test，分类仍限evaluation_eligible；Fault pre和Unknown另存。诊断不另立择优排行榜，完整保存分数、null原因及覆盖。
- [ ] 重跑本任务测试，检查数据版本与高分方向的一致性。

### Task 7：运行目录、MLflow和版本失效

**Files:** runs.py；tests/test_service_model_runs.py；后续实施中按需最小修改src/utils/logger.py，保持既有调用兼容。

**Interfaces:** `RunRecorder(output: Path, config: dict, provenance: dict)` 包装ExperimentLogger，提供context manager、`log_epoch(metrics: dict, step: int)`、`publish(artifacts: dict) -> dict`；`check_reuse(previous: dict, current: dict) -> dict` 给出可复评分/需重fit/需重cal原因，不自动执行。

- [ ] 写失败测试 `test_no_overwrite_and_atomic_status`：已存在run、raw目录及outputs/service根拒绝；写失败、OOM或中断留失败记录，不留成功manifest；未授权tracking URI不从继承环境发送到外部服务。
- [ ] 写失败测试 `test_mlflow_failure_is_visible`：追踪后端失败时本地保存错误和run身份并标失败，不静默当记录成功；本地MLflow临时目录测试不启动服务器、不依赖网络。
- [ ] 写失败测试 `test_evaluation_links_to_training_run`：fit后独立evaluate仍能通过持久化的MLflow run_id将阈值/指标关联到同一模型配置/seed，不能只创建无关联的新run；同一模型的扩展评价以独立evaluation_id区分，不覆盖共同集合指标。训练阶段FINISHED不等于完整评价已成功。
- [ ] 写失败测试 `test_version_invalidation`：仅test标签变化可固定分数重评，fit/cal资格、划分或输入/模板/掩码变更触发相应重fit/recal；模型原训练版本保留，新评价manifest绑定原模型与新评价标签，不能只改attrs冒充可复用。
- [ ] 版本测试覆盖仅evaluation变化：fit/cal不变时无需重训，经兼容核查复用原模型/阈值，新增应评分ID缺分时仅补推理；保存原训练版本、新评价版本/表内容hash/主和诊断清单hash，不伪造原分数已绑定新标签。
- [ ] manifest/MLflow保存评价表内容hash、质量规则hash、main_test和按Case/Service的诊断清单hash；主评价与诊断文件名/指标名前缀区分，和原模型run_id双向关联，不能只凭evaluation_version同名判断身份相等。
- [ ] 运行runs测试确认失败；实现data/run/evaluation/report子目录及阶段状态，配置/清单/中心/checkpoint/逐样本score/阈值/诊断/报告均有hash与版本绑定。
- [ ] 复用ExperimentLogger，本地URI由配置显式解析；为跨阶段记录可在start_run末尾新增可选run_id参数，核对当前MLflow API后透传，旧调用保持兼容。模型/seed主run及各评价ID写入manifest。若当前MLflow不接受旧文件存储，先用当前安装版本兼容的本地后端做最小修复并保持默认接口，不升级全环境。允许额外本地SQLite，但不得自动要求常驻服务。
- [ ] 记录params、step metrics、配置/manifest artifacts、失败状态；原始大日志不上传，null指标只记原因。重跑测试交付追踪协调器。

### Task 8：CLI、可离线验收的最小闭环与文档交接

**Files:** scripts/service_model.py、tt_model_matrix_v1.json；tests/test_service_model_cli.py、tests/test_service_model_smoke.py；未来实施时新增model-infrastructure-implementation.md，按history技能新增entry/index（不在本session修改）。

**Interfaces:** 显式子命令 `check-inputs --config --output`、`prepare --config --output`、`probe-fit --config --prepared --output`、`fit --config --prepared --output`、`score --config --prepared --model-run --output`、`evaluate --config --prepared --scores --calibration-scores --model-run --output`、`check-run --run`。score_scope=calibration/main_test/diagnostic/extended_test/test分别选择cal、主清单、诊断前后并集、完整可用test、主与诊断并集；默认测试执行test，重合正类只评分一次，所有用途清单记入manifest。evaluate同时生成主评价与诊断，不按模型效果临时启用。无默认train/全矩阵入口。

- [ ] 写失败测试 `test_cli_has_no_implicit_training`：帮助/check输入不创建模型、下载或读test标签选参数；各命令新目录发布，身份链写入manifest，prepare包能被fit消费。
- [ ] 写失败测试 `test_test_scoring_includes_diagnostics_once`：test评分ID等于main_test与diagnostic_pre/positive并集，正类重叠不重复评分；evaluate分开生成主/诊断结果，不能因主评价预筛丢失前参考；两用途均绑定冻结清单。
- [ ] 写失败测试 `test_offline_end_to_end`：tiny本地backbone+合成Case在临时目录完成prepare→fit→score/calibration→evaluate→check-run；重载分数一致、版本匹配、MLflow可查，确保编码器替换无需改训练器。
- [ ] 用CPU合成输入运行两个测试文件确认失败；实现CLI只编排前述接口，阶段之间明确输入artifact身份，不在evaluate中fit或下载模型。
- [ ] 运行全部新增test_service_model_*定向测试；再运行旧Service标签/准备/模型相关兼容测试，按实际影响选择，不跑全仓/DVC。
- [ ] 对本次改动运行既有black/isort检查及git diff --check；记录真正执行的命令/结果。测试超参和tiny模型只作工程验证，不能宣称真实BERT检测有效。
- [ ] 在未来实施文档说明版本、命令、未执行的真实实验与已知接口偏差；PR前history查重并新增entry。提交/PR遵循后续用户授权，不自动操作Git。

### Task 9：真实资产与 RTX 2060 预检（后续单独授权）

**Consumes:** Task 8 CLI，正式准备包、标签包和下载后固定revision模型；不改变原资产。

- [ ] 复核合入后AGENTS/history、Git状态及输入路径/hash；重新运行标签check或读取已绑定的同版本独立验收，记录采用哪一种。确认生产计数而非凑探查数字。
- [ ] 检查interface实际依赖、GPU型号/显存/PyTorch CUDA、transformers是否可用。缺依赖时仅新增兼容的transformers精确版本到项目依赖/环境文件；安装/下载按届时授权，不用make setup整体升级。
- [ ] 解析模型/tokenizer明确revision并保存文件hash，检查hidden256/4层/4头和所有文本参数可训练；没有权重时不以随机替身冒充预训练实验。
- [ ] 用新目录执行check-inputs和prepare，校验实际mask、清单、模板及日志覆盖；不读取Fault分数选择配置。合法标签已知数不等于模型有效数，未观察模态不得补全。
- [ ] 核对verified v2的479正类、2975主负类及其来源版本，确认9725 Fault pre仅在内部诊断清单；实际资源或执行共同集合另算，不能用resource_valid的2850行凑验收计数。按Case/Service报告无前参考诊断资格，尤其不从TRANSACTION缺失边界推导负类。
- [ ] 通过probe-fit命令仅以fit数据做小批量前向/反向与16窗梯度累积检查，检查BERT真实梯度、峰值显存和耗时；probe不发布正式模型。FP32作为首个尝试，OOM时只按已授权运行选项尝试FP16/梯度检查点并记录；仍失败则报告，不自动缩短日志、冻结BERT或换模型。
- [ ] 正式Fault评价前冻结实际运行配置、依赖、模型revision、清单及算法params，发布preflight报告。未过预检不启动Task 10。

Task 9的性能检查不宣称完成20轮训练或模型效果；若设备更换，仅记录运行变化，输入/模型变化须另立版本。

### Task 10：原型及三 seed 正式比较（后续单独授权）

**Consumes:** 已验收代码与Task 9冻结配置；此任务不是本轮实施的自动尾步骤。

- [ ] 先按seed42完成数值基线、三模态冻结文本基线、transformer_cls、transformer_mean及concat_svdd，保留失败与退化。原型可查看工程诊断，不据Fault成绩调参数/阈值。
- [ ] 执行正式矩阵：三种神经结构×{1,2,42}=9训练；IF数值/三模态各3 seed=6拟合；OCSVM数值/三模态各1次确定拟合；L2/Max各1次规则。三种神经结构都用同一SVDD目标，四分位阈值由同一cal分数一次计算，无需4倍重训。
- [ ] seed42原型若配置/输入/代码及模型revision与正式完全一致可显式复用，不将同一运行重复计为额外seed。失败seed不能用新seed替换。保存有效运行数及全部失败清单，不把缺失成员的均值当完整三seed结果。
- [ ] 记录共同主表、逐Fault召回、按服务/来源覆盖、完整覆盖扩展、四档阈值及Normal告警；按v2实际有效集合报告479正类的三种来源、10个Fault Case及5服务，不把窗口当独立重复。同步保存同Case/Service前后诊断；未发现混淆不证明无混淆，影响结论的证据须说明，不依结果删诊断。
- [ ] 固定分数重评只有在版本兼容核查通过后执行；不得依据测试结果挑标签包、模型结构、seed或工作点。
- [ ] 更新实施记录与后续问题清单：实测显存/耗时、坍缩、误差及输入覆盖；若需改进先形成独立配置/设计版本。Drain统计的去除消融、VAE、日志领域模型等扩展需后续计划，不在此矩阵自动展开。

## 建议命令与阶段契约

未来实施后的真实入口示例（当前文件尚未实现）：

```bash
conda run -n interface python scripts/service_model.py check-inputs --config configs/service/tt_model_v1.json --output outputs/service/model-preflight-<unique-id>
conda run -n interface python scripts/service_model.py prepare --config configs/service/tt_model_v1.json --output outputs/service/model-inputs-<unique-id>
conda run -n interface python scripts/service_model.py fit --config configs/service/tt_model_v1.json --prepared outputs/service/model-inputs-<unique-id> --output outputs/service/model-fit-<unique-id>
```

尖括号为需人工替换的唯一run后缀，不可直接作为shell命令运行。score/evaluate在各自新run目录写产物，calibration与test分数明确分开并由原清单校验；不通过移动文件或改目录名改变版本。

## Plan 自查与交接

Spec第1—2节→Task 1/2；第3节→Task 2/4；第4节→Task 3/4；第5节→Task 4/5；第6节→Task 5/9/10；第7节→Task 6；第8节→Task 4/6/10；第9节→Task 7；第10节→Task 8；第11节→Task 2—6；第12节→Task 1/8/9。

五类Review Focus分别在所指Task有具名断言；接口类型/score列/版本恢复已明确，真实输入preflight与测试/训练分阶段授权。所有检查框均未执行，测试、依赖安装、下载、模板fit、GPU探针及训练均未在本session运行。

请审阅本Plan；本session停留在文档设计。未来接续时再次核对最新master及打标接口，并由用户选择执行范围与方式，不自动启动代理、切分支或训练。
