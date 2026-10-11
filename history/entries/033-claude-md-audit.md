# 033 · CLAUDE.md 审计更新：历史管线降级为警告、补 027→031 研究结论与 AnoMod 依赖

- **日期**: 2026-10-11
- **PR**: 待开（`feature/claude-md-audit`） · **Commit**: 见本 PR merge
- **类型**: Docs
- **影响域**: `CLAUDE.md`, `history/`, DVC 操作纪律, 数据采集协议

## 做了什么

PR #31 合并后，用 `claude-md-improver` 流程审计根 CLAUDE.md，按用户确认的 9 组改动落地：
- Commands 里 v0/v1/RG/DWF/new_ep1 五个历史管线块（约 64 行）压缩为一段"不可重建、勿运行"警告。
- 研究背景新增"当前研究状态"，汇总 027→031 的平凡基线结论与 P1/P2。
- 核心任务定义改为现行口径：expand 训练池、三档正样本、per-case 宏平均主指标。
- 新增 AnoMod 外部依赖说明和 `trace_*` 错误列 gotcha。
- 修正 v2 build stage、`MAX_CONTENT_CHARS`、Hydra 示例、MLflow、ARIS 路径、目录树等过时表述。

篇幅 334→291 行，字符数 28764→29238，基本持平：删掉的历史命令被新增内容抵消了。新写入的事实都在 2026-10-11 按原始数据实测核对过，见下文。

## 关键决策（不在 commit 里）

- **历史命令压缩成警告，没有保留原命令加标注**：这些命令依赖的数据已永久丢失（entry 025），留着只会诱导运行。原文在 git 历史和 entry 011/014/015/018/019 里都还能找到。用户在两个选项里选了压缩。
- **"裸 repro 安全"改为带范围的表述**：entry 032 的三道验收门都只针对 `dvc_new_merge/dvc.yaml`，写进 CLAUDE.md 时范围丢了。实测在根目录不带 target 跑 `dvc repro --dry`：
  - 直接报错（`artifacts/contract_v1/train.parquet` 不存在）；
  - 计划执行 build_contract/train_v0/build_contract_v1，而这几个 stage 的输出 `artifacts/contract_v0/*`、`baseline_v0/scores.parquet` 不在 cache；
  - DVC 重建前会先删输出，这些文件删了无法恢复。
- **研究结论写进"研究背景"，不只留在 gotcha 和横切主题里**：027→031 的结论决定了"两个论文创新方向"还能不能成立，包括平凡基线打败全部学习模型、根因在采集模板、P1 重采是唯一根治。这些此前散落在 Known Gotchas 和 history 横切主题里，新 session 只读研究背景会得出过时的判断。
  - 数字取自 `artifacts/contract_v2_acceptance/main_table.md`（2026-10-07 canonical）。
  - 同时注明 entry 028 的下游 benchmark 用的是另一套特征和切分，两边数字不要混用。
- **AnoMod 那段写"不能当备份"，不写"孪生目录"**：调研简报说 `AnoMod/processed_data/2026_0729` 是 new_merge 的原始采集孪生目录。逐文件比对的结果：
  - 原始文件（不含 `_pipeline_out`，共 1836 个）名称和大小一致，27 个 `case_metadata.json` 的 md5 一致；
  - `_pipeline_out` 每个 case 有 16 类文件，其中 10 类在 27 个 case 上全部不同，包括 contract 的输入 `tt_endpoint_health_*`；
  - 抽样 mtime：AnoMod 侧是 07-29；new_merge 侧是 08-22（与 entry 025 同日）和 08-24（漂移重写日）。

  只写"孪生"，后来的人可能会拿它恢复数据或核对论文数字。
- **`trace_*` 错误列的表述比 AnoMod PAPER_SPEC §6.4 收窄**：§6.4 写的是"中位数为 0，两列都不可用"。new_merge `tt_fused_15s.csv` 上的实测：
  - `trace_status_coverage>0` 有 2578/23868 行（10.8%）；
  - 这些行上 `trace_error_rate` 恰好等于 coverage，其余行都是 0；
  - `trace_5xx_rate` 只在这些行非空，而且恒为 1.0。

  也就是说，带状态码的 span 看起来全是 5xx。`trace_5xx_rate` 确实是退化列；`trace_error_rate` 实际是"5xx span 占比"，和 `client_error_rate>0` 几乎同时出现（2576/2578），不能一概判为不可用。论文要不要回改由用户决定，本次没有动 AnoMod 仓库。
- **KILLPOD gotcha 是补判据，不是推翻**：AnoMod §6.3 更正的是 order/preserve：inject 阶段 trace 完好，trace_request_count 中位数从 7 涨到 62/18，属于重试风暴。本次复核 gateway 的缺陷成立：AnoMod 拷贝里该 case 的 561 行全是 baseline，inject 阶段 0 行。所以判据是"被杀的服务是否承担上报职责"，与故障类型无关。
- **这次不压缩 Known Gotchas 的长段落**：这些段落是证据链，压缩容易丢细节，适合单独做一轮，见遗留 TODO。

## 坑 / 已知问题

- **在根目录不带 target 跑 `dvc repro` 有破坏性**：
  - 表现：执行的是根 dvc.yaml 里 v0/v1 的死链，先删输出，再因缺数据失败。
  - 规避：只跑 `dvc repro dvc_new_merge/dvc.yaml`，或用 `--single-item <file>:<stage>` 跑单个 stage；验收只看 `dvc status dvc_new_merge/dvc.yaml`。全项目 `dvc status` 里的 changed / not in cache 是死链加 `data/{normal_v2,new_ep1}.dvc` 带来的预期噪音。
- **v2 build stage 没注册，依赖图断了一截**：7 个 v2 train/score stage 依赖 `artifacts/contract_new_merge_v2/*`，但没有任何 stage 产出它，DVC 把它当外部输入。所以改了 `build_contract.py`、归一化代码或 v2 config 之后，v2 stage 不会失效，必须先手动重建 contract。
- **手动跑的 seed{1,2,3} `scores.parquet` 只存在本地磁盘**：既不在 git 里，也不在 dvc cache 里。entry 030/031 的对照脚本要读它们现算指标，删了只能重跑。
- **转述会丢限定词**：本轮调研摘要里有 3 处与实测不符，都是转述时丢了限定词：
  - "孪生目录"：实际只有原始文件一致，派生文件不同；
  - "`trace_status_coverage`=0 覆盖全数据集"：PAPER_SPEC 原文是"各 case 中位数为 0"；
  - "裸 repro 零触发"：漏了"仅限 `dvc_new_merge/dvc.yaml`"。

  三处都是写进 CLAUDE.md 前逐项复核时发现的。"全绿 / 为 0 / 一致"这类结论，落笔前要按原始数据重算。
- **`configs/data/ep2_0729.yaml` 指向的 root `/tmp/ep2_0729_nolog` 已不存在**：entry 024 的产物 `artifacts/contract_ep2_0729*` 还在，但这个 config 已经无法再跑。

## 遗留 TODO

- **CLAUDE.md 进一步瘦身**：文件还有 291 行、约 2.9 万字符，其中 Known Gotchas 一节约占 43%（25 条，最长一条 1.6k 字符）。可行方向：
  - 长 gotcha 压成"规则 + entry 指针"；
  - 按目录拆到子目录 CLAUDE.md 或 `.claude/rules/`。注意 `.claude/` 目前被 `.gitignore` 排除。
- entry 031/032 的遗留项不变：注册 v2 contract build stage；开源发布前配置 DVC remote。
- AnoMod PAPER_SPEC §6.4 对 `trace_error_rate` "不可用"的说法要不要回改，由用户决定。
- `src/utils/logger.py` 的 `ExperimentLogger`（MLflow 封装）除了 `src/utils/__init__.py` 的导出，没有任何调用方；接入还是删除待定。
- `ep2_0729.yaml` 的失效 root 要不要清理或注明（低优先级）。
