# 032 · new_merge 数据漂移处置：dvc commit 承认现状 + 11 stage lock 同步

- **日期**: 2026-10-08
- **PR**: #31 · **Commit**: 见本 PR merge
- **类型**: Data
- **影响域**: `data/new_merge.dvc`, `dvc_new_merge/dvc.lock`, `CLAUDE.md`, DVC 操作纪律

## 背景与处置决策

2026-08-24 起 `data/new_merge` 有 189 个 `_pipeline_out` 派生文件内容漂移未 commit——27 case × 每个 7 个：`tt_fused_{5,10,15}s.csv`、`tt_traces_red_{5,10,15}s_report.md`、`tt_traces_red_window_comparison.md`。成因是用户在**上游 AnoMod 侧**重跑融合管线并重写了这批产物（用户本人确认）。由于 build stage 的 dep 声明是整个 `data/new_merge` 目录，目录内任何文件漂移都让它失效，裸 `dvc repro` 因此会误触发 build 并删除现有 contract 产物——已三次产生恢复成本（entry 029、031 两次），期间只能靠 `--single-item` 人肉纪律硬扛。

两个候选方向：**`dvc commit` 承认现状** vs **从上游备份恢复旧版**。取前者，依据三条：

1. **论文锚定在当前状态**：用户确认 AnoMod 路径论文的统计数字基于漂移后（2026-08-24 重写后）状态计算，且该数据集后续要开源发布——当前状态才是"论文为真"的状态。恢复旧版反而会制造"开源数据集与论文统计不一致"的真实风险。
2. **漂移与 contract 输入无关**（哈希级实证，非推测）：旧 `.dir` manifest（`2f049a3d...`，2268 文件）在 cache 中，逐一与当前磁盘比对——文件集合零增删、内容不一致恰 189 个、全部命中预期派生文件名模式、覆盖恰 27 case；contract 输入文件（`tt_endpoint_health_15s.csv`、`tt_traces_red_15s.csv`、`case_metadata.json`、原始 logs/metrics/api_responses）**零漂移**。已有 v1/v2 全部结果的可复现性不受影响。
3. commit 同时把当前状态刷入 cache，等于给论文数据集增加一份完整本地备份，也为未来开源发布的 DVC remote push 打底。

## 执行与验证

1. **术前枚举**（一次性脚本，2268 文件全量 md5，基准快照落 `/tmp/new_merge_manifest_before_d4.tsv`）：结果如上②，VERDICT PASS 才继续。
2. `dvc commit -f data/new_merge.dvc`：`-f` 是 dvc 对"不重跑 stage 直接承认输出变化"的确认闸，此处正是本意。指针 .dir md5 `2f049a3d`→`9f2080cb`，size +3.3MB（189 文件重写的净增量），nfiles 2268 不变。
3. `dvc commit -f dvc_new_merge/dvc.yaml`：同步 11 个 v1 时代 stage 的 lock dep 哈希。除 `data/new_merge` 外，stale dep 还有 #28（`eval_baseline_v0.py`/`score_trivial_baselines.py` review 修复）与 #30（`build_contract.py`/`contract_v0.py`/`normalization.py`/`src/contracts`）的代码变更——逐类核实"当前产物 == 当前代码+数据重跑的产物"成立：build 的 v1 代码路径经 6B 字节级验证（14 文件 md5 一致）；eval/trivial 的后续脚本变更均为守卫/日志/docstring 类（新增 ValueError 只在真实 contract 不出现的缺失 case 上触发，metrics/scores 内容不受影响）。train stage 与 v2 stage 的 dep 本就连贯（train 未受 #28/#30 代码影响；v2 于 C2 以修复后代码重跑），不在过期清单内。
4. **三道验收门全过**：
   - `dvc status dvc_new_merge/dvc.yaml` → "Data and pipelines are up to date"
   - 裸 `dvc repro dvc_new_merge/dvc.yaml --dry` → 全 stage "didn't change, skipping"（零触发——「禁止裸 repro」禁令解除的操作性证明）
   - commit 前后 2268 文件 md5 快照**逐字节一致**——`dvc commit` 未改动数据集任何字节（论文数据集完整性的硬验证门，不一致即中止）

## 效果与遗留

- 「禁止裸 repro / 禁止 `dvc checkout`」状态解除。CLAUDE.md 先前后矛盾的记载一并修正：所谓"cache 为空"早已过时（实际已有 183G，contract 输入 blob 本就在 cache，本次仅增量刷入 189 个漂移文件），相关 gotcha 与 Commands 注释已按处置后状态重写。
- **结构教训（已写入 CLAUDE.md gotcha）**：build stage 用整目录做 dep，目录内任何非输入文件的漂移都会作废整条链。未来新数据集接入时，`_pipeline_out` 这类派生物应拆成独立 DVC 输出或排除在 dep 之外。
- v2 contract 的 dvc build stage 补注册不再受阻（entry 031 遗留 TODO），但**本 PR 不做**——那是独立的 pipeline 变更，需要自己的验证。
- 遗留 TODO：v2 build stage 注册；开源发布前的 DVC remote 配置与 push；`dvc.yaml` 根文件三条历史死链（merged_v2 等）的 status 噪音与本 PR 无关、维持现状。
