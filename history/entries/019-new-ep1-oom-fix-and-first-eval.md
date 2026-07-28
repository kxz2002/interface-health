# 019 · new_ep1 OOM 修复 + 首次训练评估（DWF vs L0，PATCH 类型分数反向诊断）

- **日期**: 2026-07-27
- **PR**: 待开 PR（branch: `feature/deviation-weighted-fusion`）
- **类型**: Bugfix + Experiment
- **影响域**: `src/preprocessors/log_preprocessor.py`, `tests/test_log_preprocessor.py`, `configs/contract/v1_new_ep1.yaml`, `configs/data/new_ep1.yaml`, `artifacts/contract_new_ep1_expanded/`, `artifacts/baseline_new_ep1_concat/`, `artifacts/baseline_new_ep1_deviation_weighted/`

## 做了什么

用 new_ep1（独立数据集，`configs/data/new_ep1.yaml`，`roots: [data/new_ep1]`，不与 anomod_v1/endpoint_raw2 合并——entry 017 已证实跨 run 合并会污染评估）重跑整条 contract 构建 + 训练评估流程，过程中定位并修复了一次系统级 OOM kill，随后完成了 new_ep1 上的首次 L0（concat）vs DeviationWeightedFusion 训练评估对比，并诊断出 PATCH 类型故障的分数反向异常。

1. **OOM 根因修复**：`build_contract.py` 在处理 new_ep1（单个日志文件可达 4.8GB）时被内核 OOM killer 杀掉（`journalctl -k` 确认 `anon-rss:11853148kB`）。根因是 `LogPreprocessor.fit()`/`transform()` 用 `Path(path).read_text().splitlines()` 整文件读入内存（一份字符串+一份行列表，峰值翻倍到接近文件大小 2 倍）。同一次改动里还新增了 `MAX_CONTENT_CHARS=2000`（截断超长行喂给 Drain3，避免 tokenize CPU 复杂度随字符数暴涨），这两处是同一个 commit（`79712c9`）一起做的，不是本条目最初表述的"之前已修复"——`git log -S "MAX_CONTENT_CHARS"` 确认该常量在此之前的仓库历史里不存在，上一条"两次连续同源 OOM"的说法不准确，实际是同一次会话里内存问题和 CPU 问题一起被发现、一起修的，没有分两轮。改成逐行流式读取（`with path.open() as f: for line in f`）后重跑无 OOM，`fit`/`transform` 两处改动均通过既有测试（10/10）；但 `MAX_CONTENT_CHARS` 截断本身**不是**行为不变的改动——它是本分支上首次引入的新逻辑，对 `service_log__template_diversity` 特征有实测可量化的影响（截断阈值-耗时/损耗曲线见 history/entries/020），且该常量在 `master` 分支上不存在，v0/v1/v1_expanded 等既有基线的 contract 构建管线均依赖 `src/preprocessors`，本次改动会让它们相对已提交的 `metrics.json` 产生 drift。
2. **`fault_baseline_train_fraction` 推到数值上限**：new_ep1 首次实测 eval_all 正负比 8.98:91.02，与 1:1 目标偏差很大。分解 eval_all 负样本构成后确认：inject 窗口内非目标 endpoint 的 fan-out 残留（2546 行）+ recover（446 行）+ Normal holdout（176 行）三项固定不受 `fraction` 影响，只有 baseline 阶段行（受 fraction 调节）。即使 `fraction=1.0`（baseline 全部吸收进训练池），负样本下限仍是 3168，正样本 442，比例封顶 12.24:87.76——这是 endpoint 级 fan-out 的结构性稀释，`fraction` 单参数无法突破。已确认重跑后精确命中该数字。
3. **new_ep1 contract 上首次训练评估**：`fusion=concat`（L0）与 `fusion=deviation_weighted`（DWF）各跑一次（`contract_dir=artifacts/contract_new_ep1_expanded`，seed=42，epochs=50，`model=deep_svdd`），产出 `artifacts/baseline_new_ep1_concat/` 与 `artifacts/baseline_new_ep1_deviation_weighted/` 下的 scores/metrics。本次会话跑完时未纳入 DVC 追踪，2026-07-29 review 收尾时补建 `dvc_new_ep1/dvc.yaml`（同 `dvc_reliability_gate/`、`dvc_deviation_weighted/` 的隔离模式）并 `dvc commit` 补录，产物数值本身不变。

## 关键决策（不在 commit 里）

- **接受 12.24:87.76 而不改 eval_all 构成定义**：突破这个上限需要改变"是否把同窗非目标 endpoint 行计入负样本"这类 eval 集合定义本身的问题，超出单参数调节范围，且会影响与其他数据集/其他实验的可比性。本轮决定先接受这个上限，不做定义层改动。
- **流式读取而非增加内存限制/分批处理**：`LogPreprocessor` 逐行处理本身就是流式友好的（Drain3 的 `add_log_message`/`match` 天然逐条调用），唯一的内存放大点是文件读取方式，改成文件对象逐行迭代是最小改动、行为完全不变（`_parse_line()` 的 `.strip()` 已经处理逐行迭代带来的换行符），不需要引入分块读取或额外的内存监控逻辑。
- **PATCH 类型 fraction=1.0 副作用暂不回调**：诊断确认 `Lv_E_HTTPPATCH_travel2` 的 AUROC 反转（0.10~0.21，L0/DWF 两个方法都复现）根因是 `fraction=1.0` 把该 case 自己的 101 行 baseline 全部吸收进训练池——PATCH 这类"只改响应内容不改流量"的故障，其 inject 阶段流量特征与被吸收的自身 baseline 高度重合，模型把 inject 样本误判为"像训练时见过的正常样本"，反而 recover 阶段因残留 error_rate 波动打分更高，导致 AUROC 翻转。这本质是"某故障类型的 inject 特征恰好在 fraction=1.0 时与被吸收的自身 baseline 不可分"，是全局 fraction 参数在某些故障类型上的已知副作用，不是新 bug；本轮决定不回调 fraction（不引入 case-aware/per-anomaly-type 的 fraction 分级机制），因为这会重新打开"改 eval 集合构成定义"的口子，与决策 1 冲突,先如实记录副作用，留给下一轮设计取舍。

## 坑 / 已知问题

- **同一根因（`LogPreprocessor` 处理超大日志文件）在这次会话里同时暴露了两种表现不同的故障**：内存侧的整文件读入（OOM killer 杀掉进程）和 CPU 侧的单行过长 tokenize 复杂度（`MAX_CONTENT_CHARS` 截断），修复时误以为后者是"之前已经修过"的旧问题，实际两者是同一个 commit 里一起首次引入的（见上方决策 1 的更正）。经验：遇到疑似进程被杀，第一步应查 `journalctl -k`/`/var/log/syslog` 而不是假设是 agent 层面的问题；同一时间线里连续遇到的两个"相似但机制不同"的故障，写 history 时要用 `git log -S` 之类的手段核实"之前修过"这个前提是否成立，不要凭对话记忆下结论。
- **`MAX_CONTENT_CHARS` 截断不是行为不变，对既有基线构成 drift 风险**：2026-07-29 review 中发现，该常量只在本分支存在，`master` 上没有；`dvc status` 确认 v0/v1/v1_expanded（RG）三条 contract 构建管线因 `src/preprocessors` 变化处于"deps 已过期"状态——已提交的 `artifacts/baseline_v0|v1|v1_reliability_gate/metrics.json` 是旧代码（无截断）跑出来的数字，合并本分支后若重新 `dvc repro`，这些数字会漂移，且截断对 `template_diversity` 特征的影响幅度已实测非零（entry 020）。真正原因不是逻辑本身有 bug，是两个数据集里超长行的量级差了近一个数量级：`endpoint_raw2` 最大单文件 472M/9998 行，Drain3 无截断处理仅需 2.46s；`new_ep1` 最大单文件 4.5G/54057 行，同样处理需 26.39s——`endpoint_raw2` 的规模从未把 Drain3 耗时推过可观察阈值，所以截断问题在它上面从未显现，不代表旧逻辑对它是"正确"的，只是从未被触发。
- **PATCH 类型的分数反转是 L0 和 DWF 共同复现的，不是 DWF 特有缺陷**：诊断时先怀疑是 DWF 逐特征加权的问题，但对照 L0 的 `Lv_E_HTTPPATCH_travel2` AUROC（0.104）同样低于随机水平后确认根因在数据/切分层面（fraction=1.0 导致的训练池污染），不在融合机制本身。
- **DWF vs L0 在 new_ep1 上的对比不构成一致性结论**：`by_anomaly_type` 宏观看，ABORT 明显退步（0.876→0.736，主要是 `travel` case 从 0.849 掉到 0.277 拖累）、REPLACE 明显进步（0.845→0.916），DELAY/PATCH 基本持平。涨跌互抵、无一致方向，与 entry 018 在旧 `contract_v1_expanded` 上得出的"不达标"结论方向一致，但具体数字**不可跨数据集直接比较**——new_ep1 是单一 run（无跨 run 污染问题）、eval_all 比例是 12.24:87.76（旧数据集是别的比例），两者的"不达标"判定各自独立成立，不能相加或平均。

## 遗留 TODO

- `fault_baseline_train_fraction` 对不同故障类型（尤其"只改内容不改流量"的 PATCH）的副作用尚未系统评估——本次只诊断了一个具体 case（`Lv_E_HTTPPATCH_travel2`），未检查 PATCH 家族其余 case 或其他故障类型是否有类似的训练池污染模式。
- 未评估 case-aware 或 per-anomaly-type 的 `fraction` 分级机制是否值得做（能否在不改 eval_all 定义的前提下缓解 PATCH 类问题）——决策 3 只是"本轮不做"，不是"确认不需要"。
- new_ep1 缺 `assurance` 故障家族和 `PATCH_order` 组合（较早已知的数据集缺口，本次未新增排查）。
- DWF 在 new_ep1 上是否值得继续深化未决——按 entry 018 的先例（不达标不做后续深化），本次结果同样指向"不做"，但用户尚未就 new_ep1 这轮结果单独确认，留待下一步讨论。
