# 下一轮模型 agent 启动说明

先读本目录README、current-research-goal、dataset-and-protocol、baseline-findings，再检查当前AGENTS、Git状态及数据 `_handoff/migration_summary.json`；确认status passed和selected17身份/hash。研究主任务是Service×固定窗正常/异常二分类；继续Normal-only训练，成熟模型适度改进即可，不把旧No-Go当开发禁令。

数据：`data/tt_service_20261017_004546/`，只加载 `batch_control/final_acceptance/selected_cases.json`，不glob、不混new_merge/failed27。原绝对路径与补充源用 `_handoff/path_map.json` 解析，保持源hash不变；完整原始研究备份位置见archive-and-references。

下一轮经用户授权后依次做：

1. 核对并实现新批adapter/特征契约，检查run/plan/item/attempt、源与特征hash、25候选、实例/Counter/质量语义及稳定键。复用 `src/service/{protocol,metrics,baselines,evaluation}.py` 前先读接口；`configs/service/minimal_v1.yaml` 的旧4 Case/单Normal划分不能只换root复用。
2. 冻结标签与评价表：label、label_source、证据、范围及Unknown排除。注入目标代理、同源弱标签和独立窗口分开；真实Service金标不足时报告有限命题，不以目标检索取代二分类。
3. 以同输入、同有效ID、同整Case划分跑CPU-z/max/L2/IF/OCSVM及执行计数/率强基线；保存参数、seed、train/cal/test ID、逐样本分数与失败。不得按故障效果选阈值或把Normal报警直接称真实误报。
4. 做一个轻量时序模型，再验证一项有动机的改进与消融；额外模态输入改善与机制改善分别评估。上下文不能跨Case或划分边界，phase/target/注入时间等禁入特征。

本收尾没有实现adapter、标签表、新模型或重跑上述基线；已完成的分析原件仅供复现与对照。监督协议变更、未知标签审核、新补采均须另行对齐；当前没有集群、采集、DB、依赖安装或DVC重建授权。本仓库旧四 Case 基础设施已通过 PR #33 合入 master，新批交接位于 feature/service-selected17（history 035）。接入新批 S1 时核对 evaluate_s1 的 state_class 分组，避免正负类被拆成单类指标。AnoMod PR 在原交接时另有只读review进行中，本次未核查其最新状态；不把本仓库合入当作该PR已通过/已合并。
