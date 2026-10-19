# T4 实施交接报告

## 范围与交接

基于 HEAD f28ae96 和 task-4-brief，新增 src/service/evaluation.py、tests/test_service_evaluation.py，
仅在现有 Service CLI 加 evaluate 分支和参数化清单检查，更新限定研究文档。
AGENTS.md 用户文件保留 untracked。未修改 raw/data、冻结 data/run、Endpoint 管线、模型参数或分数。
未训练、未拟合、未扫描 raw。render_report 保留精确公共签名并明确 NotImplementedError，由 T6 完成。

## RED / GREEN

- 初始测试因 src.service.evaluation 缺失失败；新增 CLI 测试因 evaluate 未实现失败。
- 留出段测试发现 normal_fit 仍被纳入独立评估（10 != 0），修正为 fit/calibration 排除后通过。
- 最终 tests/test_service_evaluation.py 14 tests；连同 protocol/metrics/baselines 共 104 passed in 18.63s。
- 独立证据 fixtures 覆盖 WINDOW/RESOURCE 匹配、confirmed/heldout、来源重叠、实例 Killing、状态错配、未知标签、单类 null；CPU proxy 隔离。
- S2 fixtures 覆盖 8/10 可评分与最高2窗均值、7/10 Unknown；排名并列与未知竞争者；目标 Unknown Hit/RR0；3事件25候选8臂和固定完整35窗。
- 曾误启动 bare pytest，按 controller 要求中断，exit130/CondaError KeyboardInterrupt；没有可报告的完整 suite 结果。不将它称通过。限定104 tests随后完整通过。

## 实际 evaluate

命令：
`conda run -n interface python scripts/service_experiment.py command=evaluate data_dir=artifacts/service_minimal_v1_20261014T100628Z/data run_dir=artifacts/service_minimal_v1_20261014T100628Z/run out=artifacts/service_minimal_v1_20261014T100628Z/evaluation`

成功实际评估仅一次。此前两次在 reserve_output 之前拒绝：
1. run_manifest 是 manifest 自引用排除的 alias，不能要求它出现在 outputs；改为必须存在本地正规文件、JSON equality、完整 SHA 来源记录。
2. Parquet evidence 的 list 读回 ndarray；按 schema map(list)，再运行原 validate_evidence。
没有创建失败 evaluation 目录，也未覆盖任何已有结果。

CLI 对 prepare/run 所有 outputs 核对完整 SHA256/size；两份清单分别要求已知完整文件集合。
对 manifest 重复key、危险路径、symlink、错误hash/size使用既有严格 guard。
校验协议相同，全部 score 唯一键/8臂/schema和完整 sample集合；score/index主键与partition相同；
normal_reference_ids/index及normal_reference_scores/scores完全一致；固定3事件边界/目标来自冻结协议。
开始/结束核对全部输入哈希。evaluation_manifest 包含输出完整 SHA；source_manifest含输入、代码、配置SHA、环境、command和git状态。

## 结果与人工核验

600 行排名 = 3事件 ×25服务 ×8臂；24条 event_comparison 行；每条固定 n_expected35。
输入/output/执行code的SHA通过独立只读脚本核验（/tmp/t4_proof.py）。

|臂|CPU排名区间|全3事件Hit@3|全3事件MRR|
|---|---|---|---|
|CPU-only L2|[1,3]|1/3|.111111|
|CPU-only max|[1,3]|1/3|.111111|
|CPU-only OCSVM|[1,2]|1/3|.166667|
|CPU-only IF|[1,8]|0|.041667|
|R3 L2|[1,19]|0|.017544|
|R3 max|[1,18]|0|.018519|
|R3 OCSVM|[1,17]|0|.019608|
|R3 IF|[1,12]|0|.027778|

所有臂 Hit@1=0。两KILL目标coverage0、Unscorable、Hit/RR0，全部保留主表与宏分母3。
CPU目标coverage1；校准饱和并列不以raw分数或服务字母序改善指标。
S1独立 n_known0，AUROC/AP null；两条实例终止 independent_label1 没有变成资源Gold。
CPU proxy只有1事件、77对照+35暴露+15Unknown；CPU-only L2/max/OCSVM AUROC .988868 AP .963526，IF .987013/.961310；
R3 L2/max/OCSVM 1/1，IF .999629/.999206。此为受控时段关联弱对照，不是独立状态性能。
Normal tail原始/校准分位独立输出，未称FPR或寻找Fault最优阈值。

## 保留边界与后续

资源Unknown不代表健康；S2不是全异常集合或RCA。强CPU proxy与差S2并列检索并存，不自动证明复杂新模型必要。
独立S1当前无资源状态窗口Gold。报告图/PPT以及更细失败模式由 T6 接续，不在T4开发新机制。
本轮模型与原始数据只读。output中parquet按既有.gitignore排除，全部小文本与清单提交。

## 提交格式化记录

首次 commit 的 isort hook 修改 CLI 与测试的 import 排序，black通过，未改变计算行为。
实际执行CLI原文已保存 sibling evaluation_execution_sources/service_experiment.py.txt，完整SHA与
原evaluation/source_manifest.json一致；产物未改写。最终CLI格式化后的SHA与运行历史SHA有已记录差异。
