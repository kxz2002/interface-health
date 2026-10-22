# Task 9：gpu-b 真实预检

2026-10-22，用户授权远端运行与 subagent-driven 推进。Task 9 尚在执行，未发布正式训练效果。

## 已核验环境与同步

- 远端仓库：`~/Code/Repos/interface-health`，分支 `feature/service-model-infrastructure`；未提交代码按文件同步，不自动提交或推送。
- RTX 4090 D，24564 MiB，驱动 595.84；Python 3.12.15，torch 2.7.1+cu126，transformers 4.57.1。
- pandas 2.2.3、numpy 2.1.2、pyarrow 17.0.0、sklearn 1.5.2、MLflow 3.13.0；Drain3 可导入。
- CUDA seeded 小张量前反向通过。BERT/tokenizer 按固定 revision `387825ce42dbb39b87911cdf8e383ee3b25184f8` 离线加载，hidden256/4层/4头，11170560 参数均可训练，微型 GPU 推理输出有限。
- 运行覆盖配置：`configs/service/tt_model_gpu_b_v1.json`，CUDA FP32；输入预算、训练轮数和算法不变。
- 原始数据两端3442个文件，37683177106字节；rsync checksum dry-run 无差异。准备包与 verified v2 标签包随后单独同步，原始资产未重建。

## 身份适配与真实输入检查

首次真实 `check-inputs` 揭示合成测试与正式批次身份不一致：正式 `dataset_id=service_20261017_004546`，实现原先仅允许 `tt`。修复保留两种明确身份，拒绝未知、SN与混合身份，不改写 stable IDs 或原始包。

子代理完成修复并补测试，独立子代理静态审查通过。远端同步后核对五个变更文件的 SHA256 与本地一致；data/membership/numeric 34项测试通过（6.85秒）。

`outputs/service/model-task9-check-inputs-20261022-b` 成功，MLflow run `d9b4a0a8d93d429e967cdb8180b29e88`，发布并检查3项产物。输入绑定正式 `preparation-exploratory-20261020-task4` 与 `tt-labels-a-v2-20261020-verified/labels`。

## 真实准备与 GPU 探针结果

远端离线 `prepare` 成功，输出 `outputs/service/model-task9-prepared-20261022-a`，日志 `outputs/service/task9-logs/prepare-a.log`，16项产物检查通过。共同fit为10148窗、calibration为2853窗、main_test为3297窗（2852负/445正）；diagnostic_pre为9301窗，diagnostic_positive为445窗。主评价没有pre_intervention_reference。日志有效32010窗，extended_main_test为3309窗，排除2335窗。254个诊断Case/Service组中4组没有前参考，均为TRANSACTION的order/preserve/travel/user；不补造前参考。

探针报告补强由子代理实现、独立审查通过，远端8项training测试通过（14.02秒），同步文件SHA256一致。`outputs/service/model-task9-probe-20261022-a`成功，3项产物检查通过。16个排序fit窗中11窗有日志，实际16次反向，BERT有69个参数张量获得梯度，范数6.012055160105242且有限；总梯度范数19.42259407043457。探针耗时1.1483秒，峰值PyTorch allocated为265014784字节（约252.7 MiB），不是整卡占用，也不是全量训练最大显存保证。模型及既有梯度状态恢复成功，不保存正式checkpoint。

Task 9 的输入、准备和首批fit GPU检查已通过。冻结配置仍为CUDA FP32、16消息/128token、20轮最后checkpoint；未根据Fault分数选配置或阈值。下一步为Task 10 seed42原型。其他方法和全部fit窗的峰值、真实训练耗时及退化风险仍须实际运行报告，不把探针成功写成检测有效。

独立验收子代理在远端实际执行三个 `check-run`，全部成功，产物数分别为3/16/3；复核真实probe报告与本记录一致。

## Task 10 首轮真实结果

主模型seed42已完成：`outputs/service/model-task10-transformer-cls-seed42-20261022-a`，MLflow `8b727302edf64c49b883b2fce39b5842`，训练耗时约86分04秒。首次矩阵在评分兼容性检查阶段因 ndarray object 列哈希失败而停止；修复 `_frame_hash` 后未重训该 checkpoint，校准评分、测试评分和评估均成功完成。

首轮主测试（3297 known windows，2852 Normal/445 positive）结果为 AUROC 0.473863、AP 0.204927。评估覆盖预期12598个样本，缺失0；该结果仅代表单一模型和seed，不作为最终方法结论。

为检查数据可分性，使用同一 prepared 包、划分和校准协议运行两个经典基线：`numeric_IF` 的 AUROC/AP 为 0.660570/0.311528，`numeric_OCSVM` 为 0.563648/0.380567；两者覆盖均完整。完整结果保留在远端对应 `model-task10-baseline-20261022-{if,ocsvm}-evaluate` 目录，未将 checkpoint/parquet/MLflow 数据库纳入 Git。

准备包历史验收记录已复制至远端 `outputs/service/preparation-exploratory-20261020-task4-acceptance-final`。该历史准备验收的Unknown标签不覆盖当前verified v2标签。当前标签全网格50575行，26275负、479正、23821 Unknown；9725个Fault pre仍仅用于内部诊断。
