# 038 · Service 模型基础设施与远端首轮验收

- 日期：2026-10-22
- PR：#36；初始实现提交：d0f04eb、bf141ce
- 类型：Feature / Bugfix / Docs / Test
- 影响域：src/service/modeling/、scripts/service_model.py、configs/service/、tests/test_service_model_*、AGENTS.md

## 做了什么

增加独立 Service 模型 fit/score/evaluate 与可校验产物链，完成远端真实预检及首轮模型、数值基线运行。结果与运行身份见研究文档，不在 history 复制指标表。

## 关键决策

- 延续 035/037 的 Normal-only、整 Case 隔离及独立 Normal 校准。Fault pre 仅作内部诊断，Unknown 排除。
- 原始数据和大实验产物保留在计算服务器或研究备份，源码、配置、测试和小型结果摘要进入 Git。
- SSH 使用用户配置的 gpu-b 别名；同步保留目录层级，不读取凭证、不覆盖已有运行目录。

## 坑 / 已知问题

- Parquet 会把 list object 单元格恢复为 ndarray，pandas 原生哈希无法直接处理。嵌套数组也必须按完整内容序列化；default=str 会截断长数组，使不同内容的复用证据相同。审查补充内容变化和 Parquet 往返回归。
- 训练完成与评分/评价成功是独立状态；矩阵首次评分失败不表示后续基线已运行。仅依据各阶段成功 manifest 报告结果。

## 遗留 TODO

- 完整 Task 10 矩阵与重复 seed 尚未完成，不能把首轮结果表述为模型最终效果。
- 后续模型诊断与新实验须使用新 run_id，保留已有 checkpoint、评分和版本绑定。

## PR CI 修复

- 首轮 CI 为 806 passed、5 skipped，唯一失败为旧 self-referential fusion 单次计时加速比未达 20x；两个运行分别为18.7x和18.8x。数值一致性通过。改用 CPU profiler 核对 batch 从16增至16384时张量算子调用次数不变，直接检查逐样本循环退化；保留数值和shape断言，避免把机器速度比例当正确性标准。
