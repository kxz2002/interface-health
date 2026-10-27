# 045 · POOL/DNS 评价范围版本化与历史模型复评

- **日期**: 2026-10-27
- **PR / Commit**: 本分支 `exp/pool-dns-eval-rerun`，关键结果提交 `80ce233`；PR 编号以 GitHub 为准。
- **类型**: Feature / Bugfix / Experiment / Test / Docs
- **影响域**: configs/service/、src/service/labeling/、src/service/modeling/、scripts/label_service_data.py、scripts/service_revaluate_temporal.py、tests/test_service_*、docs/research/service-20261019/、AGENTS.md

## 做了什么

建立排除两个 Case 的 v3 评价协议，并在已保存分数及独立 Normal 校准阈值上复评历史时序模型和保留的经典基线。结果、逐 seed 摘要和限制见 [v3 结果](../../docs/research/service-20261019/pool-dns-v3-results.md)。

## 关键决策

- 用户决定长期从评价体系排除 POOL/DNS；保留原始数据、标签与 v2，使用版本化协议固定 v3，而不是改写历史产物或临时过滤指标。
- 训练和校准样本不变，复用原模型评分及阈值；新主评价、诊断和 Unknown 输出均排除两个 Case。独立 Normal 校准与 Case 隔离仍然有效。
- 旧分数和标签包通过来源 manifest、样本身份与版本绑定；分数不全或校准阈值不一致时报错。经典基线使用各模型专属样本列表，不能套用公共列表 hash。
- L2、Max 为用户废弃的候选，不纳入后续矩阵。原型记忆在已见测试集的 v3 上表现较好，尚不能据此替换原先验 GRU 或宣称独立泛化。

## 坑 / 已知问题

- 历史标签证据的路径记录在本机 Windows；远端 Linux 不应直接改写路径或伪造验收。实际在本机重放验收并生成绑定包 hash 的回执，再核对并同步包和回执。
- 历史派生变体可能在自身 provenance 中缺失列表 hash；只对已知两类变体通过已校验的原 GRU 父 manifest 获取，不能无条件接受缺失绑定。
- 最初误启动过重新拟合；相关产物未纳入正式结果。远端主仓库存在既有 dirty 文件，实验和同步采用独立 Git worktree，不覆盖原仓库。
- 四批复评使用独立暂存发布，进程完成前的临时文件不是正式结果；历史数据、逐窗分数及模型保留远端，仓库仅保存小型结果摘要。

## 遗留 TODO

- 若要判断原型记忆是否值得替代原 GRU，需要独立新数据与预先固定的对照；当前测试集已有多轮探索，不能称盲测。
- 当前标签是受控关联证据，不是完整真实 Service 异常金标；逐 Case 覆盖和故障盲区仍须与总体结果一起报告。
