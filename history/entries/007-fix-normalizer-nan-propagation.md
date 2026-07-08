# 007 · 修复 Normalizer 全 NaN group 的 NaN 传染 bug

- **日期**: 2026-07-08
- **PR**: bugfix/fix-metric（待合并）
- **类型**: Bugfix
- **影响域**: `src/data/normalization.py`, `tests/test_normalization.py`

## 做了什么

修复 PR #6 dvc repro 中发现的"metric 模态全 NaN"问题。根因不是最初怀疑的时区/时间对齐 bug，而是 `Normalizer.fit()`/`transform()` 对"某个 group 在 fit 集合（仅 Normal case）里全 NaN"没有防护：`fit()` 算出 `[nan, nan]` 统计量后，`transform()` 把这组 NaN 通过减法/除法应用到**全量数据**（不只是 Normal），导致其他 27 个数据完好的 case（如 `Lv_D_cachelimit`）的 `service_metric__*` 特征也被一起抹成 NaN。

修复：`transform()` 遇到某 group（或 `global` scope）的 `lo`/`hi` 为 NaN 时，跳过该 group 的归一化运算，保留原值不动，不让 NaN 扩散到其他 case。

## 关键决策（不在 commit 里）

- **跳过归一化保留原值，而非统一置 NaN**：另一个候选方案是该 group 全 NaN 时统一把 transform 结果置为 NaN（交给下游 `ContractDataset` 的"全 NaN 列填 0"兜底），量纲更统一但会把其他 case 本来完好的真实数值也一起丢弃。选择保留原值是因为验证时确认 `(value - nan) / max(nan, 1e-9)` 天然就是 NaN——不加判断的话代码看起来"什么都没做"实际上等价于污染，必须显式 `continue` 才能真正保留原值。
- **不处理 Normal 自身的数据缺口**：Normal 的 `service_metric__*` 全 NaN 是真实的数据采集缺口（cAdvisor 在 Normal 实验窗口开始前约 7 小时就已掉线，一直没恢复），代码层面无法修复。用户确认自行安排重采数据，本次只堵代码层的传染路径。

## 坑 / 已知问题

- **"跳过归一化"与后续 `RATE_COLUMNS` clip 的语义耦合，本次是巧合验证通过**：`scripts/build_contract.py` 在 normalize 之后对 `RATE_COLUMNS`（含 `cpu_usage_rate`/`memory_usage_ratio`/`net_rx/tx_error_rate`）做 `clip(0, 1)`，clip 的原意是"超过 Normal 学到的正常上界即视为完全异常"。跳过归一化后保留的是**原始量纲**，这次因为 `cpu_usage_rate`（实测 max≈0.69）、`memory_usage_ratio`（max≈0.63）天然落在 `[0,1]` 内，contract 校验碰巧通过。但这不是设计保证——如果未来某个 service 的原始读数超出 `[0,1]`（如 CPU usage 换算错误产生 >1 的值），"跳过归一化 + clip(0,1)"组合会把原始量纲的正常值误判成"完全异常"，语义上不成立。下次涉及全 NaN group 的 `RATE_COLUMNS` 列时需重新审视。
- **`normalization_stats.json` 里全 NaN group 仍保留 `[nan, nan]`**：这是预期行为（fit 阶段确实拿不到统计量，如实记录），`transform()` 端做跳过判断即可，不需要在 `fit()`/`save()` 端隐藏或过滤掉这些 NaN 条目。

## 遗留 TODO

- Normal 数据重采：用户自行处理，重采后 `Normal` 的 `service_metric__*` 才能真正参与训练；重采前 Deep SVDD 学不到这几维在正常情况下的分布，训练侧这几维特征等同缺失。
- 未验证 `cpu_usage_rate` 等列是否存在原始量纲超出 `[0,1]` 的边界场景（上面"坑"里提到的耦合风险），当前只是巧合验证通过，未做针对性回归测试。
