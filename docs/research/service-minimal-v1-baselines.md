# Service 最小实验：统计基线

这页可用作中期汇报的“简单基线也是研究基础、非创新”页。

- CPU-only 与 R3 共享有效 R3 样本 ID，固定 25 个 Service、4 个 case 和 8 条评分臂。
- 每 Service 的均值、尺度及 OCSVM/IF 仅拟合 Normal 早段；独立 Normal calibration 校准，tail 仅诊断。
- L2、max、OCSVM、Isolation Forest 为现成基线，不构成新方法。CPU-only 的 L2 与 max 相同。
- 校准是保留并列的经验尾部排名量，不是异常概率或独立 conformal 保证。缺测和缺正常参考保留 Unknown/Untrained。
- S2 排名与受控事件评价由后续 T4 完成。本轮不根据 CPU/KILL 表现选择模型或参数。

```bash
conda run -n interface python scripts/service_experiment.py command=score \
  data_dir=artifacts/service_minimal_v1_20261014T100628Z/data \
  out=outputs/service/<new-run-id>/run
```

输入必须带 `_SUCCESS`，所有 prepare 输出先与 manifest SHA256 核对。
输出目录必须不存在；失败写 `_FAILURE`，不写 `_SUCCESS`。产物包含模型与参数、统计、训练/校准/Normal 参考 ID、逐样本原始及校准分数、Normal 分布摘要、来源和输出清单。运行自动通过已保存模型重新评分，核对数值及稳定键。
