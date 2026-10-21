# TT 标签与主评价清单 v2 实施交接

记录日期：2026-10-20（America/Los_Angeles）；产物 UTC 时间为 2026-10-21。既有 `evaluation-scope-decision-20261021.md` 路径保留以免破坏引用。

## 已实现的协议

保持 6 个 Normal Case 的 4 拟合、1 独立校准、1 独立测试划分；所有 Fault Case 仅测试。Fault 注入前已有的参考负标签继续保留，主评价清单将其排除。Unknown 不填 0，也不作为第三个模型类别。

主评价由独立 Normal 测试 Case 的已准入负类与已知受控关联正类构成。评价范围在模型成绩产生前固定。数据的正类率不能解释为所有服务窗的真实异常率。

新增正类来源：

|来源|完整 15 秒服务窗|含义|
|---|---:|---|
|原直接应用目标规则|194|控制记录区间内的直接目标关联|
|NetworkChaos 参与对象|273|两个数据库名义事件中的应用 `.Target` 规则参与关联|
|KILL 双重就绪缺失观察|12|删除后首个 ready Pod 与 ready Endpoint 同时缺失的观察段|
|合计|479|受控关联标签，不能等同完整业务失败金标|

NetworkChaos 逐对象要求成功 Apply、Injected 状态、对象 spec 一致、唯一 runtime Pod UID→Service 映射，且 Apply 时间不晚于 effective_ms。仅取 effective_ms 至 removed_ms 内完整窗口。未取得逐对象 Remove 回执，结束使用全局控制撤销确认边界；它界定关联区间，不证明每个参与对象持续受到业务损害。7 个参与对象服务区间来自 2 个 Case，不能按 273 个窗口宣称独立重复。

KILL 使用 hash 绑定的 identity_events 和 identity_latest：事件前 30 秒内存在双 ready 观察；无已记录 observation gap、无 relationship_refresh_failed，最终关系观测 current 且无 errors。仅取首个双缺失 episode 内首末零观察之间的完整窗口；相邻零观察超过 30 秒则拒绝该 episode，未观察离开双缺失也拒绝。离开双缺失表示至少一种 ready 状态出现，不代表业务完全恢复。

30 秒预算依据采集器原始快照 `log_collector.py::refresh_relationships` 的既有 freshness budget，该文件已纳入来源 hash。EndpointSlice/关系来自刷新和离散快照，不是连续健康证明。开发期 15 秒敏感性规则得到 KILL 0、合计 467；正式 30 秒规则得到 KILL 12、合计 479。阈值依据采集约定，未使用模型成绩选择。

cachelimit 为数据库内存压力，未传播成任何应用正类。恢复期及非目标应用不默认正常。

## 产物与数量

正式包：`outputs/service/tt-labels-a-v2-20261020-verified/labels/`。

- 全网格 50,575 行：26,275 个参考负类、479 个正类、23,821 个 Unknown。
- 主评价清单：2,975 负类 + 479 正类 = 3,454 行，负∶正约 6.21∶1。
- 应用共同 `resource_valid` 掩码后：2,453 负类 + 397 正类 = 2,850 行，负∶正约 6.18∶1。
- Fault pre 参考负类 9,725 行排除在主评价外；未进入拟合或校准。
- 正类覆盖 10 个 Fault Case、5 个应用服务。仍应按 Case/来源报告，不能把窗口当独立实验。
- 其他模态和时序上下文质量掩码须在建模前声明，实际共同评价数量另报，不承诺都能保留 479。

主要文件：`labels.parquet`、`splits.parquet`、新增 `evaluation.parquet`、`evaluation_coverage.json`、`protocol.json`、`evidence.json`、`exceptions.json`、`coverage.json`、`manifest.json`。evaluation 表包含完整网格的稳定 sample_id、准入布尔值、排除原因与 evaluation_version。

配置：`configs/service/tt_labels_a_v2.json`。human-readable evaluation_version 为 `tt_normal_test_and_controlled_positive_v2`；精确来源/规则/标签内容由 manifest semantic hashes 和内容寻址 label_version 绑定。

原 A 版目录保持保留。开发中间产物 `tt-labels-a-v2-20261020` 和 `tt-labels-a-v2-20261020-final` 为未最终验收的阶段包，不作为模型输入。正式包仅使用 `-verified`。

## 模型侧接入

使用 `read_bundle` 读取正式包；从 preparation 表的预定输入质量规则获得 valid_ids，再调用新版 `join_evaluation_scores`。不要用全部 `label.notna()` 行构造主评价集。现有 `join_scores` 保留为通用旧接口；本协议主评价使用新版封装。

```python
from pathlib import Path
import pandas as pd
from src.service.labeling.bundle import read_bundle
from src.service.labeling.acceptance import join_evaluation_scores

data = read_bundle(Path("outputs/service/tt-labels-a-v2-20261020-verified"))
grid = pd.read_parquet(
    "outputs/service/preparation-exploratory-20261020-task4/tt/resource_windows.parquet"
)
valid_ids = set(grid.loc[grid.resource_valid, "sample_id"])
scores.attrs.update(data["labels"].attrs)
scores.attrs["evaluation_version"] = data["protocol"]["evaluation_version"]
evaluated = join_evaluation_scores(data, scores, valid_ids)
```

此处 scores 为模型实际输出，必须含 sample_id 与有限 score。缺少要求的评分会报错，不静默删样本。拟合和校准各自只使用指定 split 的已准入负类及预定输入质量掩码，不能由 evaluation 表倒推训练池。注入、phase、target、Case 名、控制器/probe 内容均不进入预测特征。

## 实现与复现检查

新增 `expansion.py`（证据区间）和 `evaluation.py`（固定主评价清单）；扩展 evidence/rules/bundle/acceptance，新增 v2 配置。schema 2 持久化评价清单，读取时语义重算，check 再从原始来源重放；旧 schema 1 包仍可读取。新增测试覆盖完整窗口、Unknown、准入拒绝、双缺失首 episode、间隔/闭环、评价范围、缺评分及清单语义篡改。

定向测试：`conda run -n interface python -m pytest tests/test_service_labeling_expansion.py tests/test_service_labeling_rules.py tests/test_service_labeling_bundle.py tests/test_service_labeling_acceptance.py tests/test_service_labeling_evidence.py tests/test_service_label_splits.py -q`，75 passed。

CLI 沿用 `scripts/label_service_data.py build/check/diff`；重复 build 到新目录即执行重放。正式包独立 check 已通过，50,575 行守恒且评价语义重算一致。重放包 `outputs/service/tt-labels-a-v2-20261020-replay/` 与正式包全部九项业务文件 hash 相同，内容 label_version 同为 `label:f20e82b68c5a50cb89f408a4a7f84a87edaf52a25253884700625039e41f8cde`。

A→v2 差异包 `outputs/service/tt-labels-a-v1-to-v2-20261020-diff/` 显示 14,875 行元数据或标签发生变化，多数因为补充来源引用；实际状态仅 285 行 Unknown→1，既有负类和 split 均保持。主评价额外排除 9,725 个 Fault pre 负类。通用旧 diff 比较标签/split，不比较新 evaluation 表；本次评价验收由独立重算语义与覆盖量完成。

正式包→重放包的稳定键 diff 为 0，结果位于 `outputs/service/tt-labels-a-v2-20261020-replay-diff/`。

历史 A 版文件仍由 read_bundle 核验通过，未被覆盖；用当前修改后的源码对旧包做 check 会因历史 rules hash 不同被拒绝，不能因此改写旧 manifest。

未来补采必须提供显式选择清单、迁移来源和 preparation 验收，生成新的数据/标签/评价版本。当前基础设施不会自动 glob 接入新 Case，Normal 划分仍显式固定。此次未训练模型、未补采、未改写 raw，未提交或推送。
