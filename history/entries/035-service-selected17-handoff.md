# 035 · 新17 Case数据交接与Service二分类研究目标更新

- 日期：2026-10-19
- 分支：feature/service-selected17；基于 master c087aa8（已合入 PR #33）；原交接内容保留自 feature/service-minimal-v1 的 05c57bb
- 类型：Data / Docs
- 影响域：AGENTS.md、CLAUDE.md、.gitignore、docs/research/service-20261019/、本地data快照

## 做了什么

将AnoMod已正式验收的selected17（6 Normal、11故障）复制为独立Git忽略的Service输入快照，逐文件校验source/destination SHA256；失败27的677份诊断留证另存研究备份。整理已完成分析、独立验证、参数/原始结果/源码/文献的逐文件档案，新增研究及下一轮入口。此次不实现adapter、不训练或重跑模型，不触碰旧new_merge、DVC或集群。

## 关键决策

- 当前主任务为Service×固定窗正常/异常二分类，维持Normal-only训练；Endpoint独立输出。成熟方法适度改进可通过公平实验/消融验证，不以前述033及旧报告的高新颖性No-Go作为开发禁令。033的原始结果保留，未改成当时通过。
- 采集验收与完整Service窗口真值分开。Unknown排除对应评价；注入时间/目标、非目标和recover不直接生成健康/异常全集标签。同源弱标签、受控目标检索与独立窗口分开报告。
- 正式数据以selected_cases.json allowlist加载，失败留证放独立目录；源数据和worktree不因完成复制获得删除许可。原件绝对路径/hash保持，path map记录迁移去向。
- 新批已执行分析使用前三Normal fit、第四cal、第五六heldout；旧minimal_v1四Case与单Normal 60/20/20保持历史范围，不能换root后套用。早期采集准入中的另一划分是未执行建议，不能与已执行结果混用。
- 原/tmp研究脚本及依赖保持原目录结构归档，非生产训练入口；PDF/Parquet/分数/大manifest不进普通源码Git。其他agent的/tmp/anomod-*仅登记引用。

## 坑 / 已知问题

原始补充CSV、source_manifest、_audit_implementation和源/配置/实现hash是必要链条，不能只复制features.parquet。必要输入绑定已通过；96条外部历史路径虽均存在，仍非完全可迁移闭包。全网格95%覆盖和全Service独立金标未证明；Normal报警负担不直接等于真实FPR，IF上限并列导致的失败不可事后改比较符救分。

## 遗留 TODO

1. 下一轮经用户授权实现新批adapter、冻结标签评价表、同输入ML基线、一个轻量时序模型及适度改进/消融；本轮没有完成这些开发步骤。
2. AnoMod PR另有只读review进行中；等待基准提交/dirty范围和findings后必要fix、复测，再由用户决定最终集成。当前不push/merge、不删除源/worktree。
3. 完整服务状态标签、travel偏移无害性、故障期业务能力及日志连续覆盖仍有限，不为理想结果换标签或隐藏困难事件。

复现与逐文件验证位置：docs/research/service-20261019/README.md，数据_handoff/migration_summary.json；完整本地档案见该目录archive-and-references.md。


## 2026-10-19 分支迁移

用户要求旧 PR 只提交原四 Case 工作，因此新批的 11 个交接文件此前刻意留在原工作区，切换 master 时触发 Git 覆盖保护。现先对这些小文件备份并限定路径 stash，再快进 master、建立 feature/service-selected17，恢复并解决 CLAUDE/index 冲突；保留 034 的旧数据退役事实，本条顺延为 035。只提交新批忽略规则及研究交接文档；原始数据不进 Git，不重跑研究、训练或 DVC。

原交接中 AnoMod PR 的 review 状态仅代表当时记录，本次没有核查或操作该仓库；本仓库 PR #33 的合入不意味着 AnoMod PR 已合入。后续接入新批时还需核对 evaluate_s1 的 state_class 分组是否将正负类拆开，见 033 的 PR-review 收尾记录。

迁移验证：11 个文件均已恢复；033/034 原记录与 master 字节一致；history 编号唯一，48 个相对文档链接有效；selected17 数据仍被精确忽略，生产代码、配置、测试与数据均无迁移差异。文档任务仅作结构和 Git 检查，未重跑模型测试或训练。
