# 034 · anomod/anomod_v1 数据集退役与根 v0/v1 死链下葬

- **日期**: 2026-10-15
- **PR**: 待创建（feature/retire-anomod-datasets） · **Commit**: 22fecb9 / 5931d3b / 30da4b2
- **类型**: Data + Chore
- **影响域**: `data/`, `configs/data/`, `configs/base.yaml`, `dvc.yaml`, `dvc.lock`, `dvc_new_merge/dvc.lock`, `tests/`, `.gitignore`, `history/`

## 做了什么

物理删除 `data/anomod/`（44G，项目最早期单批次数据，无任何 config 引用）与 `data/anomod_v1/`（41G，service 级故障注入，仅剩死链引用），磁盘净释放 ~85G；`dvc gc -w` 回收 cache 38733 objects（183G→124G，仅剩 new_merge 全集）。配套删除 3 个三个 root 全灭的 dataset config（`anomod.yaml`/`merged_v1.yaml`/`merged_v2.yaml`）、`configs/base.yaml` 里残留的 `- data: anomod` Hydra default、`data/anomod.dvc` 指针、`test_metric_perf_spike.py`。根 `dvc.yaml` 的 6 个 v0/v1 死 stage（依赖 merged_v2，自 entry 025 起不可重建）正式下葬，只留自包含 toy train/eval；`dvc.lock` 手动同步并清掉 entry 015 遗留的两条 stale RG lock 条目。base.yaml 改动弄脏 8 个 `train_new_merge_*` stage 的 dep 哈希，按 entry 032 先例 `dvc commit` 承认现状（零执行）。

## 关键决策（不在 commit 里）

- **anomod_v1 接受永久丢失，不先补 DVC 追踪再删**：anomod 有 .dvc 指针而 anomod_v1 没有，表面看两者可恢复性不同。但本机无有效 remote（`.dvc/config` 指向的 `~/dvc-remote/interface-health` 不可用），补追踪=把 41G 复制进本机 cache，"可恢复的删除"与"留着不删"在磁盘成本上等价。实质判断是内容价值：entry 021 已判定这批数据待重采（PATCH 类故障对现有特征全隐形）才有研究价值、entry 025 已作废其全部派生数字，且其 Normal 有 cadvisor 断流缺陷（metric 模态窗口内 0 覆盖，当年重采 normal_v2 的原因）。用户明确确认接受永久丢失。
- **删 merged_v2.yaml 倒逼根 v0/v1 stage 下葬，而不是保留 config 让守卫测试转绿**：`test_dvc_pipeline_v0.py::test_build_contract_uses_dataset_config` 断言 `--dataset` 指向真实存在的 config，删除后该测试正确变红。备选方案"保留 merged_v2.yaml 维持测试绿"被否——为一个三个 root 全灭的死链保留 config 仅为满足活守卫，本末倒置。沿用 entry 015/031 先例：stage 从执行路径移除，类代码/契约 config 留树备查。根 `dvc.yaml` 保留 toy train/eval 是因为它们自包含（deps 仅脚本），与死数据无关。
- **根 dvc.lock 手动修剪而非 `dvc remove`**：两者效果相同，手动编辑透明可控、零执行副作用；顺带清掉 `train_v1_reliability_gate`/`eval_v1_reliability_gate` 两条 stale 条目——entry 015 把这两个 stage 从根 dvc.yaml 移去 dvc_reliability_gate/ 时只改了 yaml 没动 lock，属于历史欠账。
- **`test_dvc_pipeline_v1.py` 裁剪而非整删**：5 条断言里 3 条仍有效——RG 扩容链路的 `--config` 绑定（`dvc_reliability_gate/dvc.yaml` 按 entry 031 惯例保留备查）与 v1/v1_expanded modalities 一致性。只删两条读根 dvc.yaml 的断言。
- **退役 `test_metric_perf_spike.py` 但保留 `scripts/spikes/metric_perf_spike.py`**：测试与 `data/anomod` 路径硬绑定，数据没了只能永远 skip（且它是 `@pytest.mark.slow` 的一次性内存验证，使命已完成）；脚本本身是通用 chunked-reading 工具，与具体数据集解耦。
- **`dvc gc -w` 回收 cache 而非只删工作区**：cache type=hardlink，`rm -rf data/` 只删一半链接、inode 不释放。gc 前 `--dry` 确认仅收集无引用对象；两道 "missing cache" warning 对应 new_ep1/normal_v2 的 cache 缺失——那本来就是它们的丢失状态，不是 gc 造成的。
- **base.yaml 的 `- data: anomod` default 直接删除而非改指 new_merge**：核查确认运行时无任何代码消费 `cfg.data`（训练走 `contract_dir=` 直读 contract parquet；根 dvc.yaml train stage 的 deps 从不列 `configs/data/anomod.yaml`；删后 `--cfg job` 组合正常）。给一个无人读的字段改指新目标等于给残留续命。
- **entry 编号跳 033 取 034**：`feature/service-minimal-v1` 分支已占 033（`033-service-minimal-v1.md`），两分支并行开发，跳号避免 merge 时编号冲突。

## 坑 / 已知问题

- **`configs/base.yaml` 是 `dvc_new_merge` 全部 8 个 train stage 的 dep，任何一行改动都会弄脏整条活链路**：本次仅删一行 Hydra default，v1/v2 四臂 train stage 全部 dirty（eval/score stage 不 dep 它所以不受影响）。语义上无影响（`cfg.data` 无人消费），但 DVC 只看哈希，裸 repro 会白烧 8 次训练。处置按 entry 032 先例 `dvc commit` 承认现状，验收 `dvc status dvc_new_merge/dvc.yaml` 全绿 + 裸 repro `--dry` 零触发。**教训：base.yaml 不是普通配置文件，是活链路的 dep——改它之后必须立刻 `dvc status` 检查，不能等别人裸 repro 时发现。**
- **先删 gitignore 条目再删数据目录，中间态 git status 会把 44G 数据目录显示为 untracked**：无实际危害（数据本来就该删），但容易误判为"哪里冒出来的新文件"。未来类似清理建议顺序反过来：先删数据，再摘 gitignore。
- **备查死 pipeline 的守卫测试对数据侧变化无感知**：`test_dvc_pipeline_v1.py` 保留的 RG 断言只检查 `configs/contract/*.yaml` 的存在性与绑定关系，dataset config 删了它照样绿。它们的绿只能证明"配置文件没被改坏"，不能当"链路还活着"的证据——链路死活取决于数据，而那部分已经没有守卫了。

## 遗留 TODO

- `dvc_reliability_gate/`、`dvc_deviation_weighted/`、`dvc_new_ep1/` 三个独立 pipeline 仍是死链（entry 015/018/019），本次按 031 惯例保留备查；其中 dvc_reliability_gate 引用的 `configs/data/merged_v2.yaml` 已删，若未来彻底下葬需同步处理 `test_dvc_pipeline_v1.py` 的 RG 断言与 `configs/contract/v1_expanded_pool.yaml`、`configs/contract/v1_new_ep1.yaml` 等。
- `data/new_ep1.dvc`、`data/normal_v2.dvc` 两个丢失数据集的 .dvc 指针仍在仓库（cache 本来就空，本次 gc 的两道 warning 即它们）；正式从 git 摘掉属另一次清理。
- ~~CLAUDE.md 同步~~（**本 PR 已完成**：项目定位段、data/ 目录清单、Commands 死命令块、Config System 示例、Data Rules、两条 gotcha 失配表述全部改写；claude-md-improver skill 当时在当前环境不可用，改为直接编辑 + 用户事先确认范围）。
- `scripts/build_contract.py` 的 Layout B 注释、`src/preprocessors/` 两处 anomod 相关注释、`tests/fixtures/nontarget_split_mini/_generate.py` 注释等描述性文字未清理，无功能影响。
- `docs/agent-docs/dataset-guide.md` 仍通篇描述 `data/anomod` 格式（CLAUDE.md 早已声明其过时），数据删除后它彻底成为纯历史文档，可考虑加醒目归档标注，非本次范畴。
