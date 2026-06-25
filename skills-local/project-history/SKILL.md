---
name: project-history
description: "Use BEFORE starting any new feature/refactor/bugfix work in this repo, and BEFORE merging a PR. Reads history/index.md to surface past decisions, gotchas, and unfinished TODOs that may affect the current task; appends a new entry to history/entries/ when work is complete. Trigger phrases: 开发新功能, 实现 ..., 改 ..., 重构 ..., 修复 ..., merge PR, 准备提 PR, 历史记录, 历史记忆, 查 history, project history."
allowed-tools: Read, Write, Edit, Bash, Grep, Glob
---

# Project History · 历史记忆查阅与维护

本仓库维护一份持久的项目历史，结构如下：

```
history/
├── index.md           # 轻量索引：entry 列表 + 影响域倒排
└── entries/
    ├── _template.md   # 新增 entry 时复制此模板
    ├── 001-*.md
    ├── 002-*.md
    └── ...
```

**核心动机**：commit message 只讲"做了什么"，但项目延续中真正重要的是"为什么这么选、哪里埋了坑、留了什么 TODO"。history 把这层信息持久化，让后续 agent 不重复踩坑、不破坏已有决策。

## 何时触发

### 场景 A：开发任务开始时（必读）

满足任一即应**先读 history**，再开始动手：

- 用户要求新功能、修 bug、重构、加数据/模型/评估指标
- 用户提到"改 X 模块" / "实现 Y" / "重构 Z"
- session 刚开始且本次有代码改动意图

**流程**：
1. `Read history/index.md`
2. 根据本次任务涉及的目录/主题，对照 **影响域索引** 找出相关 entry
3. `Read` 每个相关 entry，重点看「关键决策」「坑 / 已知问题」「遗留 TODO」
4. 把找到的关键约束/坑点在回复中显式提示用户："过去在 entry NNN 中决定 X 因为 Y，本次需要保持/调整"

### 场景 B：PR 即将 merge 时（必写）

满足任一即应**追加一篇 entry**：

- 用户说"准备提 PR" / "merge 前最后检查" / "/finishing-a-development-branch" 类指令
- 当前分支已完成开发，准备汇总变更

**流程**：
1. **先查重**：`Bash: ls history/entries/` 或 `Grep` 搜索本分支 commit hash / PR 号是否已在某个 entry 出现。若已存在，**不再撰写**，告知用户"history/entries/NNN-*.md 已存在对应记录，跳过"
2. `git log` 查看本分支自分叉点起的所有 commit
3. `git diff <base>...HEAD --stat` 看影响域
4. 在 `history/entries/` 下新增 `NNN-<slug>.md`（NNN 严格递增），用 `_template.md` 作骨架填充
5. 编辑 `history/index.md`：在 entry 列表表格里追加一行，在影响域索引里把新 entry 编号加到对应目录行
6. 把新增 + 修改加入本 PR 的 commit

## 写作要点

`_template.md` 已经给出结构，落笔时注意：

- **「做了什么」要短**：commit message 已经讲过的不要重复；1–2 段够了
- **「关键决策」是主体**：每条说明"选了 A 而不是 B / 为什么"。否定的方案 = 未来的诱惑，必须留痕
- **「坑」要写触发条件**：不是"X 不工作"，而是"当 X 遇到 Y 条件时会 Z，规避方式是 W"
- **「遗留 TODO」要可追溯**：每条都应能在未来某个 entry 里被"消化"——届时新 entry 应在「关键决策」里引用旧 entry 编号说明"消化了 NNN 的 TODO 第 K 项"

## 与 CLAUDE.md / git log 的边界

- **CLAUDE.md**：长期稳定的项目约定、目录、命令、坑（架构性、不随单次 PR 变化）
- **git log / commit message**：单次变更的 what
- **history entry**：单次变更的 **why + gotcha + leftover**

新增 entry 时如果发现一条信息是"架构性、长期稳定"的（例如新的目录约定、新的工具链命令），**同时**更新 CLAUDE.md，不要只写在 entry 里。

## 边界

- **写之前必须查重**：若本次 PR / commit 已存在对应 entry（按 PR 号 / commit hash / slug 匹配），告知用户并**不再撰写**，不覆盖、不追加重复内容
- 不为每次 commit 写 entry，只为 PR / 主要迭代写
- 不写实验结果数据（那是 MLflow / `outputs/` 的事）
- 不写 changelog 风格的版本号、发布说明
- entry 编号严格递增，不复用已删 entry 的编号
