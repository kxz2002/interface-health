# 004 · 引入 Project History 历史记忆系统

- **日期**: 2026-06-24
- **PR**: feature/history（待 merge）
- **类型**: Feature（基础设施）
- **影响域**: `history/`, `skills-local/project-history/`, `Makefile`, `CLAUDE.md`

## 做了什么

为仓库引入持久化的项目历史记忆机制，让 agent 在开发新功能前能查阅过去的决策与坑点。

- 新建 `history/index.md`：轻量索引，含 entry 列表 + 按目录的影响域倒排索引 + 横切主题总结
- 新建 `history/entries/`：存放每次 PR/迭代的详细文档，含 `_template.md` 模板
- 回填历史 3 篇 entry：001（初始 commit）、002（PR #1 工程基础设施）、003（PR #2 实验闭环）
- 新建 `skills-local/project-history/SKILL.md`：定义查阅（场景 A）与更新（场景 B）的完整流程；通过 `make install-skills` 链接到 `.claude/skills/`
- 更新 `CLAUDE.md`：新增 "Project History" 章节，说明用途与触发时机

## 关键决策（不在 commit 里）

- **只有两层结构（index + entries/），不做三级分类**：分类层会增加维护成本，影响域倒排索引已经能覆盖"定向查找"的需求，不需要再多一层目录
- **`history/` 而不是 `.claude/history/`**：history 是项目级文档，属于项目仓库本身；放 `.claude/` 下语义上偏向"工具配置"，不合适
- **ARIS 的 `research-wiki` 不复用**：research-wiki 实体是 Paper/Idea/Experiment/Claim，面向学术研究工作流；本需求是"PR 级别的工程决策记录"，两者粒度和语义不同，强行复用反而增加认知负担
- **index 加影响域倒排索引**：解决"我在改 X 目录，哪些历史决策与我相关"这个问题——agent 不需要逐篇扫 entry，按目录直接定位
- **SKILL 的 description 同时写中英文触发词**：agent 匹配 skill 时做字面 + 语义双层判断，触发词越丰富，自动调用成功率越高
- **查重逻辑写进 SKILL 边界**：防止 agent 在 resume / 重跑时重复写同一个 entry，污染 history
- **skill 放 `skills-local/` 而不是 `.claude/skills/`**：`.claude/` 是 ARIS 领地（gitignored），project-local skill 放 `skills-local/` 入库，`make install-skills` 自动创建 symlink。新增 project skill 只需在 `skills-local/` 下建目录，无需改 `.gitignore`

## 坑 / 已知问题

- **当前 skill 不在 ARIS 注册表里**：`history/entries/` 是本地新增，`.aris/installed-skills.txt` 不需要改；但 `.claude/skills/project-history/` 是本地目录（非 ARIS symlink），后续如果重跑 `install_aris.sh` 不会删掉它，但也不会被 ARIS 管理——这是预期行为
- **回填 entry 是从 commit message 和 git diff 反向推导的**，部分"关键决策"可能不完整（当时的内心独白没记录）。001/002/003 的主要价值是把已知的坑（drain3-improved、conda base、allow_nan 等）固化下来；深层的当时决策细节可能有遗漏

## 遗留 TODO

- 没有自动化机制保证"PR merge 前一定写了 entry"——目前靠 SKILL.md 约定 + CLAUDE.md 提醒，未来可以考虑 pre-push hook 检查 entries/ 是否有新文件
- skill 的场景 A（开发前自动查阅）还需要在实际新功能开发中验证是否真的被触发
