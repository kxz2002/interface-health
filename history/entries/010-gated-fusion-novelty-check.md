# 010 · Gated Conditional Fusion 立项前 novelty-check 结论

- **日期**: 2026-07-14
- **PR**: N/A（预实施阶段，本 entry 记录的是决策依据，不含代码改动）
- **类型**: Docs
- **影响域**: `history/`，`src/fusion/`（后续实施方向）

## 做了什么

在正式实现"粒度感知门控条件融合"（`g=sigmoid(W[e_ep;e_svc])`, `z=e_ep+g⊙(W_v·e_svc)`）之前，跑了一轮 `/novelty-check`：多源文献检索（WebSearch/WebFetch + OpenAlex API 绕过 Semantic Scholar 限流）+ 外部 LLM 交叉评审。检索并核实了 FiLM（Perez et al. 2018）、GS-Fuse（2026）、DALAD、MADGuard、TriAnomalyNet、MS-SVDD 系列等候选先行工作，得出结论后决定本分支暂缓实现门控机制本身，先转向补基础设施缺口。

## 关键决策（不在 commit 里）

- **不把门控机制本身当创新点**：结构上与 FiLM 的仿射条件调制、GS-Fuse 的门控条件融合属于同一机制家族（"一路表征生成门控/仿射参数调制另一路表征"），这是十年历史的成熟机制类别，审稿人一句"这就是 FiLM 换场景"就能打回，不能作为核心创新点宣称。
- **不把 Deep SVDD + 门控组合当创新点**：范围主动收窄——SVDD 仅作为后续验证/消融实验的下游检测器，不进入论文的创新点主张。
- **保留的唯一可能角度是问题定义本身**：metric/log 只能采集到 service 级、被 left join 广播复制到该 service 下所有 endpoint 行——这一"粒度错配"场景，在 per-endpoint 微服务异常检测文献里没见到专门处理（核实对比过 DALAD、MADGuard、TriAnomalyNet，均为不同技术路线，未处理广播复制问题）。但该问题的数学本质对应统计学里成熟的 hierarchical/panel data/mixed-effects 框架，不是全新问题定义，写作时需要正面论证"用神经网络门控做这件事"的增量价值，而非声称发现了新问题。
- **本分支改变打法**：不直接写门控实现，先补基础设施（融合机制可插拔化、group-aware split、参数量对齐 baseline），让这轮实验的方法论产出（测量协议、切分协议、对照实验设计）本身可复用，即使门控机制这个"点子"不构成独立创新。

## 坑 / 已知问题

- **现有 L0-L3 消融设计有 confounding variable 问题**（外部交叉评审抓到）：L0（裸拼接，无参数）→ L1（独立 encoder + 拼接，无门控）本身就引入了新的非线性容量/参数量，L0→L1 的性能提升不能干净归因于"解决了粒度错配"——足够表达能力的非线性网络理论上已经能从裸拼接里学到 endpoint 条件化的交互。必须用参数量对齐的 baseline 把"门控本身的贡献"隔离在 L1 vs L2 的对比里，不能只看 L0 vs L2。
- **随机行切分对 service 级广播特征无防护**：同一 service 下所有 endpoint 在 metric/log 特征上完全相同，若沿用当前按行随机切分 train/eval，模型有可能学到 service identity 而不是异常信号，构成类似数据泄漏的乐观偏差。需要 group-aware / leave-service-out 切分，目前 `src/`、`scripts/` 全库 grep 零命中，完全没有这层。
- **CLAUDE.md 与实际代码脱节**：CLAUDE.md 声称"模型通过 `hydra.utils.instantiate(cfg.model)` 实例化"，但 `configs/model/` 目录不存在，`configs/base.yaml` 的 `model: ???` 从未填充，`instantiate` 在 `src/`+`scripts/` 里零命中——实际训练脚本（`scripts/train_baseline_v0.py`）用纯 argparse 硬编码融合/模型选择。补基础设施时需要决定是真正补齐 Hydra 机制，还是更新文档反映现状，避免下次踩坑。
- **融合机制目前不是可平级扩展的**：`src/fusion/base.py` 的 `FusionModule` 抽象本身很干净（只要求 `forward()` + `output_dim`），但 `MODALITY_ORDER` 常量挂在 `EarlyConcatFusion` 类上而非独立模块常量，且 `scripts/train_baseline_v0.py` 硬编码 `from src.fusion.early_concat import EarlyConcatFusion` 并直接实例化——今天"换一个融合机制"等于改训练脚本代码，不是切一个 config 值。若不先重构，L0-L3 四个变体会被迫 fork 出 4 份几乎相同的训练脚本。

## 遗留 TODO

- 补融合机制可插拔切换（config-driven factory），让 L0-L3 及未来融合变体共享同一套训练/评估代码路径
- 补 group-aware / leave-service-out 数据切分逻辑——这是任何涉及 service 级广播特征的消融实验的前提，不做这个,后续所有实验结果都存疑
- 补参数量对齐 baseline 的构造方式/工具，用于隔离门控机制本身的贡献
- 决定 Hydra `model`/`fusion` instantiate 机制是否要真正补齐,或更新 CLAUDE.md 反映当前 argparse 现状
- 后续用 `/brainstorming` 结合本 entry 与基础设施调研产出 spec/plan；正式写代码前记得把分支名从 `feature/gated-fusion-v0` 改掉（尚未推送到远程,可安全改名，因为最终定位已经从"门控融合创新点"变成"基础设施补齐 + 消融方法论"）
