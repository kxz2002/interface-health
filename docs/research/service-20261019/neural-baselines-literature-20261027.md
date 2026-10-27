# 近期神经网络基线调研与适配记录

状态：调研中；尚无本轮基线训练或评价结果。本记录用于确定候选及复现风险，不把论文在其他数据集上的结果当作本项目结果。论文和代码核对日期为 2026-10-27。

## 候选及原始资料

| 方法 | 论文 | 结构与输入 | 本项目适配问题 |
|---|---|---|---|
| CATCH | Wu 等，ICLR 2025，[论文](https://openreview.net/forum?id=m08aK3xxdJ)、[官方代码](https://github.com/decisionintelligence/CATCH) | 多变量序列；频域 patch、通道融合和重建，时间域和频域误差用于评分 | 每个 Service 单独形成序列；频域和窗口长度需考虑短 Case、缺测及闭窗时间；官方标签接口从训练和测试分数共同计算阈值，必须替换。 |
| PatchAD | Zhong 等，IEEE Transactions on Big Data 2025，[论文](https://doi.org/10.1109/tbdata.2025.3596745)、[官方代码](https://github.com/EmorZz1G/PatchAD) | 多变量序列；多尺度 patch、MLP-Mixer 和对比表示 | 上游 trainer 用测试集做早停，且使用训练和测试分数共同定阈值，均不可沿用；patch 尺寸需整除序列长度。 |
| KAN-AD 多变量实现 | Zhou 等，ICML 2025，[论文](https://icml.cc/virtual/2025/poster/45584)、[原始代码](https://github.com/issaccv/KAN-AD)、[多变量实现](https://github.com/thuml/Time-Series-Library/blob/main/models/KANAD.py) | 原始代码用单变量历史预测下一点；Time-Series-Library 按通道共享 KAN 网络重建多变量序列 | 两者任务和输出不同，必须标为 Time-Series-Library 的多变量重建适配，不冒充作者原始预测实现；缺测数值不得计入误差。 |
| TimesNet | Wu 等，ICLR 2023，[论文](https://openreview.net/forum?id=ju_Uqw384Oq)、[官方代码](https://github.com/thuml/Time-Series-Library) | 使用频域周期选择和二维卷积进行多变量序列重建 | 官方频率选择对 batch 求均值；本项目评价逐样本推理，避免其他测试窗影响当前分数。2023 年模型较旧，但作为官方支持异常检测的架构对照值得保留。 |
| ScatterAD | Yin 等，NeurIPS 2025，[论文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/4b52b3c50110fc10f6a1a86055682ea2-Abstract-Conference.html)、[官方代码](https://github.com/jk-sounds/ScatterAD) | 多变量时间序列，将时间点构成图节点，结合图编码、时序一致性和对比融合 | 官方图连相邻时间点，评分还比较相邻节点；需处理末端评分和 batch 图边界，不能使用目标时刻之后的信息；依赖 PyTorch Geometric。 |
| DCdetector | Yang 等，KDD 2023，[官方代码和论文入口](https://github.com/DAMO-DI-ML/KDD2023-DCdetector) | 多变量序列；两种 patch 视图和双分支注意力，通过对比差异评分 | 上游 `solver.py` 用测试集做早停、混合训练和测试分数定阈值；patch 尺寸需整除窗口长度。2023 年较上述候选更早，作为架构不同的候选，最终取舍需结合更多近期文献。 |

CATCH、PatchAD 和 ScatterAD 均为 2025 年发表，DCdetector 是 2023 年发表。这些架构适合探讨时序异常检测，但论文原始数据与本项目的 Service 多模态观测、缺测机制和标签不同；不能直接声称论文基准表现会迁移到这里。上游示例通常使用几十至数百时间步，而已有 GRU 的默认历史为 4 步，直接统一窗口可能导致这些方法失去原有设计意义；使用更长窗口又会改变可用历史和覆盖，必须明确报告两者。

## 代码核查

2026-10-27 检出的上游 HEAD：CATCH `3647c69be5eb56649b072596cf89098e689e20c3`，PatchAD `38606bc1b38f4b099ea053880846e9b434171dd5`，ScatterAD `7572fa954468e46c598badf1c9cfaf6c1835c4cb`，DCdetector `9d107cae518ef24ee2eb2ad1b92107cdcd09ea10`。这是检出版本，不等于论文实验使用的版本。PatchAD 仓库有 Apache-2.0 LICENSE，ScatterAD 为 MIT；所检出的 CATCH 和 DCdetector 根目录未见 LICENSE 文件，GitHub API 也未识别许可证。引入源码前应进一步核对授权；不能默认无许可证代码允许复制。上游示例运行脚本并非直接适用于本项目，不能直接使用其评价和阈值流程。

ScatterAD 的 `compute_loss` 和 `get_anomaly_scores` 对 batch 合并后的相邻节点直接做差，即使不同图之间没有边，也可能把两个样本的边界连接起来；其双向邻接也使中间节点表示包含后续信息。复现必须验证图级切分以及当前窗的因果评分，并记录由此带来的方法改动。它的 Service 数据 loader 的 `thre` 使用验证段而非测试段，但本项目仍应以独立 Normal calibration 定阈值。本机 `interface` 环境目前缺 `einops` 和 `torch_geometric`，且仅有 CPU PyTorch；gpu-b 依赖状态尚未核对，不自动安装或更改现有环境。

实际运行 ScatterAD 上游 `7572fa954468e46c598badf1c9cfaf6c1835c4cb` 构造函数还发现：center 初始化为一维向量，随后按 `dim=1` 求范数，直接抛出 `IndexError`。论文描述固定中心，而代码定义可训练中心且目标编码器被冻结，二者还需核对；暂不将其视作已复现模型。仅为核查，`torch-geometric==2.6.1` 安装在 `/tmp/interface-neural-baseline-deps`，未改动 conda 环境。

KAN-AD 原仓库 HEAD 为 `7c3a0d96b77e0d609660a240c2b11db084ba523f`，其作者参与的 Time-Series-Library 多变量实现检出 `4e938a1767106324dd753b2a44832bf870a0252e`；二者均为 MIT 许可。TimesNet 亦使用该固定版本。后者原始 evaluation 将测试分数用于阈值且使用 point adjustment，本项目不能沿用。当前以 27 维输入进行闭窗重建，训练和评分均不把缺测数值当有效目标；对观测 mask 与可用性字段计算重建误差。缺测值及历史填充仍进入网络内部，这一适配及信息差异需在报告中说明。两种上游模型已完成合成训练和评分烟测，通用入口也通过合成数据的校准、评价和产物核对；真实数据运行仍未验证。

`configs/service/tt_kanad_reconstruction_v1.json`、`configs/service/tt_timesnet_reconstruction_v1.json` 分别参考上游 PSM 示例；两者当前使用固定 epoch 和固定学习率，未复用上游的验证早停和学习率调整，故不称完全复现原实验。TimesNet 的 FFT 周期选择会对 batch 求均值，训练批次仅含 Normal fit；校准与测试时每次只传一个目标序列，避免分数受到其他样本影响。若实际训练需要超参数选择，应先建立仅基于独立 Normal 的选择流程，不能用故障测试结果挑选。

PatchAD 的 `main_ad.py` 选择 `patchad_trainer_v2`；该 trainer 计算但未将重建 MSE 加入训练目标，而默认 `cont_beta=0` 的评分完全使用重建误差，重建头因此没有通过该目标训练。其另一 trainer 使用重建损失，但配置 `norm='ln'` 时与官方模型张量维度不兼容。本项目适配初步保留 v2 的对比训练目标并使用其对比分量评分（相当于 `cont_beta=1`），取闭窗最后一个时点，属于对上游默认评分的明确改动，不声称与其默认评测完全一致；适用性及效果尚待实际实验验证。已在本机用上游模型对合成输入执行前向/反向，并逐值对比上游目标和对比分量。`einops` 仅安装在 `/tmp/interface-neural-baseline-deps` 用于核验，未改动 conda 环境。

适配训练使用 Normal fit 中具有完整连续历史的样本，记录不足历史的 fit ID；评分保留所有指定 ID，并将不足历史的前缀填零。模型输入包含 12 个数值、12 个字段 mask、3 个模态可用性，前缀填充存在位用于筛选完整训练历史而不输入上游网络。缺测位置虽附带 mask，但上游网络不直接支持缺测掩码，其归一化仍受零填充影响；这需要与覆盖率一起报告，不能称完全解决缺测。

`configs/service/tt_patchad_neural_v1.json` 预设参考上游 PSM 示例的 105 窗、3/5/7 patch、3 epoch、学习率和 batch 大小；使用固定轮数而非测试集早停，未据本项目测试分数调整。`scripts/service_patchad_model.py` 验证上游 Git 版本及干净状态，复用 prepared 的独立 Normal 校准和评价，并保存 checkpoint、逐样本分数、训练样本 ID 和历史窗口 ID。由于旧 prepared 的标签协议为 v2，主结论仍须使用 `service_revaluate_temporal.py` 的已验证派生标签包生成 v3 结果。三种入口都已通过固定上游模型和合成 prepared 的完整训练、校准、评价、产物核对，以及复评来源/阈值验证；复评烟测沿用合成的原评价版本，并未验证真实 v3 标签包。尚未使用真实 prepared 或 gpu-b 执行，该预设对本数据能否训练及其覆盖尚未验证，不可宣称它已是有效强基线。

## 数据与公平性约束

- 当前时序输入每步包含 12 个数值（资源 3、执行 5、日志统计 4）、对应 12 个观测 mask、3 个模态可用性位；历史另含存在位。数值使用原 prepared 的 Normal 拟合状态和 `arcsinh` 变换。应分别说明哪些字段参与模型输入、重建或对比目标，不能把缺测填零后当作真实观测计算误差。
- 数据键为 Case、Service、划分和时间，15 秒一窗；上下文不能跨 Case、Service、划分、时间缺口或 fit 的有效 ID 边界。不能把多个序列拼成连续时间序列。须保存目标 ID 与上下文 ID，验证无交叉。
- 只用独立 Normal fit 学习参数；以单独留出的 Normal 数据选择停止点或超参数；独立 Normal calibration 计算 q99 阈值；Fault 仅用于最终评价，Unknown 不转负类。须冻结候选及选择规则，不能根据 Fault 分数选择设置或有意削弱模型。
- v3 排除 POOL/DNS；主评价原有 3141 个窗口。对于不足历史、缺测或模型无法评分，必须报告全部目标 ID 的覆盖和缺失原因，不能通过删除难例提高指标；比较应同时给出相同 ID 的配对指标与各方法完整覆盖。
- 论文网络通常读取包括当前时刻在内的序列并产生重建或逐点对比得分，现有 GRU 则从过去预测当前窗。闭窗离线评分可比较，但应声明信息集差异、窗口长度和 score-to-window 映射；不能使用目标窗之后的信息或重叠片段的未来分数。
- 固定多个随机种子、保存原始分数、阈值、逐 Case 统计、环境、代码版本、超参数和样本身份；与原 GRU 使用同一套标签、AUROC/AP、q99 召回及 Normal FPR 计算。按故障 Case 和覆盖共同解释，不使用 point adjustment 把整段故障填为已检出。

## 尚待完成

进一步核对论文细节、官方许可和依赖，选定至少三个可实际复现的近期模型；在独立 Normal 验证上确定合理的窗口、超参数和早停规则；实现可追溯的适配器并验证边界、缺测、计分和样本覆盖。检查远端连接和 prepared/环境后再启动训练。各候选是否可在现有数据上运行、结果如何均尚未验证。
