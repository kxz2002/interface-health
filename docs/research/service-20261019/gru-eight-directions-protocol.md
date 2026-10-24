# 八方向冻结实验协议

本轮按用户授权完成八个方向，优先级固定：双头绝对值/变化量、多尺度历史特征、分模态GRU、Normal原型记忆、机制分数组合、异方差GRU、TCN-GRU、遮蔽预训练。完整配置为 `configs/service/gru_eight_directions_v1.json`。每方向固定三个seed（1、2、42），不按首个seed、Normal loss或外层故障评分决定是否补齐；实现错误修正后采用新run目录并保留失败日志。不扩大候选或据故障结果调参。

主干为原4窗/hidden32/50轮/AdamW设置。原始GRU另做三个seed复现。模型输入是资源3项、执行统计5项、日志统计4项，以及12个mask、3个可用性和1个历史存在位。当前数据没有直接IO、DNS或连接池等待特征，方向名称不意味着能观测这些机制。

1. 双头：共享GRU，保留绝对值头并新增12维变化量头；变化量目标为当前值减4窗内最近有效历史值。变化量只在历史和当前均有效的字段计算。数值分数为两头MSE等权平均（无有效历史时使用绝对值），加原可用性BCE。
2. 多尺度：原4窗内部构建1步差分、2步斜率，各附有效mask。76维输入；不引入额外窗口或当前目标值，无跨Case/Service/split/gap参考。保持原输出与评分。
3. 分组：资源、执行、日志分别使用hidden16 GRU，各自只接收所属字段、mask、对应可用性与历史存在位；三表示拼接，经sigmoid门控和线性输出。不是三个看见全部字段的重复分支。保持原评分，参数量检查不超过原1.5倍。
4. 原型：复用同seed原GRU，按fit Service对hidden state拟合两原型KMeans（不足两种状态时减小），用fit均值/标准差白化，尺度下界0.05；距离与GRU分数组合。
5. 机制：复用同seed原GRU，增加同一输入上的最大有效字段变化平方、三模态可用性转换幅度。没有根据测试故障选择字段或翻转方向。
6. 异方差：共享GRU增加12维有界log variance头，范围(-4,4)，使用高斯NLL，加与原一致的可用性BCE。NLL加常数2保证非负；另报告原坐标MSE以免把不同损失函数直接比较。
7. TCN-GRU：两层左侧padding因果卷积，channels16、kernel2、dilation1/2，然后接hidden32 GRU，输出和评分保持原值。
8. 遮蔽：仅Normal fit历史，随机遮蔽10%有效字段并同步清除mask，10轮因果逐历史位置重建预训练；重新建立优化器，随后原预测任务50轮。校准/测试无人工遮蔽。

内层将outer fit的四个Normal Case按开始时间选最后一个整Case留出，其余三个拟合。numeric、日志模板及统计均调用现有 `prepare_inner` 重新拟合，绝不继承outer-fit预处理。外层使用原prepared及冻结fit/cal/test/main IDs和标签；Unknown不填0、context不跨边界。输入raw/旧outputs只读。

原型和机制分数组合的权重固定0.8原GRU/0.2辅助；每分量先除以内层留出Normal Case的q95（下界0.001）并log1p，不按测试调整。内层该Case同时作为组合尺度参考，故其组合分数仅诊断、不是独立组合验证；外层Normal校准Case仍独立用于阈值。内外模型、预处理坐标与原型分布的转移可能失败，需明确报告。

所有方向均先做内层Normal验证，再训练/复用外层模型并固定评价。本轮没有根据Normal loss淘汰候选，因为已知更好Normal预测不保证检测改善，不同目标的损失也不能直接作模型排名。三个seed AUROC与AP均优于对应原GRU才作为稳定替代候选；同时报告均值、样本标准差、逐seed配对差、q99误报/召回和故障分组。已看过此测试集，本轮属于探索，不能报告为新盲测、独立事件重复或有保证的泛化提升。

远端先通过本地Git bundle同步基础commit至独立worktree，保留原dirty仓库；新增源码按明确清单同步并核对SHA256。每run保存源码hash、git HEAD/dirty、环境、配置、内外fit IDs/预处理hash、上下文IDs、checkpoint、逐epoch历史、逐样本分数、阈值、报告、manifest与MLflow。续跑只接受同一源码/配置、同父prepared且check_run成功的已完成run；不覆盖已有输出。
