# Reproducibility Checklist

对照 NeurIPS Reproducibility Checklist 的逐条声明。

---

## 模型与算法

- **算法伪代码**：待补充（模型实现完成后更新）
- **完整模型规格**（层数、激活函数、超参数）：通过 `configs/` 下 YAML 文件完整记录，所有超参数均外化到配置，代码内无硬编码
- **训练目标函数**：待补充

## 理论结果

- 本项目为实证研究，无定理/命题，此项不适用

## 实验

- **随机种子**：`src/utils/seed.py` 同时设置 Python `random`、NumPy、PyTorch 三个随机源；`configs/base.yaml` 的 `seed: 42` 为默认值，命令行可覆盖
- **计算资源**：待补充（GPU 型号、训练时间）
- **超参数搜索范围**：通过 Hydra multirun 记录（`python scripts/train.py --multirun ...`），搜索范围在对应 config 文件中声明
- **评估指标**：AUROC、AUPRC；评估标签来源 `phase == 'inject'`（来自 `case_metadata.json` 的强标注），不依赖 `weak_is_anomaly`

## 数据集

- **数据集**：Train-Ticket 微服务故障注入数据集，11 个 case（1 Normal + 10 故障注入）
- **数据访问**：原始数据位于 `data/anomod/`（READ-ONLY），通过 DVC 版本化，`.dvc` 文件入 git，原始数据不入 git
- **数据预处理**：pipeline 脚本位于 `scripts/`，产物为 `_pipeline_out/tt_fused_15s.csv`，字段口径见 `docs/agent-docs/dataset-guide.md`
- **训练/测试划分**：Normal case 全程作为训练集（One-Class，label=0）；异常 case 的 inject 阶段作为正样本评估集，baseline/recover 阶段作为额外负样本
- **已知数据问题**：见 `docs/agent-docs/dataset-guide.md` 第十一节；主要问题为 `Lv_S_KILLPOD_gateway` inject 阶段数据永久缺失

## 代码

- **代码仓库**：本仓库
- **依赖环境**：`environment.yml` + `pip install -e ".[dev]"` 一键复现
- **运行说明**：`make setup` 建环境，`make test` 跑测试，`python scripts/train.py` 启动训练（训练脚本待实现）

## 待完善项

以下项目在模型实现完成后补充：

- [ ] 算法伪代码
- [ ] 完整模型规格表
- [ ] 计算资源说明
- [ ] 超参数最终取值与搜索范围
