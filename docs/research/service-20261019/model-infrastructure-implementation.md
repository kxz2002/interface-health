# Service 模型基础设施实施记录

2026-10-21，Task 8 工程集成。当前实现提供可离线验证的模型流程；合成测试使用 TinyBackbone，不能据此宣称真实 BERT 检测有效。

## 已实现入口与身份链

`scripts/service_model.py` 只有显式子命令，没有默认训练或自动矩阵执行入口：

|命令|行为|
|---|---|
|`check-inputs`|读取既有准备包和 schema 2 标签包并验证身份；不拟合预处理或加载文本模型|
|`prepare`|仅 known-normal fit 拟合数值、Drain3 和四项日志统计；冻结所有用途清单，保存 JSON/Parquet 准备包|
|`probe-fit`|仅 fit 小批前反向，恢复模型状态；不发布训练 checkpoint|
|`fit`|只执行显式选择的方法、单个 seed；神经模型保存最后 epoch checkpoint，基线保存模型包|
|`score`|重载绑定的模型；按显式用途评分，不训练|
|`evaluate`|独立 Normal calibration 分数校准，同次生成主评价及同 Case/Service 前后诊断|
|`check-run`|递归核查成功状态、配置、provenance、父 manifest 及所有 artifact 的 SHA256/字节数|

`configs/service/tt_model_matrix_v1.json` 声明三种神经融合、L2/Max、数值及冻结初始文本三模态 IF/OCSVM，共九种方法。矩阵只是显式方法清单，不会自动执行。正式 seed 设置仍为 1、2、42；确定性规则及 OCSVM 不重复生成统计意义相同的运行。

默认 `score_scope=test` 是 `main_test ∪ diagnostic_pre ∪ diagnostic_positive`，集合去重后评分；`calibration` 单独评分。另有 `main_test`、`diagnostic` 和 `extended_test`；后者使用事先冻结的完整可用 test 清单。只有 test/extended_test 产物可进入整体 evaluate；不以缺失诊断分数的 main_test 产物发布整体评价。默认 test 不强制补评分 Unknown 或日志独有扩展窗，扩展报告显示 `not_requested`；所有未请求范围仍保留冻结清单与覆盖记录。

prepared 包持久化稳定身份、数值值/掩码、日志二维 token/组/正文/覆盖、`logs['statistics']` 标准化四统计、全部 SampleLists、标签/划分/评价表及完整 protocol、版本、输入 hash 和预处理状态。fit 在 build_model 前设置 seed；checkpoint 重载核查 prepared 父身份、fit IDs、版本、配置及预处理 hash。评分包绑定所用模型和 prepared manifest，评价保存四档阈值、逐样本预测、主/诊断用途和报告。

每个阶段要求不存在的输出目录和显式本地 MLflow URI；成功才发布 manifest，失败保留状态。记录配置、Git commit/dirty、包版本、CUDA 可用性、seed、特征顺序、参数数目和训练诊断。含清洗正文/Drain 模板的 `data/prepared.json` 仅保存本地并写 hash，不复制到 MLflow artifact；配置、指标、manifest 和不含消息的表可以追踪。新建 MLflow experiment 的 artifact_location 显式设置到 tracking 数据库旁目录，测试数据库及 artifact 都留在 tmp_path，不产生仓库根 mlruns。

最终审查修复后，数值 L2/Max/IF/OCSVM 的 fit/score 不构建 BERT，也不计算冻结文本；只有神经/trimodal 方法加载 backbone。共同 prepared 包仍保留三模态输入，用于公平清单和方法比较。统一配置入口拒绝与当前固定实现不一致的 baseline 参数、校准方法/比较/主分位点、std_ddof/fallback_scale，不接受覆盖后静默忽略。

仅 test 标签/评价版本变化时，CLI 从原训练与当前 prepared 的真实内容构造完整复用证据，调用 check_reuse；比较全部观测/掩码、fit/cal 标签/资格/ID、预处理/模板、模型配置/seed、split/质量规则。兼容后可固定同一模型和已有分数独立重评，原训练版本与当前评价版本分别保存。评价投影用 source_* 列保留原分数的版本、清单及标签声明，不改写原分数文件或伪造其 attrs。新增必需 ID 缺分明确报 rescore_ids，不缩小分母；score 的 `--reuse-scores OLD_SCORE_RUN` 复用已有行，只对缺失 ID 推理，记录补评分 ID 和父链，不重训。

## 命令形式

以下是接口示例，本轮未对真实数据执行这些命令。各输出路径必须尚不存在；本地文本权重/tokenizer 必须已存在并在配置中分别绑定 40 位 Git revision。默认配置 revision 为 null，因此生产 prepare/fit 会显式失败，不下载或猜测模型版本。

```bash
conda run -n interface python scripts/service_model.py --help
conda run -n interface python scripts/service_model.py check-inputs --config CONFIG --preparation PREPARATION --label-bundle LABEL_BUNDLE --tracking-uri sqlite:////ABS/mlflow.db --output NEW_CHECK_RUN
conda run -n interface python scripts/service_model.py prepare --config CONFIG --preparation PREPARATION --label-bundle LABEL_BUNDLE --tracking-uri sqlite:////ABS/mlflow.db --output NEW_PREPARED_RUN
conda run -n interface python scripts/service_model.py probe-fit --config CONFIG --prepared PREPARED_RUN --method transformer_cls --tracking-uri sqlite:////ABS/mlflow.db --output NEW_PROBE_RUN
conda run -n interface python scripts/service_model.py fit --config CONFIG --prepared PREPARED_RUN --method transformer_cls --seed 42 --tracking-uri sqlite:////ABS/mlflow.db --output NEW_FIT_RUN
conda run -n interface python scripts/service_model.py score --config CONFIG --prepared PREPARED_RUN --model-run FIT_RUN --tracking-uri sqlite:////ABS/mlflow.db --output NEW_TEST_SCORES
conda run -n interface python scripts/service_model.py score --config CONFIG --prepared PREPARED_RUN --model-run FIT_RUN --score-scope calibration --tracking-uri sqlite:////ABS/mlflow.db --output NEW_CAL_SCORES
conda run -n interface python scripts/service_model.py evaluate --config CONFIG --prepared PREPARED_RUN --model-run FIT_RUN --scores TEST_SCORES --calibration-scores CAL_SCORES --tracking-uri sqlite:////ABS/mlflow.db --output NEW_EVALUATION_RUN
conda run -n interface python scripts/service_model.py check-run --run EXISTING_RUN
```

## 本轮验证与边界

先写 CLI/smoke 测试，实际得到两个入口缺失 ImportError；首轮实现五项测试通过。扩展全部九方法闭环时发现基线分数只有 attrs 中的配置绑定，且原 fit_baselines 会拟合全部 sklearn 方法；集成修复为验证后显式持久化配置列、按请求方法拟合。默认 test 与旧 evaluator 强制完整扩展评分的冲突通过显式 score_scope 解决，旧默认调用行为保持兼容。

Task 8 初始专项 16 项 CPU 合成测试通过，覆盖九方法 prepare→fit→score/calibration→evaluate、重载分数一致、版本/protocol/统计状态守恒、MLflow 查询、评分去重、probe 恢复、check-inputs 无模型构建、规则不拟合额外 sklearn 模型和父链篡改拒绝。最终审查后加入固定配置拒绝、数值免BERT、真实新prepared跨评价重评及补评分测试，全部新增建模与相关旧 Service 标签/准备/模型兼容回归最终为 214 passed（88.92 秒）；7文件black/isort检查通过，git diff --check无输出，使用已有工具。具体命令见 Task 8 报告。

未执行 Task 9 真实输入预检/显存预检、真实 BERT 训练/下载、Task 10 真实实验验收；未读取真实 raw/Fault、未重建准备包或 DVC、未安装依赖、未提交/推送/PR。未修改 history entry/index；后续发布流程按单独授权执行。没有真实效果、GPU峰值显存、吞吐或方法增益结论。
