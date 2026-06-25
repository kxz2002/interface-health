# 003 · 最小可复现实验闭环（PR #2，迭代 2）

- **日期**: 2026-06-24
- **PR**: #2 · **Commit**: 40cc650
- **类型**: Chore（实验闭环）
- **影响域**: `src/contracts/`, `scripts/train_baseline.py`, `scripts/eval.py`, `dvc.yaml`, `artifacts/`, `tests/`, `CLAUDE.md`

## 做了什么

把 train → eval 跑通成 DVC pipeline，**不依赖真实数据**，用 toy 合成数据走通接口。

- `src/contracts/scores_v0.py` + `metrics_v0.py`：契约校验器，固定 `scores.parquet` 和 `metrics.json` 的字段口径
- `scripts/train_baseline.py`：toy 合成数据 + L2 距离打分（占位 baseline，**不是真模型**）
- `scripts/eval.py`：`scores.parquet → metrics.json`，含 AUROC/AUPRC
- `dvc.yaml`：train/eval 两个 stage，显式 deps/outs
- 41 个测试用例覆盖契约校验和 e2e toy pipeline
- 验收：`pytest -q` 全过 + `dvc repro` 端到端跑通 + `metrics.json` 含 `protocol_version: v0`

## 关键决策（不在 commit 里）

- **契约层（`src/contracts/`）的意义**：train 和 eval 解耦。eval 不读 train 的内部数据结构，只读 `scores.parquet`；任何破坏字段口径的变更必须先升 contract 版本（v0 → v1），强制开发者意识到接口变更
- **`protocol_version: v0` 字段**：写入 metrics.json，未来切换 contract 版本时 eval 可以拒绝旧版 scores，避免静默漂移
- **toy baseline 先行的理由**：真实模型还没设计完，但 DVC pipeline、契约、CI 这些"管道"必须先稳定。等真实模型来了直接替换 `train_baseline.py`，pipeline 不需要动
- **`y_true` 用 `object` dtype 显式拒绝**：字符串标签会被 sklearn 默默吃下并算出错位的 AUROC，必须在契约层就拒掉
- **`validate_scores_df` 返回 `ScoresV0` NewType**：这是个静态类型 trick，函数签名层标记"这里返回的是已校验过的 df"，比 docstring 更可靠
- **`has_labels` 字段保留但 v0 内恒为 True**：v1 会支持纯推理场景（没有标签），现在就把开关位留好

## 坑 / 已知问题

- **`json.dumps` 必须加 `allow_nan=False`**：默认会写 `NaN` 进 json，生成无效 JSON。被 PR review 抓到（3da309a）
- **`pd.read_parquet` 必须 try/except**：原本失败会抛 stack trace，需要捕获后 `exit code 1`，遵守 CLI 合约
- **`.dvcignore` 必须排除 `__pycache__` 和 `.swp`**：否则 stage 哈希会被这些文件污染，DVC 误判 stage 变更，反复重跑
- **dvc.yaml 必须显式传 `--seed 42`**：默认随机会让 DVC stage 输出每次都不一样，破坏缓存
- **PR review 的另一类问题：silent failure**：原本几个错误路径只 log warning 不 raise，被 silent-failure-hunter agent 抓到，必须显式 raise 或 exit 1

## 遗留 TODO

- 真实模型还没写：Deep SVDD / VAE / OCSVM 这一层是空的
- 真实数据 loader 没写：`src/data/` 还没接入 `data/anomod/*/  _pipeline_out/tt_fused_15s.csv`
- 多模态融合层 `src/fusion/` 还是空目录
- 评估指标只有 AUROC/AUPRC，没有 per-endpoint 粒度细分、没有 inject phase 时间窗对齐评估
- contract v1 的设计：什么时候要升级？目前 v0 假设训练集和测试集都有 labels，纯推理场景需要 v1
