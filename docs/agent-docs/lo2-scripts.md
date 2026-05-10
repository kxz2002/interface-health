# LO2 Dataset Scripts

来源: [LO2: Microservice Dataset of Logs and Metrics (Zenodo)](https://zenodo.org/records/14938118)

**DOI**: 10.5281/zenodo.14938118 | **Version**: v3 (2025-02-28) | **License**: CC BY 4.0

## 数据集概述

LO2 是一个微服务日志与指标数据集，完整解压后约 540 GB。包含从实验运行中收集的微服务日志和指标。

**文件列表**:

| 文件 | 大小 |
|------|------|
| `lo2-data.zip` | 46.5 GB (完整数据) |
| `lo2-sample.zip` | 1.1 GB (样本，已解压到 `data/raw/lo2-sample/`) |
| `lo2-scripts.zip` | 17.2 KB (当前目录) |
| `data-appendix.pdf` | 1.6 MB |
| `README.md` | 2.1 KB |

## 脚本说明

### 数据生成类

| 脚本 | 说明 | 依赖 |
|------|------|------|
| `csv_generator.py` | 生成 CSV 格式的指标文件 | pandas, polars |
| `csv_merge_tests_to_runs.py` | 将多个 test 的 CSV 合并为 run | - |
| `csv_merge_runs_to_global.py` | 将多个 run 的 CSV 合并为全局文件 | - |

> 需按顺序执行：`csv_generator` → `merge_tests_to_runs` → `merge_runs_to_global`。最终全局合并是内存密集型操作。

### 数据分析类

| 脚本 | 说明 | 依赖 |
|------|------|------|
| `findempty.py` | 识别文件夹中的空文件，区分预期与非预期空文件 | - |
| `logstats.py` | 统计每种日志类型的行数，用于生成数据附录图表 | matplotlib |
| `pca.py` | 主成分分析，用于数据初步探查 | scikit-learn, pandas |
| `sizedist.py` | 生成每个文件名的文件大小分布，用于数据附录 | - |

### 日志处理类

| 脚本 | 说明 | 依赖 |
|------|------|------|
| `loglead_lo2.py` | 初步日志分析，用于错误检测 | LogLead v1.2.1 |
| `reduce_logs.py` | 移除初始化行（避免泄露正确性信息），确保公平分析 | - |

## 辅助文件

| 文件 | 说明 |
|------|------|
| `node_exporter_metrics.txt` | Prometheus node_exporter 导出的指标描述文本，包含 go_*, process_*, node_* 等指标类型 |
| `requirements.txt` | Python 依赖列表 |

## 依赖安装

```bash
cd scripts/lo2-scripts
pip install -r requirements.txt
```

注意：`loglead_lo2.py` 需要 LogLead v1.2.1，需单独安装。

## 本项目中文件位置

```
scripts/lo2-scripts/          # 脚本目录
data/raw/lo2-sample/          # 已解压的样本数据
├── metrics/                  # 指标 CSV 文件
└── logs/                     # 日志文件（按错误类型子目录组织）
```
