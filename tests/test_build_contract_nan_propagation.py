"""Pipeline 级回归测试：Normal fit 集合里某 service 的 metric 全 NaN（真实数据采集
缺口，如 cAdvisor 掉线）时，不应污染其他 case 里同一 service 本来完好的真实数值。

对应 history/entries/007-fix-normalizer-nan-propagation.md 记录的 bug：修复前
Normalizer.transform() 会把 fit 阶段算出的 [nan, nan] 统计量通过减法/除法应用到全量
数据，导致 27 个数据完好的 case 也被一起抹成 NaN。这个 bug 只在多 case 真实 dvc repro
时才暴露，单元测试对 Normalizer 隔离测试不会覆盖到 build_contract 这一层的实际影响。
"""

import subprocess
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parents[1]
FIXTURE_DATASET = REPO_ROOT / "tests/fixtures/nan_propagation_mini.yaml"
CONFIG = REPO_ROOT / "configs/contract/v0.yaml"


def _run_build(out_dir: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            "scripts/build_contract.py",
            "--config",
            str(CONFIG),
            "--dataset",
            str(FIXTURE_DATASET),
            "--out-dir",
            str(out_dir),
            "--seed",
            "42",
        ],
        check=True,
        cwd=str(REPO_ROOT),
    )


def test_other_case_metric_values_survive_normal_all_nan_group(tmp_path):
    """fixture：Normal 在 ts-travel-service 上完全没有 metric_data（模拟采集缺口），
    Lv_NanTest_anomaly 在同一 service 上有真实 cpu_usage_rate 数据。
    修复前：Normalizer fit 出 [nan, nan] 后 transform 全量数据，
            Lv_NanTest_anomaly 的 cpu_usage_rate 也会被抹成 NaN。
    修复后：该 group 跳过归一化保留原值，Lv_NanTest_anomaly 的数值应为真实原始值。
    """
    out_dir = tmp_path / "contract_v0"
    _run_build(out_dir)

    eval_all = pd.read_parquet(out_dir / "eval_all.parquet")

    normal_rows = eval_all[eval_all["case_id"] == "Normal"]
    anomaly_rows = eval_all[eval_all["case_id"] == "Lv_NanTest_anomaly"]

    assert not anomaly_rows.empty, "fixture 未产出 Lv_NanTest_anomaly case，检查 fixture 是否损坏"

    # Normal 自身的采集缺口是真实数据问题，代码层面理应保持 NaN（不是本次修复范畴）
    assert normal_rows["service_metric__cpu_usage_rate"].isna().all()

    # 核心断言：其他 case 的真实 metric 数值不能被 Normal 的缺口传染成 NaN
    assert not anomaly_rows["service_metric__cpu_usage_rate"].isna().any()
    assert (anomaly_rows["service_metric__cpu_usage_rate"] >= 0).all()


def test_normalization_stats_json_records_all_nan_group(tmp_path):
    """normalization_stats.json 里全 NaN 的 group 应如实记录为 NaN（fit 阶段确实拿不到
    统计量），不应被静默过滤掉——跳过归一化的判断放在 transform 端，不是 fit 端。"""
    import json

    out_dir = tmp_path / "contract_v0"
    _run_build(out_dir)

    stats = json.loads((out_dir / "normalization_stats.json").read_text())
    cpu_stats = stats["service_metric__cpu_usage_rate"]["by_group"]["ts-travel-service"]
    assert all(v != v for v in cpu_stats)  # NaN != NaN
