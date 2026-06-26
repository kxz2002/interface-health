"""Perf spike：在最大的 metric_data CSV 上跑 chunked reading，验证峰值内存 < 4GB。"""

import json
import subprocess
import sys
from pathlib import Path

import pytest


def _largest_metric_csv() -> Path:
    candidates = list(Path("data/anomod").glob("*/metric_data/*.csv"))
    if not candidates:
        pytest.skip("没有 metric_data CSV，跳过 perf spike")
    return max(candidates, key=lambda p: p.stat().st_size)


@pytest.mark.slow
def test_metric_chunked_reading_under_4gb():
    csv = _largest_metric_csv()
    out = Path("artifacts/_spike_metric.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.unlink(missing_ok=True)
    subprocess.run(
        [
            sys.executable,
            "scripts/spikes/metric_perf_spike.py",
            "--input",
            str(csv),
            "--window-start-ms",
            "0",
            "--window-end-ms",
            "9999999999999",
            "--out",
            str(out),
        ],
        check=True,
    )
    report = json.loads(out.read_text())
    assert (
        report["peak_memory_mb"] < 4096
    ), f"chunked reading 峰值 {report['peak_memory_mb']:.1f} MB 超过 4GB 上限"
    assert report["rows_kept"] > 0
