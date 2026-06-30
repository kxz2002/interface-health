"""Metric chunked reading 性能 spike。只用于验证 4GB 上限，不进 src。

metric_data CSV 的 `timestamp` 列是秒级 unix 时间戳（见 dataset-guide.md 4.3），
而 window 边界 (inject_start_ms 等) 是 epoch-ms。比较前需把秒转成毫秒，否则单位错配。
"""

import argparse
import json
import platform
import resource
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.csv as pacsv

TS_COL = "timestamp"  # 秒级 unix 时间戳


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--window-start-ms", type=int, required=True)
    parser.add_argument("--window-end-ms", type=int, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    rows_kept = 0
    # 只读 timestamp 列：稀疏的 k8s label 列类型推断不稳定（空块推成 null 再遇字符串报错），
    # 且列裁剪本身能压低峰值内存。
    reader = pacsv.open_csv(
        args.input,
        read_options=pacsv.ReadOptions(block_size=64 << 20),  # 64MB chunks
        convert_options=pacsv.ConvertOptions(include_columns=[TS_COL]),
    )
    for batch in reader:
        ts_ms = pc.multiply(pc.cast(batch[TS_COL], "int64"), 1000)
        mask = pc.and_(
            pc.greater_equal(ts_ms, args.window_start_ms),
            pc.less_equal(ts_ms, args.window_end_ms),
        )
        rows_kept += int(pc.sum(mask).as_py() or 0)

    peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux: ru_maxrss 单位是 KB；macOS: bytes
    peak_mb = peak_kb / 1024 if platform.system() == "Linux" else peak_kb / (1024 * 1024)

    Path(args.out).write_text(
        json.dumps(
            {
                "peak_memory_mb": peak_mb,
                "rows_kept": rows_kept,
                "input": args.input,
            }
        )
    )


if __name__ == "__main__":
    main()
