import os

import polars as pl

# Needs to have csv_generator.py run first

for run_name in os.listdir(os.getcwd()):
    if not os.path.isdir(run_name) or not run_name.startswith("light-oauth2"):
        continue
    run_dfs = []
    for test_name in os.listdir(run_name):
        if not os.path.isdir(os.path.join(run_name, test_name)):
            continue
        run_dfs.append(f"./metric_csvs/{run_name}-{test_name}.csv")
        print(f"Added {run_name}-{test_name}.csv")
    result = pl.scan_csv(run_dfs, infer_schema=False)
    result = result.sort(by=["timestamp"])
    print(f"Saving {run_name}.csv")
    result.sink_csv(f"{run_name}.csv", include_header=True)
