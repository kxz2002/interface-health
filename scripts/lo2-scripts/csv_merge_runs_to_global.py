import os

import polars as pl

run_dfs = []
for run_name in os.listdir(os.getcwd()):
    if not os.path.isdir(run_name) or not run_name.startswith("light-oauth2"):
        continue
    run_dfs.append(pl.scan_csv(f"{run_name}.csv", infer_schema=False))
    print(f"Added {run_name}.csv")

print("Scanning csv data")
result = pl.concat(run_dfs, how="vertical")
print("Sorting all data by timestamp")
result = result.sort(by=["timestamp"])
print("Saving light-oauth2-data.csv")
result.sink_csv("light-oauth2-data.csv", include_header=True)
