import csv
import json
import os

for run_name in os.listdir(os.getcwd()):
    if not os.path.isdir(run_name):
        continue
    if not run_name.startswith("light-oauth2"):
        continue
    for test_folder in os.listdir(run_name):
        test_path = os.path.join(run_name, test_folder)
        test_csv = run_name + "-" + test_folder + ".csv"
        if os.path.isfile(test_csv):
            print(f"{test_csv} exists, skipping...")
            continue
        if not os.path.isdir(test_path):
            continue
        METRICS = dict()
        metrics = os.path.join(test_path, "metrics")
        for file in os.listdir(metrics):
            if os.path.splitext(file)[1] != ".json":
                continue
            file = os.path.join(metrics, file)
            with open(file, "r") as f:
                try:
                    data = json.load(f)
                except:
                    print(f"Error when loading JSON from {file}")
                    continue
            for metric in data:
                if metric["metric"]["job"] != "node":
                    continue
                metric_name = (
                    metric["metric"]["__name__"]
                    + "&"
                    + "&".join(
                        [
                            f"{k}={v}"
                            for k, v in metric["metric"].items()
                            if k not in {"group", "instance", "job", "__name__"}
                        ]
                    )
                )
                metric_name = metric_name.removesuffix("&")
                for timestamp, value in metric["values"]:
                    metric_values = METRICS.setdefault(timestamp, dict())
                    metric_values[metric_name] = value

        all_metrics = set()
        for timestamp, metrics in METRICS.items():
            all_metrics = all_metrics.union(metrics.keys())
        all_metrics = sorted(list(all_metrics))
        HEADER = ["timestamp", "run_end", "test_name"]
        HEADER.extend(all_metrics)

        output_folder = "./metric_csvs/"
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
            print(f"Created folder: {output_folder}")

        print(f"Writing to {test_csv}...")
        with open(os.path.join(output_folder, test_csv), "w", newline="") as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(HEADER)
            for timestamp, metrics in METRICS.items():
                row = [timestamp, run_name, test_folder]
                for metric in all_metrics:
                    row.append(metrics.get(metric, None))
                csv_writer.writerow(row)
