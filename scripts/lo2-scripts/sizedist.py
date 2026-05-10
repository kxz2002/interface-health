import math
import os
from collections import defaultdict
from statistics import mean, median, stdev

import humanize
import matplotlib.pyplot as plt


def collect_file_sizes(directory, case_filter="both"):
    size_distributions = defaultdict(list)

    root_items = [
        item
        for item in os.listdir(directory)
        if os.path.isdir(os.path.join(directory, item))
    ]
    current_root = None

    for root, dirs, files in os.walk(directory):
        top_dir = root.replace(directory, "").strip(os.sep).split(os.sep)[0]
        if top_dir in root_items and current_root != top_dir:
            current_root = top_dir
            print(f"Processing: {top_dir}")
        if "light-oauth2-data-" not in root:  # avoid other folders
            continue
        if case_filter == "correct" and "correct" not in root:
            continue
        if case_filter == "error" and "correct" in root:
            continue

        for file in files:
            if file.startswith("metric_") or file == "last_fetch_time.txt":
                continue

            filepath = os.path.join(root, file)
            if os.path.isfile(filepath):
                size_kb = os.path.getsize(filepath) / 1024
                size_distributions[file].append(size_kb)

    return size_distributions


def plot_distributions(distributions):
    # Filter out distributions with less than 2 points
    valid_distributions = {k: v for k, v in distributions.items() if len(v) > 1}
    plt.rcParams.update({"font.size": 14})  # Increases base font size

    n_plots = len(valid_distributions)
    plots_per_row = 3
    n_rows = math.ceil(n_plots / plots_per_row)

    # Create figures, 5 plots per row
    for row in range(n_rows):
        start_idx = row * plots_per_row
        end_idx = min((row + 1) * plots_per_row, n_plots)
        current_row_items = list(valid_distributions.items())[start_idx:end_idx]

        # Create a figure for this row
        fig, axes = plt.subplots(1, len(current_row_items), figsize=(18, 6))
        fig.suptitle(f"File Size Distributions - Row {row + 1}", y=1.05)

        # Make axes iterable even if there's only one plot
        if len(current_row_items) == 1:
            axes = [axes]

        # Create histograms
        for (filename, sizes), ax in zip(current_row_items, axes):
            ax.hist(sizes, bins="auto", alpha=0.7, color="blue")
            ax.set_yscale("log")
            ax.set_title(filename)
            ax.set_xlabel("Size (KB)")
            ax.set_ylabel("Count")

            # Rotate x-axis labels for better readability
            ax.tick_params(axis="x", rotation=45)

            # Add mean and median lines
            mean_val = mean(sizes)
            median_val = median(sizes)
            ax.axvline(
                mean_val, color="red", linestyle="--", label=f"Mean: {mean_val:.1f}KB"
            )
            ax.axvline(
                median_val,
                color="green",
                linestyle="--",
                label=f"Median: {median_val:.1f}KB",
            )
            ax.legend(fontsize="small")

        # Adjust layout to prevent overlap
        plt.tight_layout()

    return n_rows


def analyze_distributions(distributions):
    analysis = {}
    for filename, sizes in distributions.items():
        if sizes:
            analysis[filename] = {
                "count": len(sizes),
                "mean": mean(sizes),
                "median": median(sizes),
                "std_dev": stdev(sizes) if len(sizes) > 1 else 0,
                "min": min(sizes),
                "max": max(sizes),
            }
    return analysis


def print_analysis(analysis):
    sorted_items = sorted(analysis.items(), key=lambda x: x[1]["count"], reverse=True)

    print("\nFile Size Distribution Analysis:")
    print("-" * 100)
    print(
        f"{'Filename':<40} {'Count':>8} {'Mean':>12} {'Median':>12} {'Std Dev':>12} {'Min':>12} {'Max':>12}"
    )
    print("-" * 100)

    for filename, stats in sorted_items:
        print(
            f"{filename[:39]:<40} "
            f"{stats['count']:>8} "
            f"{humanize.naturalsize(stats['mean']*1024):>12} "
            f"{humanize.naturalsize(stats['median']*1024):>12} "
            f"{humanize.naturalsize(stats['std_dev']*1024):>12} "
            f"{humanize.naturalsize(stats['min']*1024):>12} "
            f"{humanize.naturalsize(stats['max']*1024):>12}"
        )

    print("\nSummary:")
    print(f"Total unique filenames: {len(analysis)}")
    total_files = sum(stats["count"] for stats in analysis.values())
    print(f"Total files processed: {total_files}")


directory_path = "."  # Current directory, change this to your desired path
print(f"Starting scan in: {os.path.abspath(directory_path)}")
distributions = collect_file_sizes(directory_path, case_filter="error")

n_rows = plot_distributions(distributions)
print(f"\nCreated {n_rows} figures with histograms (5 plots per row)")

analysis = analyze_distributions(distributions)
print_analysis(analysis)

plt.show()
