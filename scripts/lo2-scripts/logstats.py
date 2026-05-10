import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import polars as pl


class LogAnalyzer:
    def __init__(self):
        self.patterns = {
            "DEBUG": r"DEBUG",
            "INFO": r"INFO",
            "WARN": r"WARN",
            "ERROR": r"ERROR",
            "STACKTRACE": r"at\s+[\w\.$/_-]+\.[^(]+\([^)]*\)",
            "EXCEPTION": r"[\w\.]+Exception:",
            "OTHER": r".*",
        }

        # Compile patterns once during initialization
        self.compiled_patterns = {
            name: re.compile(pattern) for name, pattern in self.patterns.items()
        }

    def process_logs(self, root_dir: str, pattern: str) -> pl.DataFrame:
        """Process log files recursively and return a DataFrame."""
        rows = []
        root_path = Path(root_dir)

        counter = 0

        # Skip metrics folder and use compiled patterns
        for file_path in root_path.rglob(f"*{pattern}*.log"):
            if "metrics" in str(file_path.parent):
                continue

            base_name = "-".join(file_path.name.split("-")[:-1])
            folder = file_path.parent.name

            # Smaller sample
            # counter += 1
            # if counter > 2000:
            #     return pl.DataFrame(rows)

            with open(file_path, "r") as f:
                for line in f:
                    if line.strip():
                        log_type = "OTHER"
                        for type_name, p in self.compiled_patterns.items():
                            if p.search(line):
                                log_type = type_name
                                break

                        # print line content if wanted
                        # if log_type == 'OTHER':
                        #     print(f"OTHER: {line.strip()}")

                        rows.append(
                            {
                                "subdirectory": folder,
                                "base_filename": base_name,
                                "log_type": log_type,
                            }
                        )

        return pl.DataFrame(rows)

    def plot_distributions(self, df: pl.DataFrame):
        """Create simple histograms using matplotlib."""
        # plt.style.use('classic')
        plt.rcParams.update(
            {
                "font.size": 12,  # Base font size
                "axes.labelsize": 12,  # Axis labels
                "axes.titlesize": 14,  # Subplot titles
                "xtick.labelsize": 11,  # X-axis tick labels
                "ytick.labelsize": 11,  # Y-axis tick labels
                "legend.fontsize": 11,  # Legend text
            }
        )
        pdf = df.to_pandas()

        # Overall distribution
        plt.figure(figsize=(10, 10))
        pdf["log_type"].value_counts().plot(kind="bar")
        plt.title("Overall Log Type Distribution")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig("overall_dist.png")
        plt.close()

        # Distribution by log type with grouped services
        plt.figure(figsize=(15, 8))

        # Get counts by service and log type and sort
        grouped_data = (
            pdf.groupby(["log_type", "base_filename"])
            .size()
            .reset_index()
            .pivot(index="log_type", columns="base_filename", values=0)
            .fillna(0)
        )

        # Sort by total counts and clean up name
        grouped_data = grouped_data.loc[
            grouped_data.sum(axis=1).sort_values(ascending=False).index
        ]
        grouped_data.columns = grouped_data.columns.str.replace(
            "light-oauth2-oauth2-", ""
        )

        # Plot grouped bars
        grouped_data.plot(kind="bar", width=0.8)
        plt.xlabel("Log Type")
        plt.ylabel("Count")
        plt.yscale("log")
        plt.yscale("function", functions=(np.sqrt, np.square))
        plt.xticks(rotation=30)
        plt.legend(
            title="Services",
            bbox_to_anchor=(0.98, 0.98),
            loc="upper right",
            ncol=2,
            labelspacing=0.2,
        )  # Default is 0.5, smaller value = less space between rows
        plt.tight_layout()
        plt.savefig("by_file_dist.pdf", bbox_inches="tight")
        plt.show()
        plt.close()


# Usage
ttime = time.time()
analyzer = LogAnalyzer()
df = analyzer.process_logs(".", "oauth2-oauth2")
print(df)
print(df["log_type"].value_counts())
print(df.group_by("base_filename").agg(pl.col("log_type").count()))
df.group_by(["base_filename", "log_type"]).count().write_csv("counts.csv")
analyzer.plot_distributions(df)
print(f"Total log entries processed: {len(df)}")
print(f"Time taken: {time.time()-ttime}")
