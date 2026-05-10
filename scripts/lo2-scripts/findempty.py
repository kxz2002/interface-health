import os
from collections import Counter

# List of expected empty files
EXPECTED_EMPTY = {
    "jaeger-container.log",
    "light-oauth2-mysqldb-1.log",
    "light-oauth2-node_exporter-1.log",
    "light-oauth2-prometheus-1.log",
    "traces_jaeger-all-in-one.csv",
    "traces_oauth2-client-service.csv",
    "traces_oauth2-code-service.csv",
    "traces_oauth2-key-service.csv",
    "traces_oauth2-refresh-token-service.csv",
    "traces_oauth2-service-service.csv",
    "traces_oauth2-token-service.csv",
    "traces_oauth2-user-service.csv",
}


def is_empty_file(filepath):
    size = os.path.getsize(filepath)
    if size == 0:
        return True
    if filepath.lower().endswith(".csv") and size == 58:
        return True
    return False


def is_empty_dir(dirpath):
    for root, dirs, files in os.walk(dirpath):
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath) and not is_empty_file(filepath):
                return False
    return True


def find_empty_items(directory):
    empty_items = []
    unexpected_empty = []

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

        # Check files
        for file in files:
            filepath = os.path.join(root, file)
            if os.path.isfile(filepath) and is_empty_file(filepath):
                empty_items.append(("file", file))
                # Check if this empty file is unexpected
                if file not in EXPECTED_EMPTY:
                    unexpected_empty.append(filepath)

        # Check directories
        for dir in dirs:
            if not dir.startswith("."):
                dirpath = os.path.join(root, dir)
                if is_empty_dir(dirpath):
                    empty_items.append(("dir", dir))
                    # All empty directories are unexpected
                    unexpected_empty.append(dirpath + " (directory)")

    # Print unexpected empty items first
    if unexpected_empty:
        print(
            "\nUnexpected empty items found (including empty folders or folders that only include empty files):"
        )
        print("-" * 60)
        for path in sorted(unexpected_empty):
            print(path)

    # Create and print frequency table
    freq_table = Counter(name for _, name in empty_items)
    if freq_table:
        print("\nFrequency table of empty items:")
        print("Name".ljust(40), "Count", "Type")
        print("-" * 60)
        for name, count in sorted(freq_table.items()):
            types = set(type for type, n in empty_items if n == name)
            type_str = "/".join(types)
            print(f"{name[:39].ljust(40)} {str(count).ljust(5)} {type_str}")

        print(f"\nTotal unique empty items: {len(freq_table)}")
        print(f"Total empty items: {sum(freq_table.values())}")

        # Print breakdown by type
        file_count = sum(1 for type, _ in empty_items if type == "file")
        dir_count = sum(1 for type, _ in empty_items if type == "dir")
        csv_count = sum(
            1
            for type, name in empty_items
            if type == "file" and name.lower().endswith(".csv")
        )
        expected_count = sum(
            count for name, count in freq_table.items() if name in EXPECTED_EMPTY
        )

        print("\nBreakdown:")
        print(f"Empty directories: {dir_count}")
        print(f"Empty CSV files (0 or 58 bytes): {csv_count}")
        print(f"Other empty files (0 bytes): {file_count - csv_count}")
        print(f"Expected empty files: {expected_count}")
        print(f"Unexpected empty items: {len(unexpected_empty)}")
    else:
        print("No empty items found.")


# Example usage
directory_path = "."  # Current directory, change this to your desired path
print(f"Starting scan in: {os.path.abspath(directory_path)}")
find_empty_items(directory_path)
