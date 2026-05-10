import os

logs = [
    "light-oauth2-oauth2-client-1.log",
    "light-oauth2-oauth2-code-1.log",
    "light-oauth2-oauth2-key-1.log",
    "light-oauth2-oauth2-refresh-token-1.log",
    "light-oauth2-oauth2-service-1.log",
    "light-oauth2-oauth2-token-1.log",
    "light-oauth2-oauth2-user-1.log",
]


# Function to trim a log file
def trim_file(file_path, reduce_to_sixth):
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return
    with open(file_path, "r") as file:
        lines = file.readlines()[100:]
    if reduce_to_sixth:
        lines = lines[: len(lines) // 7.05]
    with open(file_path, "w") as file:
        file.writelines(lines)


# Walk through the directory tree to find and process log files
def process_log_files(base_directory):
    """Traverse the directory tree, locate log files, and apply trimming."""
    for root, _, files in os.walk(base_directory):
        if "metrics" in root:
            continue
        for file_name in files:
            if file_name in logs:
                file_path = os.path.join(root, file_name)
                print(f"Processing file: {file_path}")
                trim_file(file_path, reduce_to_sixth=("correct" in root))


# Base directory (current directory by default)
base_directory = "."  # Adjust as needed

# Call the function to process all log files two folders down
process_log_files(base_directory)
