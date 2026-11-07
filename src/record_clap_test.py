from common import PROJECT_ROOT

import csv
from pathlib import Path
from datetime import datetime

# Project paths
PROJECT_DIR = Path(PROJECT_ROOT)
LOG_DIR = PROJECT_DIR / "logs"
LOG_FILE = LOG_DIR / "experiment_log.csv"

# Create logs directory if necessary
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Define consistent experiment-log columns
columns = [
    "experiment_id",
    "date",
    "project_week",
    "experiment_name",
    "model",
    "dataset_sample",
    "device",
    "status",
    "result",
    "notes"
]

# Create the file with headers if it does not exist
if not LOG_FILE.exists():
    with LOG_FILE.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=columns)
        writer.writeheader()

# Record the successful CLAP compatibility test
experiment = {
    "experiment_id": "CLAP-001",
    "date": datetime.now().strftime("%Y-%m-%d"),
    "project_week": "Week 1",
    "experiment_name": "LAION-CLAP audio compatibility test",
    "model": "LAION-CLAP 630k-audioset-best.pt",
    "dataset_sample": "Bells-0001.wav",
    "device": "CUDA",
    "status": "Success",
    "result": "Audio embedding generated with shape (1, 512)",
    "notes": "Technical compatibility confirmed; classification performance not evaluated."
}

# Append without overwriting existing records
with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=columns)
    writer.writerow(experiment)

print(f"Experiment recorded successfully: {LOG_FILE}")
