
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
LOG_FILE = PROJECT_DIR / "logs" / "experiment_log.csv"

df = pd.read_csv(LOG_FILE)

if "experiment_id" not in df.columns:
    raise ValueError("experiment_id column not found")

mask = df["experiment_id"].astype(str) == "SIM-002"

if not mask.any():
    raise ValueError("SIM-002 was not found")

updates = {
    "project_week": "Week 1",
    "dataset_sample": "13 manually reviewed candidate pairs",
    "notes": (
        "CLAP similarity is useful for candidate screening but does not "
        "establish recording provenance. Five pairs with similarity >= "
        "0.85 were judged Different. Do not automatically create source "
        "groups using CLAP similarity. Keep confirmed exact duplicates "
        "together in the same split. Original source identifiers are "
        "unavailable."
    ),
}

for column, value in updates.items():
    if column in df.columns:
        df.loc[mask, column] = value
    else:
        print(f"Warning: column '{column}' not found")

df.to_csv(LOG_FILE, index=False)

print("SIM-002 log entry updated.")
print(df.loc[mask].to_string(index=False))
