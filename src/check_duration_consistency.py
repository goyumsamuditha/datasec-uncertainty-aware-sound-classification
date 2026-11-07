from common import PROJECT_ROOT

from pathlib import Path
import pandas as pd

LOG_DIR = Path(PROJECT_ROOT / 'logs')

df = pd.read_csv(LOG_DIR / "audio_duration_analysis.csv")

duration = df["duration_seconds"]

print("\nDURATION CONSISTENCY CHECK")
print("=" * 50)

print(f"Total recordings: {len(df)}")

print(f"Duration >= 30 seconds: {(duration >= 30).sum()}")
print(f"Duration > 30 seconds: {(duration > 30).sum()}")
print(f"Duration >= 60 seconds: {(duration >= 60).sum()}")
print(f"Duration > 60 seconds: {(duration > 60).sum()}")

print("\nRECALCULATED DURATION GROUPS")

bins = [0, 1, 5, 10, 30, 60, float("inf")]
labels = [
    "Under 1s",
    "1-5s",
    "5-10s",
    "10-30s",
    "30-60s",
    "60s and above"
]

groups = pd.cut(
    duration,
    bins=bins,
    labels=labels,
    right=False
)

print(groups.value_counts(sort=False))

print("\nRECORDINGS AROUND 30 AND 60 SECONDS")

boundary_records = df[
    duration.between(29.5, 30.5)
    | duration.between(59.5, 60.5)
]

print(
    boundary_records[
        ["file_name", "main_class", "duration_seconds"]
    ].to_string(index=False)
)
