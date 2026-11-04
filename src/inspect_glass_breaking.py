
from pathlib import Path
import pandas as pd

LOG_DIR = Path(r"G:\My Drive\MSC\AI\Assignment\logs")

df = pd.read_csv(LOG_DIR / "audio_duration_analysis.csv")

glass = df[
    df["main_class"] == "Glass breaking"
].sort_values("duration_seconds")

print("\nGLASS BREAKING DURATION ANALYSIS")
print("=" * 65)

print(f"Total recordings: {len(glass)}")

print("\nDuration statistics:")
print(glass["duration_seconds"].describe().round(3))

print("\nShortest 10 recordings:")
print(
    glass[
        ["file_name", "duration_seconds"]
    ].head(10).to_string(index=False)
)

print("\nLongest 10 recordings:")
print(
    glass[
        ["file_name", "duration_seconds"]
    ].tail(10).to_string(index=False)
)

print("\nDuration groups:")
bins = [0, 5, 10, 30, 60, float("inf")]
labels = ["Under 5s", "5-10s", "10-30s", "30-60s", "60s and above"]

groups = pd.cut(
    glass["duration_seconds"],
    bins=bins,
    labels=labels,
    right=False
)

print(groups.value_counts(sort=False))

print("\nSaved analysis is based on the existing duration inventory.")
