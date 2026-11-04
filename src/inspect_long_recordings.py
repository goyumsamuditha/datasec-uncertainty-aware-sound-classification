
from pathlib import Path
import pandas as pd

LOG_DIR = Path(r"G:\My Drive\MSC\AI\Assignment\logs")

df = pd.read_csv(LOG_DIR / "audio_duration_analysis.csv")

# Sort by duration, longest first
longest = df.sort_values(
    "duration_seconds",
    ascending=False
)

print("\nTOP 30 LONGEST AUDIO RECORDINGS")
print("=" * 90)

print(
    longest[
        ["file_name", "main_class", "subclass", "duration_seconds"]
    ].head(30).to_string(index=False)
)

# Count recordings longer than selected thresholds
print("\nLONG RECORDING COUNTS BY CLASS")
print("=" * 90)

for threshold in [30, 60, 120, 180]:
    selected = df[df["duration_seconds"] > threshold]

    print(f"\nRecordings longer than {threshold} seconds: {len(selected)}")

    print(
        selected["main_class"]
        .value_counts()
        .head(10)
        .to_string()
    )

# Save the longest recordings
output_file = LOG_DIR / "long_recordings_inspection.csv"

longest.head(100).to_csv(output_file, index=False)

print(f"\nSaved longest recordings to: {output_file}")
