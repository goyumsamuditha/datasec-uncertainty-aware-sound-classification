
from pathlib import Path
import soundfile as sf
import pandas as pd
import matplotlib.pyplot as plt

DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
LOG_DIR = Path(r"G:\My Drive\MSC\AI\Assignment\logs")
REPORT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment\report")

records = []

for wav_path in DATASET_ROOT.rglob("*.wav"):
    info = sf.info(str(wav_path))
    duration = info.frames / info.samplerate

    folders = wav_path.relative_to(DATASET_ROOT).parts[:-1]

    records.append({
        "file_name": wav_path.name,
        "main_class": folders[0],
        "subclass": folders[1] if len(folders) > 1 else "No subclass",
        "duration_seconds": duration
    })

df = pd.DataFrame(records)

# Summary statistics
print("\nAUDIO DURATION SUMMARY")
print("=" * 50)
print(df["duration_seconds"].describe().round(3))

# Duration categories
bins = [0, 1, 5, 10, 30, 60, float("inf")]
labels = ["Under 1s", "1-5s", "5-10s", "10-30s", "30-60s", "Over 60s"]

df["duration_group"] = pd.cut(
    df["duration_seconds"],
    bins=bins,
    labels=labels,
    right=False
)

print("\nDURATION GROUP COUNTS")
print(df["duration_group"].value_counts(sort=False))

# Save report
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

df.to_csv(LOG_DIR / "audio_duration_analysis.csv", index=False)

# Plot histogram
plt.figure(figsize=(10, 6))
plt.hist(df["duration_seconds"], bins=40)
plt.xlabel("Audio duration (seconds)")
plt.ylabel("Number of recordings")
plt.title("DataSEC V4 - Audio Duration Distribution")
plt.tight_layout()

plt.savefig(
    REPORT_DIR / "audio_duration_distribution.png",
    dpi=300
)

plt.show()

print("\nDuration report and chart saved.")
