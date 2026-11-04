
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")

INPUT_FILE = PROJECT_DIR / "logs" / "training_audio_eda.csv"
OUTPUT_DIR = PROJECT_DIR / "report"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT_FILE)
df = df[df["read_status"] == "success"]

# 1. Duration distribution
plt.figure(figsize=(10, 6))
plt.hist(df["duration_seconds"], bins=50)
plt.xlabel("Duration (seconds)")
plt.ylabel("Number of recordings")
plt.title("Training Audio Duration Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_duration_distribution.png", dpi=300)
plt.close()

# 2. Duration distribution with logarithmic x-axis
plt.figure(figsize=(10, 6))
plt.hist(df["duration_seconds"], bins=50)
plt.xscale("log")
plt.xlabel("Duration (seconds, logarithmic scale)")
plt.ylabel("Number of recordings")
plt.title("Training Audio Duration Distribution (Log Scale)")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_duration_log_distribution.png", dpi=300)
plt.close()

# 3. RMS energy distribution
plt.figure(figsize=(10, 6))
plt.hist(df["rms_energy"], bins=50)
plt.xlabel("RMS Energy")
plt.ylabel("Number of recordings")
plt.title("Training Audio RMS Energy Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_rms_distribution.png", dpi=300)
plt.close()

# 4. Low-amplitude proportion distribution
plt.figure(figsize=(10, 6))
plt.hist(df["silence_proportion"], bins=50)
plt.xlabel("Low-amplitude Sample Proportion")
plt.ylabel("Number of recordings")
plt.title("Training Audio Low-Amplitude Distribution")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_low_amplitude_distribution.png", dpi=300)
plt.close()

# 5. Duration by class
plt.figure(figsize=(13, 8))
df.boxplot(
    column="duration_seconds",
    by="main_class",
    rot=90
)
plt.yscale("log")
plt.xlabel("Main class")
plt.ylabel("Duration (seconds, logarithmic scale)")
plt.title("Training Audio Duration by Class")
plt.suptitle("")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_duration_by_class.png", dpi=300)
plt.close()

print("EDA visualisations created successfully:")
for filename in [
    "training_duration_distribution.png",
    "training_duration_log_distribution.png",
    "training_rms_distribution.png",
    "training_low_amplitude_distribution.png",
    "training_duration_by_class.png"
]:
    print(OUTPUT_DIR / filename)
