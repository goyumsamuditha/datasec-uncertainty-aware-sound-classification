
import pandas as pd
import numpy as np
import soundfile as sf
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

MANIFEST_FILE = PROJECT_DIR / "logs" / "final_split_manifest.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "training_audio_eda.csv"
SUMMARY_FILE = PROJECT_DIR / "logs" / "training_audio_eda_summary.csv"

SILENCE_THRESHOLD = 0.001

manifest = pd.read_csv(MANIFEST_FILE)
train_df = manifest[manifest["partition"] == "train"].copy()

results = []

for _, row in tqdm(
    train_df.iterrows(),
    total=len(train_df),
    desc="Analysing training audio"
):
    audio_path = DATASET_DIR / row["relative_path"]

    try:
        audio, sample_rate = sf.read(
            audio_path,
            dtype="float32"
        )

        if audio.ndim == 2:
            audio = audio.mean(axis=1)

        audio = np.asarray(audio)

        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))

        silence_proportion = float(
            np.mean(np.abs(audio) < SILENCE_THRESHOLD)
        )

        duration = len(audio) / sample_rate

        results.append({
            "relative_path": row["relative_path"],
            "main_class": row["main_class"],
            "duration_seconds": duration,
            "sample_rate": sample_rate,
            "rms_energy": rms,
            "peak_amplitude": peak,
            "silence_proportion": silence_proportion,
            "read_status": "success"
        })

    except Exception as error:
        results.append({
            "relative_path": row["relative_path"],
            "main_class": row["main_class"],
            "duration_seconds": np.nan,
            "sample_rate": np.nan,
            "rms_energy": np.nan,
            "peak_amplitude": np.nan,
            "silence_proportion": np.nan,
            "read_status": str(error)
        })

eda = pd.DataFrame(results)

eda.to_csv(OUTPUT_FILE, index=False)

successful = eda[eda["read_status"] == "success"]

summary = (
    successful.groupby("main_class")
    .agg(
        file_count=("relative_path", "count"),
        mean_duration=("duration_seconds", "mean"),
        median_duration=("duration_seconds", "median"),
        mean_rms=("rms_energy", "mean"),
        median_rms=("rms_energy", "median"),
        mean_peak=("peak_amplitude", "mean"),
        mean_silence=("silence_proportion", "mean")
    )
    .round(5)
)

summary.to_csv(SUMMARY_FILE)

print("\nTRAINING AUDIO EDA")
print("==================")
print(f"Training files: {len(train_df)}")
print(f"Successfully analysed: {len(successful)}")
print(f"Failed: {len(train_df) - len(successful)}")

print("\nSample-rate distribution:")
print(successful["sample_rate"].value_counts().to_string())

print("\nOverall statistics:")
print(
    successful[
        [
            "duration_seconds",
            "rms_energy",
            "peak_amplitude",
            "silence_proportion"
        ]
    ].describe().round(5).to_string()
)

print("\nClass-level statistics:")
print(summary.to_string())

print(f"\nDetailed results: {OUTPUT_FILE}")
print(f"Class summary: {SUMMARY_FILE}")
