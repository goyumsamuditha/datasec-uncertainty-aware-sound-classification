from pathlib import Path
import pandas as pd
import torch

from train_cnn14 import DataSECDataset, NUM_SAMPLES

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
MANIFEST_PATH = (
    PROJECT_DIR / "logs"
    / "final_split_manifest_with_subclasses.csv"
)

df = pd.read_csv(MANIFEST_PATH)

# Test the problematic recording first.
target = df[
    df["relative_path"].str.endswith(
        "Airplanes-0042.wav"
    )
]

if target.empty:
    raise RuntimeError("Test recording not found in manifest.")

dataset = DataSECDataset(target, training=True)

for i in range(10):
    waveform = dataset[0] if False else dataset.load_audio(
        PROJECT_DIR.parent / "DATASEC" / "DATASEC"
        / target.iloc[0]["relative_path"]
    )

    waveform = dataset.crop_or_pad(waveform)

    assert waveform.shape == (1, NUM_SAMPLES), (
        f"Unexpected shape: {waveform.shape}"
    )
    assert torch.isfinite(waveform).all()

print("Audio preprocessing test PASSED.")
print("Recording: Airplanes-0042.wav")
print("Output shape:", tuple(waveform.shape))
print("Expected shape:", (1, NUM_SAMPLES))