from pathlib import Path
import soundfile as sf

DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

files = [
    DATASET_DIR / "Vehicle pass-by" / "Car pass-by" / "Car pass-by-0020.wav",
    DATASET_DIR / "Vehicle pass-by" / "Car pass-by" / "Car pass-by-0051.wav",
]

for path in files:
    audio, sr = sf.read(path)

    print(f"\nFile: {path.name}")
    print(f"Sample rate: {sr}")
    print(f"Duration: {len(audio) / sr:.3f} seconds")
    print(f"Samples: {len(audio)}")
    print(f"Channels: {1 if audio.ndim == 1 else audio.shape[1]}")