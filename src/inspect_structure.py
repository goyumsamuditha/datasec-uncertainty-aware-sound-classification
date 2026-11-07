from common import DATASET_ROOT

from pathlib import Path
from collections import Counter

DATASET_ROOT = Path(DATASET_ROOT)

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f"Dataset not found: {DATASET_ROOT}")

print("Dataset root:", DATASET_ROOT)
print("\nTop-level contents:")

for item in sorted(DATASET_ROOT.iterdir(), key=lambda p: p.name.lower()):
    if item.is_dir():
        wav_count = sum(
            1 for p in item.rglob("*")
            if p.is_file() and p.suffix.lower() == ".wav"
        )
        subfolders = [
            p.name for p in item.iterdir() if p.is_dir()
        ]
        print(
            f"FOLDER: {item.name} | WAV files: {wav_count} "
            f"| Subfolders: {subfolders[:10]}"
        )
    else:
        print(f"FILE: {item.name}")

print("\nMetadata files:")
for p in DATASET_ROOT.rglob("*"):
    if p.is_file() and p.suffix.lower() in {
        ".csv", ".tsv", ".json", ".xlsx", ".xls",
        ".txt", ".xml", ".yaml", ".yml"
    }:
        print(p.relative_to(DATASET_ROOT))
