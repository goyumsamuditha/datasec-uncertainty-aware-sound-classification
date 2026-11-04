
from pathlib import Path
from collections import Counter
import csv

# Change this to the extracted DATASEC folder on your computer.
DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

OUTPUT_DIR = Path("logs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Find all WAV files recursively.
wav_files = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() == ".wav"
)

print(f"Dataset folder: {DATASET_ROOT}")
print(f"Folder exists: {DATASET_ROOT.exists()}")
print(f"Total WAV files: {len(wav_files)}")

if not DATASET_ROOT.exists():
    raise FileNotFoundError(
        "Update DATASET_ROOT to the actual extracted DATASEC folder."
    )

# Count WAV files by their immediate parent folder.
class_counts = Counter(p.parent.name for p in wav_files)

print("\nWAV files by immediate parent folder:")
for class_name, count in sorted(class_counts.items()):
    print(f"{class_name}: {count}")

# Save an inventory so we can inspect the actual dataset structure.
inventory_path = OUTPUT_DIR / "dataset_inventory.csv"

with inventory_path.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["relative_path", "parent_folder", "file_size_bytes"])

    for wav in wav_files:
        writer.writerow([
            str(wav.relative_to(DATASET_ROOT)),
            wav.parent.name,
            wav.stat().st_size,
        ])

print(f"\nInventory saved to: {inventory_path.resolve()}")
