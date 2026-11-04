
from pathlib import Path
import csv

DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
OUTPUT_FILE = Path(r"G:\My Drive\MSC\AI\Assignment\logs\class_mapping.csv")

rows = []

for wav_path in sorted(DATASET_ROOT.rglob("*.wav")):
    relative_path = wav_path.relative_to(DATASET_ROOT)
    folders = relative_path.parts[:-1]

    main_folder = folders[0] if len(folders) >= 1 else ""
    subclass = folders[1] if len(folders) >= 2 else ""

    rows.append({
        "file_name": wav_path.name,
        "relative_path": str(relative_path),
        "main_folder": main_folder,
        "subclass_folder": subclass,
    })

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file_name",
            "relative_path",
            "main_folder",
            "subclass_folder",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

main_classes = set()
subclasses = set()

for row in rows:
    main_classes.add(row["main_folder"])

    if row["subclass_folder"]:
        subclasses.add(
            (row["main_folder"], row["subclass_folder"])
        )

print("\nDATASET CLASS MAPPING REPORT")
print("-" * 35)
print(f"Total WAV files: {len(rows)}")
print(f"Top-level folders: {len(main_classes)}")
print(f"Nested subclass folders: {len(subclasses)}")
print(f"Output file: {OUTPUT_FILE}")

print("\nSUBCLASS COUNTS BY MAIN FOLDER")

for main_folder in sorted(main_classes):
    count = sum(
        1 for row in rows
        if row["main_folder"] == main_folder
        and row["subclass_folder"]
    )
    if count:
        print(f"{main_folder}: {count} subclass folders")
