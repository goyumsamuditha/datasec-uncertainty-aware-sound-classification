
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

INPUT_FILE = PROJECT_DIR / "logs" / "waveform_duplicates.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "exact_duplicate_groups.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Duplicate report not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = {"number_of_files", "files"}
missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

records = []
missing_files = []

for index, row in df.iterrows():
    group_id = f"EXACT-{index + 1:03d}"

    file_paths = [
        item.strip()
        for item in str(row["files"]).split("|")
    ]

    if len(file_paths) != int(row["number_of_files"]):
        raise ValueError(
            f"File count mismatch in duplicate group {group_id}"
        )

    for relative_path in file_paths:
        full_path = DATASET_DIR / relative_path

        if not full_path.is_file():
            missing_files.append(str(full_path))
            continue

        records.append({
            "group_id": group_id,
            "relative_path": relative_path,
            "duplicate_type": "Exact waveform duplicate",
            "sample_rate": row["sample_rate"],
            "audio_shape": row["audio_shape"],
        })

if missing_files:
    print("ERROR: Some files could not be found:")
    for path in missing_files:
        print(path)
    raise SystemExit("Manifest was not created.")

manifest = pd.DataFrame(records)

manifest.to_csv(OUTPUT_FILE, index=False)

print("\nEXACT DUPLICATE MANIFEST CREATED")
print("================================")
print(f"Duplicate groups: {manifest['group_id'].nunique()}")
print(f"Files included: {len(manifest)}")
print(f"Output: {OUTPUT_FILE}")

print("\nManifest contents:")
print(manifest.to_string(index=False))
