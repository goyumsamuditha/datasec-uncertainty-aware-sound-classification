
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

DUPLICATE_FILE = PROJECT_DIR / "logs" / "exact_duplicate_groups.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "full_group_manifest.csv"

if not DUPLICATE_FILE.exists():
    raise FileNotFoundError(DUPLICATE_FILE)

# Load confirmed exact-duplicate groups
duplicates = pd.read_csv(DUPLICATE_FILE)

# Normalise paths for consistent matching
duplicate_map = {
    str(row["relative_path"]).replace("/", "\\").casefold():
    row["group_id"]
    for _, row in duplicates.iterrows()
}

# Find all dataset WAV files
audio_files = sorted(DATASET_DIR.rglob("*.wav"))

if not audio_files:
    raise RuntimeError("No WAV files found.")

records = []

for index, audio_path in enumerate(audio_files, start=1):
    relative_path = str(
        audio_path.relative_to(DATASET_DIR)
    )

    normalised_path = relative_path.replace("/", "\\").casefold()

    if normalised_path in duplicate_map:
        group_id = duplicate_map[normalised_path]
        group_type = "Confirmed exact duplicate group"
    else:
        group_id = f"SINGLE-{index:04d}"
        group_type = "Individual; source relationship unknown"

    records.append({
        "relative_path": relative_path,
        "group_id": group_id,
        "group_type": group_type,
        "main_class": audio_path.relative_to(DATASET_DIR).parts[0],
    })

manifest = pd.DataFrame(records)

# Validation
if manifest["relative_path"].duplicated().any():
    raise RuntimeError("Duplicate file paths found in manifest.")

if len(manifest) != 4292:
    raise RuntimeError(
        f"Expected 4292 files, found {len(manifest)}. "
        "Investigate before proceeding."
    )

manifest.to_csv(OUTPUT_FILE, index=False)

print("\nFULL GROUP MANIFEST CREATED")
print("===========================")
print(f"Audio files: {len(manifest)}")
print(f"Unique groups: {manifest['group_id'].nunique()}")
print(
    "Confirmed duplicate files:",
    (manifest["group_type"] == "Confirmed exact duplicate group").sum()
)
print(
    "Individual files:",
    (manifest["group_type"] == "Individual; source relationship unknown").sum()
)
print(f"Output: {OUTPUT_FILE}")

print("\nGroup type summary:")
print(manifest["group_type"].value_counts())

print("\nExact duplicate groups:")
print(
    manifest[
        manifest["group_type"] == "Confirmed exact duplicate group"
    ].groupby("group_id")["relative_path"].apply(list).to_string()
)
