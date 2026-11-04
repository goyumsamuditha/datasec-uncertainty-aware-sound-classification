from pathlib import Path

DATASET_ROOT = Path(
    r"G:\My Drive\MSC\AI\DATASEC\DATASEC"
)

metadata_extensions = {
    ".csv", ".json", ".txt", ".xlsx",
    ".xls", ".xml", ".yaml", ".yml"
}

metadata_files = [
    file for file in DATASET_ROOT.rglob("*")
    if file.is_file()
    and file.suffix.lower() in metadata_extensions
]

print("Metadata files found:", len(metadata_files))

for file in metadata_files:
    print(file.relative_to(DATASET_ROOT))