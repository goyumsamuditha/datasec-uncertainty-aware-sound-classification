
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

MANIFEST_FILE = PROJECT_DIR / "logs" / "final_split_manifest.csv"

OUTPUT_FILE = (
    PROJECT_DIR / "logs" / "final_split_integrity_check.csv"
)

if not MANIFEST_FILE.exists():
    raise FileNotFoundError(MANIFEST_FILE)

if not DATASET_DIR.exists():
    raise FileNotFoundError(DATASET_DIR)

df = pd.read_csv(MANIFEST_FILE)

required_columns = {
    "relative_path",
    "group_id",
    "main_class",
    "partition"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

results = []

for _, row in df.iterrows():
    audio_path = DATASET_DIR / str(row["relative_path"])

    results.append({
        "relative_path": row["relative_path"],
        "partition": row["partition"],
        "main_class": row["main_class"],
        "file_exists": audio_path.is_file()
    })

check_df = pd.DataFrame(results)

print("\nFINAL SPLIT FILE INTEGRITY CHECK")
print("================================")
print(f"Manifest entries: {len(check_df)}")
print(f"Files found: {check_df['file_exists'].sum()}")
print(f"Files missing: {(~check_df['file_exists']).sum()}")

print("\nResults by partition:")
print(
    check_df.groupby("partition")["file_exists"]
    .agg(["count", "sum"])
    .to_string()
)

print("\nResults by class:")
print(
    check_df.groupby("main_class")["file_exists"]
    .agg(["count", "sum"])
    .to_string()
)

check_df.to_csv(OUTPUT_FILE, index=False)

if check_df["file_exists"].all():
    print("\nOVERALL RESULT: PASS")
else:
    print("\nOVERALL RESULT: FAIL")
    print("\nMissing files:")
    print(
        check_df.loc[
            ~check_df["file_exists"],
            "relative_path"
        ].to_string(index=False)
    )

print(f"\nDetailed report saved to: {OUTPUT_FILE}")
