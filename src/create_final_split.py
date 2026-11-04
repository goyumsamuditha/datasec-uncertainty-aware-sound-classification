
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")

INPUT_FILE = PROJECT_DIR / "logs" / "full_group_manifest.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "final_split_manifest.csv"
SUMMARY_FILE = PROJECT_DIR / "logs" / "final_split_summary.csv"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

df = pd.read_csv(INPUT_FILE)

required = {"relative_path", "group_id", "main_class"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df["relative_path"].duplicated().any():
    raise ValueError("Duplicate file paths found.")

if df["group_id"].isna().any():
    raise ValueError("Missing group IDs found.")

if df.groupby("group_id")["main_class"].nunique().gt(1).any():
    raise ValueError("A group contains more than one main class.")

assignments = []

for class_name, class_df in df.groupby("main_class", sort=True):

    groups = class_df["group_id"].drop_duplicates().to_numpy()

    train_groups, temporary_groups = train_test_split(
        groups,
        test_size=VALIDATION_RATIO + TEST_RATIO,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=TEST_RATIO / (VALIDATION_RATIO + TEST_RATIO),
        random_state=RANDOM_STATE,
        shuffle=True
    )

    for group_id in train_groups:
        assignments.append((group_id, class_name, "train"))

    for group_id in validation_groups:
        assignments.append((group_id, class_name, "validation"))

    for group_id in test_groups:
        assignments.append((group_id, class_name, "test"))

group_assignments = pd.DataFrame(
    assignments,
    columns=["group_id", "main_class", "partition"]
)

result = df.merge(
    group_assignments,
    on=["group_id", "main_class"],
    how="left",
    validate="many_to_one"
)

if result["partition"].isna().any():
    raise RuntimeError("Some files were not assigned.")

# Confirm no group crosses partitions.
if result.groupby("group_id")["partition"].nunique().gt(1).any():
    raise RuntimeError("Group leakage detected.")

# Confirm exact duplicate groups remain intact.
exact_groups = result[
    result["group_id"].astype(str).str.startswith("EXACT-")
]

if not exact_groups.empty:
    if exact_groups.groupby("group_id")["partition"].nunique().gt(1).any():
        raise RuntimeError("An exact duplicate group was split.")

# Confirm every class appears in every partition.
class_counts = result.groupby(
    ["main_class", "partition"]
).size().unstack(fill_value=0)

for partition in ["train", "validation", "test"]:
    if partition not in class_counts.columns:
        class_counts[partition] = 0

if (class_counts[["train", "validation", "test"]] == 0).any().any():
    raise RuntimeError("At least one class is missing from a partition.")

# Create summary.
summary = (
    result.groupby("partition")
    .size()
    .reindex(["train", "validation", "test"])
    .reset_index(name="files")
)

summary["percentage"] = (
    summary["files"] / len(result) * 100
).round(2)

# Save the manifest and summary.
result.to_csv(OUTPUT_FILE, index=False)
summary.to_csv(SUMMARY_FILE, index=False)

print("\nFINAL SPLIT CREATED")
print("===================")
print(f"Total files: {len(result)}")
print(f"Total groups: {result['group_id'].nunique()}")
print(f"Random seed: {RANDOM_STATE}")

print("\nPartition summary:")
print(summary.to_string(index=False))

print("\nValidation:")
print("Every file assigned: PASS")
print("No group crosses partitions: PASS")
print("Exact duplicates kept together: PASS")
print("All classes represented: PASS")

print(f"\nManifest: {OUTPUT_FILE}")
print(f"Summary: {SUMMARY_FILE}")
