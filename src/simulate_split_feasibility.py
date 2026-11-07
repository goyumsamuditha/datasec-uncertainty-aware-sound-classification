from common import PROJECT_ROOT

import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

PROJECT_DIR = Path(PROJECT_ROOT)

INPUT_FILE = PROJECT_DIR / "logs" / "full_group_manifest.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "split_feasibility.csv"

RANDOM_STATE = 42

TRAIN_RATIO = 0.70
VALIDATION_RATIO = 0.15
TEST_RATIO = 0.15

assert abs(TRAIN_RATIO + VALIDATION_RATIO + TEST_RATIO - 1.0) < 1e-9

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

df = pd.read_csv(INPUT_FILE)

required = {"relative_path", "group_id", "main_class"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Each group must belong to only one main class.
if (df.groupby("group_id")["main_class"].nunique() > 1).any():
    raise ValueError("A group contains files from multiple classes.")

assignments = []

# Allocate groups separately within each class.
for class_name, class_df in df.groupby("main_class"):

    groups = class_df["group_id"].drop_duplicates().to_numpy()

    # First allocate 70% to training and 30% temporarily.
    train_groups, temporary_groups = train_test_split(
        groups,
        test_size=0.30,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    # Divide the temporary 30% equally into validation and test.
    validation_groups, test_groups = train_test_split(
        temporary_groups,
        test_size=0.50,
        random_state=RANDOM_STATE,
        shuffle=True
    )

    for group_id in train_groups:
        assignments.append({
            "group_id": group_id,
            "main_class": class_name,
            "partition": "train"
        })

    for group_id in validation_groups:
        assignments.append({
            "group_id": group_id,
            "main_class": class_name,
            "partition": "validation"
        })

    for group_id in test_groups:
        assignments.append({
            "group_id": group_id,
            "main_class": class_name,
            "partition": "test"
        })

group_assignments = pd.DataFrame(assignments)

# Map each file to the partition assigned to its group.
result = df.merge(
    group_assignments,
    on=["group_id", "main_class"],
    how="left",
    validate="many_to_one"
)

if result["partition"].isna().any():
    raise RuntimeError("Some files were not assigned to a partition.")

# Check that every group appears in exactly one partition.
group_partition_counts = result.groupby("group_id")["partition"].nunique()

if (group_partition_counts > 1).any():
    raise RuntimeError("Group leakage detected.")

# Check confirmed exact duplicates remain together.
duplicate_groups = result[
    result["group_id"].astype(str).str.startswith("EXACT-")
]

if not duplicate_groups.empty:
    duplicate_partition_counts = (
        duplicate_groups.groupby("group_id")["partition"].nunique()
    )

    if (duplicate_partition_counts > 1).any():
        raise RuntimeError("An exact-duplicate group was split.")

# Summarise file counts and percentages.
partition_summary = (
    result["partition"]
    .value_counts()
    .rename_axis("partition")
    .reset_index(name="files")
)

partition_summary["percentage"] = (
    partition_summary["files"] / len(result) * 100
).round(2)

# Summarise class representation.
class_summary = (
    result.groupby(["main_class", "partition"])
    .size()
    .unstack(fill_value=0)
)

for partition in ["train", "validation", "test"]:
    if partition not in class_summary.columns:
        class_summary[partition] = 0

class_summary = class_summary[
    ["train", "validation", "test"]
]

class_summary["total"] = class_summary.sum(axis=1)

# Every class should appear in all partitions.
missing_class_partitions = class_summary[
    (class_summary["train"] == 0) |
    (class_summary["validation"] == 0) |
    (class_summary["test"] == 0)
]

print("\nSPLIT FEASIBILITY SIMULATION")
print("============================")
print(f"Random seed: {RANDOM_STATE}")
print(f"Total files: {len(result)}")
print(f"Total groups: {result['group_id'].nunique()}")

print("\nPartition sizes:")
print(partition_summary.to_string(index=False))

print("\nClass representation:")
print(class_summary.to_string())

print("\nValidation checks:")
print("No group crosses partitions: PASS")
print("Exact duplicates remain together: PASS")
print(
    "Every class appears in all partitions:",
    "PASS" if missing_class_partitions.empty else "FAIL"
)

if not missing_class_partitions.empty:
    print("\nClasses with missing partitions:")
    print(missing_class_partitions.to_string())

print("\nClass distribution saved to:")
print(OUTPUT_FILE)

class_summary.to_csv(OUTPUT_FILE)
