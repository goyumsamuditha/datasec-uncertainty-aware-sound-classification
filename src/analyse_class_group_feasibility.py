
import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")

INPUT_FILE = PROJECT_DIR / "logs" / "full_group_manifest.csv"
OUTPUT_FILE = PROJECT_DIR / "logs" / "class_group_feasibility.csv"

if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Manifest not found: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "relative_path",
    "group_id",
    "group_type",
    "main_class"
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing columns: {missing}")

# Check whether any group contains files from multiple classes
classes_per_group = df.groupby("group_id")["main_class"].nunique()

cross_class_groups = classes_per_group[classes_per_group > 1]

if not cross_class_groups.empty:
    print("WARNING: Groups contain files from multiple main classes:")
    print(cross_class_groups)
else:
    print("PASS: Every group belongs to exactly one main class.")

# Calculate file and group counts for each class
summary = (
    df.groupby("main_class")
    .agg(
        total_files=("relative_path", "count"),
        unique_groups=("group_id", "nunique"),
    )
    .reset_index()
)

# Count files belonging to confirmed duplicate groups
duplicate_counts = (
    df[df["group_type"] == "Confirmed exact duplicate group"]
    .groupby("main_class")
    .size()
    .rename("confirmed_duplicate_files")
)

summary = summary.merge(
    duplicate_counts,
    on="main_class",
    how="left"
)

summary["confirmed_duplicate_files"] = (
    summary["confirmed_duplicate_files"].fillna(0).astype(int)
)

summary["individual_files"] = (
    summary["total_files"] - summary["confirmed_duplicate_files"]
)

summary["groups_per_file"] = (
    summary["unique_groups"] / summary["total_files"]
).round(4)

summary = summary.sort_values(
    "total_files",
    ascending=False
).reset_index(drop=True)

# Validate totals
assert summary["total_files"].sum() == len(df)
assert summary["unique_groups"].sum() == df["group_id"].nunique()

summary.to_csv(OUTPUT_FILE, index=False)

print("\nCLASS-LEVEL GROUP FEASIBILITY")
print("=============================")
print(f"Total files: {summary['total_files'].sum()}")
print(f"Total unique groups: {summary['unique_groups'].sum()}")
print(f"Number of classes: {len(summary)}")
print(
    "Confirmed duplicate files:",
    summary["confirmed_duplicate_files"].sum()
)

print("\nClass-level results:")
print(summary.to_string(index=False))

print(f"\nSaved results to: {OUTPUT_FILE}")
