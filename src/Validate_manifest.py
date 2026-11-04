from pathlib import Path
from common import manifest, maps, write_json, LOGS

# Load the existing manifest; do not regenerate it.
df = manifest()
main_to_id, sub_to_id = maps(df)

print("Total rows:", len(df))
print("\nPartition counts:")
print(df["partition"].value_counts())

print("\nMain classes:", len(main_to_id))
print("Subclass labels available in manifest:", len(sub_to_id))

# Check that every referenced audio file exists.
missing = df[~df["audio_path"].map(lambda p: Path(p).is_file())]

print("\nMissing audio files:", len(missing))

if len(missing):
    print(missing["audio_path"].head(20).to_string(index=False))

# Check that each group is assigned to only one partition.
group_partition_counts = df.groupby("group_id")["partition"].nunique()
split_groups = group_partition_counts[group_partition_counts > 1]

print("Groups appearing in multiple partitions:", len(split_groups))

if len(split_groups):
    print(split_groups.head(20).to_string())

# Check for missing labels and paths.
print("Missing relative paths:", df["relative_path"].isna().sum())
print("Missing main-class labels:", df["main_class"].isna().sum())

assert len(missing) == 0, "Some audio files are missing."
assert len(split_groups) == 0, "Some groups cross partition boundaries."
assert df["relative_path"].notna().all(), "Some relative paths are missing."
assert df["main_class"].notna().all(), "Some main-class labels are missing."

# Save the main-class mapping.
# Subclass mapping remains empty because the manifest has no subclass column.
write_json(
    {
        "main_to_id": main_to_id,
        "sub_to_id": sub_to_id,
    },
    LOGS / "label_mapping.json",
)

print("\nManifest validation passed.")
print("Label mapping saved to:", LOGS / "label_mapping.json")