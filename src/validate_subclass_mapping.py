
from pathlib import Path
import pandas as pd

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
LOG_DIR = PROJECT_DIR / "logs"

MAPPING_PATH = LOG_DIR / "class_mapping.csv"
MANIFEST_PATH = LOG_DIR / "final_split_manifest.csv"

SUMMARY_PATH = LOG_DIR / "subclass_distribution_verified.csv"
ENRICHED_PATH = LOG_DIR / "final_split_manifest_with_subclasses.csv"
REPORT_PATH = LOG_DIR / "subclass_mapping_validation.txt"


def main():
    mapping = pd.read_csv(MAPPING_PATH)
    manifest = pd.read_csv(MANIFEST_PATH)

    required_mapping = {
        "relative_path", "main_folder", "subclass_folder"
    }
    required_manifest = {
        "relative_path", "main_class", "partition"
    }

    if not required_mapping.issubset(mapping.columns):
        raise ValueError(
            f"Missing mapping columns: "
            f"{required_mapping - set(mapping.columns)}"
        )

    if not required_manifest.issubset(manifest.columns):
        raise ValueError(
            f"Missing manifest columns: "
            f"{required_manifest - set(manifest.columns)}"
        )

    # Check that each relative path occurs only once.
    if mapping["relative_path"].duplicated().any():
        raise ValueError("Duplicate paths found in class_mapping.csv")

    if manifest["relative_path"].duplicated().any():
        raise ValueError("Duplicate paths found in final split manifest")

    # Confirm both files describe the same recordings.
    mapping_paths = set(mapping["relative_path"].astype(str))
    manifest_paths = set(manifest["relative_path"].astype(str))

    if mapping_paths != manifest_paths:
        raise ValueError(
            "The mapping and split manifest contain different file paths."
        )

    # Merge labels without changing the original split assignments.
    enriched = manifest.merge(
        mapping[
            ["relative_path", "main_folder", "subclass_folder"]
        ],
        on="relative_path",
        how="left",
        validate="one_to_one"
    )

    # Confirm the main labels agree.
    mismatch = (
        enriched["main_class"].astype(str)
        != enriched["main_folder"].astype(str)
    )

    if mismatch.any():
        raise ValueError(
            f"{int(mismatch.sum())} main-class mismatches found."
        )

    enriched = enriched.drop(columns=["main_folder"])

    # Normalise missing subclass values.
    enriched["subclass_folder"] = (
        enriched["subclass_folder"].fillna("").astype(str).str.strip()
    )

    # Count distinct non-empty subclasses.
    subclass_rows = enriched[
        enriched["subclass_folder"] != ""
    ]

    subclass_counts = (
        subclass_rows.groupby(
            ["main_class", "subclass_folder"]
        )
        .size()
        .reset_index(name="recording_count")
        .sort_values(["main_class", "subclass_folder"])
    )

    unique_subclasses = enriched.loc[
        enriched["subclass_folder"] != "",
        "subclass_folder"
    ].nunique()

    missing_subclass_count = int(
        (enriched["subclass_folder"] == "").sum()
    )

    # Check that every subclass belongs to exactly one main class.
    parent_counts = (
        subclass_rows.groupby("subclass_folder")["main_class"]
        .nunique()
    )

    ambiguous_subclasses = parent_counts[parent_counts > 1]

    # Save outputs.
    subclass_counts.to_csv(SUMMARY_PATH, index=False)
    enriched.to_csv(ENRICHED_PATH, index=False)

    report = [
        "SUBCLASS MAPPING VALIDATION",
        "=" * 55,
        f"Manifest rows: {len(manifest)}",
        f"Enriched rows: {len(enriched)}",
        f"Unique main classes: {enriched['main_class'].nunique()}",
        f"Unique non-empty subclasses: {unique_subclasses}",
        f"Recordings with subclass labels: {len(subclass_rows)}",
        f"Recordings without separate subclass labels: "
        f"{missing_subclass_count}",
        f"Main-class mismatches: {int(mismatch.sum())}",
        f"Ambiguous subclass names: {len(ambiguous_subclasses)}",
        "",
        "Subclass counts by main class:",
        subclass_counts.to_string(index=False),
        "",
        "Partition counts:",
        enriched["partition"].value_counts().to_string(),
    ]

    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print("\n".join(report))
    print("\nSaved:")
    print(SUMMARY_PATH)
    print(ENRICHED_PATH)
    print(REPORT_PATH)

    # Expected checks. Stop if the mapping is inconsistent.
    assert len(enriched) == 4292
    assert enriched["main_class"].nunique() == 22
    assert int(mismatch.sum()) == 0
    assert len(ambiguous_subclasses) == 0

    print("\nBASIC SUBCLASS MAPPING VALIDATION PASSED.")


if __name__ == "__main__":
    main()
