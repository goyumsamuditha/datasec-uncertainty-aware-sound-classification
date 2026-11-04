
from pathlib import Path
import json
import pandas as pd

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
LOG_DIR = PROJECT_DIR / "logs"

MANIFEST_PATH = LOG_DIR / "final_split_manifest_with_subclasses.csv"
OUTPUT_PATH = LOG_DIR / "hierarchical_label_mapping.json"
SUMMARY_PATH = LOG_DIR / "hierarchical_label_summary.csv"


def main():
    df = pd.read_csv(MANIFEST_PATH)

    required = {
        "relative_path",
        "main_class",
        "subclass_folder",
        "partition",
    }

    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    # Stable alphabetical label order.
    main_classes = sorted(df["main_class"].dropna().unique())

    subclass_names = sorted(
        name for name in df["subclass_folder"].dropna().unique()
        if str(name).strip()
    )

    main_to_id = {
        name: index for index, name in enumerate(main_classes)
    }

    subclass_to_id = {
        name: index for index, name in enumerate(subclass_names)
    }

    df["main_label"] = df["main_class"].map(main_to_id)

    # -1 means that the dataset does not provide a subclass label.
    df["subclass_label"] = (
        df["subclass_folder"]
        .fillna("")
        .map(lambda x: subclass_to_id.get(x, -1))
    )

    if df["main_label"].isna().any():
        raise ValueError("Missing main-class numeric labels.")

    if len(main_classes) != 22:
        raise ValueError(
            f"Expected 22 main classes, found {len(main_classes)}"
        )

    if len(subclass_names) != 28:
        raise ValueError(
            f"Expected 28 subclasses, found {len(subclass_names)}"
        )

    if (df["subclass_label"] == -1).sum() != 2895:
        raise ValueError("Unexpected number of missing subclass labels.")

    # Verify each subclass belongs to one main class.
    parent_counts = (
        df[df["subclass_label"] >= 0]
        .groupby("subclass_folder")["main_class"]
        .nunique()
    )

    if (parent_counts > 1).any():
        raise ValueError("A subclass maps to multiple main classes.")

    # Save the mappings separately from the original split manifest.
    mapping = {
        "main_classes": main_classes,
        "main_to_id": main_to_id,
        "subclasses": subclass_names,
        "subclass_to_id": subclass_to_id,
        "missing_subclass_label": -1,
        "num_main_classes": len(main_classes),
        "num_subclasses": len(subclass_names),
    }

    OUTPUT_PATH.write_text(
        json.dumps(mapping, indent=2),
        encoding="utf-8",
    )

    # Summary by main class and subclass.
    summary = (
        df.groupby(
            ["main_class", "subclass_folder"],
            dropna=False
        )
        .size()
        .reset_index(name="recording_count")
    )

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(SUMMARY_PATH, index=False)

    print("HIERARCHICAL TRAINING METADATA")
    print("=" * 50)
    print(f"Total recordings: {len(df)}")
    print(f"Main classes: {len(main_classes)}")
    print(f"Subclasses: {len(subclass_names)}")
    print(
        "Recordings with subclass labels:",
        int((df["subclass_label"] >= 0).sum())
    )
    print(
        "Recordings without subclass labels:",
        int((df["subclass_label"] == -1).sum())
    )

    print("\nPartition counts:")
    print(df["partition"].value_counts())

    print("\nMain-class mapping:")
    for name, index in main_to_id.items():
        print(f"{index:2d}: {name}")

    print("\nSubclass mapping:")
    for name, index in subclass_to_id.items():
        print(f"{index:2d}: {name}")

    print("\nSaved:")
    print(OUTPUT_PATH)
    print(SUMMARY_PATH)

    print("\nMETADATA VALIDATION PASSED")


if __name__ == "__main__":
    main()
