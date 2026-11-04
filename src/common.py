from pathlib import Path
import json
import pandas as pd

# Project locations
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
LOGS = PROJECT_ROOT / "logs"

MANIFEST_PATH = LOGS / "final_split_manifest.csv"


def manifest():
    """Load the existing fixed split manifest and add absolute audio paths."""
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")

    df = pd.read_csv(MANIFEST_PATH)

    required = {
        "relative_path",
        "group_id",
        "group_type",
        "main_class",
        "partition",
    }

    missing_columns = required - set(df.columns)
    if missing_columns:
        raise ValueError(f"Manifest is missing columns: {sorted(missing_columns)}")

    # Resolve audio paths relative to the DataSEC dataset directory.
    df["audio_path"] = df["relative_path"].apply(
        lambda p: str(DATASET_ROOT / Path(str(p)))
    )

    # Keep compatibility with scripts that use split_name.
    df["split_name"] = df["partition"]

    return df


def maps(df):
    """Create deterministic label mappings from the manifest."""
    main_classes = sorted(df["main_class"].dropna().unique())
    main_to_id = {name: i for i, name in enumerate(main_classes)}

    # The current manifest contains no subclass column.
    # Do not invent subclass labels.
    subclass_column = next(
        (c for c in ["subclass", "sub_class", "secondary_class"]
         if c in df.columns),
        None,
    )

    if subclass_column:
        subclasses = sorted(df[subclass_column].dropna().unique())
        sub_to_id = {name: i for i, name in enumerate(subclasses)}
    else:
        sub_to_id = {}

    return main_to_id, sub_to_id


def write_json(data, path):
    """Write a JSON file, creating its parent folder if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)