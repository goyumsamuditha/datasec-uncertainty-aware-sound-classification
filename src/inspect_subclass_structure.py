
from pathlib import Path
from collections import Counter
import pandas as pd

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")

MANIFEST_PATH = PROJECT_DIR / "logs" / "final_split_manifest.csv"
CLASS_MAPPING_PATH = PROJECT_DIR / "logs" / "class_mapping.csv"

OUTPUT_PATH = PROJECT_DIR / "logs" / "subclass_structure_diagnostic.txt"


def main():
    lines = []

    def log(message=""):
        print(message)
        lines.append(str(message))

    log("DATASEC SUBCLASS STRUCTURE DIAGNOSTIC")
    log("=" * 60)

    # 1. Inspect the final split manifest
    manifest = pd.read_csv(MANIFEST_PATH)

    log("\n1. FINAL SPLIT MANIFEST")
    log(f"Rows: {len(manifest)}")
    log(f"Columns: {list(manifest.columns)}")
    log("\nFirst 10 relative paths:")

    for value in manifest["relative_path"].head(10):
        log(f"  {value}")

    # 2. Inspect directory structure
    log("\n2. DATASET DIRECTORY STRUCTURE")

    for item in sorted(DATASET_DIR.iterdir()):
        if item.is_dir():
            log(f"Directory: {item.name}")
            children = sorted(item.iterdir())[:8]

            for child in children:
                log(f"    {child.name}")

    # 3. Inspect relative path patterns
    log("\n3. RELATIVE PATH COMPONENTS")

    path_depths = Counter()
    parent_names = Counter()

    for value in manifest["relative_path"].dropna():
        path = Path(str(value))
        path_depths[len(path.parts)] += 1

        if len(path.parts) > 1:
            parent_names[path.parts[-2]] += 1

    log(f"Path-depth counts: {dict(path_depths)}")
    log("Most common parent folders:")

    for name, count in parent_names.most_common(30):
        log(f"  {name}: {count}")

    # 4. Inspect the previously generated class mapping
    log("\n4. EXISTING CLASS MAPPING")

    if CLASS_MAPPING_PATH.exists():
        mapping = pd.read_csv(CLASS_MAPPING_PATH)

        log(f"Rows: {len(mapping)}")
        log(f"Columns: {list(mapping.columns)}")
        log("\nFirst 40 rows:")

        for _, row in mapping.head(40).iterrows():
            log(f"  {row.to_dict()}")
    else:
        log("class_mapping.csv was not found.")

    # 5. Check whether subclass information is already in the manifest
    log("\n5. SUBCLASS LABEL CHECK")

    possible_columns = [
        col for col in manifest.columns
        if "subclass" in col.lower()
        or "sub_class" in col.lower()
    ]

    log(f"Possible subclass columns: {possible_columns}")

    if not possible_columns:
        log(
            "The final split manifest does not currently contain "
            "a recognisable subclass column."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"\nDiagnostic saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
