from common import PROJECT_ROOT

import pandas as pd
from pathlib import Path
from datetime import datetime

PROJECT_DIR = Path(PROJECT_ROOT)
LOG_DIR = PROJECT_DIR / "logs"

INPUT_FILE = LOG_DIR / "manual_similarity_review.csv"
OUTPUT_FILE = LOG_DIR / "experiment_log.csv"

# Check that the required files exist
if not INPUT_FILE.exists():
    raise FileNotFoundError(f"Missing input file: {INPUT_FILE}")

if not OUTPUT_FILE.exists():
    raise FileNotFoundError(f"Missing experiment log: {OUTPUT_FILE}")

# Load the manual review results
df = pd.read_csv(INPUT_FILE)

required_columns = {
    "manual_assessment",
    "clap_similarity",
    "file_a",
    "file_b"
}

missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing columns in review file: {missing}")

# Calculate results automatically
assessment_counts = df["manual_assessment"].value_counts()

different = int(assessment_counts.get("Different", 0))
very_similar = int(assessment_counts.get("Very similar", 0))
slightly_similar = int(assessment_counts.get("Slightly similar", 0))
identical = int(assessment_counts.get("Identical", 0))

high_similarity_different = df[
    (df["clap_similarity"] >= 0.85) &
    (df["manual_assessment"] == "Different")
]

summary = (
    f"Reviewed {len(df)} candidate pairs. "
    f"Different: {different}; "
    f"Very similar: {very_similar}; "
    f"Slightly similar: {slightly_similar}; "
    f"Identical: {identical}. "
    f"{len(high_similarity_different)} pairs had CLAP similarity >= 0.85 "
    f"but were judged Different. "
    "CLAP similarity is suitable for candidate screening, "
    "but not sufficient to establish shared recording provenance."
)

decision = (
    "Do not automatically create source groups using CLAP similarity. "
    "Keep confirmed exact-duplicate pairs together in the same split. "
    "Use similarity results only to identify candidates for further checks. "
    "Original source identifiers are unavailable, so complete "
    "source-level leakage prevention cannot be guaranteed."
)

# Read the existing experiment log
log = pd.read_csv(OUTPUT_FILE)

# Prevent duplicate entries when the script is run again
if "experiment_id" in log.columns:
    if log["experiment_id"].astype(str).eq("SIM-002").any():
        print("SIM-002 already exists. No duplicate entry added.")
        raise SystemExit

# Build a new entry using the existing CSV columns
new_entry = {}

for column in log.columns:
    name = column.strip().lower()

    if name in ["experiment_id", "id"]:
        new_entry[column] = "SIM-002"
    elif name in ["date", "timestamp"]:
        new_entry[column] = datetime.now().isoformat(timespec="seconds")
    elif name in ["experiment", "experiment_name", "task"]:
        new_entry[column] = "Manual review of CLAP similarity candidates"
    elif name in ["dataset"]:
        new_entry[column] = "DataSEC V.4"
    elif name in ["sample_size", "number_of_pairs", "n"]:
        new_entry[column] = len(df)
    elif name in ["result", "results", "summary"]:
        new_entry[column] = summary
    elif name in ["finding", "observation"]:
        new_entry[column] = (
            f"{len(high_similarity_different)} high-similarity pairs "
            "were judged Different."
        )
    elif name in ["decision", "conclusion"]:
        new_entry[column] = decision
    elif name in ["status"]:
        new_entry[column] = "Completed"
    else:
        new_entry[column] = ""

# Append without changing the existing column structure
updated_log = pd.concat(
    [log, pd.DataFrame([new_entry])],
    ignore_index=True
)

updated_log.to_csv(OUTPUT_FILE, index=False)

print("SIM-002 successfully recorded.")
print(f"Reviewed pairs: {len(df)}")
print(f"High-similarity pairs judged Different: {len(high_similarity_different)}")
print(f"Updated log: {OUTPUT_FILE}")
