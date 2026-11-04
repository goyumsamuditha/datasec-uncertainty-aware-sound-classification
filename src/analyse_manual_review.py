import pandas as pd
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
INPUT_FILE = PROJECT_DIR / "logs" / "manual_similarity_review.csv"

df = pd.read_csv(INPUT_FILE)

print("\nMANUAL REVIEW SUMMARY")
print("=====================")

print("\nAssessment counts:")
print(df["manual_assessment"].value_counts())

print("\nAverage CLAP similarity by assessment:")
print(
    df.groupby("manual_assessment")["clap_similarity"]
      .agg(["count", "mean", "min", "max"])
      .round(4)
)

print("\nPairs with high CLAP similarity (>= 0.85) "
      "but judged Different:")
print(
    df[
        (df["clap_similarity"] >= 0.85) &
        (df["manual_assessment"] == "Different")
    ][["file_a", "file_b", "clap_similarity"]]
    .to_string(index=False)
)