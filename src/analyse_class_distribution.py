
from pathlib import Path
from collections import Counter
import pandas as pd

DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
LOG_DIR = Path(r"G:\My Drive\MSC\AI\Assignment\logs")

records = []

for wav_path in DATASET_ROOT.rglob("*.wav"):
    relative_path = wav_path.relative_to(DATASET_ROOT)
    folders = relative_path.parts[:-1]

    main_class = folders[0] if folders else ""
    subclass = folders[1] if len(folders) > 1 else "No subclass"

    records.append({
        "file_name": wav_path.name,
        "main_class": main_class,
        "subclass": subclass,
    })

df = pd.DataFrame(records)

# Count recordings per main class
main_counts = (
    df.groupby("main_class")
    .size()
    .reset_index(name="audio_count")
    .sort_values("audio_count", ascending=False)
)

main_counts["percentage"] = (
    main_counts["audio_count"] / len(df) * 100
).round(2)

# Count recordings per subclass
subclass_counts = (
    df.groupby(["main_class", "subclass"])
    .size()
    .reset_index(name="audio_count")
    .sort_values("audio_count", ascending=False)
)

# Save reports
LOG_DIR.mkdir(parents=True, exist_ok=True)

main_counts.to_csv(
    LOG_DIR / "main_class_distribution.csv",
    index=False
)

subclass_counts.to_csv(
    LOG_DIR / "subclass_distribution.csv",
    index=False
)

# Display results
print("\nMAIN CLASS DISTRIBUTION")
print("=" * 65)
print(main_counts.to_string(index=False))

print("\nSUBCLASS DISTRIBUTION")
print("=" * 65)
print(subclass_counts.to_string(index=False))

print("\nSUMMARY")
print("=" * 65)
print(f"Total recordings: {len(df)}")
print(f"Main classes: {df['main_class'].nunique()}")
print(
    "Subclasses:",
    df[df["subclass"] != "No subclass"]["subclass"].nunique()
)
print(f"Largest main class: {main_counts.iloc[0]['main_class']}")
print(f"Largest class count: {main_counts.iloc[0]['audio_count']}")
print(f"Smallest main class: {main_counts.iloc[-1]['main_class']}")
print(f"Smallest class count: {main_counts.iloc[-1]['audio_count']}")

print("\nReports saved in:", LOG_DIR)
