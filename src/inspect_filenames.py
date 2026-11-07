from common import DATASET_ROOT, PROJECT_ROOT
from pathlib import Path
import pandas as pd

DATASET_ROOT = Path(
    DATASET_ROOT
)

records = []

for wav_path in DATASET_ROOT.rglob("*.wav"):
    records.append({
        "filename": wav_path.name,
        "parent_folder": wav_path.parent.name,
        "relative_path": str(wav_path.relative_to(DATASET_ROOT)),
    })

df = pd.DataFrame(records)

print("Total WAV files:", len(df))

print("\nFirst 30 filenames and paths:")
print(df.head(30).to_string(index=False))

print("\nExamples of filenames:")
print(df["filename"].head(50).to_list())

output_path = Path(
    PROJECT_ROOT / 'logs' / 'filename_inspection.csv'
)

df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)