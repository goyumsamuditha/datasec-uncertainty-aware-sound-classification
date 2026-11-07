from common import DATASET_ROOT, PROJECT_ROOT
from pathlib import Path
import hashlib
from collections import defaultdict
import pandas as pd

DATASET_ROOT = Path(
    DATASET_ROOT
)

hash_groups = defaultdict(list)

for wav_path in DATASET_ROOT.rglob("*.wav"):
    sha256 = hashlib.sha256()

    with open(wav_path, "rb") as audio_file:
        for chunk in iter(lambda: audio_file.read(1024 * 1024), b""):
            sha256.update(chunk)

    hash_groups[sha256.hexdigest()].append(
        str(wav_path.relative_to(DATASET_ROOT))
    )

duplicates = []

for file_hash, files in hash_groups.items():
    if len(files) > 1:
        duplicates.append({
            "sha256": file_hash,
            "number_of_files": len(files),
            "files": " | ".join(files)
        })

df = pd.DataFrame(duplicates)

print("Total audio files:", sum(len(v) for v in hash_groups.values()))
print("Unique audio files:", len(hash_groups))
print("Duplicate groups:", len(df))

if not df.empty:
    print("\nDuplicate groups:")
    print(df.to_string(index=False))

output_path = Path(
    PROJECT_ROOT / 'logs' / 'audio_duplicates.csv'
)

df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)