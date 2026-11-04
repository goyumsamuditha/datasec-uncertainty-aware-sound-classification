from pathlib import Path
import hashlib
import soundfile as sf
from collections import defaultdict
import pandas as pd

DATASET_ROOT = Path(
    r"G:\My Drive\MSC\AI\DATASEC\DATASEC"
)

hash_groups = defaultdict(list)

wav_files = list(DATASET_ROOT.rglob("*.wav"))

for index, wav_path in enumerate(wav_files, start=1):

    audio, sample_rate = sf.read(
        wav_path,
        dtype="float32"
    )

    # Hash the decoded audio samples
    audio_hash = hashlib.sha256(
        audio.tobytes()
    ).hexdigest()

    hash_groups[
        (sample_rate, audio.shape, audio_hash)
    ].append(
        str(wav_path.relative_to(DATASET_ROOT))
    )

    if index % 500 == 0:
        print(f"Processed {index}/{len(wav_files)} files")

duplicates = []

for key, files in hash_groups.items():
    if len(files) > 1:
        sample_rate, shape, audio_hash = key

        duplicates.append({
            "sample_rate": sample_rate,
            "audio_shape": str(shape),
            "number_of_files": len(files),
            "files": " | ".join(files)
        })

df = pd.DataFrame(duplicates)

print("\nTotal files:", len(wav_files))
print("Unique decoded waveforms:", len(hash_groups))
print("Duplicate waveform groups:", len(df))

if not df.empty:
    print("\nDuplicate waveform groups:")
    print(df.to_string(index=False))

output_path = Path(
    r"G:\My Drive\MSC\AI\Assignment\logs\waveform_duplicates.csv"
)

df.to_csv(output_path, index=False)

print("\nSaved to:", output_path)