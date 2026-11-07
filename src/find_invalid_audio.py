from common import DATASET_ROOT, PROJECT_ROOT
from pathlib import Path
import pandas as pd
import soundfile as sf

PROJECT_DIR = Path(PROJECT_ROOT)
DATASET_DIR = Path(DATASET_ROOT)
MANIFEST = PROJECT_DIR / "logs" / "final_split_manifest_with_subclasses.csv"

df = pd.read_csv(MANIFEST)

issues = []
checked = 0

for _, row in df.iterrows():
    path = DATASET_DIR / row["relative_path"]

    try:
        info = sf.info(str(path))
        checked += 1

        if info.channels < 1 or info.frames < 1:
            issues.append({
                "relative_path": row["relative_path"],
                "main_class": row["main_class"],
                "partition": row["partition"],
                "channels": info.channels,
                "frames": info.frames,
                "sample_rate": info.samplerate,
                "issue": "Zero channels or zero frames"
            })

    except Exception as exc:
        issues.append({
            "relative_path": row["relative_path"],
            "main_class": row["main_class"],
            "partition": row["partition"],
            "channels": None,
            "frames": None,
            "sample_rate": None,
            "issue": str(exc)
        })

output = PROJECT_DIR / "logs" / "invalid_audio_files.csv"
pd.DataFrame(issues).to_csv(output, index=False)

print("Audio file inspection completed.")
print("Files checked:", checked)
print("Problematic files:", len(issues))
print("Results saved to:", output)

if issues:
    print("\nProblematic files:")
    for item in issues:
        print(item)
else:
    print("No zero-channel or unreadable files found.")
