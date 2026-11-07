from common import PROJECT_ROOT

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = Path(PROJECT_ROOT / 'logs')
OUTPUT_DIR = Path(PROJECT_ROOT / 'report')

df = pd.read_csv(LOG_DIR / "main_class_distribution.csv")

df = df.sort_values("audio_count", ascending=True)

plt.figure(figsize=(12, 9))

plt.barh(df["main_class"], df["audio_count"])

plt.xlabel("Number of audio recordings")
plt.ylabel("Main sound class")
plt.title("DataSEC V4 - Main Class Distribution")

plt.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.savefig(
    OUTPUT_DIR / "main_class_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Chart saved successfully.")
