from common import PROJECT_ROOT

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

LOG_DIR = Path(PROJECT_ROOT / 'logs')
REPORT_DIR = Path(PROJECT_ROOT / 'report')

df = pd.read_csv(LOG_DIR / "audio_duration_analysis.csv")

# Duration statistics for each main class
summary = (
    df.groupby("main_class")["duration_seconds"]
    .agg(["count", "mean", "median", "min", "max"])
    .round(2)
    .sort_values("median")
)

summary.columns = [
    "recordings",
    "mean_seconds",
    "median_seconds",
    "min_seconds",
    "max_seconds"
]

print("\nDURATION STATISTICS BY MAIN CLASS")
print("=" * 85)
print(summary.to_string())

# Save results
LOG_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

summary.to_csv(LOG_DIR / "duration_by_class.csv")

# Visualise median duration
plt.figure(figsize=(12, 9))

plt.barh(
    summary.index,
    summary["median_seconds"]
)

plt.xlabel("Median duration (seconds)")
plt.ylabel("Main sound class")
plt.title("Median Audio Duration by DataSEC Main Class")

plt.tight_layout()

plt.savefig(
    REPORT_DIR / "median_duration_by_class.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nReport and chart saved successfully.")
