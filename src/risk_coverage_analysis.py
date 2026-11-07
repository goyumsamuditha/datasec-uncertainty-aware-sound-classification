
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------
# Configuration
# --------------------------------------------------

INPUT_FILE = Path(
    "models/cnn14_final123_temperature_scaling/"
    "validation_calibrated_predictions.csv"
)

OUTPUT_DIR = Path("models/cnn14_final123_risk_coverage")
REPORT_DIR = Path("report")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CURVE_FILE = OUTPUT_DIR / "risk_coverage_curve.csv"
SUMMARY_FILE = OUTPUT_DIR / "risk_coverage_summary.csv"
PLOT_FILE = REPORT_DIR / "final123_risk_coverage.png"


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = {
    "relative_path",
    "correct",
    "confidence_after",
}

missing = required_columns - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["correct"] = df["correct"].astype(bool)
df["confidence_after"] = pd.to_numeric(
    df["confidence_after"], errors="raise"
)

if df["confidence_after"].isna().any():
    raise ValueError("Missing confidence values detected.")

if not df["confidence_after"].between(0, 1).all():
    raise ValueError("Confidence values must be between 0 and 1.")

n = len(df)

if n == 0:
    raise ValueError("No validation predictions found.")


# --------------------------------------------------
# Group identical confidence scores
# --------------------------------------------------

# Highest confidence is accepted first.
# All recordings with identical confidence are
# included together to avoid arbitrary tie-breaking.

grouped = (
    df.groupby("confidence_after", as_index=False)
    .agg(
        group_size=("correct", "size"),
        group_errors=("correct", lambda x: int((~x).sum())),
    )
    .sort_values("confidence_after", ascending=False)
    .reset_index(drop=True)
)

grouped["accepted_recordings"] = grouped["group_size"].cumsum()
grouped["cumulative_errors"] = grouped["group_errors"].cumsum()

grouped["coverage"] = grouped["accepted_recordings"] / n

grouped["risk"] = (
    grouped["cumulative_errors"] /
    grouped["accepted_recordings"]
)

grouped["selective_accuracy"] = 1 - grouped["risk"]

grouped.to_csv(CURVE_FILE, index=False)


# --------------------------------------------------
# Calculate AURC
# --------------------------------------------------

# Step-wise area: risk at each coverage interval's
# right endpoint multiplied by the interval width.

coverage = grouped["coverage"].to_numpy()
risk = grouped["risk"].to_numpy()

previous_coverage = np.concatenate(([0.0], coverage[:-1]))

aurc = float(
    np.sum((coverage - previous_coverage) * risk)
)


# --------------------------------------------------
# Target coverage summary
# --------------------------------------------------

target_coverages = [0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.00]

summary_rows = []

for target in target_coverages:
    eligible = grouped[grouped["coverage"] >= target]

    if eligible.empty:
        continue

    # Select the smallest attainable coverage that
    # reaches or exceeds the requested target.
    result = eligible.iloc[0]

    summary_rows.append({
        "target_coverage": target,
        "actual_coverage": result["coverage"],
        "accepted_recordings": int(
            result["accepted_recordings"]
        ),
        "errors": int(result["cumulative_errors"]),
        "risk": result["risk"],
        "selective_accuracy": result["selective_accuracy"],
        "confidence_threshold": result["confidence_after"],
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(SUMMARY_FILE, index=False)


# --------------------------------------------------
# Plot risk-coverage curve
# --------------------------------------------------

plt.figure(figsize=(9, 6))

plt.plot(
    coverage,
    risk,
    linewidth=2,
    label="FINAL-123 (temperature-scaled MSP)"
)

# Overall error rate as a reference.
overall_risk = float((~df["correct"]).mean())

plt.axhline(
    overall_risk,
    linestyle="--",
    linewidth=1.5,
    label=f"Overall risk = {overall_risk:.3f}"
)

plt.xlabel("Coverage")
plt.ylabel("Classification risk (error rate)")
plt.title("Risk-Coverage Curve: FINAL-123")
plt.xlim(0, 1)
plt.ylim(bottom=0)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(PLOT_FILE, dpi=300)
plt.close()


# --------------------------------------------------
# Results
# --------------------------------------------------

print("\n=== RISK-COVERAGE ANALYSIS ===")
print("Input:", INPUT_FILE)
print("Validation recordings:", n)
print("Unique confidence thresholds:", len(grouped))
print("Overall accuracy:", df["correct"].mean())
print("Overall risk:", overall_risk)
print("AURC:", aurc)

print("\n=== SELECTIVE PERFORMANCE ===")
print(summary.to_string(index=False))

print("\nSaved files:")
print(CURVE_FILE)
print(SUMMARY_FILE)
print(PLOT_FILE)

print("\nValidation partition only.")
print("Held-out test partition was not accessed.")
