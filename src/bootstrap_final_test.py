from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

INPUT = Path("models/final_test_evaluation/test_predictions_all_models.csv")
OUTPUT_DIR = Path("models/final_test_evaluation")
OUTPUT = OUTPUT_DIR / "paired_bootstrap_comparison.csv"

N_BOOTSTRAPS = 10000
SEED = 42

df = pd.read_csv(INPUT)

required = [
    "actual_class",
    "cnn14_prediction",
    "clap_linear_prediction",
]

if not all(column in df.columns for column in required):
    raise ValueError("Required prediction columns are missing.")

df = df.dropna(subset=required).reset_index(drop=True)

y_true = df["actual_class"].to_numpy()
cnn_pred = df["cnn14_prediction"].to_numpy()
clap_pred = df["clap_linear_prediction"].to_numpy()

classes = np.sort(np.unique(y_true))
class_indices = [
    np.flatnonzero(y_true == class_name)
    for class_name in classes
]

if not np.array_equal(np.unique(cnn_pred), np.unique(clap_pred)):
    print("Warning: model prediction label sets differ.")

print("=" * 70)
print("PAIRED STRATIFIED BOOTSTRAP COMPARISON")
print("=" * 70)
print(f"Test recordings: {len(df)}")
print(f"Classes: {len(classes)}")
print(f"Bootstrap repetitions: {N_BOOTSTRAPS}")
print("Resampling: stratified by actual class")
print("Seed:", SEED)

# Original test-set point estimates
point_estimates = {}

for name, prediction in [
    ("CNN14", cnn_pred),
    ("CLAP linear probe", clap_pred),
]:
    point_estimates[name] = {
        "accuracy": accuracy_score(y_true, prediction),
        "macro_f1": f1_score(
            y_true,
            prediction,
            labels=classes,
            average="macro",
            zero_division=0,
        ),
    }

# Paired stratified bootstrap:
# both models receive exactly the same sampled recording indices.
rng = np.random.default_rng(SEED)

bootstrap_results = {
    "cnn_accuracy": [],
    "cnn_macro_f1": [],
    "clap_accuracy": [],
    "clap_macro_f1": [],
    "accuracy_difference": [],
    "macro_f1_difference": [],
}

for iteration in range(N_BOOTSTRAPS):

    sampled_indices = np.concatenate([
        rng.choice(indices, size=len(indices), replace=True)
        for indices in class_indices
    ])

    actual = y_true[sampled_indices]
    cnn = cnn_pred[sampled_indices]
    clap = clap_pred[sampled_indices]

    cnn_accuracy = accuracy_score(actual, cnn)
    clap_accuracy = accuracy_score(actual, clap)

    cnn_macro_f1 = f1_score(
        actual, cnn, labels=classes,
        average="macro", zero_division=0
    )

    clap_macro_f1 = f1_score(
        actual, clap, labels=classes,
        average="macro", zero_division=0
    )

    bootstrap_results["cnn_accuracy"].append(cnn_accuracy)
    bootstrap_results["cnn_macro_f1"].append(cnn_macro_f1)
    bootstrap_results["clap_accuracy"].append(clap_accuracy)
    bootstrap_results["clap_macro_f1"].append(clap_macro_f1)

    bootstrap_results["accuracy_difference"].append(
        clap_accuracy - cnn_accuracy
    )

    bootstrap_results["macro_f1_difference"].append(
        clap_macro_f1 - cnn_macro_f1
    )

    if (iteration + 1) % 2000 == 0:
        print(f"Completed {iteration + 1}/{N_BOOTSTRAPS}")

def interval(values):
    return np.percentile(values, [2.5, 97.5])

rows = []

metrics = [
    ("CNN14 accuracy", "cnn_accuracy", "CNN14", "accuracy"),
    ("CNN14 macro-F1", "cnn_macro_f1", "CNN14", "macro_f1"),
    ("CLAP accuracy", "clap_accuracy", "CLAP linear probe", "accuracy"),
    ("CLAP macro-F1", "clap_macro_f1", "CLAP linear probe", "macro_f1"),
]

for label, key, model, metric in metrics:
    lower, upper = interval(bootstrap_results[key])

    rows.append({
        "comparison": label,
        "test_estimate": point_estimates[model][metric],
        "ci_lower_95": lower,
        "ci_upper_95": upper,
    })

for label, key in [
    ("CLAP minus CNN14 accuracy", "accuracy_difference"),
    ("CLAP minus CNN14 macro-F1", "macro_f1_difference"),
]:
    values = np.asarray(bootstrap_results[key])
    lower, upper = interval(values)

    rows.append({
        "comparison": label,
        "test_estimate": (
            point_estimates["CLAP linear probe"]["accuracy"]
            - point_estimates["CNN14"]["accuracy"]
            if key == "accuracy_difference"
            else
            point_estimates["CLAP linear probe"]["macro_f1"]
            - point_estimates["CNN14"]["macro_f1"]
        ),
        "ci_lower_95": lower,
        "ci_upper_95": upper,
    })

results = pd.DataFrame(rows)
results.to_csv(OUTPUT, index=False)

print("\n" + "=" * 70)
print("BOOTSTRAP RESULTS")
print("=" * 70)
print(results.to_string(index=False, float_format=lambda x: f"{x:.5f}"))

difference = np.asarray(bootstrap_results["macro_f1_difference"])

print("\nCLAP macro-F1 advantage:")
print(f"Mean bootstrap difference: {difference.mean():.5f}")
print(f"95% CI: {np.percentile(difference, 2.5):.5f} to "
      f"{np.percentile(difference, 97.5):.5f}")
print(f"Bootstrap proportion with difference <= 0: "
      f"{np.mean(difference <= 0):.4f}")

print("\nSaved:", OUTPUT)
