from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------

INPUT_FILE = Path(
    "models/final_test_evaluation/test_predictions_all_models.csv"
)

OUTPUT_DIR = Path("models/final_test_evaluation")

N_BOOTSTRAP = 10000
RANDOM_SEED = 42
CONFIDENCE_LEVEL = 0.95

MODEL_COLUMNS = {
    "CNN14": "cnn14_prediction",
    "CLAP linear probe": "clap_linear_prediction",
    "SVM": "svm_prediction",
    "CLAP zero-shot": "clap_zero_shot_prediction",
}


# ---------------------------------------------------------
# LOAD DATA
# ---------------------------------------------------------

df = pd.read_csv(INPUT_FILE)

required_columns = ["actual_class"] + list(MODEL_COLUMNS.values())

missing = [column for column in required_columns if column not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df = df.dropna(subset=required_columns).reset_index(drop=True)

y_true = df["actual_class"].astype(str).to_numpy()
classes = np.unique(y_true)

predictions = {
    name: df[column].astype(str).to_numpy()
    for name, column in MODEL_COLUMNS.items()
}

print("=" * 70)
print("CLASS-STRATIFIED PAIRED BOOTSTRAP ANALYSIS")
print("=" * 70)
print(f"Test recordings: {len(df)}")
print(f"Classes: {len(classes)}")
print(f"Bootstrap repetitions: {N_BOOTSTRAP}")
print("Sampling: within each actual class, with replacement")
print("Comparisons: paired using identical sampled indices")


# ---------------------------------------------------------
# CREATE CLASS-SPECIFIC INDEX GROUPS
# ---------------------------------------------------------

class_indices = {
    class_name: np.flatnonzero(y_true == class_name)
    for class_name in classes
}

for class_name, indices in class_indices.items():
    print(f"  {class_name}: {len(indices)} recordings")


# ---------------------------------------------------------
# OBSERVED TEST METRICS
# ---------------------------------------------------------

observed = {}

for model_name, y_pred in predictions.items():
    observed[model_name] = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=classes,
            average="macro",
            zero_division=0,
        ),
    }


# ---------------------------------------------------------
# STRATIFIED PAIRED BOOTSTRAP
# ---------------------------------------------------------

rng = np.random.default_rng(RANDOM_SEED)

bootstrap_metrics = {
    model_name: {
        "accuracy": np.empty(N_BOOTSTRAP),
        "macro_f1": np.empty(N_BOOTSTRAP),
    }
    for model_name in predictions
}

for repetition in range(N_BOOTSTRAP):

    sampled_parts = []

    # Preserve the original number of recordings in every class.
    for class_name in classes:
        indices = class_indices[class_name]

        sampled = rng.choice(
            indices,
            size=len(indices),
            replace=True,
        )

        sampled_parts.append(sampled)

    # The same sampled indices are used for every model.
    sampled_indices = np.concatenate(sampled_parts)

    sampled_true = y_true[sampled_indices]

    for model_name, y_pred in predictions.items():
        sampled_pred = y_pred[sampled_indices]

        bootstrap_metrics[model_name]["accuracy"][repetition] = (
            accuracy_score(sampled_true, sampled_pred)
        )

        bootstrap_metrics[model_name]["macro_f1"][repetition] = (
            f1_score(
                sampled_true,
                sampled_pred,
                labels=classes,
                average="macro",
                zero_division=0,
            )
        )

    if (repetition + 1) % 2000 == 0:
        print(f"Completed {repetition + 1:,}/{N_BOOTSTRAP:,}")


# ---------------------------------------------------------
# CONFIDENCE INTERVALS
# ---------------------------------------------------------

alpha = 1.0 - CONFIDENCE_LEVEL
lower_percentile = 100 * alpha / 2
upper_percentile = 100 * (1 - alpha / 2)

summary_rows = []

print("\n" + "=" * 70)
print("MODEL PERFORMANCE WITH STRATIFIED BOOTSTRAP 95% CIs")
print("=" * 70)

for model_name in predictions:

    row = {"model": model_name}

    print(f"\n{model_name}")

    for metric in ["accuracy", "macro_f1"]:

        values = bootstrap_metrics[model_name][metric]

        lower, upper = np.percentile(
            values,
            [lower_percentile, upper_percentile],
        )

        point_estimate = observed[model_name][metric]

        row[f"{metric}_observed"] = point_estimate
        row[f"{metric}_bootstrap_mean"] = values.mean()
        row[f"{metric}_ci_lower"] = lower
        row[f"{metric}_ci_upper"] = upper

        print(
            f"  {metric.replace('_', ' ').title()}: "
            f"{point_estimate:.4f} "
            f"(95% CI {lower:.4f}–{upper:.4f})"
        )

    summary_rows.append(row)


# ---------------------------------------------------------
# PAIRED MODEL COMPARISONS
# ---------------------------------------------------------

comparisons = [
    ("CLAP linear probe", "CNN14"),
    ("CLAP linear probe", "SVM"),
    ("CLAP linear probe", "CLAP zero-shot"),
]

comparison_rows = []

print("\n" + "=" * 70)
print("PAIRED MODEL COMPARISONS")
print("=" * 70)

for model_a, model_b in comparisons:

    for metric in ["accuracy", "macro_f1"]:

        differences = (
            bootstrap_metrics[model_a][metric]
            - bootstrap_metrics[model_b][metric]
        )

        lower, upper = np.percentile(
            differences,
            [lower_percentile, upper_percentile],
        )

        observed_difference = (
            observed[model_a][metric]
            - observed[model_b][metric]
        )

        excludes_zero = lower > 0 or upper < 0

        row = {
            "comparison": f"{model_a} minus {model_b}",
            "metric": metric,
            "observed_difference": observed_difference,
            "bootstrap_mean_difference": differences.mean(),
            "ci_lower": lower,
            "ci_upper": upper,
            "ci_excludes_zero": excludes_zero,
        }

        comparison_rows.append(row)

        print(f"\n{model_a} minus {model_b} ({metric})")
        print(f"  Observed difference: {observed_difference:.4f}")
        print(f"  Bootstrap mean difference: {differences.mean():.4f}")
        print(f"  95% CI: {lower:.4f} to {upper:.4f}")
        print(f"  CI excludes zero: {excludes_zero}")


# ---------------------------------------------------------
# SAVE RESULTS
# ---------------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

summary_path = OUTPUT_DIR / "bootstrap_model_confidence_intervals.csv"
comparison_path = OUTPUT_DIR / "bootstrap_paired_model_comparisons.csv"

pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
pd.DataFrame(comparison_rows).to_csv(comparison_path, index=False)

print("\n" + "=" * 70)
print("SAVED RESULTS")
print("=" * 70)
print(summary_path)
print(comparison_path)
