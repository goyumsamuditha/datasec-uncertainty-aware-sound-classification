
from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    ROOT / "models" / "cnn14_final123_temperature_scaling"
    / "validation_logits.npz"
)

OUTPUT_DIR = ROOT / "models" / "cnn14_final123_conformal"

RANDOM_SEED = 42
TARGET_COVERAGE = 0.90
TEMPERATURE_FIT_FRACTION = 0.50


# ---------------------------------------------------------
# Fit temperature on the temperature-fitting subset only
# ---------------------------------------------------------

def fit_temperature(logits, labels):

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    logits_tensor = torch.tensor(
        logits, dtype=torch.float64, device=device
    )

    labels_tensor = torch.tensor(
        labels, dtype=torch.long, device=device
    )

    log_temperature = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.float64, device=device)
    )

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        loss = F.cross_entropy(
            logits_tensor / temperature,
            labels_tensor
        )

        loss.backward()
        return loss

    optimizer.step(closure)

    temperature = float(
        torch.exp(log_temperature.detach()).cpu().item()
    )

    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            f"Invalid temperature: {temperature}"
        )

    return temperature


# ---------------------------------------------------------
# Main procedure
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("CNN14 SPLIT CONFORMAL PREDICTION")
    print("=" * 65)

    if not INPUT_FILE.is_file():
        raise FileNotFoundError(INPUT_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load saved validation logits
    # -----------------------------------------------------

    data = np.load(INPUT_FILE)

    logits = data["logits"].astype(np.float64)
    labels = data["labels"].astype(np.int64)
    paths = data["relative_paths"]
    class_names = data["main_class_names"].tolist()

    n_samples, n_classes = logits.shape

    if n_classes != len(class_names):
        raise ValueError("Class mapping and logits mismatch.")

    if len(labels) != n_samples or len(paths) != n_samples:
        raise ValueError("Input array lengths do not match.")

    print(f"Validation recordings: {n_samples}")
    print(f"Number of classes: {n_classes}")

    # -----------------------------------------------------
    # Reproducible stratified split
    # -----------------------------------------------------

    indices = np.arange(n_samples)

    fit_indices, calibration_indices = train_test_split(
        indices,
        test_size=1.0 - TEMPERATURE_FIT_FRACTION,
        random_state=RANDOM_SEED,
        stratify=labels
    )

    fit_logits = logits[fit_indices]
    fit_labels = labels[fit_indices]

    calibration_logits = logits[calibration_indices]
    calibration_labels = labels[calibration_indices]

    print(f"Temperature-fitting recordings: {len(fit_indices)}")
    print(f"Conformal-calibration recordings: {len(calibration_indices)}")

    if set(fit_indices).intersection(set(calibration_indices)):
        raise RuntimeError("The two subsets overlap.")

    # -----------------------------------------------------
    # Fit temperature using fitting subset only
    # -----------------------------------------------------

    temperature = fit_temperature(
        fit_logits, fit_labels
    )

    print(f"\nFitted temperature: {temperature:.8f}")

    # -----------------------------------------------------
    # Apply frozen temperature to calibration subset
    # -----------------------------------------------------

    shifted_logits = calibration_logits / temperature

    shifted_logits = (
        shifted_logits
        - shifted_logits.max(axis=1, keepdims=True)
    )

    exp_logits = np.exp(shifted_logits)

    probabilities = (
        exp_logits / exp_logits.sum(axis=1, keepdims=True)
    )

    # -----------------------------------------------------
    # Calculate nonconformity scores
    # -----------------------------------------------------

    true_class_probabilities = probabilities[
        np.arange(len(calibration_labels)),
        calibration_labels
    ]

    nonconformity_scores = 1.0 - true_class_probabilities

    # Finite-sample split-conformal quantile.
    n_calibration = len(nonconformity_scores)

    alpha = 1.0 - TARGET_COVERAGE

    quantile_rank = int(
        np.ceil((n_calibration + 1) * TARGET_COVERAGE)
    )

    if quantile_rank > n_calibration:
        raise ValueError(
            "Insufficient calibration samples for this coverage."
        )

    threshold = float(
        np.sort(nonconformity_scores)[quantile_rank - 1]
    )

    print(f"\nTarget coverage: {TARGET_COVERAGE:.2%}")
    print(f"Nonconformity quantile rank: {quantile_rank}")
    print(f"Conformal threshold: {threshold:.8f}")

    # -----------------------------------------------------
    # Construct prediction sets
    # -----------------------------------------------------

    prediction_sets = probabilities >= (1.0 - threshold)

    # Include numerical tolerance at the threshold.
    prediction_sets = probabilities >= (
        1.0 - threshold - 1e-12
    )

    true_class_in_set = prediction_sets[
        np.arange(n_calibration),
        calibration_labels
    ]

    set_sizes = prediction_sets.sum(axis=1)

    empirical_coverage = float(
        true_class_in_set.mean()
    )

    average_set_size = float(set_sizes.mean())

    singleton_rate = float(
        (set_sizes == 1).mean()
    )

    full_set_rate = float(
        (set_sizes == n_classes).mean()
    )

    print("\nCalibration subset results:")
    print(f"Empirical coverage: {empirical_coverage:.4f}")
    print(f"Average prediction-set size: {average_set_size:.4f}")
    print(f"Singleton prediction rate: {singleton_rate:.4f}")
    print(f"Full-set prediction rate: {full_set_rate:.4f}")

    # -----------------------------------------------------
    # Create detailed prediction output
    # -----------------------------------------------------

    records = []

    for i, original_index in enumerate(calibration_indices):

        included_indices = np.flatnonzero(
            prediction_sets[i]
        )

        included_classes = [
            class_names[j] for j in included_indices
        ]

        predicted_class_index = int(
            probabilities[i].argmax()
        )

        records.append({
            "relative_path": paths[original_index],
            "actual_class": class_names[
                calibration_labels[i]
            ],
            "top_prediction": class_names[
                predicted_class_index
            ],
            "top_prediction_score": float(
                probabilities[i].max()
            ),
            "prediction_set": " | ".join(included_classes),
            "prediction_set_size": int(set_sizes[i]),
            "true_class_in_set": bool(
                true_class_in_set[i]
            )
        })

    prediction_df = pd.DataFrame(records)

    prediction_df.to_csv(
        OUTPUT_DIR / "calibration_prediction_sets.csv",
        index=False
    )

    # -----------------------------------------------------
    # Save summary and calibration parameters
    # -----------------------------------------------------

    metrics = {
        "method": "Global split conformal prediction",
        "nonconformity_score": "1 - calibrated true-class probability",
        "target_coverage": TARGET_COVERAGE,
        "alpha": alpha,
        "random_seed": RANDOM_SEED,
        "temperature_fit_fraction": TEMPERATURE_FIT_FRACTION,
        "temperature_fit_size": int(len(fit_indices)),
        "conformal_calibration_size": int(n_calibration),
        "number_of_classes": int(n_classes),
        "fitted_temperature": temperature,
        "conformal_quantile_rank": quantile_rank,
        "conformal_threshold": threshold,
        "empirical_calibration_coverage": empirical_coverage,
        "average_prediction_set_size": average_set_size,
        "singleton_prediction_rate": singleton_rate,
        "full_set_rate": full_set_rate,
        "test_partition_accessed": False
    }

    with open(
        OUTPUT_DIR / "conformal_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    np.savez_compressed(
        OUTPUT_DIR / "conformal_parameters.npz",
        temperature=np.asarray(temperature),
        threshold=np.asarray(threshold),
        target_coverage=np.asarray(TARGET_COVERAGE),
        random_seed=np.asarray(RANDOM_SEED),
        fit_indices=fit_indices,
        calibration_indices=calibration_indices,
        class_names=np.asarray(class_names)
    )

    print("\nSaved files:")
    print(OUTPUT_DIR / "conformal_metrics.json")
    print(OUTPUT_DIR / "conformal_parameters.npz")
    print(OUTPUT_DIR / "calibration_prediction_sets.csv")

    print("\nSplit conformal calibration completed.")
    print("The test partition was not accessed.")


if __name__ == "__main__":
    main()
