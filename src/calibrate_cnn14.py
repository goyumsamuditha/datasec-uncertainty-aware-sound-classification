
from pathlib import Path
import sys
import json

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, log_loss


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training

CHECKPOINT_PATH = (
    ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)

OUTPUT_DIR = ROOT / "models" / "cnn14_final123_temperature_scaling"

BATCH_SIZE = 4
NUM_WORKERS = 0
ECE_BINS = 15


# ---------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------

def calculate_ece(probabilities, labels, n_bins=15):
    """Calculate top-label expected calibration error."""

    confidences = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correctness = (predictions == labels).astype(float)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (
                (confidences >= bin_edges[i])
                & (confidences <= bin_edges[i + 1])
            )
        else:
            mask = (
                (confidences >= bin_edges[i])
                & (confidences < bin_edges[i + 1])
            )

        if np.any(mask):
            bin_accuracy = correctness[mask].mean()
            bin_confidence = confidences[mask].mean()

            ece += (
                mask.mean()
                * abs(bin_accuracy - bin_confidence)
            )

    return float(ece)


def calculate_brier_score(probabilities, labels):
    """Multiclass Brier score: mean sum of squared errors."""

    one_hot = np.eye(probabilities.shape[1])[labels]

    return float(
        np.mean(np.sum((probabilities - one_hot) ** 2, axis=1))
    )


def evaluate_predictions(logits, labels):
    """Calculate classification and calibration metrics."""

    probabilities = torch.softmax(
        torch.tensor(logits, dtype=torch.float64),
        dim=1
    ).numpy()

    predictions = probabilities.argmax(axis=1)

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(
            f1_score(labels, predictions, average="macro")
        ),
        "nll": float(
            log_loss(
                labels,
                probabilities,
                labels=np.arange(probabilities.shape[1])
            )
        ),
        "ece": calculate_ece(
            probabilities, labels, ECE_BINS
        ),
        "brier_score": calculate_brier_score(
            probabilities, labels
        )
    }, probabilities


# ---------------------------------------------------------
# Main procedure
# ---------------------------------------------------------

def main():

    print("=" * 65)
    print("CNN14 TEMPERATURE SCALING")
    print("=" * 65)

    device = training.DEVICE
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load mappings and validation manifest
    # -----------------------------------------------------

    main_map, sub_map = training.load_label_maps()
    df = training.load_manifest()

    val_df = (
        df[df["partition"] == "validation"]
        .copy()
        .reset_index(drop=True)
    )

    if val_df.empty:
        raise ValueError("No validation recordings found.")

    print(f"Validation recordings: {len(val_df)}")

    parents = training.build_parent_mapping(
        df, main_map, sub_map
    )

    # Retain the same preprocessing as the original analysis.
    val_dataset = training.DataSECDataset(
        val_df,
        main_map,
        sub_map,
        training=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Load the existing CNN14 checkpoint
    # -----------------------------------------------------

    model = training.DataSECCNN14().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Checkpoint main mapping does not match.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Checkpoint subclass mapping does not match.")

    main_names = [
        name for name, index in sorted(
            main_map.items(), key=lambda item: item[1]
        )
    ]

    print(f"Loaded checkpoint epoch: {checkpoint['epoch']}")
    print(f"Number of main classes: {len(main_names)}")

    # -----------------------------------------------------
    # Collect complete validation logits
    # -----------------------------------------------------

    all_logits = []
    all_labels = []
    all_paths = []

    with torch.inference_mode():

        for batch_index, (
            waveforms,
            main_labels,
            subclass_labels
        ) in enumerate(val_loader):

            waveforms = waveforms.to(
                device, non_blocking=True
            )

            main_logits, subclass_logits = model(waveforms)

            all_logits.append(main_logits.cpu().numpy())
            all_labels.extend(main_labels.tolist())

            batch_start = batch_index * BATCH_SIZE

            for i in range(len(main_labels)):
                all_paths.append(
                    val_df.iloc[batch_start + i]["relative_path"]
                )

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed validation batches: "
                    f"{batch_index + 1}"
                )

    logits = np.concatenate(all_logits, axis=0)
    labels = np.asarray(all_labels, dtype=np.int64)

    if logits.shape != (len(val_df), len(main_names)):
        raise ValueError(
            f"Unexpected logits shape: {logits.shape}"
        )

    if len(labels) != len(val_df):
        raise ValueError("Validation label count mismatch.")

    if len(all_paths) != len(val_df):
        raise ValueError("Validation path count mismatch.")

    print(f"Collected logits: {logits.shape}")

    # -----------------------------------------------------
    # Evaluate before calibration
    # -----------------------------------------------------

    before_metrics, before_probabilities = evaluate_predictions(
        logits, labels
    )

    print("\nBefore temperature scaling:")
    for name, value in before_metrics.items():
        print(f"{name}: {value:.6f}")

    # -----------------------------------------------------
    # Fit one temperature using validation NLL
    # -----------------------------------------------------

    logits_tensor = torch.tensor(
        logits,
        dtype=torch.float64,
        device=device
    )

    labels_tensor = torch.tensor(
        labels,
        dtype=torch.long,
        device=device
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
            f"Invalid fitted temperature: {temperature}"
        )

    print(f"\nFitted temperature: {temperature:.8f}")

    # -----------------------------------------------------
    # Evaluate after calibration
    # -----------------------------------------------------

    calibrated_logits = logits / temperature

    after_metrics, after_probabilities = evaluate_predictions(
        calibrated_logits, labels
    )

    print("\nAfter temperature scaling:")
    for name, value in after_metrics.items():
        print(f"{name}: {value:.6f}")

    # Temperature scaling should preserve argmax predictions.
    original_predictions = logits.argmax(axis=1)
    calibrated_predictions = calibrated_logits.argmax(axis=1)

    if not np.array_equal(
        original_predictions, calibrated_predictions
    ):
        raise RuntimeError(
            "Unexpected change in predicted classes."
        )

    # -----------------------------------------------------
    # Save outputs
    # -----------------------------------------------------

    metrics = {
        "checkpoint": str(CHECKPOINT_PATH),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "validation_recordings": int(len(labels)),
        "number_of_classes": int(len(main_names)),
        "temperature": temperature,
        "calibration_method": "Single-parameter temperature scaling",
        "fitting_objective": "Validation negative log-likelihood",
        "ece_bins": ECE_BINS,
        "before_calibration": before_metrics,
        "after_calibration": after_metrics,
        "test_partition_accessed": False
    }

    with open(
        OUTPUT_DIR / "calibration_metrics.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(metrics, file, indent=4)

    np.savez_compressed(
        OUTPUT_DIR / "validation_logits.npz",
        logits=logits,
        calibrated_logits=calibrated_logits,
        labels=labels,
        relative_paths=np.asarray(all_paths),
        main_class_names=np.asarray(main_names),
        temperature=np.asarray(temperature)
    )

    calibrated_predictions = after_probabilities.argmax(axis=1)

    prediction_df = pd.DataFrame({
        "relative_path": all_paths,
        "actual_class": [main_names[i] for i in labels],
        "predicted_class": [
            main_names[i] for i in calibrated_predictions
        ],
        "correct": calibrated_predictions == labels,
        "confidence_before": before_probabilities.max(axis=1),
        "confidence_after": after_probabilities.max(axis=1)
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "validation_calibrated_predictions.csv",
        index=False
    )

    comparison_df = pd.DataFrame([
        {"stage": "before", **before_metrics},
        {"stage": "after", **after_metrics}
    ])

    comparison_df.to_csv(
        OUTPUT_DIR / "calibration_comparison.csv",
        index=False
    )

    print("\nSaved outputs:")
    print(OUTPUT_DIR / "calibration_metrics.json")
    print(OUTPUT_DIR / "validation_logits.npz")
    print(OUTPUT_DIR / "validation_calibrated_predictions.csv")
    print(OUTPUT_DIR / "calibration_comparison.csv")

    print("\nTemperature scaling completed.")
    print("The test partition was not accessed.")


if __name__ == "__main__":
    main()
