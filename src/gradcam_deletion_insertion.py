
from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_cnn14 as training


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)

OUTPUT_DIR = ROOT / "models" / "cnn14_final123_faithfulness"
FIGURE_DIR = ROOT / "report" / "gradcam_final123_faithfulness"

N_EXAMPLES = 20
GRID_ROWS = 8
GRID_COLS = 8
RANDOM_REPEATS = 3
CURVE_POINTS = 11
MASK_VALUE = -100.0
SEED = 42

DEVICE = training.DEVICE


# ---------------------------------------------------------
# Grad-CAM
# ---------------------------------------------------------

def calculate_gradcam(model, waveform):
    """Calculate Grad-CAM for the model's original predicted class."""

    captured = {}

    def capture(module, inputs, output):
        captured["activations"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients", gradient.detach()
            )
        )

    hook = model.backbone.conv_block6.register_forward_hook(capture)

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits, _ = model(waveform)
            predicted_id = int(logits.argmax(dim=1).item())
            target_score = logits[0, predicted_id]
            target_score.backward()

        if "activations" not in captured or "gradients" not in captured:
            raise RuntimeError("Grad-CAM activations or gradients missing.")

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)

        cam = F.relu(
            (weights * activations).sum(dim=1, keepdim=True)
        )

        cam_min = cam.min()
        cam_max = cam.max()

        if (cam_max - cam_min).item() > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = torch.zeros_like(cam)

        with torch.no_grad():
            spectrogram = model.backbone.spectrogram_extractor(waveform)
            logmel = model.backbone.logmel_extractor(spectrogram)

        logmel_image = logmel.squeeze().detach().cpu().numpy()

        if logmel_image.ndim != 2:
            raise RuntimeError(
                f"Unexpected log-mel shape: {logmel_image.shape}"
            )

        cam = F.interpolate(
            cam,
            size=logmel_image.shape,
            mode="bilinear",
            align_corners=False,
        )

        cam_image = cam.squeeze().detach().cpu().numpy()

        probabilities = F.softmax(logits.detach(), dim=1)
        original_probability = float(
            probabilities[0, predicted_id].item()
        )

        return (
            cam_image,
            logmel_image,
            predicted_id,
            original_probability,
        )

    finally:
        hook.remove()


# ---------------------------------------------------------
# Region ranking and masking
# ---------------------------------------------------------

def rank_regions(cam):
    """Rank spectrogram regions by average Grad-CAM activation."""

    time_steps, mel_bins = cam.shape

    scores = []

    for row in range(GRID_ROWS):
        t_start = row * time_steps // GRID_ROWS
        t_end = (row + 1) * time_steps // GRID_ROWS

        for col in range(GRID_COLS):
            f_start = col * mel_bins // GRID_COLS
            f_end = (col + 1) * mel_bins // GRID_COLS

            region = cam[t_start:t_end, f_start:f_end]

            score = float(region.mean()) if region.size else 0.0

            scores.append((score, row, col))

    scores.sort(key=lambda item: item[0], reverse=True)

    return [(row, col) for _, row, col in scores]


def make_region_mask(order, number_of_regions, time_steps, mel_bins):
    """Create a boolean mask for selected spectrogram regions."""

    mask = torch.zeros(
        (1, 1, time_steps, mel_bins),
        dtype=torch.bool,
        device=DEVICE,
    )

    for row, col in order[:number_of_regions]:
        t_start = row * time_steps // GRID_ROWS
        t_end = (row + 1) * time_steps // GRID_ROWS

        f_start = col * mel_bins // GRID_COLS
        f_end = (col + 1) * mel_bins // GRID_COLS

        mask[:, :, t_start:t_end, f_start:f_end] = True

    return mask


# ---------------------------------------------------------
# Evaluate deletion or insertion
# ---------------------------------------------------------

def evaluate_probability(model, waveform, target_id, mask, mode):
    """
    Evaluate the original predicted class after masking log-mel regions.

    deletion:
        Selected regions are replaced by the baseline.

    insertion:
        Unselected regions are replaced by the baseline.
    """

    def apply_mask(module, inputs, output):
        modified = output.clone()

        if mode == "deletion":
            modified[mask.expand_as(modified)] = MASK_VALUE

        elif mode == "insertion":
            modified[~mask.expand_as(modified)] = MASK_VALUE

        else:
            raise ValueError(f"Unknown masking mode: {mode}")

        return modified

    hook = model.backbone.logmel_extractor.register_forward_hook(
        apply_mask
    )

    try:
        with torch.no_grad():
            logits, _ = model(waveform)
            probability = F.softmax(logits, dim=1)[0, target_id].item()

        return float(probability)

    finally:
        hook.remove()


# ---------------------------------------------------------
# Area under the curve
# ---------------------------------------------------------

def calculate_auc(fractions, probabilities):
    return float(np.trapz(probabilities, fractions))


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    print(f"Device: {DEVICE}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    main_map, sub_map = training.load_label_maps()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Main-class mapping mismatch.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Subclass mapping mismatch.")

    model = training.DataSECCNN14().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    print(f"Loaded FINAL-123 checkpoint, epoch {checkpoint['epoch']}")

    # -----------------------------------------------------
    # Validation data only
    # -----------------------------------------------------

    manifest = training.load_manifest()

    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy()

    if len(validation_df) < N_EXAMPLES:
        raise ValueError("Not enough validation recordings.")

    validation_df = validation_df.sample(
        n=N_EXAMPLES,
        random_state=SEED,
    ).reset_index(drop=True)

    dataset = training.DataSECDataset(
        validation_df,
        main_map,
        sub_map,
        training=False,
    )

    id_to_name = {value: key for key, value in main_map.items()}

    fractions = np.linspace(0.0, 1.0, CURVE_POINTS)

    curve_rows = []
    example_rows = []

    # -----------------------------------------------------
    # Process each recording
    # -----------------------------------------------------

    for index in range(len(dataset)):

        waveform, actual_id, _ = dataset[index]
        waveform = waveform.unsqueeze(0).to(DEVICE)

        (
            cam,
            logmel,
            predicted_id,
            original_probability,
        ) = calculate_gradcam(model, waveform)

        time_steps, mel_bins = logmel.shape

        cam_std = float(np.std(cam))
        gradcam_valid = cam_std >= 1e-12

        gradcam_order = rank_regions(cam)

        total_regions = GRID_ROWS * GRID_COLS

        print(
            f"\n[{index + 1}/{N_EXAMPLES}] "
            f"{id_to_name[int(actual_id.item())]} -> "
            f"{id_to_name[predicted_id]}"
        )

        print(
            f"Original probability: {original_probability:.4f} | "
            f"Grad-CAM std: {cam_std:.6f} | "
            f"Grad-CAM ranking valid: {gradcam_valid}"
        )

        gradcam_deletion = []
        gradcam_insertion = []

        # -------------------------------------------------
        # Grad-CAM-ranked deletion and insertion
        # -------------------------------------------------

        if gradcam_valid:

            for point, fraction in enumerate(fractions):

                number = int(round(fraction * total_regions))

                selected_mask = make_region_mask(
                    gradcam_order,
                    number,
                    time_steps,
                    mel_bins,
                )

                deletion_probability = evaluate_probability(
                    model,
                    waveform,
                    predicted_id,
                    selected_mask,
                    "deletion",
                )

                insertion_probability = evaluate_probability(
                    model,
                    waveform,
                    predicted_id,
                    selected_mask,
                    "insertion",
                )

                gradcam_deletion.append(deletion_probability)
                gradcam_insertion.append(insertion_probability)

                curve_rows.append({
                    "example": index + 1,
                    "relative_path": str(
                        validation_df.iloc[index]["relative_path"]
                    ),
                    "method": "gradcam_deletion",
                    "repeat": 0,
                    "fraction": fraction,
                    "probability": deletion_probability,
                })

                curve_rows.append({
                    "example": index + 1,
                    "relative_path": str(
                        validation_df.iloc[index]["relative_path"]
                    ),
                    "method": "gradcam_insertion",
                    "repeat": 0,
                    "fraction": fraction,
                    "probability": insertion_probability,
                })

        # -------------------------------------------------
        # Random-ranking deletion and insertion
        # -------------------------------------------------

        random_deletion_curves = []
        random_insertion_curves = []

        for repeat in range(RANDOM_REPEATS):

            random_order = [
                (row, col)
                for row in range(GRID_ROWS)
                for col in range(GRID_COLS)
            ]

            rng = random.Random(
                SEED + index * 1000 + repeat
            )

            rng.shuffle(random_order)

            deletion_curve = []
            insertion_curve = []

            for fraction in fractions:

                number = int(round(fraction * total_regions))

                selected_mask = make_region_mask(
                    random_order,
                    number,
                    time_steps,
                    mel_bins,
                )

                deletion_probability = evaluate_probability(
                    model,
                    waveform,
                    predicted_id,
                    selected_mask,
                    "deletion",
                )

                insertion_probability = evaluate_probability(
                    model,
                    waveform,
                    predicted_id,
                    selected_mask,
                    "insertion",
                )

                deletion_curve.append(deletion_probability)
                insertion_curve.append(insertion_probability)

                curve_rows.append({
                    "example": index + 1,
                    "relative_path": str(
                        validation_df.iloc[index]["relative_path"]
                    ),
                    "method": "random_deletion",
                    "repeat": repeat + 1,
                    "fraction": fraction,
                    "probability": deletion_probability,
                })

                curve_rows.append({
                    "example": index + 1,
                    "relative_path": str(
                        validation_df.iloc[index]["relative_path"]
                    ),
                    "method": "random_insertion",
                    "repeat": repeat + 1,
                    "fraction": fraction,
                    "probability": insertion_probability,
                })

            random_deletion_curves.append(deletion_curve)
            random_insertion_curves.append(insertion_curve)

        # -------------------------------------------------
        # Per-recording AUC metrics
        # -------------------------------------------------

        random_deletion_aucs = [
            calculate_auc(fractions, curve)
            for curve in random_deletion_curves
        ]

        random_insertion_aucs = [
            calculate_auc(fractions, curve)
            for curve in random_insertion_curves
        ]

        if gradcam_valid:

            gradcam_deletion_auc = calculate_auc(
                fractions,
                gradcam_deletion,
            )

            gradcam_insertion_auc = calculate_auc(
                fractions,
                gradcam_insertion,
            )

        else:
            gradcam_deletion_auc = np.nan
            gradcam_insertion_auc = np.nan

        random_deletion_auc = float(
            np.mean(random_deletion_aucs)
        )

        random_insertion_auc = float(
            np.mean(random_insertion_aucs)
        )

        example_rows.append({
            "example": index + 1,
            "relative_path": str(
                validation_df.iloc[index]["relative_path"]
            ),
            "actual_class": id_to_name[int(actual_id.item())],
            "predicted_class": id_to_name[predicted_id],
            "prediction_correct": (
                int(actual_id.item()) == predicted_id
            ),
            "original_probability": original_probability,
            "gradcam_std": cam_std,
            "gradcam_ranking_valid": gradcam_valid,
            "gradcam_deletion_auc": gradcam_deletion_auc,
            "random_deletion_auc": random_deletion_auc,
            "gradcam_insertion_auc": gradcam_insertion_auc,
            "random_insertion_auc": random_insertion_auc,
        })

        print(
            f"  Grad-CAM deletion AUC: {gradcam_deletion_auc:.4f} | "
            f"Random deletion AUC: {random_deletion_auc:.4f}"
        )

        print(
            f"  Grad-CAM insertion AUC: {gradcam_insertion_auc:.4f} | "
            f"Random insertion AUC: {random_insertion_auc:.4f}"
        )

    # -----------------------------------------------------
    # Save detailed results
    # -----------------------------------------------------

    curves_df = pd.DataFrame(curve_rows)
    examples_df = pd.DataFrame(example_rows)

    curves_path = OUTPUT_DIR / "deletion_insertion_curves.csv"
    examples_path = OUTPUT_DIR / "deletion_insertion_per_example.csv"

    curves_df.to_csv(curves_path, index=False)
    examples_df.to_csv(examples_path, index=False)

    # -----------------------------------------------------
    # Aggregate summary
    # -----------------------------------------------------

    valid_examples = examples_df[
        examples_df["gradcam_ranking_valid"]
    ].copy()

    summary_rows = []

    for metric, description in [
        ("gradcam_deletion_auc", "Grad-CAM deletion"),
        ("random_deletion_auc", "Random deletion"),
        ("gradcam_insertion_auc", "Grad-CAM insertion"),
        ("random_insertion_auc", "Random insertion"),
    ]:

        values = examples_df[metric].dropna()

        summary_rows.append({
            "metric": description,
            "valid_examples": len(values),
            "mean_auc": values.mean(),
            "std_auc": values.std(),
        })

    summary_df = pd.DataFrame(summary_rows)

    summary_path = OUTPUT_DIR / "deletion_insertion_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    # Paired comparison on examples with non-constant Grad-CAM.
    if len(valid_examples) > 0:

        paired_summary = pd.DataFrame([
            {
                "comparison": "Deletion: Grad-CAM minus random",
                "valid_examples": len(valid_examples),
                "mean_difference": (
                    valid_examples["gradcam_deletion_auc"]
                    - valid_examples["random_deletion_auc"]
                ).mean(),
            },
            {
                "comparison": "Insertion: Grad-CAM minus random",
                "valid_examples": len(valid_examples),
                "mean_difference": (
                    valid_examples["gradcam_insertion_auc"]
                    - valid_examples["random_insertion_auc"]
                ).mean(),
            },
        ])

    else:
        paired_summary = pd.DataFrame()

    paired_path = OUTPUT_DIR / "deletion_insertion_paired_comparison.csv"
    paired_summary.to_csv(paired_path, index=False)

    # -----------------------------------------------------
    # Aggregate deletion/insertion plots
    # -----------------------------------------------------

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 5),
        constrained_layout=True,
    )

    plot_methods = [
        ("gradcam_deletion", "Grad-CAM"),
        ("random_deletion", "Random baseline"),
        ("gradcam_insertion", "Grad-CAM"),
        ("random_insertion", "Random baseline"),
    ]

    for method, label in plot_methods:

        subset = curves_df[
            curves_df["method"] == method
        ]

        if subset.empty:
            continue

        # Average repeated random runs within each recording first.
        per_example = (
            subset.groupby(["example", "fraction"])["probability"]
            .mean()
            .reset_index()
        )

        aggregate = (
            per_example.groupby("fraction")["probability"]
            .agg(["mean", "std"])
            .reset_index()
        )

        axis = axes[0] if "deletion" in method else axes[1]

        axis.plot(
            aggregate["fraction"],
            aggregate["mean"],
            marker="o",
            label=label,
        )

        axis.fill_between(
            aggregate["fraction"],
            aggregate["mean"] - aggregate["std"].fillna(0),
            aggregate["mean"] + aggregate["std"].fillna(0),
            alpha=0.15,
        )

    axes[0].set_title("Deletion curves")
    axes[0].set_xlabel("Fraction of regions deleted")
    axes[0].set_ylabel("Original predicted-class probability")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].set_title("Insertion curves")
    axes[1].set_xlabel("Fraction of regions inserted")
    axes[1].set_ylabel("Original predicted-class probability")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plot_path = FIGURE_DIR / "deletion_insertion_curves.png"

    fig.savefig(plot_path, dpi=180)
    plt.close(fig)

    # -----------------------------------------------------
    # Final output
    # -----------------------------------------------------

    print("\nDeletion/insertion experiment completed.")
    print(f"Recordings: {len(examples_df)}")
    print(
        "Valid Grad-CAM rankings:",
        int(examples_df["gradcam_ranking_valid"].sum()),
    )
    print(f"Curve results: {curves_path}")
    print(f"Per-example results: {examples_path}")
    print(f"Summary: {summary_path}")
    print(f"Paired comparison: {paired_path}")
    print(f"Plot: {plot_path}")

    print("\nSummary:")
    print(summary_df.to_string(index=False))

    print("\nPaired comparison:")
    print(paired_summary.to_string(index=False))

    print("\nValidation partition only. Test partition was not accessed.")


if __name__ == "__main__":
    main()
