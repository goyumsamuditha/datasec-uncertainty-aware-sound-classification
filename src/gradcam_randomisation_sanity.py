
from __future__ import annotations

import copy
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT_PATH = (
    PROJECT_ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT / "models" / "cnn14_final123_gradcam_sanity"
)
FIGURE_DIR = (
    PROJECT_ROOT / "report" / "gradcam_final123_sanity"
)

SEED = 42
DEVICE = training.DEVICE

TARGET_PATHS = [
    r"Lawn mower brush cutter and olive shaker\Olive shaker\Olive shaker-0004.wav",
    r"Music\Music-0080.wav",
    r"Vehicle pass-by\car pass-by\car pass-by-0017.wav",
    r"Propeller aircrafts\Helicopters\Helicopters-0015.wav",
    r"Vehicle idling\car truck idling\car truck idling-0063.wav",
]

CONDITIONS = [
    "original",
    "randomised_main_head",
    "randomised_fc1_and_main_head",
    "randomised_conv6_fc1_and_main_head",
]


def randomise_module(module):
    """Reinitialise parameters using each layer's reset_parameters method."""
    for child in module.modules():
        if child is module:
            continue
        if hasattr(child, "reset_parameters"):
            child.reset_parameters()

    if hasattr(module, "reset_parameters"):
        module.reset_parameters()


def make_model(checkpoint, condition):
    """Create a fresh checkpoint-based model for each condition."""
    model = training.DataSECCNN14().to(DEVICE)
    model.load_state_dict(
        copy.deepcopy(checkpoint["model_state_dict"]),
        strict=True,
    )

    if condition == "randomised_main_head":
        randomise_module(model.main_head)

    elif condition == "randomised_fc1_and_main_head":
        randomise_module(model.backbone.fc1)
        randomise_module(model.main_head)

    elif condition == "randomised_conv6_fc1_and_main_head":
        randomise_module(model.backbone.conv_block6)
        randomise_module(model.backbone.fc1)
        randomise_module(model.main_head)

    elif condition != "original":
        raise ValueError(f"Unknown condition: {condition}")

    model.eval()
    return model


def calculate_gradcam(model, waveform, target_id):
    """Calculate Grad-CAM for a fixed target class."""
    captured = {}

    def capture_activations(module, inputs, output):
        captured["activations"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients", gradient.detach()
            )
        )

    hook = model.backbone.conv_block6.register_forward_hook(
        capture_activations
    )

    try:
        model.zero_grad(set_to_none=True)

        with torch.enable_grad():
            logits, _ = model(waveform)
            target_score = logits[0, target_id]
            target_score.backward()

        if "activations" not in captured or "gradients" not in captured:
            raise RuntimeError("Grad-CAM activations or gradients missing.")

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

        weights = gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

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

        cam_resized = F.interpolate(
            cam,
            size=logmel_image.shape,
            mode="bilinear",
            align_corners=False,
        )

        cam_image = cam_resized.squeeze().detach().cpu().numpy()

        return (
            cam_image,
            logmel_image,
            float(target_score.detach().item()),
            int(logits.argmax(dim=1).item()),
        )

    finally:
        hook.remove()


def heatmap_statistics(heatmap):
    """Return basic statistics for a Grad-CAM heatmap."""
    values = heatmap.flatten().astype(np.float64)

    return {
        "std": float(np.std(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "is_constant": bool(np.std(values) < 1e-12),
    }
    
    
def normalised_similarity(reference, comparison):
    """Pearson correlation, undefined for constant heatmaps."""
    a = reference.flatten().astype(np.float64)
    b = comparison.flatten().astype(np.float64)

    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")

    return float(np.corrcoef(a, b)[0, 1])


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False,
    )

    main_map, sub_map = training.load_label_maps()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Checkpoint main-class mapping does not match.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Checkpoint subclass mapping does not match.")

    # -----------------------------------------------------
    # Load validation recordings only
    # -----------------------------------------------------

    manifest = training.load_manifest()
    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy()

    validation_df["relative_path"] = (
        validation_df["relative_path"].astype(str)
    )

    available_paths = set(validation_df["relative_path"])

    missing_paths = [
        path for path in TARGET_PATHS
        if path not in available_paths
    ]

    if missing_paths:
        raise ValueError(f"Missing validation recordings: {missing_paths}")

    validation_df = (
        validation_df.set_index("relative_path")
        .loc[TARGET_PATHS]
        .reset_index()
    )

    dataset = training.DataSECDataset(
        validation_df,
        main_map,
        sub_map,
        training=False,
    )

    id_to_main = {value: key for key, value in main_map.items()}

    # -----------------------------------------------------
    # Original model: establish fixed prediction targets
    # -----------------------------------------------------

    original_model = make_model(checkpoint, "original")

    examples = []

    print("\nEstablishing original predictions...")

    for index in range(len(dataset)):
        waveform, actual_id, _ = dataset[index]
        waveform = waveform.unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            logits, _ = original_model(waveform)

        predicted_id = int(logits.argmax(dim=1).item())

        examples.append({
            "waveform": waveform,
            "actual_id": int(actual_id.item()),
            "target_id": predicted_id,
            "relative_path": TARGET_PATHS[index],
        })

        print(
            f"{index + 1}/{len(dataset)}: "
            f"{id_to_main[int(actual_id.item())]} -> "
            f"{id_to_main[predicted_id]}"
        )

    del original_model

    # -----------------------------------------------------
    # Grad-CAM randomisation experiment
    # -----------------------------------------------------

    results = []
    reference_maps = {}

    for condition in CONDITIONS:
        print(f"\nCondition: {condition}")

        # Each condition starts from the original checkpoint.
        model = make_model(checkpoint, condition)

        for index, example in enumerate(examples):
            cam, logmel, target_logit, current_prediction = (
                calculate_gradcam(
                    model,
                    example["waveform"],
                    example["target_id"],
                )
            )

            if condition == "original":
                reference_maps[index] = cam.copy()
                
            reference = reference_maps[index]
            
            reference_stats = heatmap_statistics(reference)
            current_stats = heatmap_statistics(cam)
            
            if condition == "original":
                similarity = (
                    1.0
                    if not current_stats["is_constant"]
                    else float("nan")
                )
                
            else:
                similarity = normalised_similarity(reference, cam)
                
            mean_absolute_difference = float(np.mean(np.abs(reference - cam)))


            actual_name = id_to_main[example["actual_id"]]
            target_name = id_to_main[example["target_id"]]
            current_name = id_to_main[current_prediction]

            fig, axes = plt.subplots(
                2, 1,
                figsize=(12, 8),
                sharex=True,
                constrained_layout=True,
            )

            extent = [
                0,
                training.CLIP_SECONDS,
                0,
                training.MEL_BINS,
            ]

            axes[0].imshow(
                logmel.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
            )
            axes[0].set_title("Log-mel spectrogram")
            axes[0].set_ylabel("Mel-bin index")

            axes[1].imshow(
                logmel.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="gray",
            )
            axes[1].imshow(
                cam.T,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )

            axes[1].set_title(
                f"Grad-CAM target: {target_name} | "
                f"Current prediction: {current_name}"
            )
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("Mel-bin index")

            fig.suptitle(
                f"{condition} | Actual: {actual_name} | "
                f"Original target: {target_name}",
                fontsize=11,
            )

            image_name = f"{condition}_example_{index + 1:02d}.png"
            image_path = FIGURE_DIR / image_name

            fig.savefig(image_path, dpi=160)
            plt.close(fig)

            results.append({
                "condition": condition,
                "relative_path": example["relative_path"],
                "actual_class": actual_name,
                "original_target_class": target_name,
                "current_predicted_class": current_name,
                "current_prediction_matches_original": (
                    current_prediction == example["target_id"]
                ),
                "target_logit": target_logit,
                "heatmap_pearson_correlation": similarity,
                "reference_heatmap_std": reference_stats["std"],
                "current_heatmap_std": current_stats["std"],
                "reference_heatmap_min": reference_stats["min"],
                "reference_heatmap_max": reference_stats["max"],
                "current_heatmap_min": current_stats["min"],
                "current_heatmap_max": current_stats["max"],
                "reference_heatmap_is_constant": (
                    reference_stats["is_constant"]
                ),
                "current_heatmap_is_constant": (
                    current_stats["is_constant"]
                ),
                "heatmap_mean_absolute_difference": (
                    mean_absolute_difference
                ),
                "gradcam_image": str(
                    image_path.relative_to(PROJECT_ROOT)
                ),
            })

            print(
                f"  Example {index + 1}: "
                f"prediction={current_name}, "
                f"heatmap correlation={similarity:.4f}"
            )

        del model

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)

    results_path = OUTPUT_DIR / "randomisation_sanity_results.csv"
    results_df.to_csv(results_path, index=False)

    summary = (
        results_df.groupby("condition")
        .agg(
            total_examples=(
                "heatmap_pearson_correlation", "size"
            ),
            valid_correlations=(
                "heatmap_pearson_correlation", "count"
            ),
            mean_heatmap_correlation=(
                "heatmap_pearson_correlation", "mean"
            ),
            std_heatmap_correlation=(
                "heatmap_pearson_correlation", "std"
            ),
            constant_reference_maps=(
                "reference_heatmap_is_constant", "sum"
            ),
            constant_current_maps=(
                "current_heatmap_is_constant", "sum"
            ),
            mean_heatmap_difference=(
                "heatmap_mean_absolute_difference", "mean"
            ),
            prediction_match_rate=(
                "current_prediction_matches_original", "mean"
            ),
        )
        .reset_index()
    )

    summary["undefined_correlations"] = (
        summary["total_examples"] - summary["valid_correlations"]
    )

    summary_path = OUTPUT_DIR / "randomisation_sanity_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nRandomisation sanity check completed.")
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")
    print(f"Figures: {FIGURE_DIR}")
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nValidation partition only. Test partition was not accessed.")


if __name__ == "__main__":
    main()
