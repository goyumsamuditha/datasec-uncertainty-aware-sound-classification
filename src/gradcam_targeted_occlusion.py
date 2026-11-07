
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

import train_cnn14 as training


ROOT = Path(__file__).resolve().parents[1]

CHECKPOINT = (
    ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)

OUT = ROOT / "models" / "cnn14_final123_targeted_occlusion"

MASK_FRACTION = 0.20
MASK_BASELINE = -100.0
SAMPLE_SIZE = 60
SEED = 42


def calculate_gradcam(model, waveform):
    """Calculate Grad-CAM for the model's predicted main class."""

    captured = {}

    def capture_activations(module, inputs, output):
        captured["activations"] = output
        output.register_hook(
            lambda gradient: captured.__setitem__(
                "gradients", gradient
            )
        )

    hook = model.backbone.conv_block6.register_forward_hook(
        capture_activations
    )

    model.zero_grad(set_to_none=True)

    try:
        with torch.enable_grad():
            logits, _ = model(waveform)
            predicted_id = int(logits.argmax(dim=1).item())
            target_score = logits[0, predicted_id]
            target_score.backward()

        if "activations" not in captured or "gradients" not in captured:
            raise RuntimeError("Grad-CAM activations or gradients missing.")

        activations = captured["activations"].detach()
        gradients = captured["gradients"].detach()

    finally:
        hook.remove()

    weights = gradients.mean(dim=(2, 3), keepdim=True)

    cam = F.relu((weights * activations).sum(dim=1, keepdim=True))

    cam_min = cam.min()
    cam_max = cam.max()

    if (cam_max - cam_min).item() < 1e-8:
        raise RuntimeError("Grad-CAM produced a constant heatmap.")

    cam = (cam - cam_min) / (cam_max - cam_min)

    with torch.no_grad():
        spectrogram = model.backbone.spectrogram_extractor(waveform)
        logmel = model.backbone.logmel_extractor(spectrogram)

    cam = F.interpolate(
        cam,
        size=logmel.shape[2:],
        mode="bilinear",
        align_corners=False,
    )

    return logits.detach(), predicted_id, cam.detach(), logmel.detach()


def make_region_masks(cam):
    """Create equal-sized high- and low-importance masks."""

    flat = cam.flatten()
    k = max(1, math.ceil(MASK_FRACTION * flat.numel()))

    high_indices = torch.topk(
        flat, k=k, largest=True
    ).indices

    low_indices = torch.topk(
        flat, k=k, largest=False
    ).indices

    high_mask = torch.zeros_like(flat, dtype=torch.bool)
    low_mask = torch.zeros_like(flat, dtype=torch.bool)

    high_mask[high_indices] = True
    low_mask[low_indices] = True

    return (
        high_mask.reshape_as(cam),
        low_mask.reshape_as(cam),
    )


def evaluate_mask(model, waveform, mask, predicted_id):
    """Evaluate the original predicted class after log-mel masking."""

    def apply_mask(module, inputs, output):
        masked = output.clone()
        masked.masked_fill_(mask, MASK_BASELINE)
        return masked

    hook = model.backbone.logmel_extractor.register_forward_hook(
        apply_mask
    )

    try:
        with torch.no_grad():
            logits, _ = model(waveform)
            probabilities = F.softmax(logits, dim=1)

            original_class_probability = float(
                probabilities[0, predicted_id].item()
            )

            new_predicted_id = int(logits.argmax(dim=1).item())

    finally:
        hook.remove()

    return original_class_probability, new_predicted_id


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    main_map, sub_map = training.load_label_maps()

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Checkpoint main-class mapping mismatch.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Checkpoint subclass mapping mismatch.")

    model = training.DataSECCNN14().to(training.DEVICE)
    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    model.eval()

    manifest = training.load_manifest()

    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy()

    if validation_df.empty:
        raise ValueError("Validation partition is empty.")

    sample = validation_df.sample(
        n=min(SAMPLE_SIZE, len(validation_df)),
        random_state=SEED,
    ).reset_index(drop=True)

    dataset = training.DataSECDataset(
        sample,
        main_map,
        sub_map,
        training=False,
    )

    id_to_name = {
        value: key for key, value in main_map.items()
    }

    selected = {}
    results = []

    print("Checkpoint: FINAL-123")
    print("Partition: validation only")
    print(f"Candidate examples: {len(dataset)}")
    print(f"Mask fraction: {MASK_FRACTION:.0%}")
    print(f"Device: {training.DEVICE}")

    for index in range(len(dataset)):
        waveform, actual_id, _ = dataset[index]
        waveform = waveform.unsqueeze(0).to(training.DEVICE)

        try:
            logits, predicted_id, cam, logmel = calculate_gradcam(
                model, waveform
            )
        except RuntimeError as exc:
            print(f"Skipping example {index + 1}: {exc}")
            continue

        correct = int(actual_id.item()) == predicted_id
        category = "correct" if correct else "incorrect"

        if category in selected:
            continue

        original_prob = float(
            F.softmax(logits, dim=1)[0, predicted_id].item()
        )

        high_mask, low_mask = make_region_masks(cam)

        high_prob, high_prediction = evaluate_mask(
            model, waveform, high_mask, predicted_id
        )

        low_prob, low_prediction = evaluate_mask(
            model, waveform, low_mask, predicted_id
        )

        row = {
            "relative_path": str(sample.iloc[index]["relative_path"]),
            "actual_class": id_to_name[int(actual_id.item())],
            "original_predicted_class": id_to_name[predicted_id],
            "original_prediction_correct": correct,
            "original_probability": original_prob,
            "mask_fraction": MASK_FRACTION,
            "high_importance_masked_probability": high_prob,
            "high_importance_probability_drop": original_prob - high_prob,
            "high_importance_new_prediction": id_to_name[high_prediction],
            "low_importance_masked_probability": low_prob,
            "low_importance_probability_drop": original_prob - low_prob,
            "low_importance_new_prediction": id_to_name[low_prediction],
            "high_mask_cells": int(high_mask.sum().item()),
            "low_mask_cells": int(low_mask.sum().item()),
        }

        results.append(row)
        selected[category] = True

        print(
            f"{category.upper()}: "
            f"{row['actual_class']} -> "
            f"{row['original_predicted_class']}"
        )
        print(
            f"  Original probability: {original_prob:.4f}"
        )
        print(
            f"  High-importance masked: {high_prob:.4f}"
        )
        print(
            f"  Low-importance masked:  {low_prob:.4f}"
        )

        if len(selected) == 2:
            break

    if "correct" not in selected or "incorrect" not in selected:
        raise RuntimeError(
            "Could not find both a correct and incorrect example "
            "in the sampled validation examples. No test data was used."
        )

    results_df = pd.DataFrame(results)

    results_path = OUT / "targeted_occlusion_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nTargeted Grad-CAM occlusion completed.")
    print(f"Examples evaluated: {len(results_df)}")
    print(f"Results: {results_path}")
    print("Test partition was not accessed.")


if __name__ == "__main__":
    main()
