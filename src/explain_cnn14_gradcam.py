
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "cnn14_initial" / "best_model.pt"

OUTPUT_DIR = PROJECT_ROOT / "models" / "cnn14_gradcam"
FIGURE_DIR = PROJECT_ROOT / "report" / "gradcam"

NUM_EXAMPLES = 8
SEED = 42

DEVICE = training.DEVICE


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(CHECKPOINT_PATH)

    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # -----------------------------------------------------
    # Load checkpoint and reconstruct the trained model
    # -----------------------------------------------------

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

    model = training.DataSECCNN14().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(
        "Best validation macro-F1:",
        f"{checkpoint['best_validation_main_macro_f1']:.4f}",
    )

    # -----------------------------------------------------
    # Load validation data only
    # -----------------------------------------------------

    manifest = training.load_manifest()
    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy()

    if validation_df.empty:
        raise ValueError("No validation recordings found.")

    if len(validation_df) < NUM_EXAMPLES:
        raise ValueError("Not enough validation recordings.")

    validation_df = validation_df.sample(
        n=NUM_EXAMPLES,
        random_state=SEED,
    ).reset_index(drop=True)

    dataset = training.DataSECDataset(
        validation_df,
        main_map,
        sub_map,
        training=False,
    )

    id_to_main = {value: key for key, value in main_map.items()}

    # -----------------------------------------------------
    # Grad-CAM hook on the final convolutional block
    # -----------------------------------------------------

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

    results = []

    try:
        for index in range(len(dataset)):
            waveform, actual_id, _ = dataset[index]
            waveform = waveform.unsqueeze(0).to(DEVICE)

            captured.clear()
            model.zero_grad(set_to_none=True)

            # Forward pass with gradients enabled.
            with torch.enable_grad():
                main_logits, _ = model(waveform)

                predicted_id = int(main_logits.argmax(dim=1).item())
                target_score = main_logits[0, predicted_id]

                # Explain the model's predicted class.
                target_score.backward()

            if "activations" not in captured or "gradients" not in captured:
                raise RuntimeError("Grad-CAM activations or gradients missing.")

            activations = captured["activations"].detach()
            gradients = captured["gradients"].detach()

            # Average gradients across the spatial dimensions.
            weights = gradients.mean(dim=(2, 3), keepdim=True)

            # Weighted sum of feature maps, followed by ReLU.
            cam = (weights * activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)

            # Normalise the heatmap.
            cam_min = cam.min()
            cam_max = cam.max()

            if (cam_max - cam_min).item() > 1e-8:
                cam = (cam - cam_min) / (cam_max - cam_min)
            else:
                cam = torch.zeros_like(cam)

            # Recreate the log-mel spectrogram used by the backbone.
            with torch.no_grad():
                spectrogram = model.backbone.spectrogram_extractor(
                    waveform
                )
                logmel = model.backbone.logmel_extractor(spectrogram)

            logmel_image = logmel.squeeze().detach().cpu().numpy()

            if logmel_image.ndim != 2:
                raise RuntimeError(
                    f"Unexpected log-mel shape: {logmel_image.shape}"
                )

            # Resize Grad-CAM to the spectrogram dimensions.
            cam_resized = F.interpolate(
                cam,
                size=logmel_image.shape,
                mode="bilinear",
                align_corners=False,
            )

            cam_image = cam_resized.squeeze().cpu().numpy()

            actual_name = id_to_main[int(actual_id.item())]
            predicted_name = id_to_main[predicted_id]
            relative_path = str(validation_df.iloc[index]["relative_path"])

            # -------------------------------------------------
            # Plot spectrogram and Grad-CAM overlay
            # -------------------------------------------------

            fig, axes = plt.subplots(
                2,
                1,
                figsize=(12, 8),
                sharex=True,
                constrained_layout=True,
            )

            duration = training.CLIP_SECONDS
            time_extent = [0, duration]
            frequency_extent = [0, training.MEL_BINS]

            axes[0].imshow(
                logmel_image.T,
                origin="lower",
                aspect="auto",
                extent=time_extent + frequency_extent,
                cmap="magma",
            )
            axes[0].set_title("Log-mel spectrogram")
            axes[0].set_ylabel("Mel-bin index")

            axes[1].imshow(
                logmel_image.T,
                origin="lower",
                aspect="auto",
                extent=time_extent + frequency_extent,
                cmap="gray",
            )
            axes[1].imshow(
                cam_image.T,
                origin="lower",
                aspect="auto",
                extent=time_extent + frequency_extent,
                cmap="jet",
                alpha=0.45,
                vmin=0,
                vmax=1,
            )

            axes[1].set_title(
                f"Grad-CAM for predicted class: {predicted_name}"
            )
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("Mel-bin index")

            correct = actual_id.item() == predicted_id
            status = "correct" if correct else "incorrect"

            fig.suptitle(
                f"Actual: {actual_name} | "
                f"Predicted: {predicted_name} | {status}",
                fontsize=12,
            )

            image_name = f"gradcam_{index + 1:02d}.png"
            image_path = FIGURE_DIR / image_name

            fig.savefig(image_path, dpi=160)
            plt.close(fig)

            results.append({
                "relative_path": relative_path,
                "actual_class": actual_name,
                "predicted_class": predicted_name,
                "correct": correct,
                "predicted_class_logit": float(target_score.detach().item()),
                "gradcam_image": str(image_path.relative_to(PROJECT_ROOT)),
            })

            print(
                f"[{index + 1}/{NUM_EXAMPLES}] "
                f"{actual_name} -> {predicted_name} "
                f"({'correct' if correct else 'incorrect'})"
            )

    finally:
        hook.remove()

    # -----------------------------------------------------
    # Save results
    # -----------------------------------------------------

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "gradcam_validation_results.csv"
    results_df.to_csv(results_path, index=False)

    print("\nGrad-CAM completed.")
    print(f"Recordings explained: {len(results_df)}")
    print(f"Correct predictions: {results_df['correct'].sum()}")
    print(f"Incorrect predictions: {(~results_df['correct']).sum()}")
    print(f"Results: {results_path}")
    print(f"Figures: {FIGURE_DIR}")
    print("Test partition was not accessed.")


if __name__ == "__main__":
    main()
