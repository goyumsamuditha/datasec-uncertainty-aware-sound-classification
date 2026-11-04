from __future__ import annotations

import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from lime.lime_image import LimeImageExplainer

import train_cnn14 as training


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = PROJECT_ROOT / "models" / "cnn14_initial" / "best_model.pt"
OUTPUT_DIR = PROJECT_ROOT / "models" / "cnn14_lime"
FIGURE_DIR = PROJECT_ROOT / "report" / "lime"

NUM_EXAMPLES = 6                 # Initial pilot; expand only after reviewing it.
NUM_SAMPLES = 500                # LIME perturbations per recording.
INFERENCE_BATCH_SIZE = 16
SEGMENT_ROWS = 4                 # Frequency regions.
SEGMENT_COLS = 8                 # Time regions.
TOP_REGIONS_TO_DISPLAY = 10
SEED = 42
SILENCE_FLOOR_DB = -100.0        # PANNs log-mel floor for amin=1e-10.

DEVICE = training.DEVICE


def make_time_frequency_segments(height: int, width: int) -> np.ndarray:
    """Create a fixed 4 x 8 grid of interpretable time-frequency regions."""
    segments = np.zeros((height, width), dtype=np.int32)
    for row in range(SEGMENT_ROWS):
        y0 = row * height // SEGMENT_ROWS
        y1 = (row + 1) * height // SEGMENT_ROWS
        for col in range(SEGMENT_COLS):
            x0 = col * width // SEGMENT_COLS
            x1 = (col + 1) * width // SEGMENT_COLS
            segments[y0:y1, x0:x1] = row * SEGMENT_COLS + col
    return segments


def main() -> None:
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
    # Load the existing CNN14 checkpoint (no retraining)
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

    id_to_main = {value: key for key, value in main_map.items()}
    print(f"Loaded CNN14 checkpoint from epoch {checkpoint['epoch']}.")

    # -----------------------------------------------------
    # Select validation recordings only
    # -----------------------------------------------------
    manifest = training.load_manifest()
    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy()

    if len(validation_df) < NUM_EXAMPLES:
        raise ValueError("Not enough validation recordings for the LIME pilot.")

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

    # The hook substitutes the perturbed log-mel tensor after PANNs'
    # logmel_extractor, preserving the model's remaining forward pipeline.
    hook_state: dict[str, torch.Tensor | None] = {"override": None}

    def replace_logmel_output(module, inputs, output):
        override = hook_state["override"]
        return output if override is None else override

    hook = model.backbone.logmel_extractor.register_forward_hook(
        replace_logmel_output
    )

    summary_rows: list[dict] = []
    segment_rows: list[dict] = []

    try:
        for index in range(len(dataset)):
            waveform, actual_id, _ = dataset[index]
            waveform = waveform.unsqueeze(0).to(DEVICE)  # [1, samples]

            # Recreate the exact log-mel input produced by this trained model.
            with torch.inference_mode():
                spectrogram = model.backbone.spectrogram_extractor(waveform)
                logmel_tensor = model.backbone.logmel_extractor(spectrogram)
            logmel_time_mel = logmel_tensor[0, 0].detach().cpu().numpy()

            if logmel_time_mel.ndim != 2:
                raise RuntimeError(
                    f"Unexpected log-mel shape: {logmel_time_mel.shape}"
                )

            # LIME image convention: rows = mel bins, columns = time frames.
            logmel_mel_time = logmel_time_mel.T.astype(np.float32)
            logmel_min = min(float(logmel_mel_time.min()), SILENCE_FLOOR_DB)
            logmel_max = float(logmel_mel_time.max())
            if logmel_max <= logmel_min + 1e-8:
                raise RuntimeError("Log-mel spectrogram has no usable value range.")

            scaled = np.clip(
                (logmel_mel_time - logmel_min) / (logmel_max - logmel_min),
                0.0,
                1.0,
            ).astype(np.float32)
            lime_image = np.repeat(scaled[:, :, None], 3, axis=2)
            segments = make_time_frequency_segments(*scaled.shape)

            # LIME calls this function with batches of perturbed RGB images.
            # Blacked-out regions are restored to the PANNs log-mel silence floor.
            def classifier_fn(images: np.ndarray) -> np.ndarray:
                images = np.asarray(images, dtype=np.float32)
                if images.ndim != 4 or images.shape[-1] < 1:
                    raise ValueError(
                        f"Unexpected LIME image batch shape: {images.shape}"
                    )

                batch_probabilities = []
                for start in range(0, len(images), INFERENCE_BATCH_SIZE):
                    image_batch = images[
                        start : start + INFERENCE_BATCH_SIZE
                    ]
                    scaled_batch = np.clip(image_batch[..., 0], 0.0, 1.0)

                    # Shape from LIME: [batch, mel, time]. PANNs expects
                    # [batch, 1, time, mel] at the log-mel hook.
                    mel_time_batch = (
                        scaled_batch * (logmel_max - logmel_min) + logmel_min
                    )
                    hidden = scaled_batch <= 1e-8
                    mel_time_batch[hidden] = SILENCE_FLOOR_DB
                    time_mel_batch = np.transpose(
                        mel_time_batch, (0, 2, 1)
                    )
                    override = torch.from_numpy(
                        np.ascontiguousarray(time_mel_batch)
                    ).to(device=DEVICE, dtype=waveform.dtype)
                    hook_state["override"] = override.unsqueeze(1)

                    count = len(image_batch)
                    wave_batch = waveform.expand(count, -1).contiguous()
                    try:
                        with torch.inference_mode():
                            logits, _ = model(wave_batch)
                            probabilities = torch.softmax(logits, dim=1)
                        batch_probabilities.append(
                            probabilities.cpu().numpy()
                        )
                    finally:
                        hook_state["override"] = None

                return np.concatenate(batch_probabilities, axis=0)

            # Baseline prediction is generated through the same wrapper used
            # for all LIME perturbations.
            baseline_probabilities = classifier_fn(lime_image[None, ...])[0]
            predicted_id = int(np.argmax(baseline_probabilities))
            predicted_probability = float(baseline_probabilities[predicted_id])

            explainer = LimeImageExplainer(
                random_state=SEED + index,
                feature_selection="auto",
            )
            explanation = explainer.explain_instance(
                lime_image,
                classifier_fn,
                labels=(predicted_id,),
                top_labels=None,
                hide_color=0,
                num_features=SEGMENT_ROWS * SEGMENT_COLS,
                num_samples=NUM_SAMPLES,
                batch_size=INFERENCE_BATCH_SIZE,
                segmentation_fn=lambda image: segments,
                random_seed=SEED + index,
            )

            if predicted_id not in explanation.local_exp:
                raise RuntimeError("LIME did not return the requested class explanation.")

            weights = dict(explanation.local_exp[predicted_id])
            ordered_segments = sorted(
                range(SEGMENT_ROWS * SEGMENT_COLS),
                key=lambda segment_id: abs(float(weights.get(segment_id, 0.0))),
                reverse=True,
            )
            selected_segments = ordered_segments[:TOP_REGIONS_TO_DISPLAY]

            # Convert the selected segment weights into a time-frequency map.
            weight_map = np.zeros(segments.shape, dtype=np.float32)
            for segment_id in selected_segments:
                weight_map[segments == segment_id] = float(
                    weights.get(segment_id, 0.0)
                )

            actual_name = id_to_main[int(actual_id.item())]
            predicted_name = id_to_main[predicted_id]
            correct = int(actual_id.item()) == predicted_id
            relative_path = str(validation_df.iloc[index]["relative_path"])

            # Store all 32 segment weights, not only those shown in the figure.
            for rank, segment_id in enumerate(ordered_segments, start=1):
                row = segment_id // SEGMENT_COLS
                col = segment_id % SEGMENT_COLS
                mel_start = row * scaled.shape[0] / SEGMENT_ROWS
                mel_end = (row + 1) * scaled.shape[0] / SEGMENT_ROWS
                time_start = col * training.CLIP_SECONDS / SEGMENT_COLS
                time_end = (col + 1) * training.CLIP_SECONDS / SEGMENT_COLS
                segment_rows.append({
                    "relative_path": relative_path,
                    "actual_class": actual_name,
                    "predicted_class": predicted_name,
                    "segment_id": segment_id,
                    "lime_weight": float(weights.get(segment_id, 0.0)),
                    "rank_by_absolute_weight": rank,
                    "mel_bin_start": mel_start,
                    "mel_bin_end": mel_end,
                    "time_start_seconds": time_start,
                    "time_end_seconds": time_end,
                })

            # -------------------------------------------------
            # Plot spectrogram and LIME contribution overlay
            # Positive weights support the predicted class;
            # negative weights count against the predicted class.
            # -------------------------------------------------
            fig, axes = plt.subplots(
                2,
                1,
                figsize=(13, 8),
                sharex=True,
                constrained_layout=True,
            )
            extent = [0, training.CLIP_SECONDS, 0, training.MEL_BINS]

            axes[0].imshow(
                logmel_mel_time,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="magma",
            )
            axes[0].set_title("Original log-mel spectrogram")
            axes[0].set_ylabel("Mel-bin index")

            axes[1].imshow(
                logmel_mel_time,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="gray",
            )
            max_abs_weight = max(
                [abs(float(weights.get(sid, 0.0))) for sid in selected_segments]
                + [1e-8]
            )
            alpha = np.where(weight_map != 0, 0.65, 0.0)
            overlay = axes[1].imshow(
                weight_map,
                origin="lower",
                aspect="auto",
                extent=extent,
                cmap="coolwarm",
                vmin=-max_abs_weight,
                vmax=max_abs_weight,
                alpha=alpha,
            )
            axes[1].set_title(
                f"LIME regions for predicted class: {predicted_name}\n"
                "Positive = supports prediction; negative = counts against it"
            )
            axes[1].set_xlabel("Time (seconds)")
            axes[1].set_ylabel("Mel-bin index")
            fig.colorbar(overlay, ax=axes[1], label="Local LIME weight")

            status = "correct" if correct else "incorrect"
            fig.suptitle(
                f"Actual: {actual_name} | Predicted: {predicted_name} "
                f"({status}, p={predicted_probability:.3f}) | "
                f"LIME local R²={float(explanation.score):.3f}",
                fontsize=11,
            )

            image_name = f"lime_{index + 1:02d}.png"
            image_path = FIGURE_DIR / image_name
            fig.savefig(image_path, dpi=160)
            plt.close(fig)

            summary_rows.append({
                "relative_path": relative_path,
                "actual_class": actual_name,
                "predicted_class": predicted_name,
                "correct": correct,
                "predicted_class_probability": predicted_probability,
                "lime_local_r2": float(explanation.score),
                "number_of_segments": SEGMENT_ROWS * SEGMENT_COLS,
                "num_perturbations": NUM_SAMPLES,
                "lime_figure": str(image_path.relative_to(PROJECT_ROOT)),
            })

            print(
                f"[{index + 1}/{NUM_EXAMPLES}] {actual_name} -> {predicted_name} "
                f"({'correct' if correct else 'incorrect'}); "
                f"local R²={float(explanation.score):.3f}"
            )

    finally:
        hook_state["override"] = None
        hook.remove()

    summary_path = OUTPUT_DIR / "lime_validation_results.csv"
    segments_path = OUTPUT_DIR / "lime_segment_weights.csv"
    pd.DataFrame(summary_rows).to_csv(summary_path, index=False)
    pd.DataFrame(segment_rows).to_csv(segments_path, index=False)

    print("\nLIME pilot completed.")
    print(f"Recordings explained: {len(summary_rows)}")
    print(f"Segment-weight rows: {len(segment_rows)}")
    print(f"Summary: {summary_path}")
    print(f"Segment weights: {segments_path}")
    print(f"Figures: {FIGURE_DIR}")
    print("Validation partition only; test partition was not accessed.")


if __name__ == "__main__":
    main()
