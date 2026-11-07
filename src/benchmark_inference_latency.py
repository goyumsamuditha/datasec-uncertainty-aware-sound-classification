from __future__ import annotations

import json
import sys
import time
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training
import train_svm_baseline as svm_training

CNN_CHECKPOINT = (
    ROOT / "models" / "cnn14_optimisation"
    / "FINAL-123" / "best_model.pt"
)
SVM_CHECKPOINT = (
    ROOT / "models" / "svm_baseline"
    / "mfcc_spectral_svm.joblib"
)

OUTPUT_DIR = ROOT / "logs"
DETAILS_PATH = OUTPUT_DIR / "inference_latency_benchmark.csv"
SUMMARY_PATH = OUTPUT_DIR / "inference_latency_summary.csv"
METADATA_PATH = OUTPUT_DIR / "inference_latency_metadata.json"

WARMUP_COUNT = 5
EXPECTED_VALIDATION_COUNT = 643
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def sync():
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)


def summarise(name, values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "measurement": name,
        "mean_ms": float(values.mean()),
        "median_ms": float(np.median(values)),
        "p95_ms": float(np.percentile(values, 95)),
        "throughput_per_second": float(1000.0 / values.mean()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("VALIDATION-ONLY INFERENCE LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Warm-up examples: {WARMUP_COUNT}")
    print("Batch size: 1")
    print("Held-out test partition: NOT ACCESSED")

    main_map, sub_map = training.load_label_maps()
    manifest = training.load_manifest()

    validation_df = manifest[
        manifest["partition"] == "validation"
    ].copy().reset_index(drop=True)

    if len(validation_df) != EXPECTED_VALIDATION_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_VALIDATION_COUNT} validation examples; "
            f"found {len(validation_df)}."
        )

    if "test" in set(validation_df["partition"]):
        raise RuntimeError("Test partition detected in validation data.")

    dataset = training.DataSECDataset(
        validation_df,
        main_map,
        sub_map,
        training=False,
    )

    print(f"Validation examples: {len(dataset)}")

    # Load the existing trained CNN14 checkpoint.
    checkpoint = torch.load(
        CNN_CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    if checkpoint["main_mapping"] != main_map:
        raise ValueError("CNN14 main-class mapping mismatch.")
    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("CNN14 subclass mapping mismatch.")

    cnn_model = training.DataSECCNN14().to(DEVICE)
    cnn_model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )
    cnn_model.eval()

    # Load the original SVM used in the held-out test evaluation.
    svm_model = joblib.load(SVM_CHECKPOINT)

    # Warm-up both models. Warm-up timings are excluded.
    print("\nWarming up models...")
    with torch.inference_mode():
        for index in range(min(WARMUP_COUNT, len(dataset))):
            waveform, _, _ = dataset[index]

            cnn_input = waveform.unsqueeze(0).to(DEVICE)
            _ = cnn_model(cnn_input)
            sync()

            svm_features = svm_training.extract_features(waveform)
            _ = svm_model.predict(svm_features.reshape(1, -1))

    print("Warm-up complete.")
    print("\nMeasuring validation inference...")

    rows = []
    cnn_model_only = []
    cnn_end_to_end = []
    svm_feature_times = []
    svm_prediction_times = []
    svm_model_only = []
    svm_end_to_end = []

    with torch.inference_mode():
        for index in range(len(dataset)):
            # Shared audio loading and preprocessing:
            # WAV read, mono conversion, resampling, crop/padding.
            start = time.perf_counter()
            waveform, _, _ = dataset[index]
            audio_preprocess_ms = (
                time.perf_counter() - start
            ) * 1000.0

            # CNN14 model-only forward pass.
            cnn_input = waveform.unsqueeze(0).to(DEVICE)

            sync()
            start = time.perf_counter()
            main_logits, _ = cnn_model(cnn_input)
            sync()
            cnn_forward_ms = (
                time.perf_counter() - start
            ) * 1000.0

            # SVM feature extraction.
            start = time.perf_counter()
            svm_features = svm_training.extract_features(waveform)
            feature_ms = (time.perf_counter() - start) * 1000.0

            # SVM prediction only.
            start = time.perf_counter()
            _ = svm_model.predict(svm_features.reshape(1, -1))
            prediction_ms = (time.perf_counter() - start) * 1000.0

            cnn_total_ms = audio_preprocess_ms + cnn_forward_ms
            svm_total_ms = (
                audio_preprocess_ms + feature_ms + prediction_ms
            )

            cnn_model_only.append(cnn_forward_ms)
            cnn_end_to_end.append(cnn_total_ms)
            svm_feature_times.append(feature_ms)
            svm_prediction_times.append(prediction_ms)
            svm_model_only.append(feature_ms + prediction_ms)
            svm_end_to_end.append(svm_total_ms)

            rows.append({
                "relative_path": str(
                    validation_df.iloc[index]["relative_path"]
                ),
                "audio_preprocessing_ms": audio_preprocess_ms,
                "cnn14_forward_ms": cnn_forward_ms,
                "cnn14_end_to_end_ms": cnn_total_ms,
                "svm_feature_extraction_ms": feature_ms,
                "svm_prediction_ms": prediction_ms,
                "svm_model_pipeline_ms": feature_ms + prediction_ms,
                "svm_end_to_end_ms": svm_total_ms,
            })

            if (index + 1) % 100 == 0 or index + 1 == len(dataset):
                print(f"Measured {index + 1}/{len(dataset)}")

    details = pd.DataFrame(rows)
    details.to_csv(DETAILS_PATH, index=False)

    summary_rows = [
        summarise("CNN14 model-only forward pass", cnn_model_only),
        summarise("CNN14 audio-to-prediction", cnn_end_to_end),
        summarise("SVM feature extraction", svm_feature_times),
        summarise("SVM prediction only", svm_prediction_times),
        summarise("SVM feature-plus-prediction", svm_model_only),
        summarise("SVM audio-to-prediction", svm_end_to_end),
    ]

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)

    metadata = {
        "partition": "validation",
        "examples": len(dataset),
        "warmup_examples": WARMUP_COUNT,
        "batch_size": 1,
        "device": str(DEVICE),
        "gpu": (
            torch.cuda.get_device_name(0)
            if DEVICE.type == "cuda" else None
        ),
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cnn_checkpoint": str(CNN_CHECKPOINT),
        "svm_checkpoint": str(SVM_CHECKPOINT),
        "test_partition_accessed": False,
        "timing_note": (
            "Model loading excluded. CNN14 end-to-end includes dataset "
            "audio loading/preprocessing plus forward pass. SVM end-to-end "
            "includes the same audio loading/preprocessing plus feature "
            "extraction and prediction. Batch size is one."
        ),
    }

    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 65)
    print("LATENCY BENCHMARK RESULTS")
    print("=" * 65)
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nSaved:")
    print(DETAILS_PATH)
    print(SUMMARY_PATH)
    print(METADATA_PATH)
    print("\nTest partition accessed: False")


if __name__ == "__main__":
    main()