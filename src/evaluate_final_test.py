from __future__ import annotations

import gc
import json
import sys
import time
import hashlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training
import train_svm_baseline as svm_training

OUTPUT_DIR = ROOT / "models" / "final_test_evaluation"
CHECKPOINT = ROOT / "models" / "cnn14_optimisation" / "FINAL-123" / "best_model.pt"
SVM_PATH = ROOT / "models" / "svm_baseline" / "mfcc_spectral_svm.joblib"
CLAP_LINEAR_PATH = ROOT / "models" / "clap_linear_probe" / "clap_linear_probe.joblib"
CLAP_CHECKPOINT = ROOT / "src" / "630k-audioset-best.pt"
PROMPT_PATH = ROOT / "models" / "clap_prompt_refinement" / "final_frozen_prompts.csv"

EXCLUDED_PATH = "Bells/Bells-0001.wav"
CNN_TEMPERATURE = 0.7807256153245995
CONFORMAL_TEMPERATURE = 0.9253642472130589
CONFORMAL_THRESHOLD = 0.4059282880061026
CLAP_BATCH_SIZE = 16
CNN_BATCH_SIZE = 4
ECE_BINS = 15

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def normalise_path(value):
    return str(value).replace("\\", "/").strip()


def normalise_rows(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def expected_calibration_error(probabilities, labels, bins=15):
    confidence = probabilities.max(axis=1)
    predictions = probabilities.argmax(axis=1)
    correct = predictions == labels
    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0

    for index in range(bins):
        if index == bins - 1:
            selected = (confidence >= edges[index]) & (confidence <= edges[index + 1])
        else:
            selected = (confidence >= edges[index]) & (confidence < edges[index + 1])

        if selected.any():
            bin_accuracy = correct[selected].mean()
            bin_confidence = confidence[selected].mean()
            ece += selected.mean() * abs(bin_accuracy - bin_confidence)

    return float(ece)


def probability_metrics(probabilities, labels):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    clipped = np.clip(probabilities, 1e-12, 1.0)
    nll = -np.log(clipped[np.arange(len(labels)), labels]).mean()
    one_hot = np.eye(probabilities.shape[1], dtype=np.float64)[labels]
    brier = np.square(probabilities - one_hot).sum(axis=1).mean()

    return {
        "nll": float(nll),
        "ece_15_bins": expected_calibration_error(probabilities, labels, ECE_BINS),
        "brier_score": float(brier),
    }


def evaluate_predictions(model_name, y_true, y_pred, class_names):
    labels = np.arange(len(class_names))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(OUTPUT_DIR / f"{model_name}_classification_report.csv")

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    matrix_df = pd.DataFrame(
        matrix,
        index=class_names,
        columns=class_names,
    )
    matrix_df.index.name = "actual_class"
    matrix_df.to_csv(OUTPUT_DIR / f"{model_name}_confusion_matrix.csv")

    return {
        "model": model_name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_recall": float(recall_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "macro_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="macro", zero_division=0
        )),
        "weighted_f1": float(f1_score(
            y_true, y_pred, labels=labels, average="weighted", zero_division=0
        )),
    }


def main():
    start_time = time.time()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("FINAL HELD-OUT TEST EVALUATION")
    print("=" * 72)
    print(f"Device: {DEVICE}")
    if DEVICE.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------
    # 1. Load and validate the fixed test partition
    # ---------------------------------------------------------
    print("\n[1/6] Loading labels and test manifest...")
    main_map, sub_map = training.load_label_maps()
    manifest = training.load_manifest()

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False,
    )

    checkpoint_main_map = checkpoint["main_mapping"]
    checkpoint_sub_map = checkpoint["subclass_mapping"]

    if main_map != checkpoint_main_map:
        raise ValueError("Main-class mapping differs from FINAL-123 checkpoint.")
    if sub_map != checkpoint_sub_map:
        raise ValueError("Subclass mapping differs from FINAL-123 checkpoint.")

    class_names = [
        name for name, class_id in sorted(main_map.items(), key=lambda item: item[1])
    ]
    if len(class_names) != 22:
        raise ValueError(f"Expected 22 classes, found {len(class_names)}.")

    test_df = manifest[manifest["partition"] == "test"].copy()
    test_df["normalised_relative_path"] = test_df["relative_path"].map(normalise_path)

    original_test_count = len(test_df)
    excluded_matches = test_df[
        test_df["normalised_relative_path"] == EXCLUDED_PATH
    ]

    if len(excluded_matches) != 1:
        raise ValueError(
            f"Expected exactly one excluded test file {EXCLUDED_PATH}; "
            f"found {len(excluded_matches)}."
        )

    test_df = test_df[
        test_df["normalised_relative_path"] != EXCLUDED_PATH
    ].copy()
    test_df = test_df.reset_index(drop=True)

    if original_test_count != 655 or len(test_df) != 654:
        raise ValueError(
            f"Unexpected test counts: original={original_test_count}, "
            f"eligible={len(test_df)}."
        )

    unknown_classes = sorted(set(test_df["main_class"]) - set(main_map))
    if unknown_classes:
        raise ValueError(f"Unknown test classes: {unknown_classes}")

    y_true = np.asarray(
        [main_map[str(name).strip()] for name in test_df["main_class"]],
        dtype=np.int64,
    )
    relative_paths = test_df["relative_path"].astype(str).tolist()

    if len(set(relative_paths)) != len(relative_paths):
        raise ValueError("Duplicate paths detected in final test list.")

    print(f"Original test partition: {original_test_count}")
    print(f"Excluded previously exposed file: {EXCLUDED_PATH}")
    print(f"Eligible test recordings: {len(test_df)}")
    print(f"Number of main classes: {len(class_names)}")
    print("Test class counts:")
    print(test_df["main_class"].value_counts().sort_index().to_string())

    # ---------------------------------------------------------
    # 2. CNN14 evaluation
    # ---------------------------------------------------------
    print("\n[2/6] Evaluating FINAL-123 CNN14...")
    cnn_start = time.time()

    cnn_model = training.DataSECCNN14().to(DEVICE)
    cnn_model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    cnn_model.eval()

    cnn_dataset = training.DataSECDataset(
        test_df,
        main_map,
        sub_map,
        training=False,
    )
    cnn_loader = DataLoader(
        cnn_dataset,
        batch_size=CNN_BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(DEVICE.type == "cuda"),
    )

    cnn_logits_list = []
    cnn_labels_list = []

    with torch.no_grad():
        for batch_index, (waveforms, main_labels, _) in enumerate(cnn_loader):
            waveforms = waveforms.to(DEVICE, non_blocking=True)
            main_logits, _ = cnn_model(waveforms)
            cnn_logits_list.append(main_logits.detach().cpu())
            cnn_labels_list.append(main_labels.cpu())

            processed = min(
                (batch_index + 1) * CNN_BATCH_SIZE,
                len(cnn_dataset),
            )
            if processed % 100 == 0 or processed == len(cnn_dataset):
                print(f"CNN14 processed {processed}/{len(cnn_dataset)}")

    cnn_logits = torch.cat(cnn_logits_list, dim=0).numpy()
    cnn_dataset_labels = torch.cat(cnn_labels_list, dim=0).numpy()

    if not np.array_equal(cnn_dataset_labels, y_true):
        raise ValueError("CNN14 dataset label order does not match manifest labels.")

    if cnn_logits.shape != (len(test_df), 22):
        raise ValueError(f"Unexpected CNN14 logits shape: {cnn_logits.shape}")

    cnn_raw_probabilities = torch.softmax(
        torch.tensor(cnn_logits, dtype=torch.float64), dim=1
    ).numpy()
    cnn_calibrated_probabilities = torch.softmax(
        torch.tensor(cnn_logits / CNN_TEMPERATURE, dtype=torch.float64), dim=1
    ).numpy()

    cnn_predictions = cnn_calibrated_probabilities.argmax(axis=1)
    cnn_confidence = cnn_calibrated_probabilities.max(axis=1)

    cnn_metrics = evaluate_predictions(
        "cnn14_final123",
        y_true,
        cnn_predictions,
        class_names,
    )
    cnn_metrics.update(probability_metrics(cnn_calibrated_probabilities, y_true))
    cnn_metrics["calibration_temperature"] = CNN_TEMPERATURE
    cnn_metrics["raw_accuracy"] = float(
        accuracy_score(y_true, cnn_raw_probabilities.argmax(axis=1))
    )

    # Conformal sets use their own previously fitted temperature and threshold.
    conformal_probabilities = torch.softmax(
        torch.tensor(cnn_logits / CONFORMAL_TEMPERATURE, dtype=torch.float64),
        dim=1,
    ).numpy()
    conformal_sets = (1.0 - conformal_probabilities) <= CONFORMAL_THRESHOLD
    conformal_sizes = conformal_sets.sum(axis=1)
    conformal_covered = conformal_sets[np.arange(len(y_true)), y_true]

    cnn_metrics.update({
        "conformal_temperature": CONFORMAL_TEMPERATURE,
        "conformal_threshold": CONFORMAL_THRESHOLD,
        "conformal_empirical_coverage": float(conformal_covered.mean()),
        "conformal_mean_set_size": float(conformal_sizes.mean()),
        "conformal_singleton_rate": float((conformal_sizes == 1).mean()),
        "conformal_empty_set_rate": float((conformal_sizes == 0).mean()),
    })

    conformal_rows = []
    for index, path in enumerate(relative_paths):
        included_ids = np.flatnonzero(conformal_sets[index]).tolist()
        conformal_rows.append({
            "relative_path": path,
            "actual_class": class_names[y_true[index]],
            "prediction_set": " | ".join(class_names[i] for i in included_ids),
            "set_size": int(conformal_sizes[index]),
            "true_class_in_set": bool(conformal_covered[index]),
        })

    pd.DataFrame(conformal_rows).to_csv(
        OUTPUT_DIR / "cnn14_conformal_prediction_sets.csv",
        index=False,
    )

    print(f"CNN14 macro-F1: {cnn_metrics['macro_f1']:.4f}")
    print(f"CNN14 accuracy: {cnn_metrics['accuracy']:.4f}")
    print(f"CNN14 ECE: {cnn_metrics['ece_15_bins']:.4f}")
    print(
        "CNN14 empirical conformal coverage: "
        f"{cnn_metrics['conformal_empirical_coverage']:.4f}"
    )
    print(f"CNN14 completed in {time.time() - cnn_start:.1f}s")

    # Release CNN14 GPU memory before loading CLAP.
    del cnn_model, cnn_dataset, cnn_loader
    del cnn_logits_list, cnn_labels_list
    gc.collect()
    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------------------------------------------------
    # 3. SVM evaluation
    # ---------------------------------------------------------
    print("\n[3/6] Evaluating MFCC + spectral SVM...")
    svm_start = time.time()

    svm_model = joblib.load(SVM_PATH)
    svm_features, svm_labels, svm_paths = svm_training.extract_partition_features(
        test_df,
        main_map,
        sub_map,
        "final test",
    )

    if list(svm_paths) != relative_paths:
        raise ValueError("SVM feature extraction changed the test-file order.")
    if not np.array_equal(svm_labels, y_true):
        raise ValueError("SVM labels do not match the common test labels.")

    svm_predictions = np.asarray(svm_model.predict(svm_features), dtype=np.int64)
    svm_metrics = evaluate_predictions(
        "mfcc_spectral_svm",
        y_true,
        svm_predictions,
        class_names,
    )

    print(f"SVM macro-F1: {svm_metrics['macro_f1']:.4f}")
    print(f"SVM accuracy: {svm_metrics['accuracy']:.4f}")
    print(f"SVM completed in {time.time() - svm_start:.1f}s")

    del svm_features, svm_model
    gc.collect()

    # ---------------------------------------------------------
    # 4. Load CLAP and extract the same test audio embeddings
    # ---------------------------------------------------------
    print("\n[4/6] Loading CLAP and extracting test audio embeddings...")
    clap_start = time.time()

    from laion_clap import CLAP_Module

    clap_model = CLAP_Module(
        enable_fusion=False,
        device=str(DEVICE),
    )
    clap_model.load_ckpt(
        ckpt=str(CLAP_CHECKPOINT),
        verbose=False,
    )

    test_paths = [
        Path(path) for path in test_df["audio_path"].tolist()
    ]

    audio_embedding_batches = []
    for start in range(0, len(test_paths), CLAP_BATCH_SIZE):
        batch_paths = test_paths[start:start + CLAP_BATCH_SIZE]
        batch_result = clap_model.get_audio_embedding_from_filelist(
            x=[str(path) for path in batch_paths],
            use_tensor=True,
        )
        batch_array = to_numpy(batch_result)

        if batch_array.shape != (len(batch_paths), 512):
            raise ValueError(
                f"Unexpected CLAP audio embedding shape: {batch_array.shape}"
            )
        if not np.isfinite(batch_array).all():
            raise ValueError("Non-finite CLAP audio embeddings.")

        audio_embedding_batches.append(batch_array)
        print(
            f"CLAP audio embeddings: "
            f"{min(start + len(batch_paths), len(test_paths))}/{len(test_paths)}"
        )

    audio_embeddings = np.vstack(audio_embedding_batches)

    if audio_embeddings.shape != (len(test_df), 512):
        raise ValueError(f"Unexpected combined CLAP shape: {audio_embeddings.shape}")

    print(f"CLAP audio embedding extraction: {time.time() - clap_start:.1f}s")

    # ---------------------------------------------------------
    # 5. CLAP linear probe and frozen-prompt zero-shot
    # ---------------------------------------------------------
    print("\n[5/6] Evaluating CLAP linear probe...")
    linear_model = joblib.load(CLAP_LINEAR_PATH)

    if not np.array_equal(
        np.asarray(linear_model.classes_),
        np.arange(len(class_names)),
    ):
        raise ValueError(
            "CLAP linear-probe classifier classes are not IDs 0 through 21."
        )

    linear_predictions = np.asarray(
        linear_model.predict(audio_embeddings),
        dtype=np.int64,
    )
    linear_metrics = evaluate_predictions(
        "clap_linear_probe",
        y_true,
        linear_predictions,
        class_names,
    )
    print(f"CLAP linear-probe macro-F1: {linear_metrics['macro_f1']:.4f}")
    print(f"CLAP linear-probe accuracy: {linear_metrics['accuracy']:.4f}")

    print("\nEvaluating refined CLAP zero-shot with frozen prompts...")
    prompt_df = pd.read_csv(PROMPT_PATH)

    required_prompt_columns = {"class_id", "class_name", "prompt_number", "prompt"}
    if not required_prompt_columns.issubset(prompt_df.columns):
        raise ValueError("Frozen prompt CSV is missing required columns.")

    prompt_df = prompt_df.sort_values(
        ["class_id", "prompt_number"]
    ).reset_index(drop=True)

    if len(prompt_df) != 66:
        raise ValueError(f"Expected 66 frozen prompts, found {len(prompt_df)}.")

    prompt_counts = prompt_df.groupby("class_id").size()
    if len(prompt_counts) != 22 or not (prompt_counts == 3).all():
        raise ValueError("Frozen prompts must contain exactly three prompts per class.")

    for class_id in range(22):
        rows = prompt_df[prompt_df["class_id"] == class_id]
        expected_name = class_names[class_id]
        if rows["class_name"].nunique() != 1:
            raise ValueError(f"Inconsistent prompt class names for ID {class_id}.")
        if rows["class_name"].iloc[0] != expected_name:
            raise ValueError(
                f"Prompt class order mismatch at ID {class_id}: "
                f"{rows['class_name'].iloc[0]!r} vs {expected_name!r}"
            )

    prompts = prompt_df["prompt"].astype(str).tolist()
    text_embeddings = to_numpy(
        clap_model.get_text_embedding(prompts, use_tensor=True)
    )

    if text_embeddings.shape != (66, 512):
        raise ValueError(f"Unexpected prompt embedding shape: {text_embeddings.shape}")

    # Match the validation prompt-refinement procedure:
    # normalise each prompt, average the three prompts per class, normalise again.
    normalised_prompt_embeddings = normalise_rows(text_embeddings)
    class_embeddings = []

    for class_id in range(22):
        start = class_id * 3
        group = normalised_prompt_embeddings[start:start + 3]
        class_embedding = group.mean(axis=0, keepdims=True)
        class_embedding = normalise_rows(class_embedding)
        class_embeddings.append(class_embedding[0])

    class_embeddings = np.vstack(class_embeddings)

    zero_shot_similarities = cosine_similarity(
        audio_embeddings,
        class_embeddings,
    )
    if zero_shot_similarities.shape != (len(test_df), 22):
        raise ValueError(
            f"Unexpected zero-shot similarity shape: {zero_shot_similarities.shape}"
        )

    zero_shot_predictions = zero_shot_similarities.argmax(axis=1)
    zero_shot_metrics = evaluate_predictions(
        "clap_refined_zero_shot",
        y_true,
        zero_shot_predictions,
        class_names,
    )

    print(f"Refined CLAP zero-shot macro-F1: {zero_shot_metrics['macro_f1']:.4f}")
    print(f"Refined CLAP zero-shot accuracy: {zero_shot_metrics['accuracy']:.4f}")

    # ---------------------------------------------------------
    # 6. Save the common predictions, metrics and provenance
    # ---------------------------------------------------------
    print("\n[6/6] Saving final test results...")

    prediction_df = pd.DataFrame({
        "relative_path": relative_paths,
        "actual_class": [class_names[i] for i in y_true],
        "cnn14_prediction": [class_names[i] for i in cnn_predictions],
        "cnn14_calibrated_confidence": cnn_confidence,
        "cnn14_correct": cnn_predictions == y_true,
        "svm_prediction": [class_names[i] for i in svm_predictions],
        "svm_correct": svm_predictions == y_true,
        "clap_linear_prediction": [class_names[i] for i in linear_predictions],
        "clap_linear_correct": linear_predictions == y_true,
        "clap_zero_shot_prediction": [class_names[i] for i in zero_shot_predictions],
        "clap_zero_shot_correct": zero_shot_predictions == y_true,
    })
    prediction_df.to_csv(
        OUTPUT_DIR / "test_predictions_all_models.csv",
        index=False,
    )

    metrics_rows = [
        cnn_metrics,
        svm_metrics,
        linear_metrics,
        zero_shot_metrics,
    ]
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(OUTPUT_DIR / "metrics_summary.csv", index=False)

    np.savez_compressed(
        OUTPUT_DIR / "test_probability_and_embedding_outputs.npz",
        y_true=y_true,
        cnn14_logits=cnn_logits,
        cnn14_calibrated_probabilities=cnn_calibrated_probabilities,
        cnn14_conformal_probabilities=conformal_probabilities,
        clap_audio_embeddings=audio_embeddings,
        clap_zero_shot_similarities=zero_shot_similarities,
        relative_paths=np.asarray(relative_paths),
        class_names=np.asarray(class_names),
    )

    prompt_hash = hashlib.sha256(PROMPT_PATH.read_bytes()).hexdigest()
    metadata = {
        "evaluation": "final held-out test evaluation",
        "test_partition_original_count": int(original_test_count),
        "excluded_file": EXCLUDED_PATH,
        "eligible_test_count": int(len(test_df)),
        "class_count": int(len(class_names)),
        "class_names_in_id_order": class_names,
        "test_file_order": relative_paths,
        "cnn_checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "cnn_checkpoint_epoch": int(checkpoint["epoch"]),
        "cnn_checkpoint_validation_macro_f1": float(
            checkpoint["best_validation_main_macro_f1"]
        ),
        "cnn_temperature_scaling_temperature": CNN_TEMPERATURE,
        "conformal_temperature": CONFORMAL_TEMPERATURE,
        "conformal_threshold": CONFORMAL_THRESHOLD,
        "frozen_prompt_file": str(PROMPT_PATH.relative_to(ROOT)),
        "frozen_prompt_sha256": prompt_hash,
        "clap_checkpoint": str(CLAP_CHECKPOINT.relative_to(ROOT)),
        "device": str(DEVICE),
        "test_partition_used_for_tuning": False,
        "warning": (
            "The test set is a single held-out evaluation. Do not use these "
            "results to tune models, prompts, thresholds or hyperparameters. "
            "Conformal coverage is an empirical test-set result, not a "
            "guaranteed coverage claim."
        ),
        "elapsed_seconds": float(time.time() - start_time),
    }

    with open(OUTPUT_DIR / "evaluation_metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, indent=2)

    print("\n" + "=" * 72)
    print("FINAL TEST RESULTS")
    print("=" * 72)
    display_columns = [
        "model",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]
    print(metrics_df[display_columns].to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCNN14 additional metrics:")
    for key in [
        "nll",
        "ece_15_bins",
        "brier_score",
        "conformal_empirical_coverage",
        "conformal_mean_set_size",
        "conformal_singleton_rate",
        "conformal_empty_set_rate",
    ]:
        print(f"{key}: {cnn_metrics[key]:.6f}")

    print(f"\nElapsed time: {time.time() - start_time:.1f} seconds")
    print(f"Results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
