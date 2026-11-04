
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import joblib

from laion_clap import CLAP_Module
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ---------------------------------------------------------
# Project configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
CHECKPOINT = ROOT / "src" / "630k-audioset-best.pt"

OUTPUT_DIR = ROOT / "models" / "clap_linear_probe"

BATCH_SIZE = 16
RANDOM_STATE = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# CLAP embedding extraction
# ---------------------------------------------------------

def extract_embeddings(model, dataframe, main_map, partition_name):

    embeddings = []
    labels = []
    filenames = []

    paths = [
        DATASET_DIR / relative_path
        for relative_path in dataframe["relative_path"]
    ]

    expected_labels = [
        main_map[class_name]
        for class_name in dataframe["main_class"]
    ]

    print(
        f"\nExtracting CLAP embeddings for {partition_name}: "
        f"{len(paths)} recordings"
    )

    start_time = time.time()

    for start in range(0, len(paths), BATCH_SIZE):

        batch_paths = paths[start:start + BATCH_SIZE]

        batch_embeddings = model.get_audio_embedding_from_filelist(
            x=[str(path) for path in batch_paths],
            use_tensor=True,
        )

        batch_array = batch_embeddings.detach().cpu().numpy()

        if batch_array.ndim != 2:
            raise ValueError(
                f"Unexpected embedding shape: {batch_array.shape}"
            )

        if batch_array.shape[1] != 512:
            raise ValueError(
                f"Expected 512-dimensional embeddings, "
                f"received {batch_array.shape[1]}"
            )

        if not np.isfinite(batch_array).all():
            raise ValueError("Non-finite CLAP embedding values detected.")

        embeddings.append(batch_array)

        batch_labels = expected_labels[start:start + BATCH_SIZE]
        labels.extend(batch_labels)

        filenames.extend([
            str(path.relative_to(DATASET_DIR))
            for path in batch_paths
        ])

        processed = min(start + BATCH_SIZE, len(paths))

        print(f"Processed {processed}/{len(paths)}")

    X = np.vstack(embeddings)
    y = np.asarray(labels, dtype=np.int64)

    elapsed = time.time() - start_time

    print(f"{partition_name} embedding shape: {X.shape}")
    print(f"{partition_name} extraction time: {elapsed:.2f} seconds")

    return X, y, filenames, elapsed


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("LAION-CLAP FROZEN EMBEDDINGS + LINEAR PROBE")
    print("=" * 60)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"CLAP checkpoint not found: {CHECKPOINT}"
        )

    # Reuse the verified manifest and class mappings.
    sys.path.insert(0, str(ROOT / "src"))

    import train_cnn14 as training

    main_map, _ = training.load_label_maps()
    dataframe = training.load_manifest()

    train_df = dataframe[
        dataframe["partition"] == "train"
    ].copy().reset_index(drop=True)

    val_df = dataframe[
        dataframe["partition"] == "validation"
    ].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation partition is empty.")

    # Confirm all files exist before loading CLAP.
    for partition_name, partition_df in [
        ("training", train_df),
        ("validation", val_df),
    ]:
        missing = [
            relative_path
            for relative_path in partition_df["relative_path"]
            if not (DATASET_DIR / relative_path).is_file()
        ]

        if missing:
            raise FileNotFoundError(
                f"{len(missing)} missing files in {partition_name}. "
                f"First missing file: {missing[0]}"
            )

    print(f"Training recordings: {len(train_df)}")
    print(f"Validation recordings: {len(val_df)}")
    print("Test partition reserved and will not be accessed.")

    # -----------------------------------------------------
    # Load the already-tested CLAP model
    # -----------------------------------------------------

    print("\nLoading LAION-CLAP model...")

    clap_model = CLAP_Module(
        enable_fusion=False,
        device=DEVICE,
    )

    clap_model.load_ckpt(
        ckpt=str(CHECKPOINT),
        verbose=False,
    )

    print("CLAP model loaded successfully.")

    # -----------------------------------------------------
    # Extract frozen embeddings
    # -----------------------------------------------------

    X_train, y_train, train_files, train_time = (
        extract_embeddings(
            clap_model,
            train_df,
            main_map,
            "training",
        )
    )

    X_val, y_val, val_files, val_time = (
        extract_embeddings(
            clap_model,
            val_df,
            main_map,
            "validation",
        )
    )

    # Save embeddings so that CLAP does not need to be run again
    # if we later change the linear classifier.
    np.savez_compressed(
        OUTPUT_DIR / "clap_embeddings.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_files=np.asarray(train_files),
        val_files=np.asarray(val_files),
    )

    # -----------------------------------------------------
    # Train a linear probe
    # -----------------------------------------------------

    print("\nTraining linear classifier...")

    classifier = LogisticRegression(
        C=1.0,
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )

    classifier.fit(X_train, y_train)

    print("Linear classifier training completed.")

    # -----------------------------------------------------
    # Validation evaluation
    # -----------------------------------------------------

    predictions = classifier.predict(X_val)

    class_names = [
        name for name, index in sorted(
            main_map.items(),
            key=lambda item: item[1],
        )
    ]

    accuracy = accuracy_score(y_val, predictions)

    macro_f1 = f1_score(
        y_val,
        predictions,
        labels=list(range(training.NUM_MAIN)),
        average="macro",
        zero_division=0,
    )

    print("\nValidation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # -----------------------------------------------------
    # Save predictions
    # -----------------------------------------------------

    prediction_df = pd.DataFrame({
        "relative_path": val_files,
        "actual_class": [
            class_names[index] for index in y_val
        ],
        "predicted_class": [
            class_names[index] for index in predictions
        ],
        "correct": y_val == predictions,
    })

    prediction_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_val,
        predictions,
        labels=list(range(training.NUM_MAIN)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        OUTPUT_DIR / "validation_classification_report.csv"
    )

    # -----------------------------------------------------
    # Confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        y_val,
        predictions,
        labels=list(range(training.NUM_MAIN)),
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    # -----------------------------------------------------
    # Save classifier
    # -----------------------------------------------------

    joblib.dump(
        classifier,
        OUTPUT_DIR / "clap_linear_probe.joblib",
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("CLAP LINEAR PROBE COMPLETED")
    print("=" * 60)

    print(f"Training embeddings: {X_train.shape}")
    print(f"Validation embeddings: {X_val.shape}")
    print(f"Embedding dimension: {X_train.shape[1]}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print(f"Training embedding time: {train_time:.2f} seconds")
    print(f"Validation embedding time: {val_time:.2f} seconds")

    print("\nSaved files:")
    print(OUTPUT_DIR / "clap_embeddings.npz")
    print(OUTPUT_DIR / "clap_linear_probe.joblib")
    print(OUTPUT_DIR / "validation_predictions.csv")
    print(OUTPUT_DIR / "validation_classification_report.csv")
    print(OUTPUT_DIR / "validation_confusion_matrix.csv")

    print("\nTest partition was not evaluated.")


if __name__ == "__main__":
    main()
