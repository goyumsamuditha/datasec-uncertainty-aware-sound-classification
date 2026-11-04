
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
import torchaudio

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
import joblib


# ---------------------------------------------------------
# Reuse the existing DataSEC configuration and preprocessing
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training


OUTPUT_DIR = ROOT / "models" / "svm_baseline"
LOG_DIR = ROOT / "logs"

SAMPLE_RATE = training.SAMPLE_RATE
BATCH_SIZE = 32

DEVICE = torch.device("cpu")


# ---------------------------------------------------------
# Feature extractor
# ---------------------------------------------------------

mfcc_transform = torchaudio.transforms.MFCC(
    sample_rate=SAMPLE_RATE,
    n_mfcc=20,
    melkwargs={
        "n_fft": 1024,
        "hop_length": 320,
        "n_mels": 64,
        "f_min": 50,
        "f_max": 14000,
        "power": 2.0,
    },
)


def extract_features(waveform):
    """
    Extract MFCC and basic spectral/time-domain statistics.

    Input:
        Mono waveform, already resampled and cropped/padded
        to the same 10-second format used by CNN14.

    Output:
        One fixed-length feature vector.
    """

    waveform = waveform.float().cpu()

    # MFCC: 20 coefficients, mean and standard deviation.
    mfcc = mfcc_transform(waveform.unsqueeze(0)).squeeze(0)

    mfcc_mean = mfcc.mean(dim=1)
    mfcc_std = mfcc.std(dim=1, unbiased=False)

    window = torch.hann_window(1024)

    stft = torch.stft(
        waveform,
        n_fft=1024,
        hop_length=320,
        win_length=1024,
        window=window,
        return_complex=True,
    )

    magnitude = stft.abs() + 1e-10

    frequencies = torch.linspace(
        0,
        SAMPLE_RATE / 2,
        magnitude.shape[0],
    )

    # Spectral centroid.
    centroid = (
        (frequencies[:, None] * magnitude).sum(dim=0)
        / magnitude.sum(dim=0)
    )

    # Spectral bandwidth.
    bandwidth = torch.sqrt(
        (
            (frequencies[:, None] - centroid[None, :]) ** 2
            * magnitude
        ).sum(dim=0)
        / magnitude.sum(dim=0)
    )

    # Spectral roll-off at 85%.
    cumulative = magnitude.cumsum(dim=0)
    threshold = 0.85 * magnitude.sum(dim=0)

    rolloff_indices = (
        cumulative >= threshold.unsqueeze(0)
    ).float().argmax(dim=0)

    rolloff = frequencies[rolloff_indices]

    # Frame-based RMS and zero-crossing rate.
    frame_length = 1024
    hop_length = 320

    frames = waveform.unfold(
        dimension=0,
        size=frame_length,
        step=hop_length,
    )

    rms = torch.sqrt(
        torch.mean(frames ** 2, dim=1) + 1e-10
    )

    zero_crossing = (
        (frames[:, 1:] * frames[:, :-1]) < 0
    ).float().mean(dim=1)

    # Summarise each descriptor using mean and standard deviation.
    descriptors = [
        centroid,
        bandwidth,
        rolloff,
        rms,
        zero_crossing,
    ]

    summary = []

    for values in descriptors:
        summary.extend([
            values.mean().item(),
            values.std(unbiased=False).item(),
        ])

    features = torch.cat([
        mfcc_mean,
        mfcc_std,
        torch.tensor(summary, dtype=torch.float32),
    ])

    return features.numpy().astype(np.float32)


# ---------------------------------------------------------
# Extract features for a partition
# ---------------------------------------------------------

def extract_partition_features(df, main_map, sub_map, name):

    dataset = training.DataSECDataset(
        df,
        main_map,
        sub_map,
        training=False,
    )

    features = []
    labels = []
    filenames = []

    print(f"\nExtracting {name} features: {len(dataset)} recordings")

    for index in range(len(dataset)):

        waveform, main_label, _ = dataset[index]

        vector = extract_features(waveform)

        if not np.isfinite(vector).all():
            raise ValueError(
                f"Non-finite feature values: "
                f"{df.iloc[index]['relative_path']}"
            )

        features.append(vector)
        labels.append(int(main_label))
        filenames.append(df.iloc[index]["relative_path"])

        if (index + 1) % 250 == 0:
            print(f"Processed {index + 1}/{len(dataset)}")

    return (
        np.stack(features),
        np.asarray(labels, dtype=np.int64),
        filenames,
    )


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("MFCC + SPECTRAL FEATURES + SVM BASELINE")
    print("=" * 60)

    main_map, sub_map = training.load_label_maps()
    df = training.load_manifest()

    train_df = df[df["partition"] == "train"].copy()
    val_df = df[df["partition"] == "validation"].copy()

    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation partition is empty.")

    print(f"Training recordings: {len(train_df)}")
    print(f"Validation recordings: {len(val_df)}")
    print("Test partition reserved and will not be accessed.")

    # -----------------------------------------------------
    # Extract training features
    # -----------------------------------------------------

    X_train, y_train, train_files = extract_partition_features(
        train_df, main_map, sub_map, "training"
    )

    # -----------------------------------------------------
    # Extract validation features
    # -----------------------------------------------------

    X_val, y_val, val_files = extract_partition_features(
        val_df, main_map, sub_map, "validation"
    )

    print("\nFeature dimensions:")
    print(f"Training: {X_train.shape}")
    print(f"Validation: {X_val.shape}")

    # -----------------------------------------------------
    # Train SVM
    # -----------------------------------------------------

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
        )),
    ])

    print("\nTraining SVM...")

    model.fit(X_train, y_train)

    print("SVM training completed.")

    # -----------------------------------------------------
    # Validation predictions
    # -----------------------------------------------------

    predictions = model.predict(X_val)

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
    # Save trained model and feature configuration
    # -----------------------------------------------------

    joblib.dump(
        model,
        OUTPUT_DIR / "mfcc_spectral_svm.joblib",
    )

    # Save extracted features for reproducibility and reuse.
    np.savez_compressed(
        OUTPUT_DIR / "extracted_features.npz",
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        train_files=np.asarray(train_files),
        val_files=np.asarray(val_files),
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    print("\n" + "=" * 60)
    print("SVM BASELINE COMPLETED")
    print("=" * 60)

    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print(f"Feature count: {X_train.shape[1]}")

    print("\nSaved files:")
    print(OUTPUT_DIR / "mfcc_spectral_svm.joblib")
    print(OUTPUT_DIR / "extracted_features.npz")
    print(OUTPUT_DIR / "validation_predictions.csv")
    print(OUTPUT_DIR / "validation_classification_report.csv")
    print(OUTPUT_DIR / "validation_confusion_matrix.csv")

    print("\nTest partition was not evaluated.")


if __name__ == "__main__":
    main()
