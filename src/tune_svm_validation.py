from pathlib import Path
import json
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parents[1]
FEATURE_FILE = ROOT / "models" / "svm_baseline" / "extracted_features.npz"
MANIFEST_FILE = ROOT / "logs" / "final_split_manifest.csv"
OUTPUT_DIR = ROOT / "models" / "svm_baseline"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("Loading cached features...")

    data = np.load(FEATURE_FILE)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]

    train_files = data["train_files"]
    val_files = data["val_files"]

    manifest = pd.read_csv(MANIFEST_FILE)

    train_manifest = (
        manifest[manifest["partition"] == "train"]
        .reset_index(drop=True)
    )

    val_manifest = (
        manifest[manifest["partition"] == "validation"]
        .reset_index(drop=True)
    )

    if not np.array_equal(
        train_files,
        train_manifest["relative_path"].to_numpy()
    ):
        raise ValueError("Training feature/manifest order mismatch.")

    if not np.array_equal(
        val_files,
        val_manifest["relative_path"].to_numpy()
    ):
        raise ValueError("Validation feature/manifest order mismatch.")

    groups = train_manifest["group_id"].to_numpy()
    classes = np.unique(y_train)

    print(f"Training examples: {len(X_train)}")
    print(f"Validation examples: {len(X_val)}")
    print(f"Training groups: {len(np.unique(groups))}")
    print(f"Classes: {len(classes)}")
    print("Final test partition will not be accessed.")

    cv = StratifiedGroupKFold(
        n_splits=3,
        shuffle=True,
        random_state=42,
    )

    folds = list(cv.split(X_train, y_train, groups))

    for fold_number, (train_idx, valid_idx) in enumerate(folds, 1):
        overlap = set(groups[train_idx]) & set(groups[valid_idx])

        if overlap:
            raise ValueError(
                f"Group leakage detected in fold {fold_number}."
            )

        missing = set(classes) - set(y_train[valid_idx])

        if missing:
            raise ValueError(
                f"Classes missing from fold {fold_number}: {missing}"
            )

    configurations = [
        {"C": 0.1, "gamma": "scale"},
        {"C": 1.0, "gamma": "scale"},
        {"C": 10.0, "gamma": "scale"},
        {"C": 0.1, "gamma": 0.01},
        {"C": 1.0, "gamma": 0.01},
        {"C": 10.0, "gamma": 0.01},
    ]

    results = []

    print("\nStarting three-fold SVM search...")

    for config_number, config in enumerate(configurations, 1):
        fold_scores = []
        fold_times = []

        print(f"\nConfiguration {config_number}: {config}")

        for fold_number, (train_idx, valid_idx) in enumerate(
            folds, 1
        ):
            model = Pipeline([
                ("scaler", StandardScaler()),
                ("svm", SVC(
                    kernel="rbf",
                    C=config["C"],
                    gamma=config["gamma"],
                    class_weight="balanced",
                )),
            ])

            start = time.perf_counter()

            model.fit(
                X_train[train_idx],
                y_train[train_idx],
            )

            predictions = model.predict(X_train[valid_idx])

            elapsed = time.perf_counter() - start

            score = f1_score(
                y_train[valid_idx],
                predictions,
                labels=classes,
                average="macro",
                zero_division=0,
            )

            fold_scores.append(score)
            fold_times.append(elapsed)

            print(
                f"  Fold {fold_number}: "
                f"macro-F1={score:.4f}, "
                f"time={elapsed:.2f}s"
            )

        results.append({
            "C": config["C"],
            "gamma": config["gamma"],
            "mean_macro_f1": float(np.mean(fold_scores)),
            "std_macro_f1": float(np.std(fold_scores, ddof=1)),
            "mean_fold_time_seconds": float(np.mean(fold_times)),
            "fold_1_macro_f1": fold_scores[0],
            "fold_2_macro_f1": fold_scores[1],
            "fold_3_macro_f1": fold_scores[2],
        })

    results_df = pd.DataFrame(results).sort_values(
        "mean_macro_f1",
        ascending=False,
    )

    results_path = OUTPUT_DIR / "validation_tuning_results.csv"
    results_df.to_csv(results_path, index=False)

    best = results_df.iloc[0]

    best_config = {
        "C": float(best["C"]),
        "gamma": best["gamma"],
    }

    print("\nCross-validation results:")
    print(results_df.to_string(index=False))

    print("\nBest configuration:")
    print(best_config)

    print("\nFitting selected configuration on full training data...")

    final_model = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", SVC(
            kernel="rbf",
            C=best_config["C"],
            gamma=best_config["gamma"],
            class_weight="balanced",
        )),
    ])

    start = time.perf_counter()

    final_model.fit(X_train, y_train)

    training_time = time.perf_counter() - start

    print("Evaluating selected model on validation data...")

    predictions = final_model.predict(X_val)

    validation_macro_f1 = f1_score(
        y_val,
        predictions,
        labels=classes,
        average="macro",
        zero_division=0,
    )

    validation_accuracy = accuracy_score(y_val, predictions)

    print(f"Validation macro-F1: {validation_macro_f1:.4f}")
    print(f"Validation accuracy: {validation_accuracy:.4f}")
    print(f"Full training time: {training_time:.2f} seconds")

    joblib.dump(
        final_model,
        OUTPUT_DIR / "mfcc_spectral_svm_tuned.joblib",
    )

    pd.DataFrame({
        "relative_path": val_files,
        "actual_label": y_val,
        "predicted_label": predictions,
        "correct": y_val == predictions,
    }).to_csv(
        OUTPUT_DIR / "tuned_validation_predictions.csv",
        index=False,
    )

    pd.DataFrame(
        classification_report(
            y_val,
            predictions,
            labels=classes,
            output_dict=True,
            zero_division=0,
        )
    ).T.to_csv(
        OUTPUT_DIR / "tuned_validation_classification_report.csv"
    )

    pd.DataFrame(
        confusion_matrix(
            y_val,
            predictions,
            labels=classes,
        )
    ).to_csv(
        OUTPUT_DIR / "tuned_validation_confusion_matrix.csv",
        index=False,
    )

    metadata = {
        "method": "Three-fold StratifiedGroupKFold",
        "random_state": 42,
        "training_examples": int(len(X_train)),
        "validation_examples": int(len(X_val)),
        "training_groups": int(len(np.unique(groups))),
        "configurations_tested": len(configurations),
        "best_configuration": best_config,
        "best_cv_macro_f1": float(best["mean_macro_f1"]),
        "best_cv_macro_f1_std": float(best["std_macro_f1"]),
        "validation_macro_f1": float(validation_macro_f1),
        "validation_accuracy": float(validation_accuracy),
        "full_training_time_seconds": float(training_time),
        "test_partition_accessed": False,
    }

    with open(
        OUTPUT_DIR / "tuned_validation_metadata.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(metadata, file, indent=4)

    print("\nTuning completed.")
    print("Saved results:", results_path)
    print("Test partition was not accessed.")


if __name__ == "__main__":
    main()