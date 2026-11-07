import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix
)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import train_cnn14 as training

CHECKPOINT = ROOT / "models" / "cnn14_initial" / "best_model.pt"
OUTPUT = ROOT / "models" / "initial_test_evaluation"
EXCLUDED = "Bells/Bells-0001.wav"
BATCH_SIZE = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def main():
    start = time.time()
    OUTPUT.mkdir(parents=True, exist_ok=True)

    main_map, sub_map = training.load_label_maps()
    manifest = training.load_manifest()
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)

    if main_map != checkpoint["main_mapping"]:
        raise ValueError("Main-class mapping mismatch.")
    if sub_map != checkpoint["subclass_mapping"]:
        raise ValueError("Subclass mapping mismatch.")

    class_names = [
        name for name, idx in sorted(main_map.items(), key=lambda item: item[1])
    ]

    test_df = manifest[manifest["partition"] == "test"].copy()
    test_df["normalised_path"] = test_df["relative_path"].astype(str).str.replace("\\", "/", regex=False)
    original_count = len(test_df)
    test_df = test_df[test_df["normalised_path"] != EXCLUDED].reset_index(drop=True)

    if original_count != 655 or len(test_df) != 654:
        raise ValueError(f"Unexpected test counts: {original_count}, {len(test_df)}")

    y_true = np.array([main_map[str(x).strip()] for x in test_df["main_class"]])

    dataset = training.DataSECDataset(test_df, main_map, sub_map, training=False)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=False,
        num_workers=0, pin_memory=(DEVICE.type == "cuda")
    )

    model = training.DataSECCNN14().to(DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()

    logits = []
    labels = []
    with torch.no_grad():
        for waveforms, main_labels, _ in loader:
            output, _ = model(waveforms.to(DEVICE))
            logits.append(output.cpu())
            labels.append(main_labels.cpu())

    logits = torch.cat(logits).numpy()
    observed_labels = torch.cat(labels).numpy()

    if not np.array_equal(observed_labels, y_true):
        raise ValueError("Test label order mismatch.")
    if logits.shape != (654, 22):
        raise ValueError(f"Unexpected logits shape: {logits.shape}")

    predictions = logits.argmax(axis=1)
    metrics = {
        "model": "initial_cnn14_epoch15",
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(checkpoint["best_validation_main_macro_f1"]),
        "test_recordings": len(y_true),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_precision": float(precision_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, predictions, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, average="weighted", zero_division=0)),
        "evaluation_seconds": round(time.time() - start, 2),
        "test_used_for_tuning": False,
        "excluded_previously_exposed_file": EXCLUDED,
    }

    pd.DataFrame([metrics]).to_csv(OUTPUT / "metrics_summary.csv", index=False)
    pd.DataFrame(classification_report(
        y_true, predictions, labels=np.arange(22),
        target_names=class_names, output_dict=True, zero_division=0
    )).transpose().to_csv(OUTPUT / "classification_report.csv")

    matrix = confusion_matrix(y_true, predictions, labels=np.arange(22))
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(
        OUTPUT / "confusion_matrix.csv", index_label="actual_class"
    )

    pd.DataFrame({
        "relative_path": test_df["relative_path"].astype(str),
        "actual_class": [class_names[i] for i in y_true],
        "predicted_class": [class_names[i] for i in predictions],
        "correct": predictions == y_true,
    }).to_csv(OUTPUT / "predictions.csv", index=False)

    (OUTPUT / "evaluation_metadata.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved separate results to: {OUTPUT}")

if __name__ == "__main__":
    main()
