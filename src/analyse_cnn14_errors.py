
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix


# ---------------------------------------------------------
# Import the existing training implementation
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train_cnn14 as training


# ---------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------

CHECKPOINT_PATH = (
    ROOT / "models" / "cnn14_initial" / "best_model.pt"
)

OUTPUT_DIR = ROOT / "logs"
MODEL_OUTPUT_DIR = ROOT / "models" / "cnn14_initial"

BATCH_SIZE = 4
NUM_WORKERS = 0


def main():

    print("=" * 60)
    print("CNN14 VALIDATION ERROR ANALYSIS")
    print("=" * 60)

    device = training.DEVICE
    print(f"Device: {device}")

    if not CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------
    # Load mappings and validation manifest
    # -----------------------------------------------------

    main_map, sub_map = training.load_label_maps()

    df = training.load_manifest()

    # Explicitly select validation only.
    val_df = df[df["partition"] == "validation"].copy()

    if val_df.empty:
        raise ValueError("No validation recordings found.")

    print(f"Validation recordings: {len(val_df)}")

    # Preserve the existing training class mappings.
    parents = training.build_parent_mapping(
        df, main_map, sub_map
    )

    subclass_mask = training.make_subclass_mask(
        parents, device
    )

    # -----------------------------------------------------
    # Create validation dataset and loader
    # -----------------------------------------------------

    val_dataset = training.DataSECDataset(
        val_df,
        main_map,
        sub_map,
        training=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available()
    )

    # -----------------------------------------------------
    # Load model and checkpoint
    # -----------------------------------------------------

    model = training.DataSECCNN14().to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    print(
        "Best validation macro-F1:",
        round(checkpoint["best_validation_main_macro_f1"], 4)
    )

    # Use checkpoint mappings as an additional consistency check.
    if checkpoint["main_mapping"] != main_map:
        raise ValueError("Checkpoint main mapping does not match.")

    if checkpoint["subclass_mapping"] != sub_map:
        raise ValueError("Checkpoint subclass mapping does not match.")

    main_names = [
        name for name, index in sorted(
            main_map.items(), key=lambda item: item[1]
        )
    ]

    # -----------------------------------------------------
    # Generate validation predictions
    # -----------------------------------------------------

    records = []
    all_true = []
    all_pred = []

    with torch.inference_mode():

        for batch_index, (
            waveforms,
            main_labels,
            subclass_labels
        ) in enumerate(val_loader):

            waveforms = waveforms.to(
                device, non_blocking=True
            )

            main_logits, subclass_logits = model(waveforms)

            probabilities = torch.softmax(
                main_logits, dim=1
            )

            confidence, predictions = probabilities.max(dim=1)

            top_values, top_indices = probabilities.topk(
                k=3, dim=1
            )

            true_ids = main_labels.tolist()
            pred_ids = predictions.cpu().tolist()
            confidence_values = confidence.cpu().tolist()

            top_values = top_values.cpu().tolist()
            top_indices = top_indices.cpu().tolist()

            batch_start = batch_index * BATCH_SIZE

            for i in range(len(true_ids)):

                row = val_df.iloc[batch_start + i]

                true_id = int(true_ids[i])
                pred_id = int(pred_ids[i])

                record = {
                    "relative_path": row["relative_path"],
                    "actual_class": main_names[true_id],
                    "predicted_class": main_names[pred_id],
                    "correct": true_id == pred_id,
                    "predicted_class_score": float(
                        confidence_values[i]
                    ),
                    "top_1_class": main_names[top_indices[i][0]],
                    "top_1_score": float(top_values[i][0]),
                    "top_2_class": main_names[top_indices[i][1]],
                    "top_2_score": float(top_values[i][1]),
                    "top_3_class": main_names[top_indices[i][2]],
                    "top_3_score": float(top_values[i][2]),
                }

                records.append(record)

            all_true.extend(true_ids)
            all_pred.extend(pred_ids)

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed "
                    f"{min((batch_index + 1) * BATCH_SIZE, len(val_df))}"
                    f"/{len(val_df)} recordings"
                )

    # -----------------------------------------------------
    # Save individual predictions
    # -----------------------------------------------------

    predictions_df = pd.DataFrame(records)

    predictions_path = (
        OUTPUT_DIR / "validation_predictions.csv"
    )

    predictions_df.to_csv(
        predictions_path, index=False
    )

    # -----------------------------------------------------
    # Save misclassified recordings
    # -----------------------------------------------------

    errors_df = predictions_df[
        predictions_df["correct"] == False
    ].copy()

    errors_df = errors_df.sort_values(
        by="predicted_class_score",
        ascending=False
    )

    errors_path = (
        OUTPUT_DIR / "validation_misclassified.csv"
    )

    errors_df.to_csv(errors_path, index=False)

    # -----------------------------------------------------
    # Save classification report
    # -----------------------------------------------------

    report = classification_report(
        all_true,
        all_pred,
        labels=list(range(training.NUM_MAIN)),
        target_names=main_names,
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).T

    report_path = (
        MODEL_OUTPUT_DIR / "validation_error_analysis_report.csv"
    )

    report_df.to_csv(report_path)

    # -----------------------------------------------------
    # Save confusion matrix
    # -----------------------------------------------------

    cm = confusion_matrix(
        all_true,
        all_pred,
        labels=list(range(training.NUM_MAIN))
    )

    cm_df = pd.DataFrame(
        cm,
        index=main_names,
        columns=main_names
    )

    cm_path = (
        MODEL_OUTPUT_DIR / "validation_error_analysis_confusion_matrix.csv"
    )

    cm_df.to_csv(cm_path)

    # -----------------------------------------------------
    # Summarise the most frequent class confusions
    # -----------------------------------------------------

    if not errors_df.empty:

        confusion_summary = (
            errors_df.groupby(
                ["actual_class", "predicted_class"]
            )
            .size()
            .reset_index(name="error_count")
            .sort_values(
                "error_count",
                ascending=False
            )
        )

    else:

        confusion_summary = pd.DataFrame(
            columns=[
                "actual_class",
                "predicted_class",
                "error_count"
            ]
        )

    summary_path = (
        OUTPUT_DIR / "validation_error_summary.csv"
    )

    confusion_summary.to_csv(
        summary_path, index=False
    )

    # -----------------------------------------------------
    # Final summary
    # -----------------------------------------------------

    correct_count = int(predictions_df["correct"].sum())
    error_count = len(errors_df)

    print("\n" + "=" * 60)
    print("VALIDATION ERROR ANALYSIS COMPLETED")
    print("=" * 60)

    print(f"Total validation recordings: {len(predictions_df)}")
    print(f"Correct predictions: {correct_count}")
    print(f"Incorrect predictions: {error_count}")

    print(
        f"Validation accuracy: "
        f"{correct_count / len(predictions_df):.4f}"
    )

    print("\nMost frequent class confusions:")

    if not confusion_summary.empty:
        print(confusion_summary.head(15).to_string(index=False))
    else:
        print("No misclassified recordings.")

    print("\nGenerated files:")
    print(predictions_path)
    print(errors_path)
    print(summary_path)
    print(report_path)
    print(cm_path)

    print("\nTest partition was not accessed.")


if __name__ == "__main__":
    main()
