
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from laion_clap import CLAP_Module
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.metrics.pairwise import cosine_similarity


# ---------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
CHECKPOINT = ROOT / "src" / "630k-audioset-best.pt"

OUTPUT_DIR = ROOT / "models" / "clap_zero_shot"

BATCH_SIZE = 16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------
# Fixed class prompts
#
# Keep these unchanged for this initial experiment.
# Any later prompt refinement must be documented separately.
# ---------------------------------------------------------

CLASS_PROMPTS = [
    "The sound of bells ringing.",
    "The sound of birds singing or calling.",
    "The sound of cats fighting, growling or moaning.",
    "The sound of chickens clucking in a chicken coop.",
    "The sound of cicadas and crickets chirping.",
    "The sound of crows, seagulls and magpies calling.",
    "The sound of dogs barking and howling.",
    "The sound of glass breaking and shattering.",
    "The sound of a vehicle horn honking.",
    "The sound of a jet aircraft flying overhead.",
    "The sound of a lawn mower, brush cutter or olive shaker operating.",
    "The sound of music playing.",
    "The sound of a propeller aircraft flying overhead.",
    "The sound of sirens and alarms sounding.",
    "The sound of thunder, fireworks or gunshots.",
    "The sound of a train moving along railway tracks.",
    "The sound of a vacuum cleaner, fan or hairdryer operating.",
    "The sound of a vehicle engine idling.",
    "The sound of vehicles passing by.",
    "The sound of people speaking and talking.",
    "The sound of a wind turbine operating.",
    "The sound of machinery and tools operating in a workshop.",
]


# ---------------------------------------------------------
# Main experiment
# ---------------------------------------------------------

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("LAION-CLAP ZERO-SHOT CLASSIFICATION")
    print("=" * 65)

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"CLAP checkpoint not found: {CHECKPOINT}"
        )

    # Reuse the established manifest and class mappings.
    sys.path.insert(0, str(ROOT / "src"))

    import train_cnn14 as training

    main_map, _ = training.load_label_maps()
    dataframe = training.load_manifest()

    class_names = [
        name for name, index in sorted(
            main_map.items(),
            key=lambda item: item[1],
        )
    ]

    if len(class_names) != len(CLASS_PROMPTS):
        raise ValueError(
            f"Found {len(class_names)} classes but "
            f"{len(CLASS_PROMPTS)} prompts."
        )

    # Verify that the prompt order matches the class ID order.
    print("\nClass and prompt mapping:")

    prompt_mapping = []

    for class_id, (class_name, prompt) in enumerate(
        zip(class_names, CLASS_PROMPTS)
    ):
        print(f"{class_id:02d}: {class_name} -> {prompt}")

        prompt_mapping.append({
            "class_id": class_id,
            "class_name": class_name,
            "prompt": prompt,
        })

    pd.DataFrame(prompt_mapping).to_csv(
        OUTPUT_DIR / "zero_shot_class_prompts.csv",
        index=False,
    )

    # Only validation recordings are evaluated in this experiment.
    val_df = dataframe[
        dataframe["partition"] == "validation"
    ].copy().reset_index(drop=True)

    if val_df.empty:
        raise ValueError("Validation partition is empty.")

    paths = [
        DATASET_DIR / relative_path
        for relative_path in val_df["relative_path"]
    ]

    missing = [
        str(path)
        for path in paths
        if not path.is_file()
    ]

    if missing:
        raise FileNotFoundError(
            f"{len(missing)} validation files are missing. "
            f"First missing file: {missing[0]}"
        )

    y_true = np.asarray([
        main_map[class_name]
        for class_name in val_df["main_class"]
    ], dtype=np.int64)

    relative_paths = [
        str(path.relative_to(DATASET_DIR))
        for path in paths
    ]

    print(f"\nValidation recordings: {len(paths)}")
    print(f"Number of classes: {len(class_names)}")
    print("Training partition is not used.")
    print("Test partition is not accessed.")

    # -----------------------------------------------------
    # Load CLAP
    # -----------------------------------------------------

    print("\nLoading CLAP model...")

    model = CLAP_Module(
        enable_fusion=False,
        device=DEVICE,
    )

    model.load_ckpt(
        ckpt=str(CHECKPOINT),
        verbose=False,
    )

    print("CLAP model loaded.")

    # -----------------------------------------------------
    # Extract text embeddings
    # -----------------------------------------------------

    print("\nExtracting text embeddings...")

    text_start = time.time()

    text_embeddings = model.get_text_embedding(
        CLASS_PROMPTS,
        use_tensor=True,
    )

    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu().numpy()

    text_embeddings = np.asarray(text_embeddings)

    if text_embeddings.shape != (len(class_names), 512):
        raise ValueError(
            f"Unexpected text embedding shape: "
            f"{text_embeddings.shape}"
        )

    if not np.isfinite(text_embeddings).all():
        raise ValueError("Non-finite text embeddings detected.")

    text_time = time.time() - text_start

    print(f"Text embedding shape: {text_embeddings.shape}")
    print(f"Text embedding time: {text_time:.2f} seconds")

    # -----------------------------------------------------
    # Extract validation audio embeddings
    # -----------------------------------------------------

    audio_embeddings = []

    print("\nExtracting validation audio embeddings...")

    audio_start = time.time()

    for start in range(0, len(paths), BATCH_SIZE):

        batch_paths = paths[start:start + BATCH_SIZE]

        batch_embeddings = model.get_audio_embedding_from_filelist(
            x=[str(path) for path in batch_paths],
            use_tensor=True,
        )

        batch_array = batch_embeddings.detach().cpu().numpy()

        if batch_array.shape != (len(batch_paths), 512):
            raise ValueError(
                f"Unexpected audio embedding shape: "
                f"{batch_array.shape}"
            )

        if not np.isfinite(batch_array).all():
            raise ValueError(
                "Non-finite audio embedding values detected."
            )

        audio_embeddings.append(batch_array)

        processed = min(start + BATCH_SIZE, len(paths))
        print(f"Processed {processed}/{len(paths)}")

    audio_embeddings = np.vstack(audio_embeddings)

    audio_time = time.time() - audio_start

    print(f"Audio embedding shape: {audio_embeddings.shape}")
    print(f"Audio embedding time: {audio_time:.2f} seconds")

    # -----------------------------------------------------
    # Calculate audio-text cosine similarities
    # -----------------------------------------------------

    print("\nCalculating audio-text similarities...")

    similarity_matrix = cosine_similarity(
        audio_embeddings,
        text_embeddings,
    )

    if similarity_matrix.shape != (
        len(paths),
        len(class_names),
    ):
        raise ValueError(
            f"Unexpected similarity matrix shape: "
            f"{similarity_matrix.shape}"
        )

    # Highest cosine similarity determines predicted class.
    predictions = np.argmax(similarity_matrix, axis=1)

    # -----------------------------------------------------
    # Evaluation
    # -----------------------------------------------------

    accuracy = accuracy_score(y_true, predictions)

    macro_f1 = f1_score(
        y_true,
        predictions,
        labels=list(range(len(class_names))),
        average="macro",
        zero_division=0,
    )

    print("\nZero-shot validation results:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro-F1: {macro_f1:.4f}")

    # -----------------------------------------------------
    # Save predictions and similarity scores
    # -----------------------------------------------------

    prediction_data = {
        "relative_path": relative_paths,
        "actual_class": [
            class_names[index] for index in y_true
        ],
        "predicted_class": [
            class_names[index] for index in predictions
        ],
        "correct": y_true == predictions,
    }

    for class_id, class_name in enumerate(class_names):
        prediction_data[
            f"similarity_{class_id:02d}_{class_name}"
        ] = similarity_matrix[:, class_id]

    prediction_df = pd.DataFrame(prediction_data)

    prediction_df.to_csv(
        OUTPUT_DIR / "validation_predictions_and_similarities.csv",
        index=False,
    )

    # Save raw embeddings and similarity matrix.
    np.savez_compressed(
        OUTPUT_DIR / "zero_shot_embeddings.npz",
        audio_embeddings=audio_embeddings,
        text_embeddings=text_embeddings,
        similarity_matrix=similarity_matrix,
        y_true=y_true,
        predictions=predictions,
        relative_paths=np.asarray(relative_paths),
    )

    # -----------------------------------------------------
    # Classification report
    # -----------------------------------------------------

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(len(class_names))),
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
        y_true,
        predictions,
        labels=list(range(len(class_names))),
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 65)
    print("ZERO-SHOT CLAP COMPLETED")
    print("=" * 65)

    print(f"Validation recordings: {len(paths)}")
    print(f"Audio embedding dimensions: {audio_embeddings.shape[1]}")
    print(f"Text embedding dimensions: {text_embeddings.shape[1]}")
    print(f"Validation accuracy: {accuracy:.4f}")
    print(f"Validation macro-F1: {macro_f1:.4f}")
    print(f"Text embedding time: {text_time:.2f} seconds")
    print(f"Audio embedding time: {audio_time:.2f} seconds")

    print("\nSaved files:")
    print(OUTPUT_DIR / "zero_shot_class_prompts.csv")
    print(OUTPUT_DIR / "zero_shot_embeddings.npz")
    print(OUTPUT_DIR / "validation_predictions_and_similarities.csv")
    print(OUTPUT_DIR / "validation_classification_report.csv")
    print(OUTPUT_DIR / "validation_confusion_matrix.csv")

    print("\nImportant:")
    print("- No DataSEC classifier was trained.")
    print("- No training labels were used to fit a model.")
    print("- Cosine similarities are not calibrated probabilities.")
    print("- Test partition was not evaluated.")


if __name__ == "__main__":
    main()
