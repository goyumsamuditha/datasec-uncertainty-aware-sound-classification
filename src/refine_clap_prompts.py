
import json
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


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = ROOT / "src" / "630k-audioset-best.pt"

BASELINE_DIR = ROOT / "models" / "clap_zero_shot"
OUTPUT_DIR = ROOT / "models" / "clap_prompt_refinement"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Three descriptions per class. These are designed to describe
# the sound itself while retaining the DataSEC class terminology.
PROMPT_ENSEMBLES = {
    "Bells": [
        "The sound of bells ringing clearly.",
        "A recording of bell tones, chimes and repeated bell strikes.",
        "The sound of a bell being struck and resonating.",
    ],
    "Birds": [
        "The sound of birds singing and calling outdoors.",
        "A recording of individual birds chirping and tweeting.",
        "Bird calls and bird songs in a natural environment.",
    ],
    "Cat fights and moans": [
        "The sound of cats fighting, growling and hissing.",
        "Cats yowling, moaning and making distressed vocal sounds.",
        "A recording of aggressive cat vocalisations.",
    ],
    "Chicken coop": [
        "The sound of chickens clucking, cackling and squawking.",
        "A recording of poultry making repeated chicken calls in a coop.",
        "Chickens clucking and moving around in a poultry enclosure, rather than people speaking.",
    ],
    "Cicadas and crickets": [
        "The sound of cicadas buzzing and calling.",
        "Crickets chirping repeatedly outdoors.",
        "A recording of insect sounds, including cicada and cricket calls.",
    ],
    "Crows seagulls and magpies": [
        "The sound of crows, seagulls and magpies calling.",
        "A recording of loud cawing, gull calls and magpie vocalisations.",
        "Bird calls from crows, seagulls or magpies.",
    ],
    "Dog barkings and howlings": [
        "The sound of dogs barking and howling.",
        "A recording of dog growls, barks and howls.",
        "Dogs making repeated vocal sounds.",
    ],
    "Glass breaking": [
        "The sound of glass breaking and shattering.",
        "A recording of glass objects crashing and fragments scattering.",
        "Sharp impacts followed by the sound of broken glass.",
    ],
    "Horn": [
        "The sound of a vehicle horn honking.",
        "A recording of short or sustained vehicle horn blasts.",
        "A loud warning horn sounding from a vehicle.",
    ],
    "Jet aircrafts": [
        "The sound of a jet aircraft flying overhead.",
        "A recording of jet engines producing a loud aircraft roar.",
        "The sound of a jet-powered aeroplane passing through the sky.",
    ],
    "Lawn mower brush cutter and olive shaker": [
        "The sound of a lawn mower cutting grass.",
        "A recording of a brush cutter or grass trimmer operating.",
        "The sound of an olive shaker or powered agricultural cutting equipment.",
    ],
    "Music": [
        "The sound of music being played.",
        "A recording of musical instruments, melodies or songs.",
        "Music with rhythmic and melodic sounds.",
    ],
    "Propeller aircrafts": [
        "The sound of a propeller-driven aeroplane flying overhead.",
        "A recording of an aircraft with a rotating propeller and engine.",
        "The sound of an aeroplane or helicopter producing propeller noise.",
    ],
    "Sirens and alarms": [
        "The sound of emergency sirens or warning alarms.",
        "A recording of repeated electronic alarm tones or a rising and falling siren.",
        "Loud warning signals produced by sirens and alarms.",
    ],
    "Thunder fireworks and gunshot": [
        "The sound of thunder, fireworks or gunshots.",
        "A recording of explosive bangs, thunderclaps and sharp gunfire.",
        "Sudden loud explosive sounds such as fireworks and gunshots.",
    ],
    "Train": [
        "The sound of a train travelling along railway tracks.",
        "A recording of a moving train with rail and wheel noise.",
        "Train movement, including rolling wheels and railway sounds.",
    ],
    "Vacuum cleaner fan and hairdryer": [
        "The sound of a vacuum cleaner operating.",
        "A recording of a fan or hairdryer producing continuous airflow noise.",
        "The steady mechanical and airflow noise of household electrical appliances.",
    ],
    "Vehicle idling": [
        "The sound of a stationary vehicle engine idling.",
        "A recording of a car, truck or motorbike engine running while the vehicle is stopped.",
        "An engine running at idle speed without the vehicle passing by.",
    ],
    "Vehicle pass-by": [
        "The sound of a vehicle driving past the recording location.",
        "A recording of a passing car, truck or motorbike.",
        "The changing engine and tyre noise of a moving vehicle passing by.",
    ],
    "Voices": [
        "The sound of people speaking and talking.",
        "A recording of human speech, conversations and spoken voices.",
        "People talking, conversing or speaking, rather than animal calls.",
    ],
    "Wind turbine": [
        "The sound of a wind turbine operating, with rotating blades and mechanical noise.",
        "A recording of wind turbine blade rotation and turbine machinery.",
        "The steady mechanical sound of a wind power generator operating, not an aircraft.",
    ],
    "Workshop": [
        "The sound of machinery and tools operating in a workshop.",
        "A recording of drills, grinders, saws and jackhammers being used.",
        "Workshop equipment such as air compressors and power tools operating.",
    ],
}


def normalise_rows(array):
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    return array / np.maximum(norms, 1e-12)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    embedding_file = BASELINE_DIR / "zero_shot_embeddings.npz"

    if not embedding_file.is_file():
        raise FileNotFoundError(embedding_file)

    # Reuse the original cached validation audio embeddings.
    cached = np.load(embedding_file, allow_pickle=True)

    audio_embeddings = np.asarray(cached["audio_embeddings"])
    y_true = np.asarray(cached["y_true"], dtype=np.int64)
    relative_paths = cached["relative_paths"]

    # Read the established class order from the label mapping.
    mapping_file = ROOT / "logs" / "hierarchical_label_mapping.json"

    with open(mapping_file, "r", encoding="utf-8") as file:
        mapping = json.load(file)

    class_names = mapping["main_classes"]

    if audio_embeddings.shape != (len(y_true), 512):
        raise ValueError("Unexpected cached audio embedding shape.")

    if len(class_names) != 22:
        raise ValueError("Expected exactly 22 main classes.")

    if set(PROMPT_ENSEMBLES) != set(class_names):
        raise ValueError("Prompt classes do not match DataSEC classes.")

    if len(relative_paths) != len(y_true):
        raise ValueError("Cached file paths and labels do not match.")

    print("=" * 65)
    print("LAION-CLAP PROMPT ENSEMBLE REFINEMENT")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    print(f"Validation recordings: {len(y_true)}")
    print("Cached audio embeddings will be reused.")
    print("Training and test partitions are not accessed.")

    # Preserve the exact original prompt baseline predictions.
    baseline_predictions_file = (
        BASELINE_DIR / "validation_predictions_and_similarities.csv"
    )
    baseline_df = pd.read_csv(baseline_predictions_file)

    baseline_predictions = np.asarray([
        class_names.index(name)
        for name in baseline_df["predicted_class"]
    ])

    if not np.array_equal(
        baseline_df["relative_path"].astype(str).to_numpy(),
        np.asarray(relative_paths).astype(str),
    ):
        raise ValueError("Cached paths do not match baseline predictions.")

    if not np.array_equal(
        baseline_predictions,
        np.argmax(
            np.asarray(cached["similarity_matrix"]),
            axis=1,
        ),
    ):
        raise ValueError("Baseline predictions do not match cached scores.")

    # Load CLAP only to obtain the new text embeddings.
    model = CLAP_Module(
        enable_fusion=False,
        device=DEVICE,
    )
    model.load_ckpt(ckpt=str(CHECKPOINT), verbose=False)

    all_prompts = []
    prompt_rows = []

    for class_id, class_name in enumerate(class_names):
        prompts = PROMPT_ENSEMBLES[class_name]

        for prompt_number, prompt in enumerate(prompts, start=1):
            all_prompts.append(prompt)

            prompt_rows.append({
                "class_id": class_id,
                "class_name": class_name,
                "prompt_number": prompt_number,
                "prompt": prompt,
            })

    print(f"\nExtracting {len(all_prompts)} text embeddings...")

    text_embeddings = model.get_text_embedding(
        all_prompts,
        use_tensor=True,
    )

    if isinstance(text_embeddings, torch.Tensor):
        text_embeddings = text_embeddings.detach().cpu().numpy()

    text_embeddings = np.asarray(text_embeddings)

    if text_embeddings.shape != (len(all_prompts), 512):
        raise ValueError(
            f"Unexpected text embedding shape: {text_embeddings.shape}"
        )

    if not np.isfinite(text_embeddings).all():
        raise ValueError("Non-finite text embeddings.")

    # Average the three normalised prompt embeddings per class,
    # then normalise the resulting class embedding.
    class_embeddings = []

    position = 0

    for class_name in class_names:
        number_of_prompts = len(PROMPT_ENSEMBLES[class_name])

        group = text_embeddings[
            position:position + number_of_prompts
        ]

        group = normalise_rows(group)
        mean_embedding = group.mean(axis=0, keepdims=True)
        mean_embedding = normalise_rows(mean_embedding)

        class_embeddings.append(mean_embedding[0])
        position += number_of_prompts

    class_embeddings = np.vstack(class_embeddings)

    if class_embeddings.shape != (22, 512):
        raise ValueError("Unexpected class embedding shape.")

    # Compare the cached audio embeddings with the ensemble embeddings.
    similarity_matrix = cosine_similarity(
        audio_embeddings,
        class_embeddings,
    )

    predictions = np.argmax(similarity_matrix, axis=1)

    baseline_macro_f1 = f1_score(
        y_true,
        baseline_predictions,
        labels=list(range(22)),
        average="macro",
        zero_division=0,
    )

    refined_macro_f1 = f1_score(
        y_true,
        predictions,
        labels=list(range(22)),
        average="macro",
        zero_division=0,
    )

    baseline_accuracy = accuracy_score(y_true, baseline_predictions)
    refined_accuracy = accuracy_score(y_true, predictions)

    print("\nValidation comparison:")
    print(f"Original prompt macro-F1: {baseline_macro_f1:.4f}")
    print(f"Ensemble prompt macro-F1: {refined_macro_f1:.4f}")
    print(f"Original prompt accuracy: {baseline_accuracy:.4f}")
    print(f"Ensemble prompt accuracy: {refined_accuracy:.4f}")
    print(f"Macro-F1 change: {refined_macro_f1 - baseline_macro_f1:+.4f}")

    # Save every prompt so the experiment is reproducible.
    pd.DataFrame(prompt_rows).to_csv(
        OUTPUT_DIR / "prompt_ensemble_definitions.csv",
        index=False,
    )

    report = classification_report(
        y_true,
        predictions,
        labels=list(range(22)),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    pd.DataFrame(report).T.to_csv(
        OUTPUT_DIR / "validation_classification_report.csv"
    )

    cm = confusion_matrix(
        y_true,
        predictions,
        labels=list(range(22)),
    )

    pd.DataFrame(
        cm,
        index=class_names,
        columns=class_names,
    ).to_csv(
        OUTPUT_DIR / "validation_confusion_matrix.csv"
    )

    prediction_df = pd.DataFrame({
        "relative_path": relative_paths,
        "actual_class": [class_names[i] for i in y_true],
        "original_prediction": [
            class_names[i] for i in baseline_predictions
        ],
        "refined_prediction": [
            class_names[i] for i in predictions
        ],
        "original_correct": y_true == baseline_predictions,
        "refined_correct": y_true == predictions,
    })

    for class_id, class_name in enumerate(class_names):
        prediction_df[
            f"similarity_{class_id:02d}_{class_name}"
        ] = similarity_matrix[:, class_id]

    prediction_df.to_csv(
        OUTPUT_DIR / "validation_predictions.csv",
        index=False,
    )

    np.savez_compressed(
        OUTPUT_DIR / "refined_embeddings.npz",
        class_embeddings=class_embeddings,
        similarity_matrix=similarity_matrix,
        predictions=predictions,
        y_true=y_true,
        relative_paths=np.asarray(relative_paths),
    )

    summary = pd.DataFrame([
        {
            "method": "Original generic prompts",
            "accuracy": baseline_accuracy,
            "macro_f1": baseline_macro_f1,
        },
        {
            "method": "Three-prompt class ensemble",
            "accuracy": refined_accuracy,
            "macro_f1": refined_macro_f1,
        },
    ])

    summary.to_csv(
        OUTPUT_DIR / "validation_comparison.csv",
        index=False,
    )

    print("\nSaved outputs:")
    print(OUTPUT_DIR)
    print("\nPrompt refinement completed.")
    print("The test partition has not been accessed.")


if __name__ == "__main__":
    main()
