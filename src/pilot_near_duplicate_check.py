
import csv
import random
from pathlib import Path

import numpy as np
from laion_clap import CLAP_Module
from sklearn.metrics.pairwise import cosine_similarity

# Project paths
PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
CHECKPOINT = PROJECT_DIR / "src" / "630k-audioset-best.pt"
OUTPUT_FILE = PROJECT_DIR / "logs" / "near_duplicate_pilot.csv"

# Reproducible sample
random.seed(42)

audio_files = sorted(DATASET_DIR.rglob("*.wav"))
sample_files = random.sample(audio_files, min(100, len(audio_files)))

print(f"Total dataset files: {len(audio_files)}")
print(f"Pilot sample: {len(sample_files)}")

# Load CLAP
model = CLAP_Module(enable_fusion=False, device="cuda")
model.load_ckpt(ckpt=str(CHECKPOINT), verbose=False)

# Extract embeddings in batches
embeddings = []

for start in range(0, len(sample_files), 16):
    batch = sample_files[start:start + 16]

    batch_embeddings = model.get_audio_embedding_from_filelist(
        x=[str(path) for path in batch],
        use_tensor=True
    )

    embeddings.append(batch_embeddings.detach().cpu().numpy())
    print(f"Processed {min(start + 16, len(sample_files))}/{len(sample_files)}")

embeddings = np.vstack(embeddings)

# Calculate cosine similarities
similarity_matrix = cosine_similarity(embeddings)

# Save the five nearest neighbours for each recording
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow([
        "audio_file",
        "nearest_file",
        "similarity",
        "same_class"
    ])

    for i, audio_file in enumerate(sample_files):
        similarities = similarity_matrix[i].copy()
        similarities[i] = -1

        nearest_indices = np.argsort(similarities)[-5:][::-1]

        for j in nearest_indices:
            nearest_file = sample_files[j]

            audio_class = audio_file.relative_to(DATASET_DIR).parts[0]
            nearest_class = nearest_file.relative_to(DATASET_DIR).parts[0]

            writer.writerow([
                str(audio_file.relative_to(DATASET_DIR)),
                str(nearest_file.relative_to(DATASET_DIR)),
                round(float(similarity_matrix[i, j]), 4),
                audio_class == nearest_class
            ])

print("\nPilot completed.")
print(f"Results saved to: {OUTPUT_FILE}")
print("Important: Similarity is a screening signal, not proof of shared source.")
