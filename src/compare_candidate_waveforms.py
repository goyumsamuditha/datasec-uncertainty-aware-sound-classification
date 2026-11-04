
import numpy as np
import soundfile as sf
from scipy.signal import correlate, resample_poly
from pathlib import Path
from math import gcd
import csv

DATASET_DIR = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
OUTPUT_FILE = Path(
    r"G:\My Drive\MSC\AI\Assignment\logs\candidate_partial_comparison.csv"
)

PAIRS = [
    (
        "Glass breaking/Glass breaking-0049.wav",
        "Glass breaking/Glass breaking-0027.wav",
    ),
    (
        "Vehicle idling/car truck idling/car truck idling-0055.wav",
        "Vehicle idling/car truck idling/car truck idling-0052.wav",
    ),
    (
        "Vehicle pass-by/Car pass-by/Car pass-by-0020.wav",
        "Vehicle pass-by/Car pass-by/Car pass-by-0051.wav"
    ),
    (
        "Voices/Voices-0079.wav",
        "Voices/Voices-0204.wav",
    ),
]

TARGET_RATE = 4000
WINDOW_SECONDS = 0.25


def load_audio(path):
    audio, sr = sf.read(path)

    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    common = gcd(sr, TARGET_RATE)
    audio = resample_poly(
        audio,
        TARGET_RATE // common,
        sr // common,
    )

    audio = audio.astype(np.float64)
    audio -= np.mean(audio)

    return audio


def normalise(segment):
    segment = segment - np.mean(segment)
    norm = np.linalg.norm(segment)

    if norm == 0:
        return None

    return segment / norm


def best_partial_similarity(audio_a, audio_b):
    window = int(WINDOW_SECONDS * TARGET_RATE)

    if min(len(audio_a), len(audio_b)) < window:
        return None, None

    # Use the shorter recording as the query.
    if len(audio_a) > len(audio_b):
        audio_a, audio_b = audio_b, audio_a

    best_score = -1
    best_position = None

    # Compare multiple windows from the shorter recording.
    for start in range(0, len(audio_a) - window + 1, window // 2):
        segment = normalise(audio_a[start:start + window])

        if segment is None:
            continue

        # Compare this segment against the longer recording.
        correlation = correlate(
            audio_b,
            segment,
            mode="valid",
            method="fft",
        )

        # Local normalisation reduces amplitude sensitivity.
        energy = correlate(
            audio_b ** 2,
            np.ones(window),
            mode="valid",
            method="fft",
        )

        denominator = np.sqrt(np.maximum(energy, 1e-12))
        scores = np.abs(correlation) / denominator

        position = int(np.argmax(scores))
        score = float(scores[position])

        if score > best_score:
            best_score = score
            best_position = position / TARGET_RATE

    return best_score, best_position


def main():
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for file_a, file_b in PAIRS:
        path_a = DATASET_DIR / file_a
        path_b = DATASET_DIR / file_b

        print(f"\nComparing:\n{file_a}\n{file_b}")

        audio_a = load_audio(path_a)
        audio_b = load_audio(path_b)

        score, position = best_partial_similarity(audio_a, audio_b)

        results.append({
            "file_a": file_a,
            "file_b": file_b,
            "duration_a_seconds": round(len(audio_a) / TARGET_RATE, 3),
            "duration_b_seconds": round(len(audio_b) / TARGET_RATE, 3),
            "best_partial_similarity": (
                round(score, 6) if score is not None else "Not calculated"
            ),
            "matching_position_in_longer_audio_seconds": (
                round(position, 3) if position is not None else "Not calculated"
            ),
        })

        print("Best partial similarity:", results[-1]["best_partial_similarity"])
        print("Position in longer recording:", results[-1]["matching_position_in_longer_audio_seconds"])

    with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    print("\nComparison completed.")
    print("Results saved to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()
