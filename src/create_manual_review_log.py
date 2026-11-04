import csv
from pathlib import Path

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
OUTPUT_FILE = PROJECT_DIR / "logs" / "manual_similarity_review.csv"

columns = [
    "file_a",
    "file_b",
    "clap_similarity",
    "manual_assessment",
    "waveform_similarity",
    "notes"
]

# Only manually assessed pairs are included.
# Waveform results are included only where available.
records = [
    ["Voices/Voices-1690.wav", "Voices/Voices-1641.wav", 0.9083, "Different", "", ""],
    ["Voices/Voices-1713.wav", "Voices/Voices-1641.wav", 0.9076, "Different", "", ""],
    ["Voices/Voices-1690.wav", "Voices/Voices-1713.wav", 0.8983, "Different", "", ""],
    ["Vehicle pass-by/Car pass-by-0049.wav", "Vehicle pass-by/Car pass-by-0051.wav", 0.8355, "Very similar", "", ""],
    ["Vehicle pass-by/Car pass-by-0020.wav", "Vehicle pass-by/Car pass-by-0051.wav", 0.8011, "Identical", 0.175484, "Low waveform similarity despite listening assessment."],
    ["Crows seagulls and magpies/Crows/Crows-0008.wav", "Crows seagulls and magpies/Crows/Crows-0038.wav", 0.7549, "Different", "", ""],
    ["Chicken coop/Chicken coop-0019.wav", "Chicken coop/Chicken coop-0032.wav", 0.7592, "Different", "", ""],
    ["Glass breaking/Glass breaking-0076.wav", "Glass breaking/Glass breaking-0049.wav", 0.9567, "Very similar", "", ""],
    ["Glass breaking/Glass breaking-0076.wav", "Glass breaking/Glass breaking-0027.wav", 0.9361, "Very similar", "", ""],
    ["Voices/Voices-0892.wav", "Voices/Voices-1641.wav", 0.8748, "Slightly similar", "", ""],
    ["Voices/Voices-0892.wav", "Voices/Voices-0959.wav", 0.8628, "Different", "", ""],
    ["Voices/Voices-0297.wav", "Voices/Voices-0114.wav", 0.8535, "Different", "", ""],
    ["Vehicle pass-by/Car pass-by-0010.wav", "Vehicle pass-by/Car pass-by-0049.wav", 0.7977, "Slightly similar", "", ""],
]

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(columns)
    writer.writerows(records)

print(f"Saved {len(records)} reviewed pairs to:")
print(OUTPUT_FILE)