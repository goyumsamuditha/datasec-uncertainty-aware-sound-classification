
from pathlib import Path
import wave
import csv
from collections import Counter

DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
OUTPUT_PATH = Path("logs/audio_integrity.csv")

if not DATASET_ROOT.is_dir():
    raise FileNotFoundError(f"Dataset folder not found: {DATASET_ROOT}")

OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

wav_files = sorted(
    p for p in DATASET_ROOT.rglob("*")
    if p.is_file() and p.suffix.lower() == ".wav"
)

sample_rates = Counter()
channels_count = Counter()
issues = []
durations = []
readable_count = 0

with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "relative_path", "sample_rate_hz", "channels",
        "duration_seconds", "frames", "status", "issue"
    ])

    for i, path in enumerate(wav_files, start=1):
        try:
            with wave.open(str(path), "rb") as audio:
                sample_rate = audio.getframerate()
                channels = audio.getnchannels()
                frames = audio.getnframes()
                duration = frames / sample_rate if sample_rate else 0

            readable_count += 1
            sample_rates[sample_rate] += 1
            channels_count[channels] += 1
            durations.append(duration)

            writer.writerow([
                str(path.relative_to(DATASET_ROOT)),
                sample_rate, channels, round(duration, 3),
                frames, "OK", ""
            ])

        except Exception as e:
            issues.append((path, str(e)))
            writer.writerow([
                str(path.relative_to(DATASET_ROOT)),
                "", "", "", "", "ERROR", str(e)
            ])

        if i % 500 == 0:
            print(f"Checked {i}/{len(wav_files)} files")

print("\n--- AUDIO INTEGRITY SUMMARY ---")
print(f"Total WAV files found: {len(wav_files)}")
print(f"Readable WAV files: {readable_count}")
print(f"Files with read errors: {len(issues)}")
print(f"Sample rates (Hz): {dict(sorted(sample_rates.items()))}")
print(f"Channel counts: {dict(sorted(channels_count.items()))}")

if durations:
    print(f"Shortest duration: {min(durations):.3f} seconds")
    print(f"Longest duration: {max(durations):.3f} seconds")
    print(f"Mean duration: {sum(durations) / len(durations):.3f} seconds")

if issues:
    print("\nFirst 10 files with errors:")
    for path, error in issues[:10]:
        print(f"{path}: {error}")

print(f"\nDetailed results saved to: {OUTPUT_PATH.resolve()}")
