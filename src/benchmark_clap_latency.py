
from __future__ import annotations

import hashlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from laion_clap import CLAP_Module

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from common import DATASET_ROOT
import train_cnn14 as training

DATASET_DIR = Path(DATASET_ROOT)
CHECKPOINT = ROOT / "src" / "630k-audioset-best.pt"
OUTPUT_DIR = ROOT / "logs" / "clap_latency"

SAMPLE_SIZE = 100
WARMUP_COUNT = 5
BATCH_SIZE = 16
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def synchronise():
    if DEVICE == "cuda":
        torch.cuda.synchronize()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)

    manifest = training.load_manifest()
    validation = (
        manifest[manifest["partition"] == "validation"]
        .copy()
        .reset_index(drop=True)
    )

    if len(validation) < SAMPLE_SIZE + WARMUP_COUNT:
        raise ValueError("Insufficient validation recordings.")

    # Deterministic sample; no training or test recordings.
    sample = validation.sample(
        n=SAMPLE_SIZE + WARMUP_COUNT,
        random_state=SEED,
    ).reset_index(drop=True)

    paths = [
        DATASET_DIR / str(relative_path)
        for relative_path in sample["relative_path"]
    ]

    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing[0])

    warmup_paths = paths[:WARMUP_COUNT]
    measured_paths = paths[WARMUP_COUNT:]

    print("=" * 65)
    print("LAION-CLAP INFERENCE LATENCY BENCHMARK")
    print("=" * 65)
    print(f"Device: {DEVICE}")
    if DEVICE == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Validation sample: {len(measured_paths)}")
    print(f"Warm-up recordings: {WARMUP_COUNT}")
    print("Training partition accessed: False")
    print("Test partition accessed: False")

    # Model loading time is recorded separately.
    synchronise()
    load_start = time.perf_counter()

    model = CLAP_Module(
        enable_fusion=False,
        device=DEVICE,
    )
    model.load_ckpt(
        ckpt=str(CHECKPOINT),
        verbose=False,
    )

    synchronise()
    model_load_seconds = time.perf_counter() - load_start

    # Warm-up.
    for path in warmup_paths:
        model.get_audio_embedding_from_filelist(
            x=[str(path)],
            use_tensor=True,
        )
    synchronise()

    # Batch-size-1 latency.
    individual_times = []

    for index, path in enumerate(measured_paths, start=1):
        synchronise()
        start = time.perf_counter()

        result = model.get_audio_embedding_from_filelist(
            x=[str(path)],
            use_tensor=True,
        )

        synchronise()
        elapsed = time.perf_counter() - start

        embedding = result.detach().cpu().numpy()

        if embedding.shape != (1, 512):
            raise ValueError(f"Unexpected embedding shape: {embedding.shape}")
        if not np.isfinite(embedding).all():
            raise ValueError("Non-finite embedding.")

        individual_times.append(elapsed)

        if index % 20 == 0:
            print(f"Single-recording benchmark: {index}/{SAMPLE_SIZE}")

    # Batch-16 throughput benchmark on the same sample.
    batch_times = []

    for start_index in range(0, len(measured_paths), BATCH_SIZE):
        batch_paths = measured_paths[
            start_index:start_index + BATCH_SIZE
        ]

        synchronise()
        start = time.perf_counter()

        result = model.get_audio_embedding_from_filelist(
            x=[str(path) for path in batch_paths],
            use_tensor=True,
        )

        synchronise()
        elapsed = time.perf_counter() - start

        embeddings = result.detach().cpu().numpy()

        if embeddings.shape != (len(batch_paths), 512):
            raise ValueError(f"Unexpected batch shape: {embeddings.shape}")
        if not np.isfinite(embeddings).all():
            raise ValueError("Non-finite batch embeddings.")

        batch_times.append({
            "batch_start_index": start_index,
            "batch_size": len(batch_paths),
            "elapsed_seconds": elapsed,
            "recordings_per_second": len(batch_paths) / elapsed,
        })

    times_ms = np.asarray(individual_times) * 1000
    total_batch_seconds = sum(
        row["elapsed_seconds"] for row in batch_times
    )

    summary = {
        "model": "LAION-CLAP",
        "package": "laion_clap",
        "package_version": "1.1.7",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "fusion_enabled": False,
        "device": DEVICE,
        "gpu": (
            torch.cuda.get_device_name(0)
            if DEVICE == "cuda" else None
        ),
        "partition": "validation",
        "sample_size": SAMPLE_SIZE,
        "warmup_count": WARMUP_COUNT,
        "random_seed": SEED,
        "single_recording_latency_ms_mean": float(np.mean(times_ms)),
        "single_recording_latency_ms_median": float(np.median(times_ms)),
        "single_recording_latency_ms_p95": float(np.percentile(times_ms, 95)),
        "single_recording_throughput_per_second": float(
            1000 / np.mean(times_ms)
        ),
        "batch_size": BATCH_SIZE,
        "batch_total_seconds": float(total_batch_seconds),
        "batch_throughput_per_second": float(
            SAMPLE_SIZE / total_batch_seconds
        ),
        "model_load_seconds": float(model_load_seconds),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "test_partition_accessed": False,
        "training_partition_accessed": False,
        "timing_scope": (
            "Audio embedding API call including CLAP audio preprocessing, "
            "model inference and conversion to CPU NumPy output. "
            "Model loading is measured separately."
        ),
    }

    pd.DataFrame({
        "recording_index": range(1, SAMPLE_SIZE + 1),
        "latency_ms": times_ms,
    }).to_csv(
        OUTPUT_DIR / "single_recording_latency.csv",
        index=False,
    )

    pd.DataFrame(batch_times).to_csv(
        OUTPUT_DIR / "batch_latency.csv",
        index=False,
    )

    with (OUTPUT_DIR / "benchmark_metadata.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\nBENCHMARK RESULTS")
    print(f"Model loading: {model_load_seconds:.2f}s")
    print(f"Single-recording mean: {np.mean(times_ms):.2f}ms")
    print(f"Single-recording median: {np.median(times_ms):.2f}ms")
    print(f"Single-recording p95: {np.percentile(times_ms, 95):.2f}ms")
    print(f"Batch-{BATCH_SIZE} throughput: {summary['batch_throughput_per_second']:.2f} recordings/s")
    print(f"\nSaved results to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
