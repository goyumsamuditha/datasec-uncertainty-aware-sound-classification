from common import DATASET_ROOT

from pathlib import Path
import sys
import math

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly

# --------------------------------------------------
# Project paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATASET_ROOT = Path(DATASET_ROOT)
MANIFEST_PATH = PROJECT_ROOT / "logs" / "final_split_manifest.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"
PANN_SOURCE = PROJECT_ROOT / "src" / "panns_source" / "pytorch"

sys.path.insert(0, str(PANN_SOURCE))

from models import Cnn14, init_layer

# --------------------------------------------------
# Configuration
# --------------------------------------------------
TARGET_SAMPLE_RATE = 32000
CLIP_SECONDS = 10
NUM_CLASSES = 22
BATCH_SIZE = 1
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# --------------------------------------------------
# Model
# --------------------------------------------------
class TransferCNN14(nn.Module):
    def __init__(self):
        super().__init__()

        self.base = Cnn14(
            sample_rate=32000,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527
        )

        self.fc_transfer = nn.Linear(2048, NUM_CLASSES)
        init_layer(self.fc_transfer)

    def forward(self, waveform):
        output = self.base(waveform)
        output["clipwise_output"] = self.fc_transfer(
            output["embedding"]
        )
        return output


# --------------------------------------------------
# Manifest and label checks
# --------------------------------------------------
def load_training_record():
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    print("\nManifest columns:", list(df.columns))
    print("Manifest rows:", len(df))

    partition_column = next(
        (c for c in ["partition", "split_name", "split"]
         if c in df.columns),
        None
    )

    label_column = next(
        (c for c in ["main_class", "class_name", "class", "label"]
         if c in df.columns),
        None
    )

    path_column = next(
        (c for c in ["relative_path", "audio_path", "filepath", "path"]
         if c in df.columns),
        None
    )

    if not partition_column or not label_column or not path_column:
        raise ValueError(
            "Could not identify required manifest columns. "
            f"Found: {list(df.columns)}"
        )

    train_df = df[
        df[partition_column].astype(str).str.lower().isin(
            ["train", "training"]
        )
    ].copy()

    if train_df.empty:
        raise ValueError(
            f"No training rows found in {partition_column}."
        )

    labels = sorted(train_df[label_column].dropna().astype(str).unique())

    if len(labels) != NUM_CLASSES:
        raise ValueError(
            f"Expected 22 main classes, found {len(labels)}: {labels}"
        )

    label_to_index = {
        label: index for index, label in enumerate(labels)
    }

    # Select a reproducible training recording.
    row = train_df.sample(
        n=1,
        random_state=RANDOM_SEED
    ).iloc[0]

    stored_path = Path(str(row[path_column]))

    if stored_path.is_absolute():
        audio_path = stored_path
    else:
        candidates = [
            DATASET_ROOT / stored_path,
            PROJECT_ROOT / stored_path,
            PROJECT_ROOT / "data" / stored_path
        ]
        audio_path = next(
            (p for p in candidates if p.exists()),
            candidates[0]
        )

    if not audio_path.exists():
        raise FileNotFoundError(
            f"Selected audio file not found: {audio_path}"
        )

    label = str(row[label_column])

    return audio_path, label, label_to_index


# --------------------------------------------------
# Audio loading and preprocessing
# --------------------------------------------------
def load_audio(path):
    waveform, sample_rate = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    # Convert to mono.
    waveform = waveform.mean(axis=1)

    if waveform.size == 0:
        raise ValueError("Audio file is empty.")

    if not np.isfinite(waveform).all():
        raise ValueError("Audio contains NaN or infinite values.")

    original_duration = len(waveform) / sample_rate

    # Resample to 32 kHz.
    if sample_rate != TARGET_SAMPLE_RATE:
        divisor = math.gcd(sample_rate, TARGET_SAMPLE_RATE)
        up = TARGET_SAMPLE_RATE // divisor
        down = sample_rate // divisor

        waveform = resample_poly(
            waveform,
            up=up,
            down=down
        ).astype(np.float32)

    target_length = TARGET_SAMPLE_RATE * CLIP_SECONDS

    if len(waveform) > target_length:
        # Reproducible random crop.
        rng = np.random.default_rng(RANDOM_SEED)
        start = rng.integers(
            0, len(waveform) - target_length + 1
        )
        waveform = waveform[start:start + target_length]

    elif len(waveform) < target_length:
        # Pad short recordings with zeros.
        padding = target_length - len(waveform)
        waveform = np.pad(
            waveform,
            (0, padding),
            mode="constant"
        )

    return waveform.astype(np.float32), original_duration, sample_rate


# --------------------------------------------------
# Main experiment
# --------------------------------------------------
def main():
    print("=" * 65)
    print("REAL DATASEC TRAINING-BATCH TEST")
    print("=" * 65)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    # 1. Select a real training recording.
    audio_path, label, label_to_index = load_training_record()

    print("\nSelected training recording:")
    print("File:", audio_path.name)
    print("Class:", label)
    print("Class index:", label_to_index[label])

    # 2. Load and preprocess the recording.
    waveform, original_duration, original_sample_rate = load_audio(
        audio_path
    )

    print("\nAudio preprocessing:")
    print("Original sample rate:", original_sample_rate)
    print(f"Original duration: {original_duration:.3f} seconds")
    print("Model sample rate:", TARGET_SAMPLE_RATE)
    print("Model input duration:", CLIP_SECONDS, "seconds")
    print("Waveform shape:", waveform.shape)

    # 3. Create model and load pretrained weights.
    model = TransferCNN14()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )

    model.base.load_state_dict(
        checkpoint["model"],
        strict=True
    )

    # 4. Freeze all CNN14 parameters.
    for parameter in model.base.parameters():
        parameter.requires_grad = False

    # Unfreeze only conv_block6.
    for parameter in model.base.conv_block6.parameters():
        parameter.requires_grad = True

    # Keep transfer classifier trainable.
    for parameter in model.fc_transfer.parameters():
        parameter.requires_grad = True

    model = model.to(device)

    trainable_count = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )

    print("\nTrainable parameters:", f"{trainable_count:,}")

    # 5. Keep frozen BatchNorm layers in evaluation mode.
    model.train()
    for block_name in [
        "conv_block1",
        "conv_block2",
        "conv_block3",
        "conv_block4",
        "conv_block5"
    ]:
        getattr(model.base, block_name).eval()

    # 6. Prepare real waveform and target.
    inputs = torch.from_numpy(waveform).unsqueeze(0).to(device)

    targets = torch.tensor(
        [label_to_index[label]],
        dtype=torch.long,
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    # 7. Forward pass, loss, backward pass and optimiser update.
    
    # 7. Diagnostic forward pass in full precision (FP32).
    # This temporarily disables mixed precision so we can
    # determine whether FP16 is contributing to the NaN.

    model.eval()

    # Keep only conv_block6 in training mode.
    model.base.conv_block6.train()
    model.fc_transfer.train()

    # Frozen BatchNorm layers remain in evaluation mode.
    for block_name in [
        "conv_block1",
        "conv_block2",
        "conv_block3",
        "conv_block4",
        "conv_block5"
    ]:
        getattr(model.base, block_name).eval()

    print("\nInput diagnostics:")
    print("Input finite:", torch.isfinite(inputs).all().item())
    print("Input minimum:", inputs.min().item())
    print("Input maximum:", inputs.max().item())
    print("Input contains non-zero samples:", torch.count_nonzero(inputs).item())

    optimizer.zero_grad(set_to_none=True)

    try:
        # First test: FP32, without autocast.
        with torch.autocast(
            device_type="cuda",
            enabled=False
        ):
            output = model(inputs)

            logits = output["clipwise_output"]
            embedding = output["embedding"]

            print("\nModel output diagnostics:")
            print("Embedding shape:", tuple(embedding.shape))
            print("Embedding finite:", torch.isfinite(embedding).all().item())
            print("Logits shape:", tuple(logits.shape))
            print("Logits finite:", torch.isfinite(logits).all().item())

            if torch.isfinite(logits).all():
                print("Logits minimum:", logits.min().item())
                print("Logits maximum:", logits.max().item())

            if not torch.isfinite(embedding).all():
                raise ValueError("CNN14 embedding contains NaN or infinity.")

            if not torch.isfinite(logits).all():
                raise ValueError("Classifier logits contain NaN or infinity.")

            loss = criterion(logits, targets)

        print("\nLoss:", loss.item())
        print("Loss finite:", torch.isfinite(loss).item())

        if not torch.isfinite(loss):
            raise ValueError("Loss is not finite. Optimiser update cancelled.")

        loss.backward()

        # Check gradients before changing parameters.
        invalid_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]

        if invalid_gradients:
            raise ValueError(
                "Non-finite gradients found: "
                + ", ".join(invalid_gradients)
            )

        optimizer.step()

        # Verify that trainable parameters remain finite.
        invalid_parameters = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not torch.isfinite(parameter).all()
        ]

        if invalid_parameters:
            raise ValueError(
                "Non-finite parameters found after update: "
                + ", ".join(invalid_parameters)
            )

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
        else:
            peak_memory = None

        print("\n" + "=" * 65)
        print("REAL DATASEC FP32 TRAINING-BATCH TEST PASSED")
        print("=" * 65)
        print("Target class:", label)
        print("Output shape:", tuple(logits.shape))
        print("Finite loss:", loss.item())
        print("Backward pass: PASSED")
        print("Optimiser update: PASSED")

        if peak_memory is not None:
            print(f"Peak allocated GPU memory: {peak_memory:.2f} GB")

    except Exception as error:
        print("\nREAL DATASEC TEST FAILED:")
        print(type(error).__name__, str(error))
        raise

    finally:
        del model, optimizer, inputs, targets
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
