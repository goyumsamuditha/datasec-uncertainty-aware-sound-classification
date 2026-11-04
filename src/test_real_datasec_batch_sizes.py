
from pathlib import Path
import sys
import math
import gc

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
from scipy.signal import resample_poly

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
MANIFEST_PATH = PROJECT_ROOT / "logs" / "final_split_manifest.csv"
CHECKPOINT_PATH = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"
PANN_SOURCE = PROJECT_ROOT / "src" / "panns_source" / "pytorch"

sys.path.insert(0, str(PANN_SOURCE))
from models import Cnn14, init_layer

SAMPLE_RATE = 32000
CLIP_SECONDS = 10
NUM_CLASSES = 22
SEED = 42


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


def resolve_audio_path(relative_path):
    path = Path(str(relative_path))

    if path.is_absolute():
        return path

    candidates = [
        DATASET_ROOT / path,
        PROJECT_ROOT / path,
        PROJECT_ROOT / "data" / path
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(f"Audio file not found: {relative_path}")


def load_and_prepare_audio(path):
    waveform, original_sr = sf.read(
        str(path),
        dtype="float32",
        always_2d=True
    )

    waveform = waveform.mean(axis=1)

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise ValueError(f"Invalid audio: {path}")

    if original_sr != SAMPLE_RATE:
        divisor = math.gcd(original_sr, SAMPLE_RATE)
        waveform = resample_poly(
            waveform,
            SAMPLE_RATE // divisor,
            original_sr // divisor
        ).astype(np.float32)

    target_length = SAMPLE_RATE * CLIP_SECONDS

    if len(waveform) > target_length:
        # Deterministic centre crop for this feasibility test.
        start = (len(waveform) - target_length) // 2
        waveform = waveform[start:start + target_length]

    elif len(waveform) < target_length:
        waveform = np.pad(
            waveform,
            (0, target_length - len(waveform))
        )

    return waveform.astype(np.float32)


def create_model(device):
    model = TransferCNN14()

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location="cpu",
        weights_only=False
    )
    model.base.load_state_dict(checkpoint["model"], strict=True)
    del checkpoint

    # Freeze all pretrained layers.
    for parameter in model.base.parameters():
        parameter.requires_grad = False

    # Fine-tune only the final convolution block.
    for parameter in model.base.conv_block6.parameters():
        parameter.requires_grad = True

    # Train the new 22-class classifier.
    for parameter in model.fc_transfer.parameters():
        parameter.requires_grad = True

    model.to(device)
    return model


def run_batch_test(batch_size, records, label_to_index, device):
    print("\n" + "=" * 65)
    print(f"TESTING BATCH SIZE: {batch_size}")
    print("=" * 65)

    model = None
    optimizer = None

    try:
        selected = records[:batch_size]

        waveforms = []
        targets = []

        for _, row in selected.iterrows():
            audio_path = resolve_audio_path(row["relative_path"])
            waveform = load_and_prepare_audio(audio_path)

            waveforms.append(waveform)
            targets.append(label_to_index[str(row["main_class"])])

            print(
                f"{audio_path.name} | "
                f"{row['main_class']} | "
                f"{len(waveform) / SAMPLE_RATE:.1f}s"
            )

        inputs = torch.from_numpy(
            np.stack(waveforms)
        ).to(device)

        targets = torch.tensor(
            targets,
            dtype=torch.long,
            device=device
        )

        model = create_model(device)

        # Keep frozen blocks and their BatchNorm layers in eval mode.
        model.train()

        for block_name in [
            "conv_block1",
            "conv_block2",
            "conv_block3",
            "conv_block4",
            "conv_block5"
        ]:
            getattr(model.base, block_name).eval()

        criterion = nn.CrossEntropyLoss()

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=1e-4
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

        optimizer.zero_grad(set_to_none=True)

        # FP32 only for this test.
        with torch.autocast(device_type="cuda", enabled=False):
            output = model(inputs)
            logits = output["clipwise_output"]
            loss = criterion(logits, targets)

        if logits.shape != (batch_size, NUM_CLASSES):
            raise ValueError(f"Unexpected output shape: {logits.shape}")

        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits detected.")

        if not torch.isfinite(loss):
            raise ValueError("Non-finite loss detected.")

        loss.backward()

        bad_gradients = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and parameter.grad is not None
            and not torch.isfinite(parameter.grad).all()
        ]

        if bad_gradients:
            raise ValueError(f"Non-finite gradients: {bad_gradients}")

        optimizer.step()

        bad_parameters = [
            name for name, parameter in model.named_parameters()
            if parameter.requires_grad
            and not torch.isfinite(parameter).all()
        ]

        if bad_parameters:
            raise ValueError(f"Non-finite parameters: {bad_parameters}")

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_memory = (
                torch.cuda.max_memory_allocated() / (1024 ** 3)
            )
        else:
            peak_memory = None

        print("\nRESULT: PASSED")
        print("Batch size:", batch_size)
        print("Output shape:", tuple(logits.shape))
        print("Loss:", loss.item())
        print("Finite logits, loss, gradients and parameters: YES")

        if peak_memory is not None:
            print(f"Peak allocated GPU memory: {peak_memory:.2f} GB")

        return True

    except torch.cuda.OutOfMemoryError:
        print(f"\nRESULT: OUT OF MEMORY at batch size {batch_size}")
        return False

    except Exception as error:
        print(f"\nRESULT: FAILED at batch size {batch_size}")
        print(type(error).__name__, str(error))
        return False

    finally:
        if optimizer is not None:
            del optimizer
        if model is not None:
            del model

        if "inputs" in locals():
            del inputs
        if "targets" in locals():
            del targets

        gc.collect()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(MANIFEST_PATH)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device("cuda")

    df = pd.read_csv(MANIFEST_PATH)

    train_df = df[
        df["partition"].astype(str).str.lower() == "train"
    ].copy()

    labels = sorted(train_df["main_class"].dropna().astype(str).unique())

    if len(labels) != NUM_CLASSES:
        raise ValueError(f"Expected 22 classes, found {len(labels)}")

    label_to_index = {
        label: index for index, label in enumerate(labels)
    }

    # Four distinct training recordings, reproducibly selected.
    records = train_df.sample(
        n=4,
        random_state=SEED
    ).reset_index(drop=True)

    print("Device:", torch.cuda.get_device_name(0))
    print("Training recordings available:", len(train_df))
    print("Selected recordings:", len(records))
    print("Classes:", len(labels))
    print("Precision: FP32")
    print("Input duration:", CLIP_SECONDS, "seconds")

    results = {}

    for batch_size in [2, 4]:
        passed = run_batch_test(
            batch_size,
            records,
            label_to_index,
            device
        )
        results[batch_size] = passed

        if not passed and batch_size == 2:
            print("\nBatch size 2 failed. Skipping batch size 4.")
            break

    print("\n" + "=" * 65)
    print("BATCH-SIZE FEASIBILITY SUMMARY")
    print("=" * 65)

    for batch_size, passed in results.items():
        print(f"Batch size {batch_size}: {'PASSED' if passed else 'FAILED'}")


if __name__ == "__main__":
    main()
