
from pathlib import Path
import sys
import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PANN_SOURCE = PROJECT_ROOT / "src" / "panns_source" / "pytorch"
CHECKPOINT_PATH = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"

sys.path.insert(0, str(PANN_SOURCE))

from models import Cnn14, init_layer

SAMPLE_RATE = 32000
NUM_CLASSES = 22


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


def main():
    print("=" * 60)
    print("PARTIAL FINE-TUNING AND GPU MEMORY TEST")
    print("=" * 60)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(CHECKPOINT_PATH)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

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

    # Freeze the complete pretrained CNN14 base.
    for parameter in model.base.parameters():
        parameter.requires_grad = False

    # Unfreeze only the final convolutional block.
    for parameter in model.base.conv_block6.parameters():
        parameter.requires_grad = True

    # The new 22-class classifier remains trainable.
    for parameter in model.fc_transfer.parameters():
        parameter.requires_grad = True

    model.to(device)

    print("\nTrainable parameter groups:")

    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            print(name)

    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total = sum(p.numel() for p in model.parameters())

    print("\nParameter summary")
    print(f"Trainable: {trainable:,}")
    print(f"Frozen:    {total - trainable:,}")
    print(f"Total:     {total:,}")

    if device.type != "cuda":
        print("\nCUDA unavailable. GPU memory test skipped.")
        return

    # Training-mode forward and backward test.
    model.train()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4
    )

    waveform = torch.randn(
        1, SAMPLE_RATE * 5, device=device
    )

    target = torch.tensor([0], dtype=torch.long, device=device)

    criterion = nn.CrossEntropyLoss()

    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()

    try:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16
        ):
            output = model(waveform)
            loss = criterion(output["clipwise_output"], target)

        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()

        peak_memory = (
            torch.cuda.max_memory_allocated() / (1024 ** 3)
        )

        print("\nTraining-mode test PASSED")
        print("Output shape:", tuple(output["clipwise_output"].shape))
        print("Test loss:", float(loss.detach()))
        print(f"Peak allocated GPU memory: {peak_memory:.2f} GB")
        print("Batch size: 1")
        print("Test audio duration: 5 seconds")

    except torch.cuda.OutOfMemoryError:
        print("\nGPU OUT OF MEMORY")
        print("Do not start full training with this configuration.")
        raise

    finally:
        del model, optimizer, waveform
        torch.cuda.empty_cache()

    print("\nPARTIAL FINE-TUNING TEST COMPLETED")


if __name__ == "__main__":
    main()
