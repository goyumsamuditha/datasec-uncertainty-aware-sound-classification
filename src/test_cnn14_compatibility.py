
from pathlib import Path
import sys
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent

PANN_SOURCE = PROJECT_ROOT / "src" / "panns_source" / "pytorch"
CHECKPOINT_PATH = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"

sys.path.insert(0, str(PANN_SOURCE))

import models


# 1. Environment
print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "GPU memory:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / (1024 ** 3),
            2
        ),
        "GB"
    )

if not CHECKPOINT_PATH.exists():
    raise FileNotFoundError(CHECKPOINT_PATH)


# 2. Create CNN14
print("\n" + "=" * 60)
print("CREATING CNN14")
print("=" * 60)

model = models.Cnn14(
    sample_rate=32000,
    window_size=1024,
    hop_size=320,
    mel_bins=64,
    fmin=50,
    fmax=14000,
    classes_num=527
)

print("CNN14 created.")


# 3. Load the correct PANNs checkpoint
print("\n" + "=" * 60)
print("LOADING PRETRAINED WEIGHTS")
print("=" * 60)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

state_dict = checkpoint["model"]

result = model.load_state_dict(
    state_dict,
    strict=True
)

print("Strict checkpoint loading successful.")
print("Missing keys:", result.missing_keys)
print("Unexpected keys:", result.unexpected_keys)


# 4. Forward pass
print("\n" + "=" * 60)
print("FORWARD PASS")
print("=" * 60)

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

model = model.to(device)
model.eval()

# Five seconds of test-only random mono audio at 32 kHz.
test_audio = torch.randn(
    1,
    32000 * 5,
    device=device
)

if device.type == "cuda":
    torch.cuda.reset_peak_memory_stats()

with torch.no_grad():
    output = model(test_audio)

print("Device:", device)
print("Output type:", type(output).__name__)

if isinstance(output, dict):
    for key, value in output.items():
        if torch.is_tensor(value):
            print(key, "shape:", tuple(value.shape))
else:
    print("Output:", output)

if device.type == "cuda":
    peak_memory = (
        torch.cuda.max_memory_allocated() / (1024 ** 3)
    )
    print("Peak GPU memory:", round(peak_memory, 3), "GB")

print("\nCNN14 COMPATIBILITY TEST PASSED.")
