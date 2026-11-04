
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "src" / "630k-audioset-best.pt"

print("Loading checkpoint:", CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint Python type:")
print(type(checkpoint))

if isinstance(checkpoint, dict):
    print("\nTop-level keys:")
    for key, value in checkpoint.items():
        print(
            f"{key}: type={type(value).__name__}",
            f"shape={tuple(value.shape) if torch.is_tensor(value) else ''}"
        )

    print("\nChecking possible state dictionaries:")

    for key, value in checkpoint.items():
        if isinstance(value, dict):
            tensor_count = sum(
                torch.is_tensor(v) for v in value.values()
            )

            print(f"\nCandidate: {key}")
            print("Entries:", len(value))
            print("Tensor entries:", tensor_count)
            print("First keys:", list(value.keys())[:10])

elif isinstance(checkpoint, (list, tuple)):
    print("Number of elements:", len(checkpoint))
