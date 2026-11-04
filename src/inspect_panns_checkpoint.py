
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CHECKPOINT = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"

print("Checkpoint:", CHECKPOINT)
print("File exists:", CHECKPOINT.exists())

if not CHECKPOINT.exists():
    raise FileNotFoundError(CHECKPOINT)

checkpoint = torch.load(
    CHECKPOINT,
    map_location="cpu",
    weights_only=False
)

print("\nCheckpoint type:", type(checkpoint))

if isinstance(checkpoint, dict):
    print("Top-level keys:", list(checkpoint.keys()))

    state_dict = checkpoint.get("model", checkpoint)

    print("\nState dictionary entries:", len(state_dict))
    print("First 15 keys:")

    for key in list(state_dict.keys())[:15]:
        print(key)

    print("\nTensor entries:", sum(
        torch.is_tensor(value)
        for value in state_dict.values()
    ))
