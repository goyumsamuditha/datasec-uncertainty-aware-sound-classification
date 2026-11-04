
from pathlib import Path
import inspect
import sys
import torch

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")
PANN_SOURCE = PROJECT_DIR / "src" / "panns_source"
CHECKPOINT = PROJECT_DIR / "src" / "Cnn14_mAP_0.431.pth"

sys.path.insert(0, str(PANN_SOURCE / "pytorch"))

from models import Cnn14


def main():
    print("CNN14 TRAINING SETUP CHECK")
    print("=" * 55)

    print("PyTorch:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "GPU memory:",
            round(
                torch.cuda.get_device_properties(0).total_memory
                / 1024**3,
                2
            ),
            "GB"
        )

    print("\nCheckpoint exists:", CHECKPOINT.exists())

    if not CHECKPOINT.exists():
        raise FileNotFoundError(CHECKPOINT)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location="cpu",
        weights_only=False
    )

    print("Checkpoint type:", type(checkpoint).__name__)

    if isinstance(checkpoint, dict):
        print("Checkpoint keys:", list(checkpoint.keys())[:20])

        state_dict = checkpoint.get("model", checkpoint)

        if isinstance(state_dict, dict):
            print("Number of parameter entries:", len(state_dict))
            print("First 10 parameter names:")
            for key in list(state_dict.keys())[:10]:
                print(" ", key)

    print("\nCNN14 constructor:")
    print(inspect.signature(Cnn14))

    model = Cnn14(
        sample_rate=32000,
        window_size=1024,
        hop_size=320,
        mel_bins=64,
        fmin=50,
        fmax=14000,
        classes_num=527
    )

    print("\nCNN14 model created successfully.")
    print("Total parameters:", sum(p.numel() for p in model.parameters()))

    print("\nCNN14 forward method:")
    print(inspect.getsource(model.forward))

    print("\nSETUP CHECK COMPLETED")


if __name__ == "__main__":
    main()
