
from pathlib import Path
import sys
import torch
import torch.nn as nn

# --------------------------------------------------
# Paths
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent

PANN_SOURCE = PROJECT_ROOT / "src" / "panns_source" / "pytorch"
CHECKPOINT_PATH = PROJECT_ROOT / "src" / "Cnn14_mAP_0.431.pth"

sys.path.insert(0, str(PANN_SOURCE))

from models import Cnn14, init_layer

# --------------------------------------------------
# Configuration
# --------------------------------------------------
SAMPLE_RATE = 32000
WINDOW_SIZE = 1024
HOP_SIZE = 320
MEL_BINS = 64
FMIN = 50
FMAX = 14000
NUM_CLASSES = 22


# --------------------------------------------------
# Transfer CNN14 wrapper
# --------------------------------------------------
class TransferCNN14(nn.Module):

    def __init__(self):
        super().__init__()

        self.base = Cnn14(
            sample_rate=SAMPLE_RATE,
            window_size=WINDOW_SIZE,
            hop_size=HOP_SIZE,
            mel_bins=MEL_BINS,
            fmin=FMIN,
            fmax=FMAX,
            classes_num=527
        )

        self.fc_transfer = nn.Linear(2048, NUM_CLASSES)

        init_layer(self.fc_transfer)

    def forward(self, waveform):
        output = self.base(waveform)
        embedding = output["embedding"]

        output["clipwise_output"] = self.fc_transfer(embedding)

        return output


# --------------------------------------------------
# Inspect architecture
# --------------------------------------------------
print("=" * 60)
print("TRANSFER CNN14 ARCHITECTURE")
print("=" * 60)

model = TransferCNN14()

print("\nModel created successfully.")

# --------------------------------------------------
# Load official pretrained weights
# --------------------------------------------------
checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=False
)

model.base.load_state_dict(
    checkpoint["model"],
    strict=True
)

print("Pretrained CNN14 checkpoint loaded successfully.")
print("Strict loading: PASSED")

# --------------------------------------------------
# Inspect convolutional blocks
# --------------------------------------------------
print("\nCNN14 CONVOLUTIONAL BLOCKS")

for block_name in [
    "conv_block1",
    "conv_block2",
    "conv_block3",
    "conv_block4",
    "conv_block5",
    "conv_block6"
]:
    block = getattr(model.base, block_name, None)

    if block is not None:
        count = sum(p.numel() for p in block.parameters())
        print(f"{block_name}: {count:,} parameters")
    else:
        print(f"{block_name}: NOT FOUND")

# --------------------------------------------------
# Inspect classifier
# --------------------------------------------------
print("\nTRANSFER CLASSIFIER")
print(model.fc_transfer)

# --------------------------------------------------
# Parameter summary
# --------------------------------------------------
trainable = sum(
    p.numel() for p in model.parameters() if p.requires_grad
)

total = sum(p.numel() for p in model.parameters())

print("\nPARAMETER SUMMARY")
print(f"Trainable: {trainable:,}")
print(f"Total:     {total:,}")

# --------------------------------------------------
# Forward test
# --------------------------------------------------
print("\nRUNNING FORWARD TEST")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
model.eval()

test_waveform = torch.randn(1, SAMPLE_RATE * 5, device=device)

with torch.no_grad():
    output = model(test_waveform)

print("Device:", device)
print("Embedding:", tuple(output["embedding"].shape))
print("Classifier output:", tuple(output["clipwise_output"].shape))

print("\nTRANSFER CNN14 INSPECTION PASSED")
