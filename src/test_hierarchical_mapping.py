
from pathlib import Path
import json
import sys

import pandas as pd
import torch

PROJECT_DIR = Path(r"G:\My Drive\MSC\AI\Assignment")

sys.path.insert(0, str(PROJECT_DIR / "src"))

from train_cnn14 import (
    build_subclass_parent_mapping,
    create_subclass_mask,
    mask_subclass_logits,
)

manifest_path = (
    PROJECT_DIR / "logs"
    / "final_split_manifest_with_subclasses.csv"
)

label_path = (
    PROJECT_DIR / "logs"
    / "hierarchical_label_mapping.json"
)

df = pd.read_csv(manifest_path)

with open(label_path, encoding="utf-8") as f:
    mapping = json.load(f)

main_to_id = mapping["main_to_id"]
subclass_to_id = mapping["subclass_to_id"]

df["main_label"] = df["main_class"].map(main_to_id)

df["subclass_label"] = (
    df["subclass_folder"]
    .fillna("")
    .map(lambda x: subclass_to_id.get(x, -1))
)

parent_mapping = build_subclass_parent_mapping(
    df,
    len(subclass_to_id)
)

mask = create_subclass_mask(
    parent_mapping,
    len(main_to_id),
    len(subclass_to_id),
    torch.device("cpu")
)

assert mask.shape == (22, 28)
assert len(parent_mapping) == 28

# Every subclass must have exactly one parent.
assert torch.all(mask.sum(dim=0) == 1)

# Check that every known subclass belongs to its true main class.
labelled = df[df["subclass_label"] >= 0]

for row in labelled.itertuples():
    assert mask[
        int(row.main_label),
        int(row.subclass_label)
    ].item()

# Test that masking prevents invalid subclass predictions.
test_logits = torch.zeros(2, 28)

test_main_labels = torch.tensor([
    main_to_id["Propeller aircrafts"],
    main_to_id["Vehicle pass-by"],
])

masked_logits, valid_mask = mask_subclass_logits(
    test_logits,
    test_main_labels,
    mask
)

for i in range(2):
    assert torch.all(masked_logits[i, ~valid_mask[i]] == -1e9)

    # Argmax must always select an allowed subclass.
    prediction = masked_logits[i].argmax().item()
    assert valid_mask[i, prediction].item()

print("HIERARCHICAL MAPPING TEST PASSED")
print("Main classes:", len(main_to_id))
print("Subclasses:", len(subclass_to_id))
print("Subclass-parent relationships:", len(parent_mapping))
print("Mask shape:", tuple(mask.shape))
print("Recordings with subclass labels:", len(labelled))
print("Invalid subclass predictions prevented.")
