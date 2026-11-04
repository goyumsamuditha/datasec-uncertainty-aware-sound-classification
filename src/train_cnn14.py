from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import Dataset, DataLoader

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = Path(r"G:\My Drive\MSC\AI\DATASEC\DATASEC")
PANN_SOURCE = ROOT / "src" / "panns_source" / "pytorch"
PRETRAINED_PATH = ROOT / "src" / "Cnn14_mAP_0.431.pth"
MANIFEST_PATH = ROOT / "logs" / "final_split_manifest_with_subclasses.csv"
LABELS_PATH = ROOT / "logs" / "hierarchical_label_mapping.json"
OUTPUT_DIR = ROOT / "models" / "cnn14_initial"

SAMPLE_RATE = 32_000
CLIP_SECONDS = 10
NUM_SAMPLES = SAMPLE_RATE * CLIP_SECONDS
WINDOW_SIZE, HOP_SIZE, MEL_BINS = 1024, 320, 64
FMIN, FMAX = 50, 14_000
NUM_MAIN = 22
NUM_SUB = 28
BATCH_SIZE = 4
EPOCHS = 20
PATIENCE = 5
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
SUBCLASS_LOSS_WEIGHT = 1.0
SEED = 42
NUM_WORKERS = 0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _find_mapping(data, candidate_keys, mapping_name):
    for key in candidate_keys:
        if key in data and isinstance(data[key], dict):
            return {
                str(name): int(index)
                for name, index in data[key].items()
            }

    raise ValueError(
        f"Cannot find {mapping_name}. "
        f"Available keys: {list(data.keys())}"
    )


def load_label_maps():
    import json

    mapping_path = (
        PROJECT_ROOT
        / "logs"
        / "hierarchical_label_mapping.json"
    )

    if not mapping_path.exists():
        raise FileNotFoundError(
            f"Label mapping file not found: {mapping_path}"
        )

    # Load the JSON dictionary first.
    with open(mapping_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    # Find main-class mapping.
    main_raw = _find_mapping(
        data,
        ["main_to_id", "main_classes", "main_class_to_id"],
        "main-class mapping",
    )

    # Find subclass mapping.
    sub_raw = _find_mapping(
        data,
        ["subclass_to_id", "subclasses", "subclass_to_id"],
        "subclass mapping",
    )

    # Validate mapping sizes.
    if len(main_raw) != 22:
        raise ValueError(
            f"Expected 22 main classes, found {len(main_raw)}"
        )

    if len(sub_raw) != 28:
        raise ValueError(
            f"Expected 28 subclasses, found {len(sub_raw)}"
        )

    print("Label mappings loaded successfully.")
    print(f"Main classes: {len(main_raw)}")
    print(f"Subclasses: {len(sub_raw)}")

    return main_raw, sub_raw


def load_manifest() -> pd.DataFrame:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    df = pd.read_csv(MANIFEST_PATH)
    required = {"relative_path", "main_class", "partition", "subclass_folder"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest missing columns: {sorted(missing)}")
    df["main_class"] = df["main_class"].astype(str).str.strip()
    df["subclass_folder"] = df["subclass_folder"].fillna("").astype(str).str.strip()
    df["partition"] = df["partition"].astype(str).str.lower().str.strip()
    if df["relative_path"].duplicated().any():
        raise ValueError("Duplicate relative_path values in manifest.")
    if not set(df["partition"].unique()).issubset({"train", "validation", "test"}):
        raise ValueError("Manifest contains unexpected partition values.")
    df["audio_path"] = df["relative_path"].map(lambda p: DATASET_ROOT / str(p))
    missing_files = [str(p) for p in df["audio_path"] if not Path(p).is_file()]
    if missing_files:
        raise FileNotFoundError(f"{len(missing_files)} missing audio files; first: {missing_files[0]}")
    return df


def build_parent_mapping(df: pd.DataFrame, main_map: dict, sub_map: dict) -> Dict[int, int]:
    """Map each subclass ID to exactly one main-class ID."""
    parents: Dict[int, int] = {}
    labelled = df[df["subclass_folder"].ne("")]
    for row in labelled.itertuples(index=False):
        main_name = str(row.main_class).strip()
        sub_name = str(row.subclass_folder).strip()
        if main_name not in main_map:
            raise ValueError(f"Unknown main class in manifest: {main_name}")
        if sub_name not in sub_map:
            raise ValueError(f"Unknown subclass in manifest: {sub_name}")
        sid, mid = sub_map[sub_name], main_map[main_name]
        if sid in parents and parents[sid] != mid:
            raise ValueError(f"Subclass {sub_name!r} has multiple parent classes.")
        parents[sid] = mid
    expected = set(range(NUM_SUB))
    if set(parents) != expected:
        raise ValueError(f"Missing subclass parent IDs: {sorted(expected - set(parents))}")
    return parents


def make_subclass_mask(parents: Dict[int, int], device: torch.device) -> torch.Tensor:
    mask = torch.zeros(NUM_MAIN, NUM_SUB, dtype=torch.bool, device=device)
    for subclass_id, parent_id in parents.items():
        mask[parent_id, subclass_id] = True
    if not torch.all(mask.sum(dim=0) == 1):
        raise ValueError("Each subclass must map to exactly one parent.")
    return mask


def mask_subclass_logits(logits: torch.Tensor, parent_ids: torch.Tensor,
                         subclass_mask: torch.Tensor) -> torch.Tensor:
    """Set logits for subclasses outside each selected parent to -1e9."""
    allowed = subclass_mask[parent_ids]
    return logits.masked_fill(~allowed, -1e9)


class DataSECDataset(Dataset):
    def __init__(self, df: pd.DataFrame, main_map: dict, sub_map: dict, training: bool):
        self.df = df.reset_index(drop=True)
        self.main_map = main_map
        self.sub_map = sub_map
        self.training = training
        self.resamplers = {}

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: Path) -> torch.Tensor:
        audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # mono
        waveform = torch.from_numpy(audio.copy()).float().unsqueeze(0)
        if sr != SAMPLE_RATE:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(sr, SAMPLE_RATE)
            waveform = self.resamplers[sr](waveform)
        length = waveform.shape[-1]
        if length > NUM_SAMPLES:
            start = random.randint(0, length - NUM_SAMPLES) if self.training else (length - NUM_SAMPLES) // 2
            waveform = waveform[:, start:start + NUM_SAMPLES]  # crop time axis
        elif length < NUM_SAMPLES:
            waveform = F.pad(waveform, (0, NUM_SAMPLES - length))
        if waveform.shape != (1, NUM_SAMPLES):
            raise RuntimeError(f"Unexpected waveform shape {tuple(waveform.shape)} for {path}")
        return waveform.squeeze(0)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        waveform = self._load_audio(Path(row.audio_path))
        main_id = self.main_map[str(row.main_class).strip()]
        subclass_name = str(row.subclass_folder).strip()
        subclass_id = self.sub_map[subclass_name] if subclass_name else -1
        return waveform, torch.tensor(main_id, dtype=torch.long), torch.tensor(subclass_id, dtype=torch.long)


# PANNs official implementation
if not PANN_SOURCE.is_dir():
    raise FileNotFoundError(f"PANNs source directory not found: {PANN_SOURCE}")
sys.path.insert(0, str(PANN_SOURCE))
from models import Cnn14  # noqa: E402


class DataSECCNN14(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Cnn14(
            sample_rate=SAMPLE_RATE, window_size=WINDOW_SIZE, hop_size=HOP_SIZE,
            mel_bins=MEL_BINS, fmin=FMIN, fmax=FMAX, classes_num=527,
        )
        if not PRETRAINED_PATH.is_file():
            raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAINED_PATH}")
        checkpoint = torch.load(PRETRAINED_PATH, map_location="cpu", weights_only=False)
        if isinstance(checkpoint, dict) and "model" in checkpoint:
            checkpoint = checkpoint["model"]
        self.backbone.load_state_dict(checkpoint, strict=True)
        for p in self.backbone.parameters():
            p.requires_grad = False
        for p in self.backbone.conv_block6.parameters():
            p.requires_grad = True
        for p in self.backbone.fc1.parameters():
            p.requires_grad = True
        self.main_head = nn.Linear(2048, NUM_MAIN)
        self.subclass_head = nn.Linear(2048, NUM_SUB)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            # Frozen convolutional blocks remain in eval mode (BatchNorm stability).
            for name in ("conv_block1", "conv_block2", "conv_block3", "conv_block4", "conv_block5", "fc_audioset"):
                getattr(self.backbone, name).eval()
        return self

    def forward(self, waveforms):
        features = self.backbone(waveforms)["embedding"]
        return self.main_head(features), self.subclass_head(features)


def class_weights(train_df: pd.DataFrame, main_map: dict, device) -> torch.Tensor:
    counts = train_df["main_class"].value_counts()
    total = len(train_df)
    weights = np.zeros(NUM_MAIN, dtype=np.float32)
    for name, idx in main_map.items():
        count = int(counts.get(name, 0))
        if count == 0:
            raise ValueError(f"No training examples for main class {name!r}.")
        weights[idx] = total / (NUM_MAIN * count)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32, device=device)


def run_training_epoch(model, dataloader, optimizer, main_criterion,
                       subclass_criterion, subclass_mask, device):
    """Train one epoch. Returns a metrics dictionary."""
    model.train()
    sums = {"loss": 0.0, "main_loss": 0.0, "subclass_loss": 0.0}
    main_true, main_pred = [], []
    sub_true, sub_pred = [], []

    for waveforms, main_labels, subclass_labels in dataloader:
        waveforms = waveforms.to(device, non_blocking=True)
        main_labels = main_labels.to(device, non_blocking=True)
        subclass_labels = subclass_labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        main_logits, subclass_logits = model(waveforms)
        loss_main = main_criterion(main_logits, main_labels)
        labelled = subclass_labels >= 0
        if labelled.any():
            true_main = main_labels[labelled]
            true_sub = subclass_labels[labelled]
            if not torch.all(subclass_mask[true_main, true_sub]):
                raise ValueError("A true subclass is invalid for its true parent class.")
            valid_logits = mask_subclass_logits(subclass_logits[labelled], true_main, subclass_mask)
            loss_sub = subclass_criterion(valid_logits, true_sub)
            sub_pred.extend(valid_logits.argmax(1).detach().cpu().tolist())
            sub_true.extend(true_sub.detach().cpu().tolist())
        else:
            loss_sub = main_logits.new_zeros(())
        loss = loss_main + SUBCLASS_LOSS_WEIGHT * loss_sub
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite training loss detected.")
        loss.backward()
        optimizer.step()
        sums["loss"] += float(loss.item())
        sums["main_loss"] += float(loss_main.item())
        sums["subclass_loss"] += float(loss_sub.item())
        main_true.extend(main_labels.detach().cpu().tolist())
        main_pred.extend(main_logits.argmax(1).detach().cpu().tolist())

    n = max(len(dataloader), 1)
    return {
        "loss": sums["loss"] / n,
        "main_loss": sums["main_loss"] / n,
        "subclass_loss": sums["subclass_loss"] / n,
        "main_macro_f1": f1_score(main_true, main_pred, average="macro", zero_division=0),
        "main_accuracy": accuracy_score(main_true, main_pred),
        "subclass_macro_f1": f1_score(sub_true, sub_pred, labels=list(range(NUM_SUB)), average="macro", zero_division=0) if sub_true else float("nan"),
    }


@torch.no_grad()
def evaluate(model, dataloader, subclass_mask, device):
    """Evaluate validation only; report conditional and end-to-end subclass scores."""
    model.eval()
    main_true, main_pred = [], []
    conditional_true, conditional_pred = [], []
    e2e_true, e2e_pred = [], []
    total_loss = 0.0
    batches = 0
    main_criterion = nn.CrossEntropyLoss()
    subclass_criterion = nn.CrossEntropyLoss()

    for waveforms, main_labels, subclass_labels in dataloader:
        waveforms = waveforms.to(device, non_blocking=True)
        main_labels = main_labels.to(device, non_blocking=True)
        subclass_labels = subclass_labels.to(device, non_blocking=True)
        main_logits, subclass_logits = model(waveforms)
        loss_main = main_criterion(main_logits, main_labels)
        labelled = subclass_labels >= 0
        if labelled.any():
            true_main = main_labels[labelled]
            true_sub = subclass_labels[labelled]
            true_parent_logits = mask_subclass_logits(subclass_logits[labelled], true_main, subclass_mask)
            loss_sub = subclass_criterion(true_parent_logits, true_sub)
            conditional_pred.extend(true_parent_logits.argmax(1).cpu().tolist())
            conditional_true.extend(true_sub.cpu().tolist())

            predicted_main = main_logits[labelled].argmax(1)
            predicted_parent_logits = mask_subclass_logits(subclass_logits[labelled], predicted_main, subclass_mask)
            has_subclasses = subclass_mask[predicted_main].any(dim=1)
            predicted_sub = torch.full_like(true_sub, -1)
            if has_subclasses.any():
                predicted_sub[has_subclasses] = predicted_parent_logits[has_subclasses].argmax(1)
            e2e_pred.extend(predicted_sub.cpu().tolist())
            e2e_true.extend(true_sub.cpu().tolist())
        else:
            loss_sub = main_logits.new_zeros(())
        total_loss += float((loss_main + SUBCLASS_LOSS_WEIGHT * loss_sub).item())
        batches += 1
        main_true.extend(main_labels.cpu().tolist())
        main_pred.extend(main_logits.argmax(1).cpu().tolist())

    return {
        "loss": total_loss / max(batches, 1),
        "main_macro_f1": f1_score(main_true, main_pred, average="macro", zero_division=0),
        "main_accuracy": accuracy_score(main_true, main_pred),
        "conditional_subclass_macro_f1": f1_score(conditional_true, conditional_pred, labels=list(range(NUM_SUB)), average="macro", zero_division=0) if conditional_true else float("nan"),
        "end_to_end_subclass_macro_f1": f1_score(e2e_true, e2e_pred, labels=list(range(NUM_SUB)), average="macro", zero_division=0) if e2e_true else float("nan"),
        "main_true": main_true,
        "main_pred": main_pred,
        "conditional_true": conditional_true,
        "conditional_pred": conditional_pred,
    }


def save_reports(metrics: dict, main_map: dict, sub_map: dict, out_dir: Path):
    main_names = [k for k, _ in sorted(main_map.items(), key=lambda x: x[1])]
    report = classification_report(
        metrics["main_true"], metrics["main_pred"], labels=list(range(NUM_MAIN)),
        target_names=main_names, output_dict=True, zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(out_dir / "validation_main_class_report.csv")
    cm = confusion_matrix(metrics["main_true"], metrics["main_pred"], labels=list(range(NUM_MAIN)))
    pd.DataFrame(cm, index=main_names, columns=main_names).to_csv(out_dir / "validation_main_confusion_matrix.csv")
    if metrics["conditional_true"]:
        sub_names = [k for k, _ in sorted(sub_map.items(), key=lambda x: x[1])]
        sub_report = classification_report(
            metrics["conditional_true"], metrics["conditional_pred"], labels=list(range(NUM_SUB)),
            target_names=sub_names, output_dict=True, zero_division=0,
        )
        pd.DataFrame(sub_report).T.to_csv(out_dir / "validation_subclass_conditional_report.csv")


def main():
    set_seed()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    main_map, sub_map = load_label_maps()
    df = load_manifest()
    unknown = set(df.main_class.unique()) - set(main_map)
    if unknown:
        raise ValueError(f"Unknown main classes: {sorted(unknown)}")
    parents = build_parent_mapping(df, main_map, sub_map)
    subclass_mask = make_subclass_mask(parents, DEVICE)
    print(f"Training recordings: {(df.partition == 'train').sum()}")
    print(f"Validation recordings: {(df.partition == 'validation').sum()}")
    print(f"Training recordings with subclasses: {((df.partition == 'train') & df.subclass_folder.ne('')).sum()}")
    print(f"Subclass-parent relationships: {len(parents)}")
    print(f"Mask shape: {tuple(subclass_mask.shape)}")
    print("Test partition reserved; it will not be evaluated here.")

    train_df = df[df.partition == "train"].copy()
    val_df = df[df.partition == "validation"].copy()
    if train_df.empty or val_df.empty:
        raise ValueError("Training or validation partition is empty.")

    train_ds = DataSECDataset(train_df, main_map, sub_map, training=True)
    val_ds = DataSECDataset(val_df, main_map, sub_map, training=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                            num_workers=NUM_WORKERS, pin_memory=torch.cuda.is_available())

    model = DataSECCNN14().to(DEVICE)
    weights = class_weights(train_df, main_map, DEVICE)
    main_criterion = nn.CrossEntropyLoss(weight=weights)
    subclass_criterion = nn.CrossEntropyLoss()
    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = []
    best_score = -float("inf")
    best_epoch = 0
    no_improve = 0
    checkpoint_path = OUTPUT_DIR / "best_model.pt"

    for epoch in range(1, EPOCHS + 1):
        print(f"\nEpoch {epoch}/{EPOCHS}")
        train_metrics = run_training_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            main_criterion=main_criterion,
            subclass_criterion=subclass_criterion,
            subclass_mask=subclass_mask,  # FIX: explicitly pass the mask
            device=DEVICE,
        )
        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            subclass_mask=subclass_mask,  # FIX: validation receives same mask
            device=DEVICE,
        )
        print(f"Train loss: {train_metrics['loss']:.4f} | main macro-F1: {train_metrics['main_macro_f1']:.4f} | subclass macro-F1: {train_metrics['subclass_macro_f1']:.4f}")
        print(f"Validation loss: {val_metrics['loss']:.4f} | main macro-F1: {val_metrics['main_macro_f1']:.4f} | main accuracy: {val_metrics['main_accuracy']:.4f}")
        print(f"Validation subclass F1 (conditional): {val_metrics['conditional_subclass_macro_f1']:.4f}")
        print(f"Validation subclass F1 (end-to-end): {val_metrics['end_to_end_subclass_macro_f1']:.4f}")

        record = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_main_loss": train_metrics["main_loss"],
            "train_subclass_loss": train_metrics["subclass_loss"],
            "train_main_macro_f1": train_metrics["main_macro_f1"],
            "train_main_accuracy": train_metrics["main_accuracy"],
            "train_subclass_macro_f1": train_metrics["subclass_macro_f1"],
            "validation_loss": val_metrics["loss"],
            "validation_main_macro_f1": val_metrics["main_macro_f1"],
            "validation_main_accuracy": val_metrics["main_accuracy"],
            "validation_subclass_conditional_macro_f1": val_metrics["conditional_subclass_macro_f1"],
            "validation_subclass_end_to_end_macro_f1": val_metrics["end_to_end_subclass_macro_f1"],
        }
        history.append(record)
        pd.DataFrame(history).to_csv(OUTPUT_DIR / "training_history.csv", index=False)

        score = val_metrics["main_macro_f1"]
        if score > best_score:
            best_score = score
            best_epoch = epoch
            no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_validation_main_macro_f1": best_score,
                "main_mapping": main_map,
                "subclass_mapping": sub_map,
                "subclass_parent_mapping": parents,
                "config": {
                    "sample_rate": SAMPLE_RATE,
                    "clip_seconds": CLIP_SECONDS,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE,
                    "weight_decay": WEIGHT_DECAY,
                    "subclass_loss_weight": SUBCLASS_LOSS_WEIGHT,
                    "seed": SEED,
                },
            }, checkpoint_path)
            save_reports(val_metrics, main_map, sub_map, OUTPUT_DIR)
            print(f"Saved new best checkpoint: {checkpoint_path}")
        else:
            no_improve += 1
            print(f"No validation improvement: {no_improve}/{PATIENCE}")

        if no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

    print("\nTRAINING FINISHED")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation main macro-F1: {best_score:.4f}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"History: {OUTPUT_DIR / 'training_history.csv'}")
    print("Test partition remains untouched.")


if __name__ == "__main__":
    main()
