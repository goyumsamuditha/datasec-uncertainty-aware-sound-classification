from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_cnn14 as training


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "models" / "cnn14_optimisation"
LOG_PATH = PROJECT_ROOT / "logs" / "cnn14_optimisation_runs.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one controlled CNN14 optimisation experiment."
    )
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--optimizer", choices=["adamw", "sgd"], required=True)
    parser.add_argument("--learning-rate", type=float, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--fine-tune-depth", type=int, choices=[1, 2, 3], required=True,
                        help="Number of final convolutional blocks to unfreeze: 1=block6, 2=blocks5-6, 3=blocks4-6.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--momentum", type=float, default=0.9)
    return parser.parse_args()


def validate_experiment_id(value: str) -> str:
    safe = "".join(ch for ch in value if ch.isalnum() or ch in "-_")
    if not safe or safe != value:
        raise ValueError("Experiment ID may contain only letters, numbers, hyphens and underscores.")
    return safe


def configure_fine_tuning(model: torch.nn.Module, depth: int) -> None:
    """Unfreeze the requested final convolutional blocks; fc1 remains trainable."""
    blocks = {
        1: ["conv_block6"],
        2: ["conv_block5", "conv_block6"],
        3: ["conv_block4", "conv_block5", "conv_block6"],
    }[depth]
    for block_name in blocks:
        block = getattr(model.backbone, block_name)
        for parameter in block.parameters():
            parameter.requires_grad = True

    # Verify the requested blocks are trainable and earlier blocks stay frozen.
    for block_number in range(1, 7):
        block = getattr(model.backbone, f"conv_block{block_number}")
        should_train = block_number >= (7 - depth)
        actual_train = any(p.requires_grad for p in block.parameters())
        if actual_train != should_train:
            raise RuntimeError(
                f"Fine-tuning configuration mismatch for conv_block{block_number}: "
                f"expected trainable={should_train}, observed={actual_train}."
            )


def append_run_log(row: dict) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame([row])
    if LOG_PATH.exists():
        old = pd.read_csv(LOG_PATH)
        if row["experiment_id"] in old.get("experiment_id", pd.Series(dtype=str)).astype(str).values:
            raise ValueError(f"Experiment ID already exists in {LOG_PATH}: {row['experiment_id']}")
        frame = pd.concat([old, frame], ignore_index=True)
    frame.to_csv(LOG_PATH, index=False)


def main() -> None:
    args = parse_args()
    experiment_id = validate_experiment_id(args.experiment_id)
    if args.learning_rate <= 0 or args.batch_size <= 0 or args.epochs <= 0:
        raise ValueError("Learning rate, batch size and epochs must be positive.")

    run_dir = OUTPUT_ROOT / experiment_id
    if run_dir.exists():
        raise FileExistsError(
            f"Output already exists: {run_dir}. Use a new experiment ID; existing results are protected."
        )
    run_dir.mkdir(parents=True, exist_ok=False)
    curve_path = run_dir / "learning_curve.csv"
    checkpoint_path = run_dir / "best_model.pt"
    config_path = run_dir / "configuration.json"

    config = {
        "experiment_id": experiment_id,
        "optimizer": args.optimizer,
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        "fine_tune_depth": args.fine_tune_depth,
        "fine_tune_definition": {
            "1": "conv_block6",
            "2": "conv_block5 and conv_block6",
            "3": "conv_block4, conv_block5 and conv_block6",
        }[str(args.fine_tune_depth)],
        "epochs": args.epochs,
        "seed": args.seed,
        "momentum": args.momentum if args.optimizer == "sgd" else None,
        "weight_decay": training.WEIGHT_DECAY,
        "subclass_loss_weight": training.SUBCLASS_LOSS_WEIGHT,
        "data_partitions": ["train", "validation"],
        "test_partition_accessed": False,
    }
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    status = "Failed"
    error_message = ""
    started = time.perf_counter()
    history: list[dict] = []
    best_score = -float("inf")
    best_epoch = 0
    best_accuracy = float("nan")
    best_loss = float("nan")
    epochs_completed = 0

    try:
        training.set_seed(args.seed)
        device = training.DEVICE
        print(f"Device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        main_map, sub_map = training.load_label_maps()
        manifest = training.load_manifest()
        train_df = manifest[manifest["partition"] == "train"].copy()
        validation_df = manifest[manifest["partition"] == "validation"].copy()
        if train_df.empty or validation_df.empty:
            raise ValueError("Training or validation partition is empty.")

        parents = training.build_parent_mapping(manifest, main_map, sub_map)
        subclass_mask = training.make_subclass_mask(parents, device)
        train_dataset = training.DataSECDataset(
            train_df, main_map, sub_map, training=True
        )
        validation_dataset = training.DataSECDataset(
            validation_df, main_map, sub_map, training=False
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=training.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=training.NUM_WORKERS,
            pin_memory=torch.cuda.is_available(),
        )

        model = training.DataSECCNN14().to(device)
        configure_fine_tuning(model, args.fine_tune_depth)

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        trainable_count = sum(p.numel() for p in trainable_parameters)
        if args.optimizer == "adamw":
            optimizer = torch.optim.AdamW(
                trainable_parameters,
                lr=args.learning_rate,
                weight_decay=training.WEIGHT_DECAY,
            )
        else:
            optimizer = torch.optim.SGD(
                trainable_parameters,
                lr=args.learning_rate,
                momentum=args.momentum,
                weight_decay=training.WEIGHT_DECAY,
            )

        weights = training.class_weights(train_df, main_map, device)
        main_criterion = nn.CrossEntropyLoss(weight=weights)
        subclass_criterion = nn.CrossEntropyLoss()

        print(f"Experiment: {experiment_id}")
        print(
            f"Optimizer={args.optimizer}; lr={args.learning_rate:g}; "
            f"batch={args.batch_size}; fine-tune depth={args.fine_tune_depth}; "
            f"epochs={args.epochs}; seed={args.seed}"
        )
        print(f"Trainable parameters: {trainable_count:,}")
        print(f"Training recordings: {len(train_dataset)}; validation recordings: {len(validation_dataset)}")
        print("Test partition reserved and not accessed.")

        for epoch in range(1, args.epochs + 1):
            epoch_start = time.perf_counter()
            train_metrics = training.run_training_epoch(
                model=model,
                dataloader=train_loader,
                optimizer=optimizer,
                main_criterion=main_criterion,
                subclass_criterion=subclass_criterion,
                subclass_mask=subclass_mask,
                device=device,
            )
            validation_metrics = training.evaluate(
                model=model,
                dataloader=validation_loader,
                subclass_mask=subclass_mask,
                device=device,
            )
            epoch_seconds = time.perf_counter() - epoch_start
            epochs_completed = epoch

            row = {
                "epoch": epoch,
                "train_loss": float(train_metrics["loss"]),
                "train_main_macro_f1": float(train_metrics["main_macro_f1"]),
                "train_main_accuracy": float(train_metrics["main_accuracy"]),
                "validation_loss": float(validation_metrics["loss"]),
                "validation_main_macro_f1": float(validation_metrics["main_macro_f1"]),
                "validation_main_accuracy": float(validation_metrics["main_accuracy"]),
                "epoch_seconds": epoch_seconds,
            }
            history.append(row)
            pd.DataFrame(history).to_csv(curve_path, index=False)

            score = float(validation_metrics["main_macro_f1"])
            print(
                f"Epoch {epoch:02d}/{args.epochs}: "
                f"train loss={row['train_loss']:.4f}, "
                f"val macro-F1={score:.4f}, "
                f"val accuracy={row['validation_main_accuracy']:.4f}, "
                f"{epoch_seconds:.1f}s"
            )

            if score > best_score:
                best_score = score
                best_epoch = epoch
                best_accuracy = float(validation_metrics["main_accuracy"])
                best_loss = float(validation_metrics["loss"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "best_validation_main_macro_f1": best_score,
                        "main_mapping": main_map,
                        "subclass_mapping": sub_map,
                        "subclass_parent_mapping": parents,
                        "config": config,
                    },
                    checkpoint_path,
                )

        status = "Completed"
        print("\nExperiment completed.")
        print(f"Best epoch: {best_epoch}")
        print(f"Best validation macro-F1: {best_score:.4f}")
        print(f"Best validation accuracy: {best_accuracy:.4f}")
        print(f"Best checkpoint: {checkpoint_path}")
        print(f"Learning curve: {curve_path}")

    except Exception as exc:
        error_message = f"{type(exc).__name__}: {exc}"
        print(f"\nEXPERIMENT FAILED: {error_message}")
        raise

    finally:
        elapsed = time.perf_counter() - started
        run_row = {
            "experiment_id": experiment_id,
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "optimizer": args.optimizer,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "fine_tune_depth": args.fine_tune_depth,
            "epochs_requested": args.epochs,
            "epochs_completed": epochs_completed,
            "seed": args.seed,
            "best_epoch": best_epoch,
            "best_validation_macro_f1": best_score if np.isfinite(best_score) else np.nan,
            "best_validation_accuracy": best_accuracy,
            "best_validation_loss": best_loss,
            "training_time_seconds": elapsed,
            "trainable_parameters": trainable_count if "trainable_count" in locals() else np.nan,
            "device": str(training.DEVICE),
            "status": status,
            "error": error_message,
            "checkpoint_path": str(checkpoint_path) if checkpoint_path.exists() else "",
            "learning_curve_path": str(curve_path) if curve_path.exists() else "",
            "test_partition_accessed": False,
        }
        append_run_log(run_row)
        print(f"Run log updated: {LOG_PATH}")


if __name__ == "__main__":
    main()
