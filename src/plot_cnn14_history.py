from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = PROJECT_ROOT / "models" / "cnn14_initial" / "training_history.csv"
OUTPUT_DIR = PROJECT_ROOT / "report"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(HISTORY_PATH)

# 1. Training and validation loss
plt.figure(figsize=(10, 6))
plt.plot(df["epoch"], df["train_loss"], marker="o", label="Training loss")
plt.plot(df["epoch"], df["validation_loss"], marker="o", label="Validation loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("CNN14 Training and Validation Loss")
plt.xticks(df["epoch"])
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn14_loss_curves.png", dpi=300)
plt.close()

# 2. Main-class macro-F1
plt.figure(figsize=(10, 6))
plt.plot(
    df["epoch"],
    df["train_main_macro_f1"],
    marker="o",
    label="Training main macro-F1"
)
plt.plot(
    df["epoch"],
    df["validation_main_macro_f1"],
    marker="o",
    label="Validation main macro-F1"
)

best_row = df.loc[df["validation_main_macro_f1"].idxmax()]
plt.scatter(
    best_row["epoch"],
    best_row["validation_main_macro_f1"],
    marker="*",
    s=180,
    label=f"Best validation: epoch {int(best_row['epoch'])}"
)

plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("CNN14 Main-Class Macro-F1")
plt.xticks(df["epoch"])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn14_main_macro_f1.png", dpi=300)
plt.close()

# 3. Subclass macro-F1
plt.figure(figsize=(10, 6))
plt.plot(
    df["epoch"],
    df["train_subclass_macro_f1"],
    marker="o",
    label="Training subclass macro-F1"
)
plt.plot(
    df["epoch"],
    df["validation_subclass_conditional_macro_f1"],
    marker="o",
    label="Validation conditional subclass F1"
)
plt.plot(
    df["epoch"],
    df["validation_subclass_end_to_end_macro_f1"],
    marker="o",
    label="Validation end-to-end subclass F1"
)
plt.xlabel("Epoch")
plt.ylabel("Macro-F1")
plt.title("CNN14 Subclass Performance")
plt.xticks(df["epoch"])
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "cnn14_subclass_f1.png", dpi=300)
plt.close()

print("CNN14 learning curves generated successfully.")
print(f"Output folder: {OUTPUT_DIR}")
print("Files:")
print("- cnn14_loss_curves.png")
print("- cnn14_main_macro_f1.png")
print("- cnn14_subclass_f1.png")
print(
    f"Best validation main macro-F1: "
    f"{best_row['validation_main_macro_f1']:.4f} "
    f"at epoch {int(best_row['epoch'])}"
)