from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from .dataset import load_and_prepare
from .model import NoShowMLP


def plot_confusion(cm: np.ndarray, out_path: Path) -> None:
    plt.figure()
    plt.imshow(cm)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    out_dir = Path("outputs")
    ckpt_path = out_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError("Run training first: python -m src.train")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    data = load_and_prepare("data/appointments.csv")

    model = NoShowMLP(ckpt["n_features"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    X_test = torch.tensor(data.X_test, dtype=torch.float32)
    with torch.no_grad():
        logits = model(X_test).numpy()
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= 0.5).astype(int)

    y_true = data.y_test.astype(int)

    metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
    }

    cm = confusion_matrix(y_true, preds)
    plot_confusion(cm, out_dir / "confusion_matrix.png")

    with open(out_dir / "eval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Metrics:", metrics)
    print("Saved:", out_dir / "confusion_matrix.png")
    print("Saved:", out_dir / "eval_metrics.json")


if __name__ == "__main__":
    main()
