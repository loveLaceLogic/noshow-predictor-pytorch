from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

from .dataset import load_and_prepare
from .model import NoShowMLP


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def main() -> None:
    set_seed(42)
    out_dir = Path("outputs")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_and_prepare("data/appointments.csv")  # uses No-show as label
    n_features = data.X_train.shape[1]

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print("Device:", device)
    print("Target column:", data.target_col)
    print("Train/Val/Test:", data.X_train.shape, data.X_val.shape, data.X_test.shape)

    X_train = torch.tensor(data.X_train, dtype=torch.float32)
    y_train = torch.tensor(data.y_train, dtype=torch.float32)
    X_val = torch.tensor(data.X_val, dtype=torch.float32)
    y_val = torch.tensor(data.y_val, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    model = NoShowMLP(n_features).to(device)
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 12
    train_losses, val_losses = [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()
            total += loss.item() * xb.size(0)

        tr_loss = total / len(train_loader.dataset)
        train_losses.append(tr_loss)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(device))
            v_loss = loss_fn(val_logits, y_val.to(device)).item()
        val_losses.append(v_loss)

        print(f"Epoch {epoch:02d}/{epochs} | train loss {tr_loss:.4f} | val loss {v_loss:.4f}")

    torch.save({"state_dict": model.state_dict(), "n_features": n_features}, out_dir / "model.pt")

    plt.figure()
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_curve.png")
    plt.close()

    with open(out_dir / "train_metrics.json", "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=2)

    print("Saved:", out_dir / "model.pt")
    print("Saved:", out_dir / "loss_curve.png")


if __name__ == "__main__":
    main()


