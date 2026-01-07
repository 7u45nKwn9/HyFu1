import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from data_loader.bearing_dataset import FusionDataset
from models.vanilla_mamba_T import VMamba


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")

    # ---------------- Paths ----------------
    train_root = "data/split/train/fusion_pt"
    val_root   = "data/split/val/fusion_pt"

    save_dir = Path("results/final_model")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---------------- Dataset ----------------
    train_ds = FusionDataset(train_root)
    val_ds   = FusionDataset(val_root)

    train_loader = DataLoader(
        train_ds,
        batch_size=8,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=8,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # ---------------- Model ----------------
    model = VMamba(
        in_chans=6,       # fusion = 6 channels
        num_classes=5     # 5 bearing faults
    ).to(device)

    # ---------------- Loss & Optimizer ----------------
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-4,
        weight_decay=1e-4
    )

    # ---------------- Training ----------------
    best_val_acc = 0.0
    epochs = 100

    for epoch in range(1, epochs + 1):
        # ===== TRAIN =====
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

        train_acc = correct / total

        # ===== VAL =====
        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for x, y in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)

        val_acc = correct / total

        print(
            f"[Epoch {epoch:02d}] "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        # ===== SAVE BEST =====
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / "vmamba_fusion_best.pt"
            torch.save(model.state_dict(), save_path)
            print(f"[SAVE] Best model saved to {save_path}")

    print("[DONE] Training finished.")
    print(f"[BEST] Best Val Acc = {best_val_acc:.4f}")


if __name__ == "__main__":
    train()
