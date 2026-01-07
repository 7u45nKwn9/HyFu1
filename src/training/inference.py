import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import numpy as np

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc
)
from sklearn.preprocessing import label_binarize

import matplotlib.pyplot as plt
from tqdm import tqdm

# ====== IMPORT PROJECT ======
from data_loader.bearing_dataset import FusionDataset
from models.vanilla_mamba_T import VMamba


def inference():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---------- PATHS ----------
    test_root  = "data/split/test/fusion_pt"
    model_path = "results/final_model/vmamba_fusion_best.pt"

    table_dir = Path("results/tables")
    plot_dir  = Path("results/plots")

    table_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ---------- DATA ----------
    test_dataset = FusionDataset(test_root)
    test_loader = DataLoader(
        test_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=4
    )

    # ---------- MODEL ----------
    model = VMamba(in_chans=6, num_classes=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # ---------- INFERENCE ----------
    all_preds  = []
    all_labels = []
    all_probs  = []

    with torch.no_grad():
        for x, y in tqdm(test_loader, desc="Inference [Test]"):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs  = torch.softmax(logits, dim=1)
            preds  = torch.argmax(probs, dim=1)

            all_preds.append(preds.cpu().numpy())
            all_labels.append(y.cpu().numpy())
            all_probs.append(probs.cpu().numpy())

    y_true = np.concatenate(all_labels)
    y_pred = np.concatenate(all_preds)
    y_prob = np.concatenate(all_probs)

    # ---------- CLASS NAMES ----------
    class_names = [
        "Healthy",
        "Inner",
        "Outer",
        "Ball",
        "Cage"
    ]

    # ---------- CLASSIFICATION REPORT ----------
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    report_path = table_dir / "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    print("\n===== Classification Report =====")
    print(report)

    # ---------- CONFUSION MATRIX ----------
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    disp.plot(ax=ax, cmap="Blues", colorbar=True)
    plt.title("Confusion Matrix - Test Set")
    plt.tight_layout()

    cm_path = plot_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # ---------- SCORE DISTRIBUTION ----------
    plt.figure(figsize=(8, 5))
    for i, name in enumerate(class_names):
        plt.hist(
            y_prob[:, i],
            bins=50,
            alpha=0.6,
            label=name,
            density=True
        )

    plt.xlabel("Predicted probability")
    plt.ylabel("Density")
    plt.title("Prediction Score Distribution (Test)")
    plt.legend()
    plt.grid(True)

    score_path = plot_dir / "score_distribution.png"
    plt.tight_layout()
    plt.savefig(score_path, dpi=300)
    plt.close()

    # ---------- ROC / AUC (One-vs-Rest) ----------
    n_classes = len(class_names)

    # (N,) -> (N, C)
    y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fpr = {}
    tpr = {}
    roc_auc = {}

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # ---------- PLOT ROC ----------
    plt.figure(figsize=(8, 6))

    for i, name in enumerate(class_names):
        plt.plot(
            fpr[i],
            tpr[i],
            label=f"{name} (AUC = {roc_auc[i]:.4f})"
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves (One-vs-Rest) - Test Set")
    plt.legend(loc="lower right")
    plt.grid(True)

    roc_path = plot_dir / "roc_auc.png"
    plt.tight_layout()
    plt.savefig(roc_path, dpi=300)
    plt.close()

    # ---------- LOG ----------
    print("\n✅ Inference completed")
    print(f"📄 Report saved to: {report_path}")
    print(f"📊 Confusion matrix saved to: {cm_path}")
    print(f"📈 Score distribution saved to: {score_path}")
    print(f"📉 ROC/AUC curves saved to: {roc_path}")


if __name__ == "__main__":
    inference()
