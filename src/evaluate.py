
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix as sk_confusion_matrix,
)
from tqdm import tqdm

from src.dataset import CLASS_NAMES, NUM_CLASSES



@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    #ruleaza modelul pe test_loader si colectam etichete si predictii
    model.eval()
    labels_list, preds_list, confs_list = [], [], []

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(dim=1)

        labels_list.append(labels.numpy())
        preds_list.append(pred.cpu().numpy())
        confs_list.append(conf.cpu().numpy())

    return (
        np.concatenate(labels_list),
        np.concatenate(preds_list),
        np.concatenate(confs_list),
    )


#calc metrice

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> dict:
    
    accuracy = (labels == preds).mean()
    report = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=False,
        zero_division=0,
    )
    report_dict = classification_report(
        labels, preds,
        target_names=CLASS_NAMES,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    return {"accuracy": accuracy, "report_str": report, "report_dict": report_dict}




#matricea de confuzie
def plot_confusion_matrix(
    labels: np.ndarray,
    preds: np.ndarray,
    save_path: str,
    normalize: bool = True,
) -> None:
    cm = sk_confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    if normalize:
        cm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1 if normalize else None)
    plt.colorbar(im, ax=ax)

    ax.set(
        xticks=range(NUM_CLASSES),
        yticks=range(NUM_CLASSES),
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        xlabel="Predicted label",
        ylabel="True label",
        title="Confusion Matrix" + (" (normalized)" if normalize else ""),
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)

    thresh = 0.5 if normalize else cm.max() / 2
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, val, ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black", fontsize=9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved {save_path}")



#curbele de antrenament
def plot_training_curves(history: dict, save_path: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val loss")
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Loss curves")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, [a * 100 for a in history["train_acc"]], label="Train acc")
    axes[1].plot(epochs, [a * 100 for a in history["val_acc"]],   label="Val acc")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy curves")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Training curves saved {save_path}")


#functie principala care ruleaza tot
def run_evaluation(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    results_dir: str = "./results",
) -> dict:
    
    os.makedirs(results_dir, exist_ok=True)

    labels, preds, confs = collect_predictions(model, test_loader, device)
    metrics = compute_metrics(labels, preds)

    print(f"\n{'='*55}")
    print(f"  Test accuracy : {metrics['accuracy']*100:.2f}%")
    print(f"{'='*55}")
    print(metrics["report_str"])

    plot_confusion_matrix(
        labels, preds,
        save_path=os.path.join(results_dir, "confusion_matrix.png"),
    )
    return metrics
