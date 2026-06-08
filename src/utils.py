
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.dataset import CLASS_NAMES, MEAN, STD


#functie de denormalizare a imaginilor pentru vizualizare

def denormalize(tensor: torch.Tensor) -> np.ndarray:
    mean = np.array(MEAN, dtype=np.float32).reshape(3, 1, 1)
    std  = np.array(STD,  dtype=np.float32).reshape(3, 1, 1)
    img = tensor.cpu().numpy() * std + mean          # CHW float
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8).transpose(1, 2, 0)  # HWC uint8
    return img


#vizualizare predictii pe un grid de imagini

def show_sample_grid(
    images: torch.Tensor,
    true_labels: torch.Tensor,
    pred_labels: torch.Tensor | None = None,
    confs: torch.Tensor | None = None,
    save_path: str | None = None,
    n_cols: int = 5,
    title: str = "Sample Predictions",
) -> None:

    n = min(len(images), n_cols * 4)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 2.4, n_rows * 2.6))
    axes = np.array(axes).reshape(-1)   # flatten for indexing

    for i in range(len(axes)):
        ax = axes[i]
        ax.axis("off")
        if i >= n:
            continue

        img = denormalize(images[i])
        true_cls = CLASS_NAMES[true_labels[i].item()]

        ax.imshow(img)

        if pred_labels is not None:
            pred_cls  = CLASS_NAMES[pred_labels[i].item()]
            correct   = (pred_labels[i] == true_labels[i]).item()
            color     = "green" if correct else "red"
            conf_str  = f" ({confs[i].item():.0%})" if confs is not None else ""
            cell_title = f"GT: {true_cls}\nPred: {pred_cls}{conf_str}"
        else:
            color = "black"
            cell_title = true_cls

        ax.set_title(cell_title, fontsize=7.5, color=color, pad=2)

    fig.suptitle(title, fontsize=11, y=1.01)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Sample grid saved {save_path}")
    else:
        plt.show()
    plt.close()



#demo vizualizare predictii
@torch.no_grad()
def show_demo_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    save_path: str = "./results/demo_predictions.png",
    n_images: int = 20,
) -> None:

    model.eval()
    images, labels = next(iter(loader))
    images_dev = images[:n_images].to(device)
    labels     = labels[:n_images]

    logits = model(images_dev)
    probs  = torch.softmax(logits, dim=1)
    confs, preds = probs.max(dim=1)

    show_sample_grid(
        images=images[:n_images],
        true_labels=labels,
        pred_labels=preds.cpu(),
        confs=confs.cpu(),
        save_path=save_path,
        title="Test-set Predictions",
    )
