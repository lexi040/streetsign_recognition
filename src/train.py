
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import TrafficSignCNN, build_model, model_summary
from src.dataset import get_loaders, NUM_CLASSES


#ruleaza o epoca

def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    desc: str,
) -> tuple[float, float]:
    #returneaza avg_loss si accuratete
    training = optimizer is not None
    model.train(training)

    total_loss = 0.0
    correct = 0
    total = 0

    with torch.set_grad_enabled(training):
        for images, labels in tqdm(loader, desc=desc, leave=False):
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


#functia principala de antrenare
def train(
    data_root: str = "./data",
    checkpoint_dir: str = "./checkpoints",
    epochs: int = 30,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    num_workers: int = 4,
    device_str: str = "auto",
) -> dict:
    #antreneaza modelul si salveaza cel mai bun checkpoint pe baza acuratetei pe setul de validare
    
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)
    print(f"Using device: {device}")

    #date
    print("Loading dataset ...")
    train_loader, val_loader, _, class_weights = get_loaders(
        data_root, batch_size=batch_size, num_workers=num_workers
    )
    print(f"  Train batches : {len(train_loader)}  /  Val batches: {len(val_loader)}")

    #init model
    model = build_model(device)
    model_summary(model)

    #loss, optimizer, scheduler
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training lopp
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_val_acc = 0.0
    best_path = os.path.join(checkpoint_dir, "best.pth")

    history: dict[str, list[float]] = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "lr": [],
    }

    print(f"\nTraining for {epochs} epochs ...\n")
    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = _run_epoch(
            model, train_loader, criterion, optimizer, device,
            desc=f"Epoch {epoch:02d}/{epochs} [train]",
        )
        val_loss, val_acc = _run_epoch(
            model, val_loader, criterion, None, device,
            desc=f"Epoch {epoch:02d}/{epochs} [val]  ",
        )
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["lr"].append(current_lr)

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"loss {train_loss:.4f}/{val_loss:.4f} | "
            f"acc {train_acc:.3f}/{val_acc:.3f} | "
            f"lr {current_lr:.2e} | "
            f"{elapsed:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                },
                best_path,
            )
            print(f"Saved new best checkpoint (val_acc={val_acc:.3f})")

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.3f}")
    print(f"Checkpoint saved to: {best_path}")
    return history


def load_best_model(checkpoint_dir: str, device: torch.device) -> TrafficSignCNN:
    #returneaza modelul cu cea mai buna performanta
    path = os.path.join(checkpoint_dir, "best.pth")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}. Run training first.")

    ckpt = torch.load(path, map_location=device)
    model = build_model(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"Loaded checkpoint from epoch {ckpt['epoch']} (val_acc={ckpt['val_acc']:.3f})")
    return model
