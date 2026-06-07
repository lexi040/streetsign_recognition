"""
Traffic Sign Recognition — main entry point.

Usage:
  python main.py train                # download data, train, save checkpoint
  python main.py evaluate             # load best checkpoint, evaluate on test set
  python main.py demo                 # save a prediction grid from the test set
  python main.py all                  # train → evaluate → demo in one go

Options (all commands):
  --data-root   DIR   where to store the GTSRB dataset  (default: ./data)
  --ckpt-dir    DIR   directory for model checkpoints   (default: ./checkpoints)
  --results-dir DIR   directory for output plots        (default: ./results)
  --epochs      N     number of training epochs         (default: 30)
  --batch-size  N     batch size                        (default: 64)
  --lr          F     initial learning rate             (default: 1e-3)
  --workers     N     DataLoader workers                (default: 4)
  --device      STR   cpu | cuda | auto                 (default: auto)
"""

import argparse
import os
import sys
import torch

# ── Argument parser ───────────────────────────────────────────────────────────

def make_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Traffic Sign Recognition (CV4V project)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("command", choices=["train", "evaluate", "demo", "all"],
                   help="Action to run")
    p.add_argument("--data-root",    default="./data")
    p.add_argument("--ckpt-dir",     default="./checkpoints")
    p.add_argument("--results-dir",  default="./results")
    p.add_argument("--epochs",       type=int,   default=30)
    p.add_argument("--batch-size",   type=int,   default=64)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--workers",      type=int,   default=4)
    p.add_argument("--device",       default="auto")
    return p


# ── Command implementations ───────────────────────────────────────────────────

def cmd_train(args) -> dict:
    from src.train import train
    from src.evaluate import plot_training_curves

    history = train(
        data_root=args.data_root,
        checkpoint_dir=args.ckpt_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        num_workers=args.workers,
        device_str=args.device,
    )
    plot_training_curves(
        history,
        save_path=os.path.join(args.results_dir, "training_curves.png"),
    )
    return history


def cmd_evaluate(args) -> None:
    from src.train import load_best_model
    from src.dataset import get_loaders
    from src.evaluate import run_evaluation

    device = _resolve_device(args.device)
    model = load_best_model(args.ckpt_dir, device)
    _, _, test_loader, _ = get_loaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.workers
    )
    run_evaluation(model, test_loader, device, results_dir=args.results_dir)


def cmd_demo(args) -> None:
    from src.train import load_best_model
    from src.dataset import get_loaders
    from src.utils import show_demo_predictions

    device = _resolve_device(args.device)
    model = load_best_model(args.ckpt_dir, device)
    _, _, test_loader, _ = get_loaders(
        args.data_root, batch_size=args.batch_size, num_workers=args.workers
    )
    show_demo_predictions(
        model, test_loader, device,
        save_path=os.path.join(args.results_dir, "demo_predictions.png"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = make_parser()
    args = parser.parse_args()

    # Convert hyphenated arg names to underscore attributes
    args.ckpt_dir = args.ckpt_dir

    os.makedirs(args.data_root,   exist_ok=True)
    os.makedirs(args.ckpt_dir,    exist_ok=True)
    os.makedirs(args.results_dir, exist_ok=True)

    if args.command == "train":
        cmd_train(args)

    elif args.command == "evaluate":
        cmd_evaluate(args)

    elif args.command == "demo":
        cmd_demo(args)

    elif args.command == "all":
        print("=" * 60)
        print("STEP 1 / 3 — Training")
        print("=" * 60)
        cmd_train(args)

        print("\n" + "=" * 60)
        print("STEP 2 / 3 — Evaluation")
        print("=" * 60)
        cmd_evaluate(args)

        print("\n" + "=" * 60)
        print("STEP 3 / 3 — Demo")
        print("=" * 60)
        cmd_demo(args)

        print("\n✓ All done! Outputs saved to:", args.results_dir)


if __name__ == "__main__":
    main()
