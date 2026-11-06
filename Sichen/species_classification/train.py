"""Train a ResNet-50 baseline for aegypti vs albopictus with transfer learning.

Two stages:
  1. Freeze the pretrained backbone, train only the new 2-class layer (fast, safe start).
  2. Unfreeze everything and fine-tune with a small learning rate (adapts features to mosquitoes).
After every epoch we evaluate on the val set and keep the checkpoint with the lowest val loss.

Usage:
  python train.py                                        # full run
  python train.py --epochs-head 1 --epochs-finetune 1 --limit-batches 5   # quick smoke test
"""
import argparse
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models

from dataset import CLASSES, IMAGENET_MEAN, IMAGENET_STD, INPUT_SIZE, MosquitoDataset, build_transforms

ROOT = Path(__file__).resolve().parent


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")   # NVIDIA GPU (Windows laptop)
    if torch.backends.mps.is_available():
        return torch.device("mps")    # Apple GPU (Mac)
    return torch.device("cpu")


def build_model():
    # Load ResNet-50 with weights learned on ImageNet (1.2M photos, 1000 classes).
    # Its early layers already detect edges, textures, patterns -- useful for mosquitoes too.
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Replace the last layer (2048 features -> 1000 ImageNet classes) with 2048 -> 2 classes.
    model.fc = nn.Linear(model.fc.in_features, len(CLASSES))
    return model


def load_checkpoint(path, device):
    """Rebuild the model from a best.pt saved by this script. Returns (model, checkpoint dict)."""
    checkpoint = torch.load(path, map_location=device)
    model = models.resnet50(weights=None)  # no need to download ImageNet weights: we load our own
    model.fc = nn.Linear(model.fc.in_features, len(checkpoint["classes"]))
    model.load_state_dict(checkpoint["model_state"])
    return model.to(device).eval(), checkpoint


def set_backbone_trainable(model, trainable):
    for name, param in model.named_parameters():
        if not name.startswith("fc."):
            param.requires_grad = trainable


def run_epoch(model, loader, criterion, device, optimizer=None, limit_batches=None):
    """One pass over the data. Trains if an optimizer is given, otherwise only evaluates."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, seen = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for i, (images, labels) in enumerate(loader):
            if limit_batches and i >= limit_batches:
                break
            images, labels = images.to(device), labels.to(device)
            logits = model(images)                 # raw scores, shape (batch, 2)
            loss = criterion(logits, labels)       # how wrong the scores are
            if training:
                optimizer.zero_grad()
                loss.backward()                    # compute gradients
                optimizer.step()                   # update weights a little
            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += len(labels)
    return total_loss / seen, correct / seen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs-head", type=int, default=5)
    parser.add_argument("--epochs-finetune", type=int, default=15)
    parser.add_argument("--lr-head", type=float, default=1e-3)
    parser.add_argument("--lr-finetune", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--limit-batches", type=int, default=None, help="only for quick tests")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()
    run_dir = ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True)
    print(f"device: {device} | saving to {run_dir}")

    train_loader = DataLoader(MosquitoDataset("train", build_transforms(train=True)),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(MosquitoDataset("val", build_transforms(train=False)),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = build_model().to(device)
    criterion = nn.CrossEntropyLoss()

    stages = [("head", args.epochs_head, args.lr_head, False),
              ("finetune", args.epochs_finetune, args.lr_finetune, True)]
    best_val_loss, history, epoch = float("inf"), [], 0
    for stage, n_epochs, lr, backbone_trainable in stages:
        set_backbone_trainable(model, backbone_trainable)
        # Only parameters with requires_grad=True are given to the optimizer
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
        for _ in range(n_epochs):
            epoch += 1
            start = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, criterion, device, optimizer, args.limit_batches)
            val_loss, val_acc = run_epoch(model, val_loader, criterion, device, None, args.limit_batches)
            history.append({"epoch": epoch, "stage": stage, "train_loss": train_loss, "train_acc": train_acc,
                            "val_loss": val_loss, "val_acc": val_acc})
            mark = ""
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                # Save everything needed to use the model later, not just the weights
                torch.save({"model_state": model.state_dict(), "arch": "resnet50", "classes": CLASSES,
                            "input_size": INPUT_SIZE, "mean": IMAGENET_MEAN, "std": IMAGENET_STD,
                            "epoch": epoch, "val_loss": val_loss, "val_acc": val_acc},
                           run_dir / "best.pt")
                mark = "  <- best, saved"
            print(f"[{stage:8}] epoch {epoch:2d} | train loss {train_loss:.3f} acc {train_acc:.3f} | "
                  f"val loss {val_loss:.3f} acc {val_acc:.3f} | {time.time() - start:.0f}s{mark}", flush=True)

    with open(run_dir / "history.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    print(f"done. best val loss {best_val_loss:.3f}, checkpoint: {run_dir / 'best.pt'}")


if __name__ == "__main__":  # required on Mac/Windows when DataLoader uses num_workers > 0
    main()
