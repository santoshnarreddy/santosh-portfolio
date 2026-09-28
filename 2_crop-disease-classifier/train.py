"""
Crop Disease Classifier — Training Script
Author: Santosh Narreddy

Two-phase training strategy:
  Phase 1 — Freeze backbone, train custom head (10 epochs, high lr)
  Phase 2 — Unfreeze last 3 ResNet blocks, fine-tune (20 epochs, low lr)

Usage:
    python train.py --data_dir ./data/PlantVillage --epochs_phase1 10 --epochs_phase2 20
"""

import os
import argparse
import json
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import build_resnet50, unfreeze_last_n_layers, count_params, NUM_CLASSES

# ── Reproducibility ────────────────────────────────────────────────────────────
torch.manual_seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[INFO] Using device: {DEVICE}")


# ── Data Loaders ───────────────────────────────────────────────────────────────

def get_loaders(data_dir, batch_size=32, num_workers=4):
    # ImageNet normalization (since we use pretrained ResNet50)
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    val_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dir = os.path.join(data_dir, 'train')
    val_dir   = os.path.join(data_dir, 'val')

    train_ds = datasets.ImageFolder(train_dir, transform=train_transforms)
    val_ds   = datasets.ImageFolder(val_dir,   transform=val_transforms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"[INFO] Train: {len(train_ds)} images | Val: {len(val_ds)} images")
    return train_loader, val_loader


# ── Train / Eval Loops ─────────────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer, epoch):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for batch_idx, (images, labels) in enumerate(loader):
        images, labels = images.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

        if batch_idx % 50 == 0:
            print(f"  Epoch {epoch} | Batch {batch_idx}/{len(loader)} "
                  f"| Loss: {loss.item():.4f}")

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += images.size(0)

    return total_loss / total, correct / total


# ── Training Phase ─────────────────────────────────────────────────────────────

def run_phase(model, train_loader, val_loader, optimizer, scheduler,
              num_epochs, phase_name, save_dir):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc     = evaluate(model, val_loader, criterion)
        scheduler.step()

        elapsed = time.time() - t0
        print(f"[{phase_name}] Epoch {epoch}/{num_epochs} | "
              f"Train: loss={train_loss:.4f} acc={train_acc:.4f} | "
              f"Val: loss={val_loss:.4f} acc={val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        history.append({
            'epoch': epoch, 'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = os.path.join(save_dir, f'best_model_{phase_name}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
            }, ckpt_path)
            print(f"  ✔ Saved best model ({val_acc:.4f}) → {ckpt_path}")

    return history, best_val_acc


# ── Main ───────────────────────────────────────────────────────────────────────

def main(args):
    os.makedirs(args.save_dir, exist_ok=True)

    train_loader, val_loader = get_loaders(
        args.data_dir, batch_size=args.batch_size, num_workers=args.workers
    )

    model = build_resnet50(num_classes=NUM_CLASSES, freeze_backbone=True)
    count_params(model)
    model = model.to(DEVICE)

    # ── Phase 1: Train only the head ──────────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 1 — Training classification head (backbone frozen)")
    print("="*60)

    optimizer_p1 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
    scheduler_p1 = CosineAnnealingLR(optimizer_p1, T_max=args.epochs_phase1)

    hist_p1, best_p1 = run_phase(model, train_loader, val_loader,
                                  optimizer_p1, scheduler_p1,
                                  args.epochs_phase1, 'phase1', args.save_dir)

    # ── Phase 2: Fine-tune last 3 blocks ──────────────────────────────────────
    print("\n" + "="*60)
    print("PHASE 2 — Fine-tuning (backbone partially unfrozen)")
    print("="*60)

    model = unfreeze_last_n_layers(model, n=3)
    count_params(model)

    optimizer_p2 = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)
    scheduler_p2 = CosineAnnealingLR(optimizer_p2, T_max=args.epochs_phase2)

    hist_p2, best_p2 = run_phase(model, train_loader, val_loader,
                                  optimizer_p2, scheduler_p2,
                                  args.epochs_phase2, 'phase2', args.save_dir)

    # Save history
    all_history = {'phase1': hist_p1, 'phase2': hist_p2,
                   'best_phase1': best_p1, 'best_phase2': best_p2}
    with open(os.path.join(args.save_dir, 'training_history.json'), 'w') as f:
        json.dump(all_history, f, indent=2)

    print(f"\n[DONE] Best val acc — Phase1: {best_p1:.4f} | Phase2: {best_p2:.4f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Crop Disease Classifier Training')
    parser.add_argument('--data_dir',       default='data/PlantVillage')
    parser.add_argument('--save_dir',       default='checkpoints')
    parser.add_argument('--batch_size',     type=int, default=32)
    parser.add_argument('--workers',        type=int, default=4)
    parser.add_argument('--epochs_phase1',  type=int, default=10)
    parser.add_argument('--epochs_phase2',  type=int, default=20)
    args = parser.parse_args()
    main(args)
