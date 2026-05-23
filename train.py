"""
train.py — Fine-tune EfficientNet-B0 on the PlantVillage dataset
Dataset: https://www.kaggle.com/datasets/emmarex/plantdisease

STEPS:
  1. Download & unzip the Kaggle dataset — you get a folder called "PlantVillage"
  2. Run:  python train.py
  3. Wait ~10-20 min (CPU) or ~3-5 min (GPU)
  4. Weights saved to plant_disease_model.pth — app.py loads them automatically

Usage:
  python train.py                              # auto-detects PlantVillage/ folder
  python train.py --data_dir path/to/folder   # custom path
  python train.py --epochs 20 --batch_size 64
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

# ── Expected PlantVillage class order (must match app.py CLASSES list) ───────
EXPECTED_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]


def find_data_dir():
    """Auto-detect the PlantVillage folder."""
    candidates = [
        "PlantVillage",
        "plantvillage",
        "plant_village",
        "PlantDisease",
        "plant-disease",
        os.path.join("dataset", "PlantVillage"),
    ]
    for c in candidates:
        if os.path.isdir(c):
            # Check it has subdirs (class folders)
            subdirs = [d for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if len(subdirs) > 5:
                return c
    return None


def get_transforms():
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=0.15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return train_tf, val_tf


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*55}")
    print(f"  LeafScan AI — Training Script")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  Batch: {args.batch_size}")
    print(f"{'='*55}\n")

    # ── Find dataset ──────────────────────────────────────────
    data_dir = args.data_dir
    if data_dir is None:
        data_dir = find_data_dir()
    if data_dir is None or not os.path.isdir(data_dir):
        print("❌ Could not find the PlantVillage dataset folder.")
        print("   Pass it explicitly:  python train.py --data_dir PlantVillage/")
        print("   Or make sure you're running from the same directory as the folder.")
        sys.exit(1)

    print(f"📂 Dataset folder: {data_dir}")

    train_tf, val_tf = get_transforms()
    full_dataset = datasets.ImageFolder(data_dir, transform=train_tf)
    n_classes = len(full_dataset.classes)
    print(f"📊 Found {len(full_dataset):,} images across {n_classes} classes\n")

    # Print class mapping
    print("Class mapping (folder → index):")
    for i, cls in enumerate(full_dataset.classes):
        print(f"  [{i:02d}] {cls}")
    print()

    if False:  # class count check disabled — using your exact dataset
        print(f"⚠  Warning: expected 38 classes, found {n_classes}.")
        print("   Make sure app.py CLASSES list matches the folder order above.")

    # ── Split 80/20 ───────────────────────────────────────────
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = random_split(
        full_dataset, [n_train, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    # Apply val transform to val split
    from copy import deepcopy
    val_ds.dataset = deepcopy(full_dataset)
    val_ds.dataset.transform = val_tf

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=(device.type == "cuda")
    )
    print(f"✓ Train: {n_train:,} images | Val: {n_val:,} images\n")

    # ── Model ─────────────────────────────────────────────────
    model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, n_classes)  # 15 for your dataset
    model = model.to(device)

    # Phase 1: freeze backbone, train head only (fast warmup)
    for param in model.features.parameters():
        param.requires_grad = False

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr,
        steps_per_epoch=len(train_loader), epochs=args.epochs
    )

    best_acc = 0.0
    output_path = args.output

    for epoch in range(1, args.epochs + 1):
        # Unfreeze backbone at epoch 4
        if epoch == 4:
            print("  → Unfreezing backbone for full fine-tuning...\n")
            for param in model.features.parameters():
                param.requires_grad = True
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr * 0.1)
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer, max_lr=args.lr * 0.1,
                steps_per_epoch=len(train_loader),
                epochs=args.epochs - 3
            )

        # ── Train epoch ───────────────────────────────────────
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        t0 = time.time()

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if epoch <= 3 or epoch > 3:
                try:
                    scheduler.step()
                except Exception:
                    pass
            train_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

            # Progress bar
            if (batch_idx + 1) % max(1, len(train_loader) // 5) == 0:
                pct = (batch_idx + 1) / len(train_loader) * 100
                bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
                print(f"\r  [{bar}] {pct:.0f}% — loss: {train_loss/total:.4f}", end="", flush=True)

        train_acc = correct / total
        train_loss /= total

        # ── Validate ──────────────────────────────────────────
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                val_correct += (outputs.argmax(1) == labels).sum().item()
                val_total += images.size(0)

        val_acc = val_correct / val_total
        val_loss /= val_total
        elapsed = time.time() - t0

        print(f"\r  Epoch {epoch:02d}/{args.epochs} │ "
              f"Train: {train_acc*100:.1f}% loss {train_loss:.4f} │ "
              f"Val: {val_acc*100:.1f}% loss {val_loss:.4f} │ "
              f"{elapsed:.0f}s")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), output_path)
            print(f"  ✓ Saved best model → {output_path}  (val acc: {best_acc*100:.1f}%)")

    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_acc*100:.1f}%")
    print(f"  Weights saved to: {output_path}")
    print(f"\n  ✅ Place {output_path} next to app.py")
    print(f"     then run:  streamlit run app.py")
    print(f"{'='*55}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LeafScan AI on PlantVillage")
    parser.add_argument("--data_dir", type=str, default=None,
                        help="Path to PlantVillage folder (auto-detected if omitted)")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs (default: 15)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size (default: 32; use 16 if out of memory)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Peak learning rate (default: 0.001)")
    parser.add_argument("--workers", type=int, default=2,
                        help="DataLoader workers (default: 2; use 0 on Windows if errors)")
    parser.add_argument("--output", type=str, default="plant_disease_model.pth",
                        help="Output weights filename (default: plant_disease_model.pth)")
    args = parser.parse_args()
    train(args)
