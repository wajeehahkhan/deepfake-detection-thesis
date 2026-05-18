"""
thesis_train.py — Main Training Pipeline
=========================================
Trains EfficientNet-B4 and XceptionNet on the Real-vs-Fake face dataset
for binary deepfake detection (real=0, fake=1).

Both models are loaded with ImageNet pretrained weights and fine-tuned
on the full 140k dataset using full fine-tuning (all layers unfrozen).

Usage:
    python thesis_train.py

Requirements:
    - Dataset downloaded from Kaggle and placed in ./data/
      (see README.md for exact folder structure)
    - All dependencies installed: pip install -r requirements.txt

Outputs (saved to ./results/):
    - final_summary.csv          — accuracy, F1, AUC per model
    - gender_bias.csv            — FNR/FPR broken down by gender
    - age_bias.csv               — FNR/FPR broken down by age group
    - *_training_curves.png      — loss/accuracy/AUC per epoch
    - *_confusion_matrix.png     — confusion matrix heatmap

Checkpoints saved to ./checkpoints/:
    - efficientnet_best.pth      — best EfficientNet-B4 weights
    - xceptionnet_best.pth       — best XceptionNet weights

Author: Wajeeha Khan
Institution: Tilburg University — MSc Data Science & Society 2026
"""

import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import timm

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for cluster (no display)
import matplotlib.pyplot as plt
import seaborn as sns

from deepface import DeepFace

# ─────────────────────────────────────────
# SECTION 1: REPRODUCIBILITY SETUP
# ─────────────────────────────────────────
# Setting all random seeds ensures results are reproducible across runs.
# SEED=42 is used throughout — same value as config.py.

SEED = 42

def set_seed(seed=SEED):
    """Fix all random seeds for full reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True   # deterministic convolutions
    torch.backends.cudnn.benchmark = False       # disable auto-tuning for consistency

set_seed()

# Use GPU if available, otherwise fall back to CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # mixed precision only works on GPU
print(f"Using device: {DEVICE} | AMP: {USE_AMP}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Create output directories if they don't exist
os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ─────────────────────────────────────────
# SECTION 2: DATASET LOADING
# ─────────────────────────────────────────
# The Kaggle dataset is pre-split into train/valid/test folders.
# We load all paths and labels here, then use them to build DataLoaders.
#
# Dataset: Real-vs-Fake Face Detection (Kaggle)
# URL: https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection
# Structure: data/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/

DATA_DIR = Path("./data")

all_images = []
all_labels = []

for split in ["train", "valid", "test"]:
    for label, class_name in enumerate(["real", "fake"]):
        # label=0 for real, label=1 for fake (binary classification)
        folder = DATA_DIR / "real_vs_fake" / "real-vs-fake" / split / class_name
        if folder.exists():
            imgs = list(folder.glob("*.jpg")) + list(folder.glob("*.png"))
            all_images.extend([str(p) for p in imgs])
            all_labels.extend([label] * len(imgs))

print(f"Total images found: {len(all_images)}")
print(f"Real (0): {all_labels.count(0)} | Fake (1): {all_labels.count(1)}")

# ─────────────────────────────────────────
# SECTION 3: TRAIN / VAL / TEST SPLIT (70/15/15)
# ─────────────────────────────────────────
# Although the Kaggle dataset comes pre-split, we re-split here to ensure
# a consistent, stratified 70/15/15 ratio with a fixed random seed.
# stratify=all_labels ensures equal class balance across all three splits.

train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
    all_images, all_labels,
    test_size=0.30,          # 30% goes to val+test
    random_state=SEED,
    stratify=all_labels      # maintain 50/50 real/fake ratio
)

val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
    temp_imgs, temp_lbls,
    test_size=0.50,          # split remaining 30% equally → 15% val, 15% test
    random_state=SEED,
    stratify=temp_lbls
)

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# ─────────────────────────────────────────
# SECTION 4: TRANSFORMS & DATASET CLASS
# ─────────────────────────────────────────
# EfficientNet-B4: designed for 224×224 input
# XceptionNet: originally designed for 299×299 but trained at 224×224
#              to match EfficientNet for fair comparison (slight performance trade-off)
#
# Training augmentations (applied only to training set):
#   - RandomHorizontalFlip: increases variety without distorting faces
#   - RandomRotation(10): handles slightly tilted images
#   - ColorJitter: robustness to lighting/colour variations
#   - RandomErasing: prevents model from relying on single pixel regions
#
# Validation/test transforms: only resize + normalize (no augmentation)
#
# ImageNet normalization: required because both models use ImageNet pretrained
# weights. The mean/std values match ImageNet statistics.

IMG_SIZE     = 224  # used for both models (fair comparison)
XCEPTION_SIZE = 299  # XceptionNet's native input size (kept for reference)

_NORM = dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(**_NORM),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

val_test_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(**_NORM),
])

# XceptionNet uses the same transforms as EfficientNet for fair comparison
xception_train_transform   = train_transform
xception_val_test_transform = val_test_transform


class FaceDataset(Dataset):
    """
    Custom PyTorch Dataset for loading face images with binary labels.

    Args:
        image_paths: list of file paths to images
        labels: list of integer labels (0=real, 1=fake)
        transform: torchvision transform pipeline to apply
    """
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label, self.image_paths[idx]  # path returned for error analysis


# Build datasets
train_dataset = FaceDataset(train_imgs, train_lbls, transform=train_transform)
val_dataset   = FaceDataset(val_imgs,   val_lbls,   transform=val_test_transform)
test_dataset  = FaceDataset(test_imgs,  test_lbls,  transform=val_test_transform)

xception_train_dataset = FaceDataset(train_imgs, train_lbls, transform=xception_train_transform)
xception_val_dataset   = FaceDataset(val_imgs,   val_lbls,   transform=xception_val_test_transform)
xception_test_dataset  = FaceDataset(test_imgs,  test_lbls,  transform=xception_val_test_transform)

# BATCH_SIZE=64: larger than Colab (32) since cluster has more RAM
# NUM_WORKERS=4: parallel data loading for faster throughput on cluster
BATCH_SIZE  = 64
NUM_WORKERS = 4

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
test_loader  = DataLoader(test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

xception_train_loader = DataLoader(xception_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
xception_val_loader   = DataLoader(xception_val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)
xception_test_loader  = DataLoader(xception_test_dataset,  batch_size=BATCH_SIZE, shuffle=False,
                                   num_workers=NUM_WORKERS, pin_memory=True, persistent_workers=True)

print(f"Batches — Train: {len(train_loader)} | Val: {len(val_loader)} | Test: {len(test_loader)}")

# ─────────────────────────────────────────
# SECTION 5: MODEL DEFINITIONS
# ─────────────────────────────────────────
# Both models loaded via timm with ImageNet pretrained weights.
# num_classes=1 → single sigmoid output for binary classification.
#
# Design choice: we use timm instead of torchvision for XceptionNet
# because torchvision does not include Xception. timm provides a
# well-maintained implementation consistent with the original paper.

def build_efficientnet(num_classes=1):
    """
    Load EfficientNet-B4 with ImageNet pretrained weights.
    Compound scaling (Tan & Le, 2019) makes it highly parameter-efficient.
    """
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model


def build_xceptionnet(num_classes=1):
    """
    Load XceptionNet with ImageNet pretrained weights.
    Depthwise separable convolutions (Chollet, 2017) make it efficient
    at capturing spatial correlations — relevant for GAN artifact detection.
    """
    model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    return model

# ─────────────────────────────────────────
# SECTION 6: TRAINING UTILITIES
# ─────────────────────────────────────────
# Label smoothing (0.1): instead of hard 0/1 targets, use 0.05/0.95.
# This prevents overconfidence and improves calibration — important
# for a model that will be applied to real-world deepfake detection.

LABEL_SMOOTHING = 0.1

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    """Run one training epoch, return average loss and predictions."""
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        # Apply label smoothing: push targets away from hard 0/1 boundaries
        labels_smooth = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        optimizer.zero_grad(set_to_none=True)

        # Automatic mixed precision (AMP): speeds up training on GPU
        # by using float16 where possible without sacrificing accuracy
        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(imgs)
            loss = criterion(outputs, labels_smooth)

        if USE_AMP:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs) > 0.5).int().squeeze(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.squeeze(1).int().cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc, all_labels, all_preds


def evaluate(model, loader, criterion, device):
    """Evaluate model on a dataloader, return loss, accuracy, F1, AUC."""
    model.eval()
    total_loss = 0
    all_probs, all_preds, all_labels, all_paths = [], [], [], []

    with torch.no_grad():
        for imgs, labels, paths in loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).squeeze(1).cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze(1).int().cpu().numpy())
            all_paths.extend(paths)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc, all_labels, all_preds, all_paths

# ─────────────────────────────────────────
# SECTION 7: FULL TRAINING LOOP
# ─────────────────────────────────────────

def train_model(model_name, model, train_loader, val_loader, epochs, lr, save_path):
    """
    Full training loop with early stopping and learning rate scheduling.

    Args:
        model_name: string identifier for logging
        model: PyTorch model to train
        train_loader: training DataLoader
        val_loader: validation DataLoader
        epochs: maximum number of epochs
        lr: initial learning rate (1e-4 chosen as standard for fine-tuning)
        save_path: path to save best model checkpoint

    Returns:
        model: trained model (best weights loaded)
        history: dict of training metrics per epoch
    """
    model = model.to(DEVICE)

    # Adam optimizer: adaptive learning rates per parameter, well-suited for CNNs
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    # ReduceLROnPlateau: halve LR when val AUC plateaus for 2 epochs
    # Monitors AUC (not loss) since AUC is a more stable metric for imbalanced data
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2, verbose=True
    )

    criterion = nn.BCEWithLogitsLoss()  # numerically stable sigmoid + BCE
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    best_auc = 0.0
    patience_counter = 0
    patience = 3  # stop if no AUC improvement for 3 consecutive epochs

    history = {"train_loss": [], "val_loss": [], "val_acc": [], "val_f1": [], "val_auc": []}

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _, _ = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        val_loss, val_acc, val_f1, val_auc, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(f"[{model_name}] Epoch {epoch}/{epochs} | "
              f"Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | "
              f"Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}", flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (AUC: {best_auc:.4f})")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  → Early stopping at epoch {epoch}")
                break

    # Load best weights before returning
    model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, history

# ─────────────────────────────────────────
# SECTION 8: VISUALISATION UTILITIES
# ─────────────────────────────────────────

def plot_training_curves(history, model_name):
    """Plot and save training loss, accuracy, and AUC curves per epoch."""
    epochs = range(1, len(history["val_loss"]) + 1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"],   label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    axes[1].plot(epochs, history["val_acc"], label="Val Accuracy")
    axes[1].set_title("Accuracy"); axes[1].legend()

    axes[2].plot(epochs, history["val_auc"], label="Val AUC")
    axes[2].set_title("AUC-ROC"); axes[2].legend()

    fig.suptitle(f"{model_name} — Training Curves", fontweight="bold")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(f"./results/{safe_name}_training_curves.png", dpi=100)
    plt.close()
    print(f"Saved: ./results/{safe_name}_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, model_name):
    """Plot and save confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("True Label"); plt.xlabel("Predicted Label")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(f"./results/{safe_name}_confusion_matrix.png", dpi=100)
    plt.close()
    print(f"Saved: ./results/{safe_name}_confusion_matrix.png")

# ─────────────────────────────────────────
# SECTION 9: RUN TRAINING — FULL 140k DATASET
# ─────────────────────────────────────────

# Train EfficientNet-B4 first
print("\n" + "="*60)
print("Training EfficientNet-B4...")
print("="*60)
efficientnet = build_efficientnet()
efficientnet, eff_history = train_model(
    model_name="EfficientNet-B4",
    model=efficientnet,
    train_loader=train_loader,
    val_loader=val_loader,
    epochs=10,
    lr=1e-4,
    save_path="./checkpoints/efficientnet_best.pth"
)

plot_training_curves(eff_history, "EfficientNet-B4")

# Evaluate on test set
criterion = nn.BCEWithLogitsLoss()
_, eff_acc, eff_f1, eff_auc, eff_true, eff_pred, _ = evaluate(
    efficientnet, test_loader, criterion, DEVICE
)
print(f"\nEfficientNet-B4 Test — Acc: {eff_acc:.4f} F1: {eff_f1:.4f} AUC: {eff_auc:.4f}")
print(classification_report(eff_true, eff_pred, target_names=["Real", "Fake"]))
plot_confusion_matrix(eff_true, eff_pred, "EfficientNet-B4")

# Free GPU memory before loading XceptionNet
del efficientnet
torch.cuda.empty_cache()
import gc; gc.collect()
print("EfficientNet done. Memory freed. Starting XceptionNet...", flush=True)

# Train XceptionNet
print("\n" + "="*60)
print("Training XceptionNet...")
print("="*60)
xceptionnet = build_xceptionnet()
xceptionnet, xcp_history = train_model(
    model_name="XceptionNet",
    model=xceptionnet,
    train_loader=xception_train_loader,
    val_loader=xception_val_loader,
    epochs=10,
    lr=1e-4,
    save_path="./checkpoints/xceptionnet_best.pth"
)

plot_training_curves(xcp_history, "XceptionNet")

xceptionnet.load_state_dict(
    torch.load("./checkpoints/xceptionnet_best.pth", map_location=DEVICE, weights_only=True)
)
_, xcp_acc, xcp_f1, xcp_auc, xcp_true, xcp_pred, _ = evaluate(
    xceptionnet, xception_test_loader, criterion, DEVICE
)
print(f"\nXceptionNet Test — Acc: {xcp_acc:.4f} F1: {xcp_f1:.4f} AUC: {xcp_auc:.4f}")
print(classification_report(xcp_true, xcp_pred, target_names=["Real", "Fake"]))
plot_confusion_matrix(xcp_true, xcp_pred, "XceptionNet")

# ─────────────────────────────────────────
# SECTION 10: FINAL SUMMARY
# ─────────────────────────────────────────

summary_rows = []
for name, acc, f1, auc, y_true, y_pred in [
    ("EfficientNet-B4", eff_acc, eff_f1, eff_auc, eff_true, eff_pred),
    ("XceptionNet",     xcp_acc, xcp_f1, xcp_auc, xcp_true, xcp_pred),
]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    summary_rows.append({
        "Model":                          name,
        "Accuracy":                       round(acc, 4),
        "F1 Score":                       round(f1, 4),
        "AUC-ROC":                        round(auc, 4),
        "True Positives (Fake caught)":   tp,
        "False Negatives (Fake missed)":  fn,
        "False Positives (Real flagged)": fp,
    })

summary_df = pd.DataFrame(summary_rows)
print("\n" + "="*60)
print("FINAL RESULTS SUMMARY")
print("="*60)
print(summary_df.to_string(index=False))
summary_df.to_csv("./results/final_summary.csv", index=False)
print("\nAll results saved to ./results/")

# ─────────────────────────────────────────
# SECTION 11: DEMOGRAPHIC BIAS ANALYSIS
# ─────────────────────────────────────────
# Uses DeepFace to estimate age and gender for each test image.
# Results are cached to avoid re-running the slow DeepFace inference.
#
# IMPORTANT LIMITATION: DeepFace predictions are pseudo-labels, not ground truth.
# Validation against UTKFace shows MAE of 12.21 years for age and
# 79.10% accuracy for gender. Bias results should be interpreted accordingly.

def get_demographics(image_paths, cache_path=None):
    """
    Run DeepFace age/gender estimation on each image.
    Results are cached to a CSV so this only runs once.

    Args:
        image_paths: list of image file paths
        cache_path: path to save/load cached results (optional)

    Returns:
        ages: list of estimated ages (int or None if failed)
        genders: list of estimated genders ('Man'/'Woman' or None)
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached demographics from {cache_path}", flush=True)
        df = pd.read_csv(cache_path)
        return df["age"].tolist(), df["gender"].tolist()

    ages, genders = [], []
    n = len(image_paths)
    for i, path in enumerate(image_paths):
        if i % 500 == 0:
            print(f"  DeepFace analysis: {i}/{n}", flush=True)
        try:
            result = DeepFace.analyze(
                img_path=path,
                actions=["age", "gender"],
                enforce_detection=False,  # don't fail if face not detected
            )
            if isinstance(result, list):
                result = result[0]
            ages.append(result["age"])
            genders.append(result["dominant_gender"])
        except Exception:
            ages.append(None)
            genders.append(None)

    if cache_path:
        pd.DataFrame({"path": image_paths, "age": ages, "gender": genders}).to_csv(
            cache_path, index=False
        )
        print(f"Demographics cached to {cache_path}", flush=True)

    return ages, genders


def assign_age_group(age):
    """Map raw age estimate to one of four age groups."""
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    age = int(age)
    if age < 20:   return "Under 20"
    elif age < 35: return "20-34"
    elif age < 50: return "35-49"
    else:          return "50+"


def compute_bias_metrics(y_true, y_pred, group_labels, group_col):
    """
    Compute accuracy, FNR, and FPR for each demographic subgroup.

    FNR (False Negative Rate) = proportion of real fakes missed
    FPR (False Positive Rate) = proportion of real faces incorrectly flagged

    Args:
        y_true: ground truth labels
        y_pred: predicted labels
        group_labels: demographic group per sample
        group_col: column name for the group (e.g. 'Gender', 'Age Group')

    Returns:
        DataFrame with per-group bias metrics
    """
    rows = []
    unique_groups = sorted(set(g for g in group_labels if g is not None))
    for group in unique_groups:
        idx = [i for i, g in enumerate(group_labels) if g == group]
        gt = [y_true[i] for i in idx]
        gp = [y_pred[i] for i in idx]
        if len(gt) == 0:
            continue
        cm_g = confusion_matrix(gt, gp, labels=[0, 1])
        tn, fp, fn, tp = cm_g.ravel()
        n_total = len(gt)
        acc = (tp + tn) / n_total
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None
        rows.append({
            group_col:            group,
            "N":                  n_total,
            "Accuracy":           round(acc, 4),
            "FNR (Fake missed)":  round(fnr, 4) if fnr is not None else None,
            "FPR (Real flagged)": round(fpr, 4) if fpr is not None else None,
        })
    return pd.DataFrame(rows)


print("\nRunning DeepFace demographic analysis on test set...", flush=True)
demo_ages, demo_genders = get_demographics(
    test_imgs,
    cache_path="./results/test_demographics.csv",
)
age_groups = [assign_age_group(a) for a in demo_ages]

# Compute bias tables for both models
eff_gender_df = compute_bias_metrics(eff_true, eff_pred, demo_genders, "Gender")
eff_gender_df.insert(0, "Model", "EfficientNet-B4")

eff_age_df = compute_bias_metrics(eff_true, eff_pred, age_groups, "Age Group")
eff_age_df.insert(0, "Model", "EfficientNet-B4")

xcp_gender_df = compute_bias_metrics(xcp_true, xcp_pred, demo_genders, "Gender")
xcp_gender_df.insert(0, "Model", "XceptionNet")

xcp_age_df = compute_bias_metrics(xcp_true, xcp_pred, age_groups, "Age Group")
xcp_age_df.insert(0, "Model", "XceptionNet")

# Combine and save
gender_bias_df = pd.concat([eff_gender_df, xcp_gender_df], ignore_index=True)
age_bias_df    = pd.concat([eff_age_df,    xcp_age_df],    ignore_index=True)

print("\n" + "="*60)
print("GENDER BIAS BREAKDOWN")
print("="*60)
print(gender_bias_df.to_string(index=False))
gender_bias_df.to_csv("./results/gender_bias.csv", index=False)
print("Saved: ./results/gender_bias.csv")

print("\n" + "="*60)
print("AGE BIAS BREAKDOWN")
print("="*60)
print(age_bias_df.to_string(index=False))
age_bias_df.to_csv("./results/age_bias.csv", index=False)
print("Saved: ./results/age_bias.csv")
