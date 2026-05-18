"""
Deepfake Detection: EfficientNet-B4 vs XceptionNet
Adapted from Colab notebook for GPU cluster (SLURM)
- Removed Google Drive / Colab-specific code
- Removed 10k subset limitation → uses full 140k dataset
- Increased batch size and workers for cluster
- Saves all results to ./results/
"""

# pip install timm deepface tf-keras

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
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
import seaborn as sns

from deepface import DeepFace

# ─────────────────────────────────────────
# SECTION 1: SETUP
# ─────────────────────────────────────────

SEED = 42

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed()

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = torch.cuda.is_available()  # mixed precision only on GPU
print(f"Using device: {DEVICE} | AMP: {USE_AMP}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

os.makedirs("./checkpoints", exist_ok=True)
os.makedirs("./results", exist_ok=True)

# ─────────────────────────────────────────
# SECTION 2: DATASET LOADING
# ─────────────────────────────────────────

# The Kaggle dataset should be unzipped to ./data/
# Expected structure: data/real_vs_fake/real-vs-fake/{train,valid,test}/{real,fake}/
DATA_DIR = Path("./data")

all_images = []
all_labels = []

for split in ["train", "valid", "test"]:
    for label, class_name in enumerate(["real", "fake"]):
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

train_imgs, temp_imgs, train_lbls, temp_lbls = train_test_split(
    all_images, all_labels,
    test_size=0.30,
    random_state=SEED,
    stratify=all_labels
)

val_imgs, test_imgs, val_lbls, test_lbls = train_test_split(
    temp_imgs, temp_lbls,
    test_size=0.50,
    random_state=SEED,
    stratify=temp_lbls
)

print(f"Train: {len(train_imgs)} | Val: {len(val_imgs)} | Test: {len(test_imgs)}")

# ─────────────────────────────────────────
# SECTION 4: TRANSFORMS & DATASET CLASS
# ─────────────────────────────────────────

IMG_SIZE = 224
XCEPTION_SIZE = 299  # XceptionNet was designed for 299×299

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

xception_train_transform = transforms.Compose([
    transforms.Resize((XCEPTION_SIZE, XCEPTION_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
    transforms.ToTensor(),
    transforms.Normalize(**_NORM),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
])

xception_val_test_transform = transforms.Compose([
    transforms.Resize((XCEPTION_SIZE, XCEPTION_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(**_NORM),
])


class FaceDataset(Dataset):
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
        return img, label, self.image_paths[idx]


train_dataset = FaceDataset(train_imgs, train_lbls, transform=train_transform)
val_dataset   = FaceDataset(val_imgs,   val_lbls,   transform=val_test_transform)
test_dataset  = FaceDataset(test_imgs,  test_lbls,  transform=val_test_transform)

xception_train_dataset = FaceDataset(train_imgs, train_lbls, transform=xception_train_transform)
xception_val_dataset   = FaceDataset(val_imgs,   val_lbls,   transform=xception_val_test_transform)
xception_test_dataset  = FaceDataset(test_imgs,  test_lbls,  transform=xception_val_test_transform)

# Larger batch size and more workers — safe on the cluster
BATCH_SIZE  = 64   # was 32 in Colab; cluster can handle 64
NUM_WORKERS = 4    # was 2 in Colab

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

def build_efficientnet(num_classes=1):
    model = timm.create_model("efficientnet_b4", pretrained=True, num_classes=num_classes)
    return model


def build_xceptionnet(num_classes=1):
    model = timm.create_model("xception", pretrained=True, num_classes=num_classes)
    return model

# ─────────────────────────────────────────
# SECTION 6: TRAINING UTILITIES
# ─────────────────────────────────────────

LABEL_SMOOTHING = 0.1  # pushes targets from 0/1 to 0.05/0.95

def train_one_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for batch_idx, (imgs, labels, _) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.float().unsqueeze(1).to(device)
        labels_smooth = labels * (1 - LABEL_SMOOTHING) + 0.5 * LABEL_SMOOTHING

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=USE_AMP):
            outputs = model(imgs)
            loss = criterion(outputs, labels_smooth)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()
        preds = (torch.sigmoid(outputs.detach()) > 0.5).long().squeeze(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.long().squeeze(1).cpu().numpy())

        if batch_idx % 200 == 0:
            print(f"  Batch {batch_idx}/{len(loader)} — Loss: {loss.item():.4f}", flush=True)

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    return avg_loss, acc, f1


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for imgs, labels, _ in loader:
            imgs = imgs.to(device)
            labels = labels.float().unsqueeze(1).to(device)

            with torch.cuda.amp.autocast(enabled=USE_AMP):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            total_loss += loss.item()

            probs = torch.sigmoid(outputs).squeeze(1)
            preds = (probs > 0.5).long()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.long().squeeze(1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    avg_loss = total_loss / len(loader)
    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds, zero_division=0)
    auc = roc_auc_score(all_labels, all_probs)
    return avg_loss, acc, f1, auc, all_labels, all_preds, all_probs

# ─────────────────────────────────────────
# SECTION 7: TRAINING LOOP
# ─────────────────────────────────────────

def train_model(model_name, model, train_loader, val_loader, epochs=10, lr=1e-4, save_path=None):
    print(f"\n{'='*50}")
    print(f"Training: {model_name}")
    print(f"{'='*50}", flush=True)

    model = model.to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )
    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP)

    history = {
        "train_loss": [], "val_loss": [],
        "train_acc":  [], "val_acc":  [],
        "train_f1":   [], "val_f1":   [],
        "val_auc": []
    }

    best_auc = 0
    patience_counter = 0
    PATIENCE = 3
    start_epoch = 1

    safe_name = model_name.replace(" ", "_")
    resume_path = f"./checkpoints/{safe_name}_resume.pth"

    if os.path.exists(resume_path):
        print(f"Resuming from checkpoint: {resume_path}", flush=True)
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        if "scaler_state" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state"])
        history          = ckpt["history"]
        best_auc         = ckpt["best_auc"]
        patience_counter = ckpt["patience_counter"]
        start_epoch      = ckpt["epoch"] + 1
        print(f"Resumed at epoch {start_epoch}, best AUC so far: {best_auc:.4f}", flush=True)

    for epoch in range(start_epoch, epochs + 1):
        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, DEVICE
        )
        val_loss, val_acc, val_f1, val_auc, _, _, _ = evaluate(
            model, val_loader, criterion, DEVICE
        )

        scheduler.step(val_auc)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_f1"].append(train_f1)
        history["val_f1"].append(val_f1)
        history["val_auc"].append(val_auc)

        print(f"Epoch {epoch:02d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} F1: {train_f1:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f} AUC: {val_auc:.4f}",
              flush=True)

        if val_auc > best_auc:
            best_auc = val_auc
            patience_counter = 0
            if save_path:
                torch.save(model.state_dict(), save_path)
                print(f"  ✓ Best model saved (AUC: {best_auc:.4f})", flush=True)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping at epoch {epoch}", flush=True)
                # Save resume checkpoint before stopping so it can be inspected,
                # but mark it so we don't accidentally resume into more early-stop loops
                torch.save({
                    "epoch": epoch,
                    "model_state":      model.state_dict(),
                    "optimizer_state":  optimizer.state_dict(),
                    "scheduler_state":  scheduler.state_dict(),
                    "scaler_state":     scaler.state_dict(),
                    "history":          history,
                    "best_auc":         best_auc,
                    "patience_counter": patience_counter,
                }, resume_path)
                break

        # Save full training state after every epoch so a crash can be resumed
        torch.save({
            "epoch": epoch,
            "model_state":      model.state_dict(),
            "optimizer_state":  optimizer.state_dict(),
            "scheduler_state":  scheduler.state_dict(),
            "scaler_state":     scaler.state_dict(),
            "history":          history,
            "best_auc":         best_auc,
            "patience_counter": patience_counter,
        }, resume_path)
    else:
        # Loop completed without early stopping — remove the resume checkpoint
        if os.path.exists(resume_path):
            os.remove(resume_path)
            print(f"Training complete. Resume checkpoint removed.", flush=True)

    return model, history

# ─────────────────────────────────────────
# SECTION 8: PLOTTING HELPERS
# ─────────────────────────────────────────

def plot_training_curves(history, model_name):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    epochs = range(1, len(history["train_loss"]) + 1)

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"],   label="Val")
    axes[0].set_title(f"{model_name} — Loss")
    axes[0].legend()

    axes[1].plot(epochs, history["train_acc"], label="Train")
    axes[1].plot(epochs, history["val_acc"],   label="Val")
    axes[1].set_title(f"{model_name} — Accuracy")
    axes[1].legend()

    axes[2].plot(epochs, history["val_auc"], label="Val AUC", color="green")
    axes[2].set_title(f"{model_name} — Val AUC-ROC")
    axes[2].legend()

    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(f"./results/{safe_name}_training_curves.png", dpi=100)
    plt.close()
    print(f"Saved: ./results/{safe_name}_training_curves.png")


def plot_confusion_matrix(y_true, y_pred, model_name):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"])
    plt.title(f"{model_name} — Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    safe_name = model_name.replace(" ", "_")
    plt.savefig(f"./results/{safe_name}_confusion_matrix.png", dpi=100)
    plt.close()
    print(f"Saved: ./results/{safe_name}_confusion_matrix.png")

# ─────────────────────────────────────────
# SECTION 9: RUN TRAINING — FULL 140k DATASET
# ─────────────────────────────────────────

# Train EfficientNet-B4
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

# Reload best checkpoint for evaluation
efficientnet.load_state_dict(
    torch.load("./checkpoints/efficientnet_best.pth", map_location=DEVICE, weights_only=True)
)
criterion = nn.BCEWithLogitsLoss()
_, eff_acc, eff_f1, eff_auc, eff_true, eff_pred, _ = evaluate(
    efficientnet, test_loader, criterion, DEVICE
)
print(f"\nEfficientNet-B4 Test — Acc: {eff_acc:.4f} F1: {eff_f1:.4f} AUC: {eff_auc:.4f}")
print(classification_report(eff_true, eff_pred, target_names=["Real", "Fake"]))
plot_confusion_matrix(eff_true, eff_pred, "EfficientNet-B4")

# Free memory before XceptionNet
del efficientnet
torch.cuda.empty_cache()
import gc; gc.collect()
print("EfficientNet done. Memory freed. Starting XceptionNet...", flush=True)

# Train XceptionNet (uses 299×299 loaders — correct native resolution)
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
        "Model": name,
        "Accuracy": round(acc, 4),
        "F1 Score": round(f1, 4),
        "AUC-ROC": round(auc, 4),
        "True Positives (Fake caught)": tp,
        "False Negatives (Fake missed)": fn,
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

def get_demographics(image_paths, cache_path=None):
    """Run DeepFace age/gender on each image. Results cached so this only runs once."""
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
                enforce_detection=False,
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
    if age is None or (isinstance(age, float) and np.isnan(age)):
        return None
    age = int(age)
    if age < 20:
        return "Under 20"
    elif age < 35:
        return "20-34"
    elif age < 50:
        return "35-49"
    else:
        return "50+"


def compute_bias_metrics(y_true, y_pred, group_labels, group_col):
    """Accuracy, FNR, FPR per demographic group."""
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
        fnr = fn / (fn + tp) if (fn + tp) > 0 else None  # fake missed / all fake
        fpr = fp / (fp + tn) if (fp + tn) > 0 else None  # real flagged / all real
        rows.append({
            group_col:            group,
            "N":                  n_total,
            "Accuracy":           round(acc, 4),
            "FNR (Fake missed)":  round(fnr, 4) if fnr is not None else None,
            "FPR (Real flagged)": round(fpr, 4) if fpr is not None else None,
        })
    return pd.DataFrame(rows)


# test_imgs order matches test_loader and xception_test_loader (both shuffle=False),
# so eff_true[i] / xcp_true[i] both correspond to test_imgs[i].
print("\nRunning DeepFace demographic analysis on test set...", flush=True)
demo_ages, demo_genders = get_demographics(
    test_imgs,
    cache_path="./results/test_demographics.csv",
)
age_groups = [assign_age_group(a) for a in demo_ages]

# Per-model bias tables
eff_gender_df = compute_bias_metrics(eff_true, eff_pred, demo_genders, "Gender")
eff_gender_df.insert(0, "Model", "EfficientNet-B4")

eff_age_df = compute_bias_metrics(eff_true, eff_pred, age_groups, "Age Group")
eff_age_df.insert(0, "Model", "EfficientNet-B4")

xcp_gender_df = compute_bias_metrics(xcp_true, xcp_pred, demo_genders, "Gender")
xcp_gender_df.insert(0, "Model", "XceptionNet")

xcp_age_df = compute_bias_metrics(xcp_true, xcp_pred, age_groups, "Age Group")
xcp_age_df.insert(0, "Model", "XceptionNet")

# Combine both models into one CSV per breakdown type
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
