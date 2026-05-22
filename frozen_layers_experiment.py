"""
frozen_layers_experiment.py — Frozen Layers Training (EfficientNet-B4)
=======================================================================
Investigates the effect of transfer learning depth on EfficientNet-B4
by freezing early convolutional layers and only training the last 2
blocks + classifier head.

Motivation:
    Full fine-tuning of XceptionNet led to model collapse (see thesis_train.py).
    This experiment tests whether restricting trainable parameters affects
    EfficientNet-B4 similarly or differently addresses the question of
    whether catastrophic forgetting is architecture-specific.

Frozen layers strategy:
    - ALL layers frozen except: blocks 5, blocks.6, conv_head, bn2, classifier
    - This leaves 79.4% of parameters trainable (13.9M / 17.6M)
    - Early layers retain ImageNet feature representations
    - Only task-specific higher-level features are adapted

Usage:
    python frozen_layers_experiment.py

Outputs:
    - ./results/frozen_layers_results.csv   — val metrics per epoch + test results
    - ./checkpoints/efficientnet_frozen_best.pth — best model checkpoint


"""

import os
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import numpy as np

# ─────────────────────────────────────────
# PATHS & HYPERPARAMETERS
# ─────────────────────────────────────────
# Update DATA_DIR if running on a different machine
DATA_DIR   = os.path.expanduser('~/data/real_vs_fake/real-vs-fake')
TRAIN_DIR  = os.path.join(DATA_DIR, 'train')
VAL_DIR    = os.path.join(DATA_DIR, 'valid')
TEST_DIR   = os.path.join(DATA_DIR, 'test')
OUTPUT_CSV = os.path.expanduser('~/results/frozen_layers_results.csv')
CHECKPOINT = os.path.expanduser('~/checkpoints/efficientnet_frozen_best.pth')

os.makedirs(os.path.expanduser('~/results'), exist_ok=True)
os.makedirs(os.path.expanduser('~/checkpoints'), exist_ok=True)

# Hyperparameters — identical to full fine-tuning for fair comparison
IMG_SIZE     = 224
BATCH_SIZE   = 32
LR           = 1e-4   # standard learning rate for fine-tuning pretrained CNNs
WEIGHT_DECAY = 1e-4   # L2 regularisation
EPOCHS       = 10     # max epochs — early stopping typically triggers before this
PATIENCE     = 3      # stop if no AUC improvement for 3 consecutive epochs
RANDOM_SEED  = 42

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
# Uses torchvision ImageFolder which expects:
#   train/real/, train/fake/, valid/real/, valid/fake/, test/real/, test/fake/
# ImageNet normalisation required since model uses ImageNet pretrained weights

train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),   # basic augmentation to prevent overfitting
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading datasets...")
train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_transform)
val_ds   = datasets.ImageFolder(VAL_DIR,   transform=val_transform)
test_ds  = datasets.ImageFolder(TEST_DIR,  transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}")

# ─────────────────────────────────────────
# MODEL — FROZEN LAYERS SETUP
# ─────────────────────────────────────────
# Step 1: Load EfficientNet-B4 with ImageNet pretrained weights
# Step 2: Freeze ALL parameters
# Step 3: Selectively unfreeze last 2 blocks + classifier head
#
# Design choice — why last 2 blocks?
#   Early layers (blocks.0-4) capture low-level features (edges, textures)
#   that transfer well from ImageNet. Later layers capture high-level
#   task-specific features. Freezing early layers preserves the
#   generalizable representations while allowing task adaptation.
#
# EfficientNet-B4 block structure:
#   blocks.0 → blocks.6 (7 MBConv blocks, increasing complexity)
#   conv_head → final conv before classifier
#   bn2 → batch norm after conv_head
#   classifier → final linear layer (num_classes=1 for binary)

print("Loading EfficientNet-B4 with ImageNet weights...")
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
model = model.to(device)

# Freeze all parameters first
print("Freezing layers...")
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze last 2 blocks and classifier head
UNFREEZE_LAYERS = ['blocks.6', 'blocks.5', 'conv_head', 'bn2', 'classifier']
for name, param in model.named_parameters():
    if any(x in name for x in UNFREEZE_LAYERS):
        param.requires_grad = True

# Report parameter counts
total     = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

# ─────────────────────────────────────────
# OPTIMISER & LOSS
# ─────────────────────────────────────────
# filter(requires_grad) ensures frozen parameters don't receive gradient updates
optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)

# Halve LR if val AUC doesn't improve for 2 epochs
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)

# BCEWithLogitsLoss: numerically stable sigmoid + binary cross-entropy
criterion = nn.BCEWithLogitsLoss()

# ─────────────────────────────────────────
# TRAINING LOOP
# ─────────────────────────────────────────
best_auc        = 0
patience_counter = 0
history         = []

print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):

    # --- Training phase ---
    model.train()
    train_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        optimizer.zero_grad()
        outputs = model(imgs).squeeze(1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # --- Validation phase ---
    model.eval()
    all_labels, all_probs = [], []
    val_loss = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.float().to(device)
            outputs = model(imgs).squeeze(1)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    all_preds = (np.array(all_probs) > 0.5).astype(int)
    val_acc   = (np.array(all_preds) == np.array(all_labels)).mean()
    val_f1    = f1_score(all_labels, all_preds)
    val_auc   = roc_auc_score(all_labels, all_probs)

    scheduler.step(val_auc)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {val_loss/len(val_loader):.4f} | "
          f"Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.6f}")

    history.append({'epoch': epoch, 'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc})

    # Save best checkpoint
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  Saved best model (AUC: {best_auc:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch} (no improvement for {PATIENCE} epochs)")
            break

# ─────────────────────────────────────────
# TEST SET EVALUATION
# ─────────────────────────────────────────
# Load best checkpoint (not last epoch) for final evaluation
print("\nEvaluating best checkpoint on test set...")
model.load_state_dict(torch.load(CHECKPOINT, map_location=device))
model.eval()
all_labels, all_probs = [], []
with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.float().to(device)
        outputs = model(imgs).squeeze(1)
        probs = torch.sigmoid(outputs).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

all_preds = (np.array(all_probs) > 0.5).astype(int)
test_acc  = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1   = f1_score(all_labels, all_preds)
test_auc  = roc_auc_score(all_labels, all_probs)
cm        = confusion_matrix(all_labels, all_preds)
tn, fp, fn, tp = cm.ravel()

print(f"\n=== EFFICIENTNET-B4 FROZEN LAYERS — TEST RESULTS ===")
print(f"Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)")
print(f"F1-score:  {test_f1:.4f}")
print(f"AUC-ROC:   {test_auc:.6f}")
print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
print(f"FNR: {fn/(fn+tp)*100:.2f}% | FPR: {fp/(fp+tn)*100:.2f}%")
print(f"Confusion Matrix:\n{cm}")

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch', 'val_acc', 'val_f1', 'val_auc'])
    writer.writeheader()
    writer.writerows(history)
    # Append test results as final row
    writer.writerow({
        'epoch': 'TEST',
        'val_acc': test_acc,
        'val_f1': test_f1,
        'val_auc': test_auc
    })

print(f"\nResults saved to {OUTPUT_CSV}")
print(f"Best checkpoint saved to {CHECKPOINT}")
