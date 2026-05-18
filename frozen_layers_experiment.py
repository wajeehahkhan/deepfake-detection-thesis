"""
Frozen Layers Experiment
- Freezes all EfficientNet-B4 layers except last 2 blocks + classifier
- Trains for up to 10 epochs with early stopping
- Saves results to ~/results/frozen_layers_results.csv
- Saves best model to ~/checkpoints/efficientnet_frozen_best.pth
"""

import os, csv, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix
import numpy as np

# Paths
TRAIN_DIR = os.path.expanduser('~/data/real_vs_fake/real-vs-fake/train')
VAL_DIR = os.path.expanduser('~/data/real_vs_fake/real-vs-fake/valid')
TEST_DIR = os.path.expanduser('~/data/real_vs_fake/real-vs-fake/test')
OUTPUT_CSV = os.path.expanduser('~/results/frozen_layers_results.csv')
CHECKPOINT = os.path.expanduser('~/checkpoints/efficientnet_frozen_best.pth')

# Hyperparameters — same as original
IMG_SIZE = 224
BATCH_SIZE = 32
LR = 1e-4
WEIGHT_DECAY = 1e-4
EPOCHS = 10
PATIENCE = 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Transforms — same as original
train_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
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
val_ds = datasets.ImageFolder(VAL_DIR, transform=val_transform)
test_ds = datasets.ImageFolder(TEST_DIR, transform=val_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

# Load model
print("Loading EfficientNet-B4 with ImageNet weights...")
model = timm.create_model('efficientnet_b4', pretrained=True, num_classes=1)
model = model.to(device)

# FREEZE all layers except last 2 blocks + classifier
print("Freezing layers...")
for name, param in model.named_parameters():
    param.requires_grad = False

# Unfreeze last 2 blocks and classifier
for name, param in model.named_parameters():
    if any(x in name for x in ['blocks.6', 'blocks.5', 'conv_head', 'bn2', 'classifier']):
        param.requires_grad = True

# Report frozen vs trainable
total = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total params: {total:,} | Trainable: {trainable:,} ({100*trainable/total:.1f}%)")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='max', factor=0.5, patience=2
)
criterion = nn.BCEWithLogitsLoss()

# Training loop
best_auc = 0
patience_counter = 0
history = []

print("\nStarting training...")
for epoch in range(1, EPOCHS + 1):
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

    # Validation
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
    val_acc = (np.array(all_preds) == np.array(all_labels)).mean()
    val_f1 = f1_score(all_labels, all_preds)
    val_auc = roc_auc_score(all_labels, all_probs)

    scheduler.step(val_auc)
    print(f"Epoch {epoch}/{EPOCHS} | Loss: {val_loss/len(val_loader):.4f} | Acc: {val_acc:.4f} | F1: {val_f1:.4f} | AUC: {val_auc:.6f}")

    history.append({'epoch': epoch, 'val_acc': val_acc, 'val_f1': val_f1, 'val_auc': val_auc})

    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model.state_dict(), CHECKPOINT)
        print(f"  Saved best model (AUC: {best_auc:.6f})")
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print(f"Early stopping at epoch {epoch}")
            break

# Test evaluation
print("\nEvaluating on test set...")
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
test_acc = (np.array(all_preds) == np.array(all_labels)).mean()
test_f1 = f1_score(all_labels, all_preds)
test_auc = roc_auc_score(all_labels, all_probs)
cm = confusion_matrix(all_labels, all_preds)

print(f"\n=== FROZEN LAYERS TEST RESULTS ===")
print(f"Accuracy: {test_acc:.4f}")
print(f"F1: {test_f1:.4f}")
print(f"AUC: {test_auc:.6f}")
print(f"Confusion Matrix:\n{cm}")

# Save results
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['epoch','val_acc','val_f1','val_auc'])
    writer.writeheader()
    writer.writerows(history)
    writer.writerow({'epoch': 'TEST', 'val_acc': test_acc, 'val_f1': test_f1, 'val_auc': test_auc})

print(f"\nSaved to {OUTPUT_CSV}")
