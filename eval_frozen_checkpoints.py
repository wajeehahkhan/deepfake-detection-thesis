"""
eval_frozen_checkpoints.py — Evaluate Frozen Layer Checkpoints
===============================================================
Loads the best saved checkpoints from the frozen layers experiments
and evaluates them on the held-out test set.

This script is separate from the training scripts so that evaluation
can be re-run independently without retraining. It also allows
evaluation of partially-trained checkpoints (e.g. if training was
interrupted before completing all epochs).

Results reported:
    - Accuracy, F1-score, AUC-ROC
    - Confusion matrix (TP, FP, FN, TN)
    - False Negative Rate (FNR): proportion of fakes missed
    - False Positive Rate (FPR): proportion of real faces falsely flagged

Usage:
    python eval_frozen_checkpoints.py

Prerequisites:
    - Run frozen_layers_experiment.py first to generate:
        ./checkpoints/efficientnet_frozen_best.pth
    - Run frozen_layers_xception.py first to generate:
        ./checkpoints/xception_frozen_best.pth

Author: Wajeeha Khan
Institution: Tilburg University — MSc Data Science & Society 2026
"""

import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
# Update these paths if running on a different machine
TEST_DIR       = os.path.expanduser('~/data/real_vs_fake/real-vs-fake/test')
EFF_CHECKPOINT = os.path.expanduser('~/checkpoints/efficientnet_frozen_best.pth')
XCP_CHECKPOINT = os.path.expanduser('~/checkpoints/xception_frozen_best.pth')

IMG_SIZE   = 224
BATCH_SIZE = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# ─────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────
# No augmentation for evaluation — only resize and normalise
# ImageNet normalisation required to match training preprocessing

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_ds     = datasets.ImageFolder(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Test set: {len(test_ds):,} images")

# ─────────────────────────────────────────
# EVALUATION FUNCTION
# ─────────────────────────────────────────

def evaluate(model, loader, name):
    """
    Evaluate a model on the test set and print all metrics.

    Args:
        model: trained PyTorch model
        loader: DataLoader for test set
        name: model name string for display

    Returns:
        tuple of (accuracy, f1, auc)
    """
    model.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze(1)
            # Convert raw logits to probabilities using sigmoid
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    # Threshold at 0.5: prob > 0.5 → predicted fake (1), else real (0)
    all_preds = (np.array(all_probs) > 0.5).astype(int)

    acc = accuracy_score(all_labels, all_preds)
    f1  = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm  = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} FROZEN — TEST RESULTS ===")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.6f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"FNR (fakes missed):      {fn/(fn+tp)*100:.2f}%")
    print(f"FPR (real faces flagged): {fp/(fp+tn)*100:.2f}%")

    return acc, f1, auc

# ─────────────────────────────────────────
# EVALUATE EFFICIENTNET FROZEN
# ─────────────────────────────────────────
# pretrained=False because we load our own fine-tuned weights,
# not the original ImageNet weights
if os.path.exists(EFF_CHECKPOINT):
    print("\nLoading EfficientNet-B4 frozen checkpoint...")
    eff_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    eff_model.load_state_dict(torch.load(EFF_CHECKPOINT, map_location=device))
    eff_model = eff_model.to(device)
    evaluate(eff_model, test_loader, 'EfficientNet-B4')
else:
    print(f"EfficientNet checkpoint not found at {EFF_CHECKPOINT}")
    print("Run frozen_layers_experiment.py first to generate the checkpoint.")

# ─────────────────────────────────────────
# EVALUATE XCEPTIONNET FROZEN
# ─────────────────────────────────────────
if os.path.exists(XCP_CHECKPOINT):
    print("\nLoading XceptionNet frozen checkpoint...")
    xcp_model = timm.create_model('xception', pretrained=False, num_classes=1)
    xcp_model.load_state_dict(torch.load(XCP_CHECKPOINT, map_location=device))
    xcp_model = xcp_model.to(device)
    evaluate(xcp_model, test_loader, 'XceptionNet')
else:
    print(f"XceptionNet checkpoint not found at {XCP_CHECKPOINT}")
    print("Run frozen_layers_xception.py first to generate the checkpoint.")

print("\nDone.")
