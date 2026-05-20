"""
plot_combined_roc.py — Combined ROC Curve for Both Models
=========================================================
Generates a single ROC curve plot comparing EfficientNet-B4 and
XceptionNet on the same axes, allowing direct visual comparison
of their discriminative performance.

The ROC curve plots True Positive Rate (sensitivity) against False
Positive Rate (1-specificity) at all possible classification thresholds.
AUC-ROC summarises this into a single number:
    - AUC = 1.0: perfect classifier
    - AUC = 0.5: random classifier (diagonal line)

Key finding:
    Despite XceptionNet's poor accuracy (71.72%) due to model collapse,
    its AUC of 0.9839 shows it has genuine discriminative ability —
    the threshold (0.5) is simply miscalibrated, not the underlying model.

Usage:
    python plot_combined_roc.py

Prerequisites:
    - predictions_cache.csv must exist (from thesis_train.py)

Outputs:
    - ~/results/combined_roc_curve.png

Author: Wajeeha Khan
Institution: Tilburg University — MSc Data Science & Society 2026
"""

import os
import csv
import numpy as np

import matplotlib
matplotlib.use('Agg')   # non-interactive backend for cluster
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# ─────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────
PREDICTIONS_CSV = os.path.expanduser('~/results/predictions_cache.csv')
OUTPUT          = os.path.expanduser('~/results/combined_roc_curve.png')

os.makedirs(os.path.expanduser('~/results'), exist_ok=True)

# ─────────────────────────────────────────
# LOAD PREDICTIONS
# ─────────────────────────────────────────
# predictions_cache.csv contains one row per test image with:
# path, label, eff_pred, xcp_pred, eff_prob, xcp_prob

print("Loading predictions...")
labels, eff_probs, xcp_probs = [], [], []

with open(PREDICTIONS_CSV) as f:
    for row in csv.DictReader(f):
        labels.append(int(row['label']))
        eff_probs.append(float(row['eff_prob']))
        xcp_probs.append(float(row['xcp_prob']))

print(f"Loaded {len(labels):,} predictions.")

# ─────────────────────────────────────────
# COMPUTE ROC CURVES
# ─────────────────────────────────────────
# roc_curve returns (fpr, tpr, thresholds) at all decision thresholds
# auc computes the area under the curve using the trapezoidal rule

eff_fpr, eff_tpr, _ = roc_curve(labels, eff_probs)
xcp_fpr, xcp_tpr, _ = roc_curve(labels, xcp_probs)
eff_auc = auc(eff_fpr, eff_tpr)
xcp_auc = auc(xcp_fpr, xcp_tpr)

print(f"EfficientNet-B4 AUC: {eff_auc:.6f}")
print(f"XceptionNet AUC:     {xcp_auc:.6f}")

# ─────────────────────────────────────────
# PLOT
# ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 7))

# EfficientNet curve — teal
ax.plot(eff_fpr, eff_tpr,
        color='#2a9d8f', lw=2.5,
        label=f'EfficientNet-B4 (AUC = {eff_auc:.6f})')

# XceptionNet curve — orange
ax.plot(xcp_fpr, xcp_tpr,
        color='#e76f51', lw=2.5,
        label=f'XceptionNet (AUC = {xcp_auc:.6f})')

# Random classifier baseline (diagonal)
ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random classifier')

ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — EfficientNet-B4 vs XceptionNet', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT}")
