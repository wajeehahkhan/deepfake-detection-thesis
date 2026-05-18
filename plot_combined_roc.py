import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import csv

PREDICTIONS_CSV = '/home/u773837/results/predictions_cache.csv'
OUTPUT = '/home/u773837/results/combined_roc_curve.png'

print("Loading predictions...")
labels, eff_probs, xcp_probs = [], [], []
with open(PREDICTIONS_CSV) as f:
    for row in csv.DictReader(f):
        labels.append(int(row['label']))
        eff_probs.append(float(row['eff_prob']))
        xcp_probs.append(float(row['xcp_prob']))

eff_fpr, eff_tpr, _ = roc_curve(labels, eff_probs)
xcp_fpr, xcp_tpr, _ = roc_curve(labels, xcp_probs)
eff_auc = auc(eff_fpr, eff_tpr)
xcp_auc = auc(xcp_fpr, xcp_tpr)

fig, ax = plt.subplots(figsize=(8, 7))
ax.plot(eff_fpr, eff_tpr, color='#2a9d8f', lw=2.5, label=f'EfficientNet-B4 (AUC = {eff_auc:.6f})')
ax.plot(xcp_fpr, xcp_tpr, color='#e76f51', lw=2.5, label=f'XceptionNet (AUC = {xcp_auc:.6f})')
ax.plot([0,1],[0,1], 'k--', lw=1, alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=13)
ax.set_ylabel('True Positive Rate', fontsize=13)
ax.set_title('ROC Curve — EfficientNet-B4 vs XceptionNet', fontsize=14, fontweight='bold')
ax.legend(fontsize=12)
ax.set_facecolor('#f9f9f9')
fig.patch.set_facecolor('white')
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight')
print(f"Saved to {OUTPUT}")
