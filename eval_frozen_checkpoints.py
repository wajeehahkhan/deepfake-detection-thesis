"""
Evaluate frozen layer checkpoints on test set
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import timm
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, accuracy_score
import numpy as np

TEST_DIR = os.path.expanduser('~/data/real_vs_fake/real-vs-fake/test')
EFF_CHECKPOINT = os.path.expanduser('~/checkpoints/efficientnet_frozen_best.pth')
XCP_CHECKPOINT = os.path.expanduser('~/checkpoints/xception_frozen_best.pth')

IMG_SIZE = 224
BATCH_SIZE = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

val_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_ds = datasets.ImageFolder(TEST_DIR, transform=val_transform)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
print(f"Test set: {len(test_ds)} images")

def evaluate(model, loader, name):
    model.eval()
    all_labels, all_probs = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs).squeeze(1)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

    all_preds = (np.array(all_probs) > 0.5).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)
    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n=== {name} FROZEN — TEST RESULTS ===")
    print(f"Accuracy:  {acc:.4f} ({acc*100:.2f}%)")
    print(f"F1-score:  {f1:.4f}")
    print(f"AUC-ROC:   {auc:.6f}")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"FNR: {fn/(fn+tp)*100:.2f}% | FPR: {fp/(fp+tn)*100:.2f}%")
    return acc, f1, auc

# EfficientNet
if os.path.exists(EFF_CHECKPOINT):
    print("\nLoading EfficientNet frozen checkpoint...")
    eff_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
    eff_model.load_state_dict(torch.load(EFF_CHECKPOINT, map_location=device))
    eff_model = eff_model.to(device)
    evaluate(eff_model, test_loader, 'EfficientNet-B4')
else:
    print(f"EfficientNet checkpoint not found at {EFF_CHECKPOINT}")

# XceptionNet
if os.path.exists(XCP_CHECKPOINT):
    print("\nLoading XceptionNet frozen checkpoint...")
    xcp_model = timm.create_model('xception', pretrained=False, num_classes=1)
    xcp_model.load_state_dict(torch.load(XCP_CHECKPOINT, map_location=device))
    xcp_model = xcp_model.to(device)
    evaluate(xcp_model, test_loader, 'XceptionNet')
else:
    print(f"XceptionNet checkpoint not found at {XCP_CHECKPOINT}")

print("\nDone.")
