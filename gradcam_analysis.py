"""
gradcam_analysis.py — Grad-CAM Spatial Attention Visualisation
==============================================================
Generates Gradient-weighted Class Activation Mapping (Grad-CAM) heatmaps
for both EfficientNet-B4 and XceptionNet to visualise which image regions
drive classification decisions.

What is Grad-CAM?
    Grad-CAM (Selvaraju et al., 2017) computes the gradient of the class
    score with respect to the final convolutional feature maps. The resulting
    heatmap highlights which spatial regions contributed most to the model's
    decision — making the "black box" CNN interpretable.

    For deepfake detection, Grad-CAM reveals:
    - True Positives: which GAN artifacts the model detected (e.g., jawline, hairline)
    - False Negatives: why subtle deepfakes were missed (e.g., spatial distraction)
    - False Positives: why real faces were incorrectly flagged

Categories analysed (5 examples each):
    - correct_real:    real face correctly classified as real
    - correct_fake:    fake face correctly classified as fake
    - false_negative:  fake face incorrectly classified as real (missed deepfake)
    - false_positive:  real face incorrectly classified as fake

Target layers:
    - EfficientNet-B4: eff.blocks[-1][-1] — final MBConv block
    - XceptionNet: xcp.act4 — activation after final separable conv block
    These are the deepest convolutional layers before the classifier,
    capturing the highest-level semantic representations.

Note on XceptionNet False Positives:
    XceptionNet produced 0 false positives (FPR = 0.00%) due to model
    collapse. No FP heatmaps are generated — this is documented as a finding.

Usage:
    python gradcam_analysis.py

Prerequisites:
    - predictions_cache.csv must exist (from thesis_train.py)
    - Trained checkpoints in ~/checkpoints/

Outputs:
    - ~/results/gradcam/eff_*.png — EfficientNet heatmaps (20 images)
    - ~/results/gradcam/xcp_*.png — XceptionNet heatmaps (15 images, no FP)

References:
    Selvaraju et al. (2017). Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization. ICCV.

"""

import os
import sys
import random
import csv
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import timm

import matplotlib
matplotlib.use('Agg')   # non-interactive backend — required for cluster (no display)
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────
OUTPUT_DIR      = os.path.expanduser('~/results/gradcam/')
PREDICTIONS_CSV = os.path.expanduser('~/results/predictions_cache.csv')
IMG_SIZE        = 224
RANDOM_SEED     = 42
N_SAMPLES       = 5   # number of examples per category

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalisation — must match training preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# GRAD-CAM IMPLEMENTATION
# ─────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM implementation using PyTorch hooks.

    Registers forward and backward hooks on a target convolutional layer
    to capture activations and gradients during the forward/backward pass.

    Args:
        model: trained PyTorch model
        target_layer: the convolutional layer to attach hooks to
                      (should be the last conv layer for best results)
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        # Forward hook: captures the feature map output of the target layer
        target_layer.register_forward_hook(
            lambda m, i, o: setattr(self, 'activations', o.detach())
        )

        # Backward hook: captures the gradients flowing back through the target layer
        target_layer.register_full_backward_hook(
            lambda m, i, o: setattr(self, 'gradients', o[0].detach())
        )

    def generate(self, tensor):
        """
        Generate a Grad-CAM heatmap for a single input image.

        Algorithm:
            1. Forward pass → get class score
            2. Backward pass → compute gradients w.r.t. target layer
            3. Global average pool gradients → importance weights per channel
            4. Weighted sum of activation maps → class activation map
            5. ReLU → keep only positive contributions
            6. Normalise to [0, 1]

        Args:
            tensor: preprocessed image tensor, shape (1, 3, H, W)

        Returns:
            cam: numpy array of shape (H', W'), normalised to [0, 1]
        """
        self.model.zero_grad()
        out = self.model(tensor)
        out.backward()  # backprop to compute gradients

        # Global average pool gradients across spatial dimensions [C, H, W] → [C]
        weights = self.gradients[0].mean(dim=(1, 2))

        # Weighted sum of activation maps: each channel weighted by its importance
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0][i]

        # ReLU: only keep positive contributions (regions that increase "fake" score)
        cam = torch.relu(cam).cpu().numpy()

        # Normalise to [0, 1] for visualisation
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam


# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────

def save_heatmap(img_path, cam, save_path, title=''):
    """
    Save a Grad-CAM visualisation with three panels:
        1. Original image
        2. Grad-CAM heatmap (jet colormap)
        3. Overlay (heatmap blended with original image)

    Args:
        img_path: path to original image file
        cam: normalised Grad-CAM array (H', W')
        save_path: output path for the PNG file
        title: figure title (model name, prediction type, probability)
    """
    # Load and resize original image
    img = np.array(Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))

    # Resize CAM to match image dimensions
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Panel 1: Original image
    axes[0].imshow(img)
    axes[0].set_title('Original')
    axes[0].axis('off')

    # Panel 2: Grad-CAM heatmap
    axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    # Panel 3: Alpha-blended overlay (50% image + 50% heatmap)
    heatmap = cm.jet(cam_resized)[:, :, :3]   # RGB heatmap
    overlay = np.clip(0.5 * img / 255.0 + 0.5 * heatmap, 0, 1)
    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()


# ─────────────────────────────────────────
# SAMPLE SELECTION
# ─────────────────────────────────────────

def select_samples(rows, pred_col, n=5):
    """
    Select n examples from each of four prediction categories.

    Categories:
        correct_real:    label=0, pred=0 (real face correctly classified)
        correct_fake:    label=1, pred=1 (fake face correctly classified)
        false_negative:  label=1, pred=0 (fake face missed — most important for thesis)
        false_positive:  label=0, pred=1 (real face incorrectly flagged)

    Args:
        rows: list of prediction dicts from predictions_cache.csv
        pred_col: column name for predictions ('eff_pred' or 'xcp_pred')
        n: number of samples per category

    Returns:
        list of (row, category_string) tuples
    """
    random.seed(RANDOM_SEED)

    correct_real   = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==0]
    correct_fake   = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==1]
    false_negative = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==0]
    false_positive = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==1]

    samples  = [(r, 'correct_real')   for r in random.sample(correct_real,   min(n, len(correct_real)))]
    samples += [(r, 'correct_fake')   for r in random.sample(correct_fake,   min(n, len(correct_fake)))]
    samples += [(r, 'false_negative') for r in random.sample(false_negative, min(n, len(false_negative)))]
    samples += [(r, 'false_positive') for r in random.sample(false_positive, min(n, len(false_positive)))]

    return samples


# ─────────────────────────────────────────
# LOAD PREDICTIONS
# ─────────────────────────────────────────
rows = []
with open(PREDICTIONS_CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print(f"Loaded {len(rows):,} predictions from cache.")

# ─────────────────────────────────────────
# EFFICIENTNET GRAD-CAM
# ─────────────────────────────────────────
# Target layer: eff.blocks[-1][-1] — the last MBConv block
# This is the final feature extraction layer before the classifier head,
# capturing the highest-level representations relevant to deepfake detection

print("\nLoading EfficientNet-B4...")
eff = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
eff.load_state_dict(
    torch.load(os.path.expanduser('~/checkpoints/efficientnet_best.pth'), map_location=device)
)
eff = eff.to(device).eval()
gradcam_eff = GradCAM(eff, eff.blocks[-1][-1])

print(f"Generating {N_SAMPLES} heatmaps per category for EfficientNet...")
for i, (row, stype) in enumerate(select_samples(rows, 'eff_pred', n=N_SAMPLES)):
    try:
        img    = Image.open(row['path']).convert('RGB')
        tensor = transform(img).unsqueeze(0).to(device)
        tensor.requires_grad_(True)

        cam   = gradcam_eff.generate(tensor)
        fname = f"eff_{stype}_{i}.png"
        title = (f"EfficientNet | {stype.replace('_',' ').title()} | "
                 f"Label={'Fake' if int(row['label'])==1 else 'Real'} | "
                 f"p(fake)={float(row['eff_prob']):.3f}")
        save_heatmap(row['path'], cam, os.path.join(OUTPUT_DIR, fname), title)
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  Error on {row['path']}: {e}")

# ─────────────────────────────────────────
# XCEPTIONNET GRAD-CAM
# ─────────────────────────────────────────
# Target layer: xcp.act4 — activation after the final separable conv block
# Note: XceptionNet produced 0 false positives due to model collapse,
# so no false_positive heatmaps will be generated for XceptionNet

print("\nLoading XceptionNet...")
for checkpoint_path in [
    os.path.expanduser('~/checkpoints/xceptionnet_best.pth'),
    os.path.expanduser('~/checkpoints/XceptionNet_resume.pth')
]:
    if os.path.exists(checkpoint_path):
        try:
            xcp = timm.create_model('xception', pretrained=False, num_classes=1)
            xcp.load_state_dict(torch.load(checkpoint_path, map_location=device))
            xcp = xcp.to(device).eval()
            gradcam_xcp = GradCAM(xcp, xcp.act4)

            print(f"Generating {N_SAMPLES} heatmaps per category for XceptionNet...")
            for i, (row, stype) in enumerate(select_samples(rows, 'xcp_pred', n=N_SAMPLES)):
                try:
                    img    = Image.open(row['path']).convert('RGB')
                    tensor = transform(img).unsqueeze(0).to(device)
                    tensor.requires_grad_(True)

                    cam   = gradcam_xcp.generate(tensor)
                    fname = f"xcp_{stype}_{i}.png"
                    title = (f"XceptionNet | {stype.replace('_',' ').title()} | "
                             f"Label={'Fake' if int(row['label'])==1 else 'Real'} | "
                             f"p(fake)={float(row['xcp_prob']):.3f}")
                    save_heatmap(row['path'], cam, os.path.join(OUTPUT_DIR, fname), title)
                    print(f"  Saved: {fname}")
                except Exception as e:
                    print(f"  Error on {row['path']}: {e}")
            break

        except Exception as e:
            print(f"XceptionNet load error: {e}")

print(f"\nDone. All heatmaps saved to {OUTPUT_DIR}")
