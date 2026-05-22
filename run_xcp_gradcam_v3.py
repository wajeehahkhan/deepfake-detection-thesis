"""
run_xcp_gradcam_v3.py — XceptionNet Grad-CAM (Fixed Version)
=============================================================
Alternative XceptionNet Grad-CAM implementation that resolves the
"inplace view" error that occurs when using register_full_backward_hook
on XceptionNet's act4 layer.

Why this separate script exists:
    The main gradcam_analysis.py uses register_full_backward_hook which
    causes a PyTorch error with XceptionNet's architecture due to how
    depthwise separable convolutions handle gradient computation.

    This version uses register_backward_hook (non-full) on a different
    target layer (xcp.block4.rep[-1] instead of xcp.act4), which avoids
    the inplace modification error.

    Note: register_backward_hook is deprecated in newer PyTorch versions
    but remains functional. A FutureWarning is expected — this is harmless.

Target layer change:
    Original attempt: xcp.act4 (activation after final conv) — caused error
    Fixed version: xcp.block4.rep[-1] (last conv in block4) — works correctly

Device note:
    This script forces CPU inference (device = 'cpu') because the
    backward hook issue was more reliably avoided on CPU in our testing.

Usage:
    python run_xcp_gradcam_v3.py

Prerequisites:
    - predictions_cache.csv must exist (from thesis_train.py)
    - ~/checkpoints/xceptionnet_best.pth must exist

Outputs:
    - ~/results/gradcam/xcp_*.png — XceptionNet Grad-CAM heatmaps

"""

import os
import csv
import random
import numpy as np

import torch
from torchvision import transforms
from PIL import Image
import timm

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# ─────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────
OUTPUT_DIR      = os.path.expanduser('~/results/gradcam/')
PREDICTIONS_CSV = os.path.expanduser('~/results/predictions_cache.csv')
IMG_SIZE        = 224
RANDOM_SEED     = 42
N_SAMPLES       = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Force CPU — avoids backward hook issues on this cluster's GPU driver
device = torch.device('cpu')

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# GRAD-CAM (FIXED IMPLEMENTATION)
# ─────────────────────────────────────────

class GradCAM:
    """
    Grad-CAM using non-full backward hook.

    Uses register_backward_hook instead of register_full_backward_hook
    to avoid the inplace view error in XceptionNet's architecture.

    The .clone() calls in hooks ensure gradients/activations are not
    modified inplace by subsequent operations.
    """

    def __init__(self, model, target_layer):
        self.model       = model
        self.gradients   = None
        self.activations = None

        target_layer.register_forward_hook(self._forward_hook)
        target_layer.register_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Capture and clone feature map activations."""
        self.activations = output.detach().clone()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture and clone gradients — clone prevents inplace modification."""
        self.gradients = grad_output[0].detach().clone()

    def generate(self, tensor):
        """Generate normalised Grad-CAM heatmap for input tensor."""
        self.model.zero_grad()
        out = self.model(tensor)
        out.backward()

        weights = self.gradients[0].mean(dim=(1, 2))
        cam     = torch.zeros(self.activations.shape[2:])
        for i, w in enumerate(weights):
            cam += w * self.activations[0][i]

        cam = torch.relu(cam).numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ─────────────────────────────────────────
# VISUALISATION
# ─────────────────────────────────────────

def save_heatmap(img_path, cam, save_path, title=''):
    """Save 3-panel Grad-CAM visualisation: original | heatmap | overlay."""
    img   = np.array(Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    cam_r = np.array(
        Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))
    ) / 255.0

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img);     axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cam_r, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    overlay = np.clip(0.5 * img / 255.0 + 0.5 * cm.jet(cam_r)[:, :, :3], 0, 1)
    axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')

    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()


# ─────────────────────────────────────────
# SAMPLE SELECTION
# ─────────────────────────────────────────

def select_samples(rows, pred_col, n=5):
    """Select n examples from each prediction category."""
    random.seed(RANDOM_SEED)
    cr = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==0]
    cf = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==1]
    fn = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==0]
    fp = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==1]
    s  = [(r, 'correct_real')   for r in random.sample(cr, min(n, len(cr)))]
    s += [(r, 'correct_fake')   for r in random.sample(cf, min(n, len(cf)))]
    s += [(r, 'false_negative') for r in random.sample(fn, min(n, len(fn)))]
    s += [(r, 'false_positive') for r in random.sample(fp, min(n, len(fp)))]
    return s


# ─────────────────────────────────────────
# LOAD PREDICTIONS & RUN
# ─────────────────────────────────────────
rows = []
with open(PREDICTIONS_CSV) as f:
    for r in csv.DictReader(f):
        rows.append(r)

print("Loading XceptionNet...")
xcp = timm.create_model('xception', pretrained=False, num_classes=1)
xcp.load_state_dict(
    torch.load(os.path.expanduser('~/checkpoints/xceptionnet_best.pth'), map_location='cpu')
)
xcp = xcp.eval()

# Target layer: xcp.block4.rep[-1]
# block4 is the 4th Xception block in the middle flow
# rep[-1] is the last depthwise separable conv in that block
# This was chosen after xcp.act4 caused inplace hook errors
target_layer = xcp.block4.rep[-1]
gradcam_xcp  = GradCAM(xcp, target_layer)
print("GradCAM ready, running samples...")

for i, (row, stype) in enumerate(select_samples(rows, 'xcp_pred', n=N_SAMPLES)):
    try:
        img    = Image.open(row['path']).convert('RGB')
        tensor = transform(img).unsqueeze(0)
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

print("Done.")
