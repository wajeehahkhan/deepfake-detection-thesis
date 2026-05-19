"""
both_models_utk.py — Out-of-Distribution (OOD) Evaluation on UTKFace
=====================================================================
Tests whether EfficientNet-B4 and XceptionNet generalise to completely
unseen real human faces from the UTKFace dataset.

Motivation:
    The primary test set uses StyleGAN-generated fakes paired with
    CelebA/FFHQ real faces. To determine whether detected demographic
    biases are specific to the training distribution or reflect genuine
    generalised limitations, we evaluate on UTKFace — a dataset the
    models have never seen during training.

Experimental design:
    - All 2,000 UTKFace images are REAL (ground truth label = real)
    - Any prediction of "fake" = False Positive
    - FPR by age group and gender reveals whether bias generalises
    - UTKFace provides ground-truth age and gender labels in filenames
      (unlike the StyleGAN dataset which requires DeepFace estimation)

UTKFace filename format: age_gender_race_timestamp.jpg
    - age: integer (0-116)
    - gender: 0=Male, 1=Female

Dataset: UTKFace (Zhang et al., 2017)
Download: https://susanqq.github.io/UTKFace/

Usage:
    python both_models_utk.py

Prerequisites:
    - UTKFace images in ~/data/utk/UTKFace/
    - Trained checkpoints from thesis_train.py

Outputs:
    - ~/results/both_models_utk_results.csv — per-image predictions + demographics

Author: Wajeeha Khan
Institution: Tilburg University — MSc Data Science & Society 2026
"""

import os
import csv
import random
from collections import defaultdict

import torch
from torchvision import transforms
from PIL import Image
import timm

# ─────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────
# Update UTK_DIR if running on a different machine
UTK_DIR    = os.path.expanduser('~/data/utk/UTKFace/')
OUTPUT_CSV = os.path.expanduser('~/results/both_models_utk_results.csv')

# Sample 2,000 images — sufficient for statistically meaningful FPR estimates
# while keeping compute time manageable
SAMPLE_SIZE = 2000
RANDOM_SEED = 42   # fixed seed for reproducibility
IMG_SIZE    = 224

os.makedirs(os.path.expanduser('~/results'), exist_ok=True)

random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ImageNet normalisation — must match training preprocessing
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ─────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────
# pretrained=False: we load our own fine-tuned weights, not ImageNet weights

print("Loading EfficientNet-B4...")
eff_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
eff_model.load_state_dict(
    torch.load(os.path.expanduser('~/checkpoints/efficientnet_best.pth'), map_location=device)
)
eff_model = eff_model.to(device).eval()
print("EfficientNet loaded.")

# XceptionNet: try multiple checkpoint paths for robustness
xcp_model = None
for checkpoint_path in [
    os.path.expanduser('~/checkpoints/xceptionnet_best.pth'),
    os.path.expanduser('~/checkpoints/XceptionNet_resume.pth')
]:
    if os.path.exists(checkpoint_path):
        try:
            xcp_model = timm.create_model('xception', pretrained=False, num_classes=1)
            xcp_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            xcp_model = xcp_model.to(device).eval()
            print(f"XceptionNet loaded from {checkpoint_path}.")
            break
        except Exception as e:
            print(f"XceptionNet load error: {e}")

if xcp_model is None:
    print("WARNING: XceptionNet checkpoint not found. Only EfficientNet will be evaluated.")

# ─────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────

def parse_filename(fname):
    """
    Extract ground-truth age and gender from UTKFace filename.

    UTKFace naming convention: age_gender_race_timestamp.jpg
    Returns (age, gender) where gender: 0=Male, 1=Female
    Returns (None, None) if parsing fails.
    """
    try:
        parts = fname.split('_')
        age    = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except (ValueError, IndexError):
        return None, None


def get_age_group(age):
    """Map raw age to one of four age groups used throughout the thesis."""
    if age < 20:   return 'Under 20'
    elif age < 35: return '20-34'
    elif age < 50: return '35-49'
    else:          return '50+'

# ─────────────────────────────────────────
# LOAD & FILTER UTKFace IMAGES
# ─────────────────────────────────────────
# Filter to only images with valid age (0-120) and binary gender (0 or 1)
# Some UTKFace filenames are malformed — these are skipped

all_files   = [f for f in os.listdir(UTK_DIR) if f.endswith('.jpg')]
valid_files = [
    f for f in all_files
    if parse_filename(f)[0] is not None
    and 0 <= parse_filename(f)[0] <= 120
    and parse_filename(f)[1] in [0, 1]
]

print(f"Found {len(all_files)} total UTKFace images, {len(valid_files)} with valid labels.")

# Random sample for evaluation
sample = random.sample(valid_files, min(SAMPLE_SIZE, len(valid_files)))
print(f"Running both models on {len(sample)} images...")

# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────
# For each image:
#   - Load and preprocess image
#   - Run through EfficientNet and XceptionNet
#   - Record p(fake) probability and binary prediction
#   - Any prediction of "fake" on a real UTKFace image = False Positive

results = []
errors  = 0

with torch.no_grad():
    for i, fname in enumerate(sample):
        if i % 200 == 0:
            print(f"  {i}/{len(sample)}...")

        gt_age, gt_gender = parse_filename(fname)

        try:
            img    = Image.open(os.path.join(UTK_DIR, fname)).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)

            # EfficientNet prediction
            eff_prob = torch.sigmoid(eff_model(tensor)).item()
            eff_pred = 1 if eff_prob > 0.5 else 0  # 1=fake, 0=real

            # XceptionNet prediction (if loaded)
            xcp_prob = xcp_pred = xcp_fp = None
            if xcp_model is not None:
                xcp_prob = torch.sigmoid(xcp_model(tensor)).item()
                xcp_pred = 1 if xcp_prob > 0.5 else 0
                xcp_fp   = 1 if xcp_pred == 1 else 0  # FP since all images are real

            results.append({
                'filename':         fname,
                'gt_age':           gt_age,
                'gt_gender':        gt_gender,
                'gender_str':       'Man' if gt_gender == 0 else 'Woman',
                'age_group':        get_age_group(gt_age),
                'eff_prob_fake':    eff_prob,
                'eff_pred':         eff_pred,
                'eff_false_positive': 1 if eff_pred == 1 else 0,  # all real → any fake = FP
                'xcp_prob_fake':    xcp_prob,
                'xcp_pred':         xcp_pred,
                'xcp_false_positive': xcp_fp,
            })

        except Exception as e:
            errors += 1
            continue

print(f"\nDone. Processed {len(results):,}, skipped {errors} errors.")

# ─────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────

age_order = ['Under 20', '20-34', '35-49', '50+']

print("\n=== EfficientNet FPR by Age Group ===")
age_eff = defaultdict(list)
for r in results:
    age_eff[r['age_group']].append(r['eff_false_positive'])
for g in age_order:
    v = age_eff[g]
    if v:
        print(f"  {g}: {sum(v)/len(v)*100:.2f}% FPR (n={len(v)})")

print("\n=== EfficientNet FPR by Gender ===")
gen_eff = defaultdict(list)
for r in results:
    gen_eff[r['gender_str']].append(r['eff_false_positive'])
for g in ['Man', 'Woman']:
    v = gen_eff[g]
    if v:
        print(f"  {g}: {sum(v)/len(v)*100:.2f}% FPR (n={len(v)})")

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────
fieldnames = [
    'filename', 'gt_age', 'gt_gender', 'gender_str', 'age_group',
    'eff_prob_fake', 'eff_pred', 'eff_false_positive',
    'xcp_prob_fake', 'xcp_pred', 'xcp_false_positive'
]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nResults saved to {OUTPUT_CSV}")
