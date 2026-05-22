"""
deepface_utk_validation.py — DeepFace Demographic Label Validation
===================================================================
Validates the reliability of DeepFace demographic predictions against
ground-truth labels from the UTKFace dataset.

Purpose:
    The primary StyleGAN dataset lacks ground-truth demographic metadata.
    To analyse demographic bias, we use DeepFace to estimate age and gender
    for each test image. However, DeepFace predictions are not perfect —
    this script quantifies their accuracy so bias findings can be interpreted
    with appropriate uncertainty bounds.

    This is a mandatory methodological step for rigorous algorithmic fairness
    research (Buolamwini & Gebru, 2018): demographic analysis based on
    estimated labels must explicitly acknowledge label noise.

What this script does:
    1. Loads 2,000 UTKFace images with ground-truth age/gender in filenames
    2. Runs DeepFace age and gender estimation on each image
    3. Compares predictions against ground truth
    4. Reports Mean Age Error (MAE) and gender classification accuracy

Results from our validation:
    - Mean Age Error (MAE): 12.21 years
    - Gender Accuracy: 79.10%
    → ~1 in 5 gender labels may be incorrect
    → Age cohort assignments have overlapping margins of error

UTKFace filename format: age_gender_race_timestamp.jpg
    - age: integer (0-116)
    - gender: 0=Male, 1=Female

Dataset: UTKFace (Zhang et al., 2017)
Download: https://susanqq.github.io/UTKFace/

Usage:
    python deepface_utk_validation.py

Outputs:
    - ~/results/deepface_utk_validation.csv — per-image predictions vs ground truth


"""

import os
import csv
import sys
import random

# ─────────────────────────────────────────
# PATHS & SETTINGS
# ─────────────────────────────────────────
UTK_DIR    = os.path.expanduser('~/data/utk/UTKFace/')
OUTPUT_CSV = os.path.expanduser('~/results/deepface_utk_validation.csv')

# 2,000 images is sufficient for robust validation statistics
# while being much faster than running all 23,000 UTKFace images
SAMPLE_SIZE = 2000
RANDOM_SEED = 42

os.makedirs(os.path.expanduser('~/results'), exist_ok=True)

# ─────────────────────────────────────────
# IMPORT DEEPFACE
# ─────────────────────────────────────────
# DeepFace requires TensorFlow — if not installed, exit with helpful message
try:
    from deepface import DeepFace
    print("DeepFace imported successfully.")
except ImportError as e:
    print(f"DeepFace not available: {e}")
    print("Install with: pip install deepface tf-keras")
    sys.exit(1)

# ─────────────────────────────────────────
# LOAD & FILTER UTKFace IMAGES
# ─────────────────────────────────────────

def parse_filename(fname):
    """
    Extract ground-truth age and gender from UTKFace filename.

    UTKFace naming convention: age_gender_race_timestamp.jpg
    Some files have malformed names — these return (None, None).

    Args:
        fname: filename string

    Returns:
        (age, gender) tuple where gender: 0=Male, 1=Female
        (None, None) if parsing fails
    """
    try:
        parts  = fname.split('_')
        age    = int(parts[0])
        gender = int(parts[1])
        return age, gender
    except (ValueError, IndexError):
        return None, None


all_files = [f for f in os.listdir(UTK_DIR) if f.endswith('.jpg')]
print(f"Found {len(all_files)} images in UTKFace.")

# Filter to images with valid age (0-120) and binary gender label
valid_files = [
    f for f in all_files
    if parse_filename(f)[0] is not None
    and 0 <= parse_filename(f)[0] <= 120
    and parse_filename(f)[1] in [0, 1]
]
print(f"{len(valid_files)} files with valid ground truth labels.")

# Random sample
random.seed(RANDOM_SEED)
sample = random.sample(valid_files, min(SAMPLE_SIZE, len(valid_files)))
print(f"Running DeepFace on {len(sample)} images...")

# ─────────────────────────────────────────
# RUN DEEPFACE VALIDATION
# ─────────────────────────────────────────
# enforce_detection=False: don't fail if face detection is uncertain
# silent=True: suppress verbose DeepFace logging
# actions=['age', 'gender']: only estimate what we need (faster than all actions)

results = []
errors  = 0

for i, fname in enumerate(sample):
    if i % 100 == 0:
        print(f"  {i}/{len(sample)}...")

    gt_age, gt_gender = parse_filename(fname)
    img_path = os.path.join(UTK_DIR, fname)

    try:
        analysis = DeepFace.analyze(
            img_path=img_path,
            actions=['age', 'gender'],
            enforce_detection=False,
            silent=True
        )

        # DeepFace returns a list when multiple faces detected — take first
        if isinstance(analysis, list):
            analysis = analysis[0]

        pred_age        = analysis['age']
        pred_gender_str = analysis['dominant_gender']           # 'Man' or 'Woman'
        pred_gender     = 0 if pred_gender_str == 'Man' else 1  # convert to 0/1

        age_error      = abs(pred_age - gt_age)
        gender_correct = 1 if pred_gender == gt_gender else 0

        results.append({
            'filename':       fname,
            'gt_age':         gt_age,
            'gt_gender':      gt_gender,        # 0=Male, 1=Female (ground truth)
            'pred_age':       pred_age,
            'pred_gender':    pred_gender,
            'pred_gender_str': pred_gender_str,
            'age_error':      age_error,
            'gender_correct': gender_correct,
        })

    except Exception as e:
        # Skip images where DeepFace fails entirely (e.g. no face detected)
        errors += 1
        continue

print(f"\nDone. Processed {len(results):,}, skipped {errors} errors.")

# ─────────────────────────────────────────
# SUMMARY STATISTICS
# ─────────────────────────────────────────

age_errors  = [r['age_error'] for r in results]
mae         = sum(age_errors) / len(age_errors)
gender_acc  = sum(r['gender_correct'] for r in results) / len(results) * 100

print(f"\n=== DeepFace Validation Results ===")
print(f"Validation set:     {len(results)} UTKFace images")
print(f"Mean Age Error (MAE): {mae:.2f} years")
print(f"Gender Accuracy:      {gender_acc:.2f}%")
print(f"\nInterpretation:")
print(f"  ~1 in {100//(100-gender_acc+1)} gender labels may be incorrect")
print(f"  Age estimates have a mean error of {mae:.1f} years")
print(f"  These limitations are acknowledged in the bias analysis (Chapter 4)")

# ─────────────────────────────────────────
# SAVE RESULTS
# ─────────────────────────────────────────
fieldnames = [
    'filename', 'gt_age', 'gt_gender', 'pred_age', 'pred_gender',
    'pred_gender_str', 'age_error', 'gender_correct'
]

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"\nSaved to {OUTPUT_CSV}")
