"""
DeepFace Validation on UTKFace
- Parses ground truth age/gender from filenames
- Runs DeepFace on each image
- Compares predictions vs ground truth
- Saves results to ~/results/deepface_utk_validation.csv
"""

import os
import csv
import sys
import random

UTK_DIR = os.path.expanduser('~/data/utk/UTKFace/')
OUTPUT_CSV = os.path.expanduser('~/results/deepface_utk_validation.csv')

# Sample size — 2000 images is enough for validation, faster than all 23k
SAMPLE_SIZE = 2000
RANDOM_SEED = 42

try:
    from deepface import DeepFace
    print("DeepFace imported successfully.")
except ImportError as e:
    print(f"DeepFace not available: {e}")
    sys.exit(1)

# Get all valid image files
all_files = [f for f in os.listdir(UTK_DIR) if f.endswith('.jpg')]
print(f"Found {len(all_files)} images in UTKFace.")

# Parse ground truth from filename: age_gender_race_timestamp.jpg.chip.jpg
def parse_filename(fname):
    try:
        parts = fname.split('_')
        age = int(parts[0])
        gender = int(parts[1])  # 0=Male, 1=Female
        return age, gender
    except:
        return None, None

valid_files = []
for f in all_files:
    age, gender = parse_filename(f)
    if age is not None and 0 <= age <= 120 and gender in [0, 1]:
        valid_files.append(f)

print(f"{len(valid_files)} files with valid ground truth labels.")

# Sample
random.seed(RANDOM_SEED)
sample = random.sample(valid_files, min(SAMPLE_SIZE, len(valid_files)))
print(f"Running DeepFace on {len(sample)} images...")

results = []
errors = 0

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
        if isinstance(analysis, list):
            analysis = analysis[0]

        pred_age = analysis['age']
        pred_gender_str = analysis['dominant_gender']  # 'Man' or 'Woman'
        pred_gender = 0 if pred_gender_str == 'Man' else 1

        age_error = abs(pred_age - gt_age)
        gender_correct = 1 if pred_gender == gt_gender else 0

        results.append({
            'filename': fname,
            'gt_age': gt_age,
            'gt_gender': gt_gender,  # 0=Male, 1=Female
            'pred_age': pred_age,
            'pred_gender': pred_gender,
            'pred_gender_str': pred_gender_str,
            'age_error': age_error,
            'gender_correct': gender_correct,
        })

    except Exception as e:
        errors += 1
        continue

print(f"\nDone. Processed {len(results)}, skipped {errors} errors.")

# Summary stats
age_errors = [r['age_error'] for r in results]
gender_acc = sum(r['gender_correct'] for r in results) / len(results) * 100
mae = sum(age_errors) / len(age_errors)
print(f"Mean Age Error (MAE): {mae:.2f} years")
print(f"Gender Accuracy: {gender_acc:.2f}%")

# Save
fieldnames = ['filename', 'gt_age', 'gt_gender', 'pred_age', 'pred_gender',
              'pred_gender_str', 'age_error', 'gender_correct']
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Saved to {OUTPUT_CSV}")
