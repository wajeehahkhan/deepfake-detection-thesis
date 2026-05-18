"""
Pixel Intensity vs Error Rate Analysis
Runs on cluster using only system python3 + PIL + numpy (no conda needed)
Saves results to ~/results/pixel_intensity_results.csv
"""

import csv
import os
import sys

# Try to import required libs
try:
    from PIL import Image
    import numpy as np
except ImportError as e:
    print(f"Missing library: {e}")
    print("Try: python3 -m pip install --user Pillow numpy")
    sys.exit(1)

PREDICTIONS_CSV = os.path.expanduser('~/results/predictions_cache.csv')
OUTPUT_CSV = os.path.expanduser('~/results/pixel_intensity_results.csv')

print("Loading predictions...")

rows = []
with open(PREDICTIONS_CSV, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        rows.append(row)

print(f"Loaded {len(rows)} rows. Processing images...")

results = []
errors_skipped = 0

for i, row in enumerate(rows):
    if i % 1000 == 0:
        print(f"  {i}/{len(rows)}...")

    path = row['path']
    label = int(row['label'])
    eff_pred = int(row['eff_pred'])
    xcp_pred = int(row['xcp_pred'])
    eff_prob = float(row['eff_prob'])
    xcp_prob = float(row['xcp_prob'])

    try:
        img = Image.open(path).convert('RGB')
        img = img.resize((224, 224))
        arr = np.array(img, dtype=np.float32)

        # Grayscale for brightness
        gray = 0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]

        mean_brightness = float(np.mean(gray))
        std_brightness = float(np.std(gray))   # proxy for contrast
        mean_r = float(np.mean(arr[:,:,0]))
        mean_g = float(np.mean(arr[:,:,1]))
        mean_b = float(np.mean(arr[:,:,2]))

        # Error flags
        eff_error = 1 if eff_pred != label else 0
        xcp_error = 1 if xcp_pred != label else 0

        results.append({
            'path': path,
            'label': label,
            'eff_pred': eff_pred,
            'xcp_pred': xcp_pred,
            'eff_prob': eff_prob,
            'xcp_prob': xcp_prob,
            'eff_error': eff_error,
            'xcp_error': xcp_error,
            'mean_brightness': mean_brightness,
            'std_brightness': std_brightness,
            'mean_r': mean_r,
            'mean_g': mean_g,
            'mean_b': mean_b,
        })

    except Exception as e:
        errors_skipped += 1
        continue

print(f"\nDone. Processed {len(results)} images, skipped {errors_skipped}.")

# Save results
fieldnames = ['path', 'label', 'eff_pred', 'xcp_pred', 'eff_prob', 'xcp_prob',
              'eff_error', 'xcp_error', 'mean_brightness', 'std_brightness',
              'mean_r', 'mean_g', 'mean_b']

with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)

print(f"Saved to {OUTPUT_CSV}")
