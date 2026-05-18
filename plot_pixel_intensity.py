import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

INPUT = '/home/u773837/results/pixel_intensity_results.csv'
OUTPUT = '/home/u773837/results/pixel_intensity_plot.png'

print("Loading data...")
brightness, contrast, eff_errors, xcp_errors = [], [], [], []
with open(INPUT) as f:
    for row in csv.DictReader(f):
        brightness.append(float(row['mean_brightness']))
        contrast.append(float(row['std_brightness']))
        eff_errors.append(int(row['eff_error']))
        xcp_errors.append(int(row['xcp_error']))

brightness = np.array(brightness)
contrast = np.array(contrast)
eff_errors = np.array(eff_errors)
xcp_errors = np.array(xcp_errors)

bins = np.linspace(brightness.min(), brightness.max(), 20)
bin_idx = np.digitize(brightness, bins)
eff_rate, xcp_rate, bin_centers = [], [], []
for i in range(1, len(bins)):
    mask = bin_idx == i
    if mask.sum() > 50:
        eff_rate.append(eff_errors[mask].mean() * 100)
        xcp_rate.append(xcp_errors[mask].mean() * 100)
        bin_centers.append(bins[i-1])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')
axes[0].plot(bin_centers, eff_rate, color='#2a9d8f', lw=2.5, marker='o', label='EfficientNet-B4')
axes[0].plot(bin_centers, xcp_rate, color='#e76f51', lw=2.5, marker='o', label='XceptionNet')
axes[0].set_xlabel('Mean Brightness', fontsize=12)
axes[0].set_ylabel('Error Rate (%)', fontsize=12)
axes[0].set_title('Error Rate vs Image Brightness', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].set_facecolor('#f9f9f9')

bins2 = np.linspace(contrast.min(), contrast.max(), 20)
bin_idx2 = np.digitize(contrast, bins2)
eff_rate2, xcp_rate2, bin_centers2 = [], [], []
for i in range(1, len(bins2)):
    mask = bin_idx2 == i
    if mask.sum() > 50:
        eff_rate2.append(eff_errors[mask].mean() * 100)
        xcp_rate2.append(xcp_errors[mask].mean() * 100)
        bin_centers2.append(bins2[i-1])

axes[1].plot(bin_centers2, eff_rate2, color='#2a9d8f', lw=2.5, marker='o', label='EfficientNet-B4')
axes[1].plot(bin_centers2, xcp_rate2, color='#e76f51', lw=2.5, marker='o', label='XceptionNet')
axes[1].set_xlabel('Contrast (Std Brightness)', fontsize=12)
axes[1].set_ylabel('Error Rate (%)', fontsize=12)
axes[1].set_title('Error Rate vs Image Contrast', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_facecolor('#f9f9f9')

plt.suptitle('Pixel Intensity Analysis — Error Rate by Image Properties', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUTPUT}")
