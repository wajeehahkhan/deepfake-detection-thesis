import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

DEMO_CSV = '/home/u773837/results/demographics_cache.csv'
OUTPUT = '/home/u773837/results/age_distribution_histogram.png'

df = pd.read_csv(DEMO_CSV)
df = df.dropna(subset=['age'])
df['age'] = df['age'].astype(int)
df['label'] = df['path'].apply(lambda p: 'Fake (StyleGAN)' if '/fake/' in p else 'Real')

def age_group(age):
    if age < 20: return 'Under 20'
    elif age < 35: return '20-34'
    elif age < 50: return '35-49'
    else: return '50+'

df['age_group'] = df['age'].apply(age_group)
order = ['Under 20', '20-34', '35-49', '50+']

real_df = df[df['label'] == 'Real']
fake_df = df[df['label'] == 'Fake (StyleGAN)']

print(f"Real: {len(real_df)}, Fake: {len(fake_df)}")
print(df['age_group'].value_counts())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.patch.set_facecolor('white')

# Raw age histogram
axes[0].hist(real_df['age'], bins=40, color='#2a9d8f', alpha=0.7, label='Real', edgecolor='white')
axes[0].hist(fake_df['age'], bins=40, color='#e76f51', alpha=0.7, label='Fake (StyleGAN)', edgecolor='white')
axes[0].axvline(real_df['age'].mean(), color='#2a9d8f', lw=2, linestyle='--', label=f'Real mean={real_df["age"].mean():.1f}')
axes[0].axvline(fake_df['age'].mean(), color='#e76f51', lw=2, linestyle='--', label=f'Fake mean={fake_df["age"].mean():.1f}')
axes[0].set_xlabel('Age (estimated by DeepFace)', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Age Distribution — Test Set (Real vs Fake)', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].set_facecolor('#f9f9f9')

# Age group bar chart
real_counts = real_df['age_group'].value_counts().reindex(order)
fake_counts = fake_df['age_group'].value_counts().reindex(order)
x = np.arange(len(order))
width = 0.35
bars1 = axes[1].bar(x - width/2, real_counts.values, width, color='#2a9d8f', alpha=0.85, label='Real')
bars2 = axes[1].bar(x + width/2, fake_counts.values, width, color='#e76f51', alpha=0.85, label='Fake')
for bar in list(bars1) + list(bars2):
    axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                str(int(bar.get_height())), ha='center', fontsize=9, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(order)
axes[1].set_xlabel('Age Group', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Age Group Distribution — Test Set', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=11)
axes[1].set_facecolor('#f9f9f9')

plt.suptitle('Test Set Age Distribution (DeepFace Estimated)', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUTPUT, dpi=150, bbox_inches='tight', facecolor='white')
print(f"Saved to {OUTPUT}")
