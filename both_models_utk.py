import os, sys, csv, random
from collections import defaultdict
import torch
from torchvision import transforms
from PIL import Image
import timm

UTK_DIR = os.path.expanduser('~/data/utk/UTKFace/')
OUTPUT_CSV = os.path.expanduser('~/results/both_models_utk_results.csv')
SAMPLE_SIZE = 2000
RANDOM_SEED = 42
IMG_SIZE = 224

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

print("Loading EfficientNet-B4...")
eff_model = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
eff_model.load_state_dict(torch.load(os.path.expanduser('~/checkpoints/efficientnet_best.pth'), map_location=device))
eff_model = eff_model.to(device).eval()
print("EfficientNet loaded.")

xcp_model = None
for p in [os.path.expanduser('~/checkpoints/xceptionnet_best.pth'),
          os.path.expanduser('~/checkpoints/XceptionNet_resume.pth')]:
    if os.path.exists(p):
        try:
            xcp_model = timm.create_model('xception', pretrained=False, num_classes=1)
            xcp_model.load_state_dict(torch.load(p, map_location=device))
            xcp_model = xcp_model.to(device).eval()
            print(f"XceptionNet loaded from {p}.")
            break
        except Exception as e:
            print(f"XceptionNet error: {e}")

def parse_filename(fname):
    try:
        parts = fname.split('_')
        return int(parts[0]), int(parts[1])
    except:
        return None, None

def get_age_group(age):
    if age < 20: return 'Under 20'
    elif age < 35: return '20-34'
    elif age < 50: return '35-49'
    else: return '50+'

all_files = [f for f in os.listdir(UTK_DIR) if f.endswith('.jpg')]
valid_files = [f for f in all_files if parse_filename(f)[0] is not None and parse_filename(f)[1] in [0,1]]
random.seed(RANDOM_SEED)
sample = random.sample(valid_files, min(SAMPLE_SIZE, len(valid_files)))
print(f"Running on {len(sample)} images...")

results, errors = [], 0
with torch.no_grad():
    for i, fname in enumerate(sample):
        if i % 200 == 0: print(f"  {i}/{len(sample)}...")
        gt_age, gt_gender = parse_filename(fname)
        try:
            img = Image.open(os.path.join(UTK_DIR, fname)).convert('RGB')
            tensor = transform(img).unsqueeze(0).to(device)
            eff_prob = torch.sigmoid(eff_model(tensor)).item()
            eff_pred = 1 if eff_prob > 0.5 else 0
            xcp_prob = xcp_pred = xcp_fp = None
            if xcp_model:
                xcp_prob = torch.sigmoid(xcp_model(tensor)).item()
                xcp_pred = 1 if xcp_prob > 0.5 else 0
                xcp_fp = 1 if xcp_pred == 1 else 0
            results.append({
                'filename': fname, 'gt_age': gt_age, 'gt_gender': gt_gender,
                'gender_str': 'Man' if gt_gender==0 else 'Woman',
                'age_group': get_age_group(gt_age),
                'eff_prob_fake': eff_prob, 'eff_pred': eff_pred,
                'eff_false_positive': 1 if eff_pred==1 else 0,
                'xcp_prob_fake': xcp_prob, 'xcp_pred': xcp_pred,
                'xcp_false_positive': xcp_fp,
            })
        except Exception as e:
            errors += 1

print(f"Done. Processed {len(results)}, skipped {errors}.")
age_order = ['Under 20','20-34','35-49','50+']
print("\n=== EfficientNet FPR by Age ===")
age_eff = defaultdict(list)
for r in results: age_eff[r['age_group']].append(r['eff_false_positive'])
for g in age_order:
    v = age_eff[g]
    if v: print(f"  {g}: {sum(v)/len(v)*100:.2f}% (n={len(v)})")
print("\n=== EfficientNet FPR by Gender ===")
gen_eff = defaultdict(list)
for r in results: gen_eff[r['gender_str']].append(r['eff_false_positive'])
for g in ['Man','Woman']:
    v = gen_eff[g]
    if v: print(f"  {g}: {sum(v)/len(v)*100:.2f}% (n={len(v)})")

fieldnames = ['filename','gt_age','gt_gender','gender_str','age_group',
              'eff_prob_fake','eff_pred','eff_false_positive',
              'xcp_prob_fake','xcp_pred','xcp_false_positive']
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(results)
print(f"Saved to {OUTPUT_CSV}")
