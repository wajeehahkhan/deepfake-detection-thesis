import os, sys, random, csv
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import timm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

OUTPUT_DIR = os.path.expanduser('~/results/gradcam/')
PREDICTIONS_CSV = os.path.expanduser('~/results/predictions_cache.csv')
IMG_SIZE = 224
RANDOM_SEED = 42
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(lambda m,i,o: setattr(self, 'activations', o.detach()))
        target_layer.register_full_backward_hook(lambda m,i,o: setattr(self, 'gradients', o[0].detach()))

    def generate(self, tensor):
        self.model.zero_grad()
        out = self.model(tensor)
        out.backward()
        weights = self.gradients[0].mean(dim=(1,2))
        cam = torch.zeros(self.activations.shape[2:], device=self.activations.device)
        for i, w in enumerate(weights):
            cam += w * self.activations[0][i]
        cam = torch.relu(cam).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam

def save_heatmap(img_path, cam, save_path, title=''):
    img = np.array(Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE)))
    cam_r = np.array(Image.fromarray((cam*255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE))) / 255.0
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img); axes[0].set_title('Original'); axes[0].axis('off')
    axes[1].imshow(cam_r, cmap='jet'); axes[1].set_title('Grad-CAM'); axes[1].axis('off')
    overlay = np.clip(0.5*img/255.0 + 0.5*cm.jet(cam_r)[:,:,:3], 0, 1)
    axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
    fig.suptitle(title, fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

rows = []
with open(PREDICTIONS_CSV) as f:
    for r in csv.DictReader(f): rows.append(r)

def select_samples(rows, pred_col, n=5):
    cr = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==0]
    cf = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==1]
    fn = [r for r in rows if int(r['label'])==1 and int(r[pred_col])==0]
    fp = [r for r in rows if int(r['label'])==0 and int(r[pred_col])==1]
    random.seed(RANDOM_SEED)
    s = [(r,'correct_real') for r in random.sample(cr, min(n,len(cr)))]
    s += [(r,'correct_fake') for r in random.sample(cf, min(n,len(cf)))]
    s += [(r,'false_negative') for r in random.sample(fn, min(n,len(fn)))]
    s += [(r,'false_positive') for r in random.sample(fp, min(n,len(fp)))]
    return s

# EfficientNet
print("Loading EfficientNet-B4...")
eff = timm.create_model('efficientnet_b4', pretrained=False, num_classes=1)
eff.load_state_dict(torch.load(os.path.expanduser('~/checkpoints/efficientnet_best.pth'), map_location=device))
eff = eff.to(device).eval()
gradcam_eff = GradCAM(eff, eff.blocks[-1][-1])

for i, (row, stype) in enumerate(select_samples(rows, 'eff_pred')):
    try:
        img = Image.open(row['path']).convert('RGB')
        t = transform(img).unsqueeze(0).to(device)
        t.requires_grad_(True)
        cam = gradcam_eff.generate(t)
        fname = f"eff_{stype}_{i}.png"
        title = f"EfficientNet | {stype} | Label={'Fake' if int(row['label'])==1 else 'Real'} | p={float(row['eff_prob']):.3f}"
        save_heatmap(row['path'], cam, os.path.join(OUTPUT_DIR, fname), title)
        print(f"  Saved: {fname}")
    except Exception as e:
        print(f"  Error: {e}")

# XceptionNet
print("\nLoading XceptionNet...")
for p in [os.path.expanduser('~/checkpoints/xceptionnet_best.pth'),
          os.path.expanduser('~/checkpoints/XceptionNet_resume.pth')]:
    if os.path.exists(p):
        try:
            xcp = timm.create_model('xception', pretrained=False, num_classes=1)
            xcp.load_state_dict(torch.load(p, map_location=device))
            xcp = xcp.to(device).eval()
            gradcam_xcp = GradCAM(xcp, xcp.act4)
            for i, (row, stype) in enumerate(select_samples(rows, 'xcp_pred')):
                try:
                    img = Image.open(row['path']).convert('RGB')
                    t = transform(img).unsqueeze(0).to(device)
                    t.requires_grad_(True)
                    cam = gradcam_xcp.generate(t)
                    fname = f"xcp_{stype}_{i}.png"
                    title = f"XceptionNet | {stype} | Label={'Fake' if int(row['label'])==1 else 'Real'} | p={float(row['xcp_prob']):.3f}"
                    save_heatmap(row['path'], cam, os.path.join(OUTPUT_DIR, fname), title)
                    print(f"  Saved: {fname}")
                except Exception as e:
                    print(f"  Error: {e}")
            break
        except Exception as e:
            print(f"XceptionNet error: {e}")

print(f"\nDone. Results in {OUTPUT_DIR}")
