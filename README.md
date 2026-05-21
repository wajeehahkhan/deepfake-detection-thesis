# Explainable Deepfake Detection Using CNN-Based Models

**Author:** Wajeeha Khan  
**Institution:** Tilburg University - MSc Data Science & Society  
**Year:** 2026  

---

## Overview

This repository contains the full implementation for my MSc thesis on explainable deepfake detection using CNN-based models. The project compares **EfficientNet-B4** and **XceptionNet** on a large-scale balanced dataset of 140,000 real and StyleGAN-generated faces, analysing detection performance across demographic subgroups (age and gender) and explaining model decisions using Grad-CAM.

---

## Research Questions

**Main RQ:** How effectively can CNN-based deepfake detectors identify StyleGAN-generated faces, and to what extent does an individual's demographic group influence detection performance?

- **SRQ1:** How does classification performance (accuracy, F1-score, AUC-ROC) differ between EfficientNet-B4 and XceptionNet on a large-scale balanced dataset?
- **SRQ2:** Do the models perform differently across demographic subgroups (age, gender), and does this disparity persist at their intersection?
- **SRQ3:** What patterns can be identified through Grad-CAM maps and misclassified examples to explain model errors?

---

## Key Results

| Model | Training Strategy | Accuracy | F1 | AUC | FNR | FPR |
|-------|----------|----------|-----|-----|-----|-----|
| EfficientNet-B4 | Full fine-tuning | 99.90% | 99.90% | 0.9999 | 0.05% | 0.15% |
| EfficientNet-B4 | Frozen layers | 97.16% | 97.16% | 0.9966 | 2.76% | 2.93% |
| XceptionNet | Full fine-tuning | 71.72% | 60.57% | 0.9839 | 56.56% | 0.00% |
| XceptionNet | Frozen layers | 98.21% | 98.23% | 0.9987 | 0.94% | 2.64% |

**Key finding:** XceptionNet's full fine-tuning collapse (71.72%) was caused by catastrophic forgetting — when early layers were frozen, accuracy jumped to 98.21%, proving the architecture is capable but sensitive to transfer learning strategy.

**Demographic bias finding:** EfficientNet-B4's highest False Negative Rate was in the *Woman / Under 20* subgroup (FNR 0.60%), confirmed by out-of-distribution testing on UTKFace (Under-20 FPR: 0.96% vs 0.00% for 20–34).

---

## Repository Structure

```
├── thesis_train.py                  # Main training pipeline (full fine-tuning, both models)
├── frozen_layers_experiment.py      # EfficientNet-B4 frozen layers experiment
├── frozen_layers_xception.py        # XceptionNet frozen layers experiment
├── eval_frozen_checkpoints.py       # Test set evaluation of frozen layer checkpoints
├── both_models_utk.py               # Out-of-distribution evaluation on UTKFace
├── deepface_utk_validation.py       # DeepFace demographic label validation
├── pixel_intensity_analysis.py      # Pixel intensity vs error rate analysis
├── gradcam_analysis.py              # Grad-CAM heatmap generation (EfficientNet + XceptionNet)
├── run_xcp_gradcam_v3.py            # XceptionNet Grad-CAM (fixed backward hook version)
├── plot_combined_roc.py             # Combined ROC curve plot
├── plot_pixel_intensity.py          # Pixel intensity error rate plots
├── plot_histogram_final.py          # Age distribution histogram
├── config.py                        # Central configuration (paths, hyperparameters)
├── run_pipeline.sh                  # End-to-end pipeline script
└── requirements.txt                 # Python dependencies
```

---

## Dataset

**Real-vs-Fake Face Detection Dataset** — available on Kaggle:  
https://www.kaggle.com/datasets/ciplab/real-and-fake-face-detection

- 140,000 images total (70,000 real, 70,000 StyleGAN-generated)
- Split: 70% train / 15% validation / 15% test
- Test set: 20,000 images (10,000 real, 10,000 fake) — perfectly balanced

**UTKFace Dataset** (for OOD evaluation and DeepFace validation):  
https://susanqq.github.io/UTKFace/

Expected directory structure:
```
data/
├── real_vs_fake/
│   └── real-vs-fake/
│       ├── train/
│       │   ├── real/
│       │   └── fake/
│       ├── valid/
│       │   ├── real/
│       │   └── fake/
│       └── test/
│           ├── real/
│           └── fake/
└── utk/
    └── UTKFace/
        └── *.jpg
```

---

## Setup

```bash
# Clone the repository
git clone https://github.com/wajeehahkhan/deepfake-detection-thesis.git
cd deepfake-detection-thesis

# Create and activate virtual environment
python3 -m venv thesis_env
source thesis_env/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

Update `config.py` with your local paths if needed, then run:

```bash
# Run everything end-to-end
bash run_pipeline.sh
```

Or run individual steps:

```bash
# Step 1: Train both models (full fine-tuning)
python3 thesis_train.py

# Step 2: Frozen layers experiments
python3 frozen_layers_experiment.py
python3 frozen_layers_xception.py

# Step 3: Evaluate frozen checkpoints
python3 eval_frozen_checkpoints.py

# Step 4: Out-of-distribution evaluation on UTKFace
python3 both_models_utk.py

# Step 5: Validate DeepFace demographic labels
python3 deepface_utk_validation.py

# Step 6: Pixel intensity analysis
python3 pixel_intensity_analysis.py

# Step 7: Generate Grad-CAM heatmaps
python3 gradcam_analysis.py
python3 run_xcp_gradcam_v3.py

# Step 8: Generate all plots
python3 plot_combined_roc.py
python3 plot_pixel_intensity.py
python3 plot_histogram_final.py
```

---

## Reproducibility

- All random seeds fixed at `42` (set in `config.py`)
- Dependencies pinned in `requirements.txt`
- Checkpoints saved automatically during training to `./checkpoints/`
- Results cached to `./results/` to avoid re-running expensive computations

**Hardware used:** Tilburg University GPU cluster (Byzantium node), Python 3.11.2, PyTorch 2.11. Due to a CUDA driver mismatch, all inference and Grad-CAM generation was run on CPU — results are mathematically identical to GPU execution.

---

## Experimental Design Notes

**Why 224×224 for both models?**  
XceptionNet was originally designed for 299×299 input. We standardised both to 224×224 for a fair comparison. This is acknowledged as a methodological limitation — the reduced resolution may have constrained XceptionNet's spatial feature extraction.

**Why frozen layers?**  
XceptionNet's full fine-tuning collapse led to a secondary experiment investigating catastrophic forgetting. Freezing early layers preserved ImageNet representations and resolved the collapse entirely (71.72% → 98.21%), confirming the failure was a training artefact, not an architectural weakness.

**Why DeepFace for demographics?**  
The StyleGAN dataset contains no ground-truth demographic labels. DeepFace pseudo-labels were validated against UTKFace (MAE: 12.21 years, gender accuracy: 79.10%). All bias findings are interpreted with this uncertainty acknowledged.

---

## Citation

If you use this code, please cite:

```
Khan, W. (2026). Explainable Deepfake Detection Using CNN-Based Models.
MSc Thesis, Tilburg University — Data Science & Society.
GitHub: https://github.com/wajeehahkhan/deepfake-detection-thesis
```

---

## Acknowledgements

- Real-vs-Fake Face Detection Dataset (Kaggle / CIPLAB)
- UTKFace Dataset (Zhang et al., 2017)
- DeepFace library (Serengil & Ozpinar, 2021)
- PyTorch Image Models / timm (Wightman, 2019)
- Grad-CAM (Selvaraju et al., 2017)
