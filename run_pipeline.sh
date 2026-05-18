#!/bin/bash
# =============================================================
# run_pipeline.sh — Full pipeline for deepfake detection thesis
#
# Run this script to reproduce all results from start to finish.
# Make sure you have:
#   1. Installed all dependencies: pip install -r requirements.txt
#   2. Downloaded the dataset (see README.md for instructions)
#   3. Updated config.py with your local paths if needed
# =============================================================

echo "======================================================"
echo " Explainable Deepfake Detection — Full Pipeline"
echo "======================================================"

# STEP 1 — Train both models (full fine-tuning)
echo ""
echo "STEP 1: Training EfficientNet-B4 and XceptionNet (full fine-tuning)..."
python3 thesis_train.py

# STEP 2 — Frozen layers experiment
echo ""
echo "STEP 2a: Training EfficientNet-B4 with frozen layers..."
python3 frozen_layers_experiment.py

echo ""
echo "STEP 2b: Training XceptionNet with frozen layers..."
python3 frozen_layers_xception.py

# STEP 3 — Evaluate frozen checkpoints on test set
echo ""
echo "STEP 3: Evaluating frozen layer checkpoints..."
python3 eval_frozen_checkpoints.py

# STEP 4 — Demographic bias analysis (UTKFace OOD)
echo ""
echo "STEP 4: Running UTKFace out-of-distribution evaluation..."
python3 both_models_utk.py

# STEP 5 — DeepFace label validation
echo ""
echo "STEP 5: Validating DeepFace demographic labels on UTKFace..."
python3 deepface_utk_validation.py

# STEP 6 — Pixel intensity analysis
echo ""
echo "STEP 6: Running pixel intensity vs error rate analysis..."
python3 pixel_intensity_analysis.py

# STEP 7 — Grad-CAM heatmaps
echo ""
echo "STEP 7a: Generating EfficientNet Grad-CAM heatmaps..."
python3 gradcam_analysis.py

echo ""
echo "STEP 7b: Generating XceptionNet Grad-CAM heatmaps..."
python3 run_xcp_gradcam_v3.py

# STEP 8 — Generate all plots
echo ""
echo "STEP 8: Generating result plots..."
python3 plot_combined_roc.py
python3 plot_pixel_intensity.py
python3 plot_histogram_final.py

echo ""
echo "======================================================"
echo " Pipeline complete! Results saved to ~/results/"
echo "======================================================"
