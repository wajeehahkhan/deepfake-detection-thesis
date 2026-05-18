"""
config.py — Central configuration file for the deepfake detection pipeline.

All paths, hyperparameters, and settings are defined here.
To run on a different machine, only this file needs to be updated.
"""

import os

# ============================================================
# DATA PATHS
# Set BASE_DIR to wherever you stored the dataset
# ============================================================
BASE_DIR = os.path.expanduser("~")  # defaults to home directory

DATA_DIR = os.path.join(BASE_DIR, "data", "real_vs_fake", "real-vs-fake")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR   = os.path.join(DATA_DIR, "valid")
TEST_DIR  = os.path.join(DATA_DIR, "test")

UTK_DIR   = os.path.join(BASE_DIR, "data", "utk", "UTKFace")

# ============================================================
# OUTPUT PATHS
# ============================================================
RESULTS_DIR     = os.path.join(BASE_DIR, "results")
CHECKPOINTS_DIR = os.path.join(BASE_DIR, "checkpoints")
GRADCAM_DIR     = os.path.join(RESULTS_DIR, "gradcam")

# Ensure output directories exist
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
os.makedirs(GRADCAM_DIR, exist_ok=True)

# ============================================================
# MODEL CHECKPOINTS
# ============================================================
EFF_CHECKPOINT        = os.path.join(CHECKPOINTS_DIR, "efficientnet_best.pth")
XCP_CHECKPOINT        = os.path.join(CHECKPOINTS_DIR, "xceptionnet_best.pth")
EFF_FROZEN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "efficientnet_frozen_best.pth")
XCP_FROZEN_CHECKPOINT = os.path.join(CHECKPOINTS_DIR, "xception_frozen_best.pth")

# ============================================================
# TRAINING HYPERPARAMETERS
# ============================================================
IMG_SIZE     = 224      # Input image size (both models trained at 224x224)
BATCH_SIZE   = 32       # Batch size — reduced from 64 due to CPU memory constraints
LR           = 1e-4     # Learning rate — standard for fine-tuning pretrained CNNs
WEIGHT_DECAY = 1e-4     # L2 regularization to prevent overfitting
EPOCHS       = 10       # Maximum epochs — early stopping typically triggers before this
PATIENCE     = 3        # Early stopping patience (epochs without AUC improvement)

# ImageNet normalization — required since both models use ImageNet pretrained weights
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# ============================================================
# REPRODUCIBILITY
# ============================================================
RANDOM_SEED = 42        # Fixed seed for all random operations

# ============================================================
# ANALYSIS SETTINGS
# ============================================================
DEEPFACE_SAMPLE_SIZE = 2000   # Number of UTKFace images for DeepFace validation
UTK_SAMPLE_SIZE      = 2000   # Number of UTKFace images for OOD evaluation
GRADCAM_SAMPLES      = 5      # Number of Grad-CAM examples per category
