"""
Configuration module for Diabetic Retinopathy Detection System.
All configurable parameters are centralized here.
"""

from pathlib import Path

# ──────────────────────────────────────────────
# Base Paths
# ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
TRAIN_IMAGES_DIR = DATA_DIR / "train_images"
TEST_IMAGES_DIR = DATA_DIR / "test_images"
TRAIN_CSV = DATA_DIR / "train.csv"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"
DATABASE_DIR = BASE_DIR / "database"
UPLOADS_DIR = BASE_DIR / "uploads"
STATIC_DIR = BASE_DIR / "static"
TEMPLATES_DIR = BASE_DIR / "templates"

# ──────────────────────────────────────────────
# Model Paths
# ──────────────────────────────────────────────
MODEL_PATH = MODELS_DIR / "best_model.keras"
MODEL_HISTORY_PATH = MODELS_DIR / "training_history.json"

# ──────────────────────────────────────────────
# Image Settings
# ──────────────────────────────────────────────
IMAGE_SIZE = 224
CHANNELS = 3
SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg"}
MAX_UPLOAD_SIZE_MB = 16

# ──────────────────────────────────────────────
# Training Hyperparameters (Optimized)
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 20
INITIAL_LEARNING_RATE = 1e-3
FINE_TUNE_LEARNING_RATE = 5e-6
DROPOUT_RATE = 0.3
RANDOM_SEED = 42
VALIDATION_SPLIT = 0.2

# ──────────────────────────────────────────────
# Class Definitions
# ──────────────────────────────────────────────
NUM_CLASSES = 5
CLASS_NAMES = {
    0: "No Diabetic Retinopathy",
    1: "Mild Diabetic Retinopathy",
    2: "Moderate Diabetic Retinopathy",
    3: "Severe Diabetic Retinopathy",
    4: "Proliferative Diabetic Retinopathy",
}
CLASS_SHORT_NAMES = {
    0: "No DR",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}

# ──────────────────────────────────────────────
# Risk Mapping
# ──────────────────────────────────────────────
RISK_LEVELS = {
    0: "Low",
    1: "Low-Moderate",
    2: "Moderate-High",
    3: "High",
    4: "Very High",
}

# ──────────────────────────────────────────────
# Augmentation Settings
# ──────────────────────────────────────────────
AUGMENTATION_CONFIG = {
    "horizontal_flip": True,
    "rotation_range": 15,
    "zoom_range": 0.15,
    "width_shift_range": 0.1,
    "height_shift_range": 0.1,
    "brightness_range": [0.85, 1.15],
    "fill_mode": "nearest",
}

# ──────────────────────────────────────────────
# Grad-CAM Settings
# ──────────────────────────────────────────────
GRADCAM_LAST_CONV_LAYER_INDEX = -1
GRADCAM_HEATMAP_ALPHA = 0.5
GRADCAM_COLORMAP = "jet"

# ──────────────────────────────────────────────
# Flask Settings
# ──────────────────────────────────────────────
SECRET_KEY = "diabetic-retinopathy-ai-secret-key-change-in-production"
DATABASE_PATH = DATABASE_DIR / "predictions.db"
MAX_CONTENT_LENGTH = MAX_UPLOAD_SIZE_MB * 1024 * 1024
DEBUG_MODE = True
FLASK_HOST = "127.0.0.1"
FLASK_PORT = 5000

# ──────────────────────────────────────────────
# Demo Mode (False = Real AI model)
# ──────────────────────────────────────────────
DEMO_MODE = False
