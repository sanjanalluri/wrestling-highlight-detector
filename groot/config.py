"""
Configuration file for LSTM highlight detection pipeline.
All hyperparameters and model settings are defined here for easy tuning.
"""

import torch

# ============================================================================
# MODEL ARCHITECTURE CONFIGURATIONS
# ============================================================================

# Segmentation Model (DeepLabV3+)
SEGMENTATION_MODEL = "deeplabv3plus"
SEGMENTATION_BACKBONE = "resnet50"  # or "resnet101", "mobilenet_v3_large"
SEGMENTATION_NUM_CLASSES = 2  # background + person
SEGMENTATION_INPUT_SIZE = (512, 512)  # (height, width)

# 2D CNN Feature Extraction (EfficientNet)
CNN_MODEL = "efficientnet_b3"  # Options: "efficientnet_b0" to "efficientnet_b7"
CNN_FEATURE_DIM = 1536  # Feature dimension for EfficientNet-B3
CNN_INPUT_SIZE = (224, 224)  # Standard input size for EfficientNet
CNN_PRETRAINED = True  # Use ImageNet pre-trained weights

# Bi-LSTM Configuration
LSTM_HIDDEN_SIZE = 384  # 1/4 of CNN_FEATURE_DIM (1536 / 4 = 384)
LSTM_NUM_LAYERS = 2
LSTM_BIDIRECTIONAL = True
LSTM_DROPOUT = 0.2

# Dense Classification Network
DENSE_LAYER_1_SIZE = 512
DENSE_LAYER_2_SIZE = 128
DENSE_DROPOUT_RATE = 0.3

# ============================================================================
# DATA PROCESSING CONFIGURATIONS
# ============================================================================

# Frame Sampling
NUM_FRAMES = 16  # Number of uniform frames to sample from each clip
MIN_CLIP_DURATION = 1.0  # Minimum clip duration in seconds
TARGET_FPS = 12  # Target frame rate for all videos

# Sliding Window (for future match processing)
WINDOW_SIZE_SECONDS = 5.0
OVERLAP_SECONDS = 2.5
WINDOW_STRIDE_SECONDS = WINDOW_SIZE_SECONDS - OVERLAP_SECONDS

# ============================================================================
# TRAINING CONFIGURATIONS
# ============================================================================

# Cross-Validation
K_FOLD_CV = 5
RANDOM_SEED = 42

# Training Parameters
BATCH_SIZE = 16
NUM_EPOCHS = 100
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4

# Loss Function
LOSS_FUNCTION = "focal_loss"  # Options: "binary_cross_entropy", "focal_loss"
FOCAL_LOSS_ALPHA = 0.25  # For focal loss
FOCAL_LOSS_GAMMA = 2.0   # For focal loss

# Learning Rate Scheduling
LR_SCHEDULER = "cosine_annealing"  # Options: "step", "cosine_annealing", "plateau"
LR_STEP_SIZE = 30  # For step scheduler
LR_GAMMA = 0.1     # For step scheduler
MIN_LR = 1e-6      # Minimum learning rate

# Early Stopping
EARLY_STOPPING_PATIENCE = 15
EARLY_STOPPING_MIN_DELTA = 0.001

# ============================================================================
# DATA PROCESSING CONFIGURATIONS
# ============================================================================

# Frame Deduplication Parameters
SSIM_THRESHOLD = 0.985  # Structural Similarity Index threshold
OPTICAL_FLOW_THRESHOLD = 0.5  # Average optical flow magnitude threshold (pixels)
CONTEXT_EXTENSION_SECONDS = 0.25  # Additional context when clip is too short

# Video Processing
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mov', '.mkv', '.wmv']
MAX_VIDEO_SIZE_GB = 10  # Maximum video file size to process

# ============================================================================
# DATASET CONFIGURATIONS
# ============================================================================

# Dataset Paths
DATASET_PATHS = {
    "kaggle_person_segmentation": "datasets/kaggle_person_segmentation",
    "freestyle_wrestling": "datasets/freestyle_wrestling",
    "training_clips": "datasets/training_clips",
    "validation_clips": "datasets/validation_clips"
}

# Data Augmentation
AUGMENTATION_ENABLED = True
AUGMENTATION_PROBABILITY = 0.5

# ============================================================================
# DATABASE CONFIGURATIONS
# ============================================================================

# Database Path
DATABASE_PATH = "../../database/wrestling_pipeline.db"

# Table Names
TABLE_NAMES = {
    # Data Processing Tables
    "groot_videos": "groot_videos",  # Video metadata and frame rate info
    "groot_highlight_clips": "groot_highlight_clips",  # Manually labeled highlight clips
    "groot_processed_frames": "groot_processed_frames",  # Final 16-frame samples
    
    # Model Tables
    "lstm_training_clips": "lstm_training_clips",
    "lstm_models": "lstm_models",
    "lstm_training_runs": "lstm_training_runs",
    "lstm_predictions": "lstm_predictions",
    "lstm_match_predictions": "lstm_match_predictions",
    "lstm_features": "lstm_features"
}

# ============================================================================
# INFERENCE CONFIGURATIONS
# ============================================================================

# Prediction Threshold
CONFIDENCE_THRESHOLD = 0.5  # Threshold for binary classification

# Model Checkpoint
MODEL_CHECKPOINT_PATH = "models/lstm_highlight_detector.pth"
BEST_MODEL_PATH = "models/best_lstm_model.pth"

# ============================================================================
# VISUALIZATION CONFIGURATIONS
# ============================================================================

# Training Plots
PLOT_SAVE_PATH = "plots/"
PLOT_FORMAT = "png"  # Options: "png", "jpg", "pdf"

# Metrics to Plot
METRICS_TO_PLOT = ["loss", "accuracy", "precision", "recall", "f1_score"]

# ============================================================================
# COMPUTATION CONFIGURATIONS
# ============================================================================

# Device Configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4  # For data loading

# Mixed Precision Training
USE_MIXED_PRECISION = True  # Use AMP for faster training

# ============================================================================
# LOGGING CONFIGURATIONS
# ============================================================================

# Logging
LOG_LEVEL = "INFO"
LOG_FILE = "logs/lstm_pipeline.log"

# TensorBoard
USE_TENSORBOARD = True
TENSORBOARD_LOG_DIR = "runs/lstm_experiments"

# ============================================================================
# VALIDATION CONFIGURATIONS
# ============================================================================

# Validation Metrics
VALIDATION_METRICS = ["accuracy", "precision", "recall", "f1_score", "auc"]

# Model Selection
MODEL_SELECTION_METRIC = "f1_score"  # Metric to use for selecting best model

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_summary():
    """Print a summary of the current model configuration"""
    summary = f"""
🤖 LSTM Highlight Detection Model Configuration:
    
📊 Model Architecture:
   • Segmentation: {SEGMENTATION_MODEL} ({SEGMENTATION_BACKBONE})
   • CNN: {CNN_MODEL} (features: {CNN_FEATURE_DIM})
   • LSTM: {LSTM_NUM_LAYERS} layers, {LSTM_HIDDEN_SIZE} hidden size, {'Bidirectional' if LSTM_BIDIRECTIONAL else 'Unidirectional'}
   • Dense: {DENSE_LAYER_1_SIZE} → {DENSE_LAYER_2_SIZE} → 1

🎯 Training Parameters:
   • Frames per clip: {NUM_FRAMES}
   • Batch size: {BATCH_SIZE}
   • Learning rate: {LEARNING_RATE}
   • Epochs: {NUM_EPOCHS}
   • K-fold CV: {K_FOLD_CV}

🔧 Processing:
   • Window size: {WINDOW_SIZE_SECONDS}s
   • Overlap: {OVERLAP_SECONDS}s
   • Device: {DEVICE}
    """
    return summary

def validate_config():
    """Validate configuration parameters"""
    assert CNN_FEATURE_DIM > 0, "CNN_FEATURE_DIM must be positive"
    assert LSTM_HIDDEN_SIZE > 0, "LSTM_HIDDEN_SIZE must be positive"
    assert NUM_FRAMES > 0, "NUM_FRAMES must be positive"
    assert BATCH_SIZE > 0, "BATCH_SIZE must be positive"
    assert LEARNING_RATE > 0, "LEARNING_RATE must be positive"
    assert K_FOLD_CV > 1, "K_FOLD_CV must be greater than 1"
    
    print("✅ Configuration validation passed!")

if __name__ == "__main__":
    print(get_model_summary())
    validate_config() 