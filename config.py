import torch
from pathlib import Path

# Project Paths
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
LOG_DIR = BASE_DIR / "logs"

# Training Configuration
BATCH_SIZE = 64
LEARNING_RATE = 0.001
EPOCHS = 10

# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model File
MODEL_PATH = MODEL_DIR / "best_model.pth"