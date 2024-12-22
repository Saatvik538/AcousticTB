import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

# Raw data paths
RAW_DIR = DATA_DIR / "raw"
SOLICITED_DIR = RAW_DIR / "solicited_coughs"
LONGITUDINAL_DIR = RAW_DIR / "longitudinal_coughs"

# Processed data paths
PROCESSED_DIR = DATA_DIR / "processed"
SOLICITED_PROCESSED = PROCESSED_DIR / "solicited"
LONGITUDINAL_PROCESSED = PROCESSED_DIR / "longitudinal"

# Feature paths
FEATURE_TYPES = ['mfcc', 'spectrograms', 'wavelets']

# Audio parameters
SAMPLE_RATE = 22050
DURATION = 0.5  # Duration in seconds
N_MFCC = 13
N_MELS = 128
N_FFT = 2048
HOP_LENGTH = 512

# Model parameters
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories if they don't exist
for directory in [SOLICITED_DIR, LONGITUDINAL_DIR, 
                 SOLICITED_PROCESSED, LONGITUDINAL_PROCESSED]:
    directory.mkdir(parents=True, exist_ok=True)

    # Create feature directories
    if "processed" in str(directory):
        for feature_type in FEATURE_TYPES:
            (directory / "features" / feature_type).mkdir(parents=True, exist_ok=True)
        
        # Create split directories for solicited data
        if "solicited" in str(directory):
            for split in ['train', 'val', 'test']:
                (directory / split).mkdir(parents=True, exist_ok=True)