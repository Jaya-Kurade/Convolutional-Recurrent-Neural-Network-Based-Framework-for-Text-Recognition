import string
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = PROJECT_ROOT / 'models'
CHECKPOINTS_DIR = PROJECT_ROOT / 'checkpoints'
LOGS_DIR = PROJECT_ROOT / 'logs'

# SynthText Dataset
SYNTHTEXT_URL = 'https://thor.robots.ox.ac.uk/~vgg/data/scenetext/SynthText.zip'
SYNTHTEXT_MAT = RAW_DATA_DIR / 'SynthText' / 'gt.mat'

# Model hyperparameters
IMG_HEIGHT = 32
IMG_WIDTH = 128
IMG_CHANNELS = 1  # Grayscale

# CRNN Architecture
CNN_FEATURE_DIM = 512
RNN_HIDDEN_SIZE = 256
RNN_NUM_LAYERS = 2
RNN_BIDIRECTIONAL = True

# Character set (vocabulary)
CHARS = string.digits + string.ascii_lowercase + string.ascii_uppercase + "():,.;?!'-\""
NUM_CLASSES = len(CHARS) + 1  # +1 for CTC blank token
CHAR_TO_IDX = {char: idx + 1 for idx, char in enumerate(CHARS)}  # 0 reserved for blank
IDX_TO_CHAR = {idx: char for char, idx in CHAR_TO_IDX.items()}
BLANK_TOKEN = 0

# Training parameters
BATCH_SIZE = 64
NUM_EPOCHS = 50
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 0 # was causing problems

# Data split
TRAIN_SPLIT = 0.75
VAL_SPLIT   = 0.10
TEST_SPLIT  = 0.15
RANDOM_SEED = 42

# Training settings
DEVICE = 'cuda'  # or 'cpu'
CHECKPOINT_FREQ = 5  # Save every N epochs
LOG_FREQ = 100  # Log every N batches
