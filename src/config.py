import os

# Project root (parent of src/)
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Put archive.zip in the project root (next to README, run.py, src/)
ZIP_PATH = os.path.join(ROOT, 'archive.zip')

# Data and outputs inside the project root
EXTRACT_DIR = os.path.join(ROOT, 'data', 'lung_images')
OUTPUT_DIR = os.path.join(ROOT, 'outputs')

# Tweak these for quick testing
MAX_PER_CLASS = 500
BATCH_SIZE = 32
RANDOM_STATE = 42
