import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Path to the uploaded zip file (put archive.zip next to project root or change this)
ZIP_PATH = os.path.join(os.path.dirname(ROOT), 'archive.zip')
EXTRACT_DIR = os.path.join(os.path.dirname(ROOT), 'data', 'lung_images')
OUTPUT_DIR = os.path.join(os.path.dirname(ROOT), 'outputs')
MAX_PER_CLASS = 500  # set to 100 or 200 for quick tests
BATCH_SIZE = 32
RANDOM_STATE = 42