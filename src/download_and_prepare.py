# src/download_and_prepare.py

import os
import zipfile
import shutil
import logging
from pathlib import Path
from PIL import Image

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
Image.MAX_IMAGE_PIXELS = None # Allows Pillow to handle very large images

DATASETS = {
    "lung": "adityamahimkar/iqothnccd-lung-cancer-dataset",
    "breast": "aryashah2k/breast-ultrasound-images-dataset",
    "brain": "masoudnickparvar/brain-tumor-mri-dataset",
    "blood": "sumithsingh/blood-cell-images-for-cancer-detection",
    "colon": "kmader/colorectal-histology-mnist",
}

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
CLEAN_DATA_DIR = BASE_DIR / "data" / "clean"

# --- STAGE 1: DUMB EXTRACTION ---

def download_and_unzip():
    """Downloads all datasets and extracts them into their own raw folders."""
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
    except Exception as e:
        logging.error(f"Kaggle API authentication failed. Is kaggle.json configured? Error: {e}")
        return False

    for name, slug in DATASETS.items():
        logging.info(f"--- Processing: {name.upper()} ---")
        extract_path = RAW_DATA_DIR / name
        if extract_path.exists():
            logging.info(f"Raw data for '{name}' already extracted. Skipping download.")
            continue

        try:
            zip_path = RAW_DATA_DIR / f"{slug.replace('/', '_')}.zip"
            logging.info(f"Downloading '{slug}'...")
            api.dataset_download_files(slug, path=RAW_DATA_DIR, unzip=False, quiet=True)
            
            # Find and rename the downloaded zip for consistency
            original_zip_name = RAW_DATA_DIR / f"{slug.split('/')[1]}.zip"
            if original_zip_name.exists():
                original_zip_name.rename(zip_path)

            logging.info(f"Extracting '{zip_path.name}'...")
            with zipfile.ZipFile(zip_path, 'r') as z:
                z.extractall(extract_path)
            os.remove(zip_path) # Clean up the zip file
            logging.info("Extraction complete.")

        except Exception as e:
            if "403" in str(e):
                logging.error(f"PERMISSION DENIED for '{name}'. ACTION REQUIRED:")
                logging.error(f"Please visit: https://www.kaggle.com/datasets/{slug}, log in, and accept the terms.")
            else:
                logging.error(f"Failed to process '{name}'. Error: {e}")
    return True

# --- STAGE 2: INTELLIGENT CLEANING ---

def clean_and_organize():
    """
    Scans the raw data folders, identifies valid images, sanitizes labels,
    converts formats, and copies them to the clean directory. This is the robust part.
    """
    logging.info("\n--- Starting intelligent cleaning and organization ---")
    CLEAN_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for name in DATASETS.keys():
        logging.info(f"Cleaning dataset: {name.upper()}")
        raw_folder = RAW_DATA_DIR / name
        clean_folder = CLEAN_DATA_DIR / name

        if not raw_folder.exists():
            logging.warning(f"Raw data for '{name}' not found. Skipping.")
            continue
        
        # Find ALL image files, regardless of nesting depth
        image_extensions = ["*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff"]
        all_images = []
        for ext in image_extensions:
            all_images.extend(list(raw_folder.rglob(ext)))

        if not all_images:
            logging.error(f"Could not find any images for '{name}' in {raw_folder}. This is unexpected.")
            continue
            
        logging.info(f"Found {len(all_images)} total image files. Filtering and processing...")
        
        for src_path in all_images:
            # ROBUST MASK FILTERING
            if "mask" in src_path.name.lower():
                continue

            # RELIABLE LABEL EXTRACTION
            label = src_path.parent.name
            
            # Sanitize labels for consistency
            label = label.lower().replace(" cases", "").replace("bengin", "benign")
            if name == "colon": label = label.split('_')[-1] if '_' in label else label

            # Create destination and copy/convert
            dest_dir = clean_folder / label
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest_path = dest_dir / f"{src_path.stem}.png" # Standardize to PNG

            try:
                if src_path.suffix.lower() in [".tif", ".tiff"]:
                    with Image.open(src_path) as img:
                        img.convert("RGB").save(dest_path, "PNG")
                else:
                    # For png/jpg, open and save to standardize and remove potential corruption
                    with Image.open(src_path) as img:
                        img.convert("RGB").save(dest_path, "PNG")
            except Exception as e:
                logging.warning(f"Could not process file {src_path}. Skipping. Error: {e}")
    
    logging.info("--- Intelligent cleaning complete. ---")

# --- MAIN ORCHESTRATOR ---

def main():
    if download_and_unzip():
        clean_and_organize()
        # Optional: Clean up the entire raw directory at the end
        # logging.info("Cleaning up raw data directory...")
        # shutil.rmtree(RAW_DATA_DIR)
        logging.info("--- Data preparation finished successfully. ---")
    else:
        logging.error("--- Data preparation failed due to download/unzip errors. ---")

if __name__ == "__main__":
    main()