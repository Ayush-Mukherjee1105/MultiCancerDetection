import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging

# Import the feature extractor we already built
from transformer_model import VisionTransformerExtractor

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths (This is where the variable is defined) ---
BASE_DIR = Path(__file__).resolve().parent.parent
CLEAN_DATA_DIR = BASE_DIR / "data" / "clean"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

# --- Define labels to explicitly ignore ---
LABELS_TO_IGNORE = {"test", "10"}

# --- Main Embedding Generation Logic ---

def generate_embeddings():
    """
    Iterates through the cleaned data, extracts embeddings using the ViT model,
    and saves the embeddings, labels, and a manifest file.
    """
    logging.info("--- Starting Embedding Generation ---")
    
    try:
        extractor = VisionTransformerExtractor()
    except Exception as e:
        logging.error(f"Failed to initialize VisionTransformerExtractor: {e}")
        return

    image_formats = ['*.png', '*.jpg', '*.jpeg']
    all_image_paths = []
    for fmt in image_formats:
        all_image_paths.extend(CLEAN_DATA_DIR.rglob(fmt))

    if not all_image_paths:
        logging.error(f"No images found in {CLEAN_DATA_DIR}. Please check the data preparation step.")
        return
        
    logging.info(f"Found {len(all_image_paths)} total image files. Filtering and processing...")

    all_embeddings = []
    all_labels = []
    manifest_data = []
    skipped_count = 0

    for path in tqdm(all_image_paths, desc="Extracting Embeddings"):
        label = path.parent.name
        
        # Skip ignored labels and any remaining mask files
        if label in LABELS_TO_IGNORE or '_mask' in path.name:
            skipped_count += 1
            continue
            
        dataset = path.parent.parent.name
        embedding = extractor.get_embedding(str(path))
        
        if embedding is not None:
            all_embeddings.append(embedding)
            all_labels.append(label)
            manifest_data.append({
                "image_path": str(path.relative_to(BASE_DIR)),
                "label": label,
                "dataset": dataset
            })
        else:
            logging.warning(f"Skipping image due to an error: {path}")

    if skipped_count > 0:
        logging.info(f"Skipped {skipped_count} images from ignored directories or mask files.")

    if not all_embeddings:
        logging.error("No embeddings were generated. Please check your data folders and the ignore list.")
        return

    embeddings_array = np.array(all_embeddings)
    labels_array = np.array(all_labels)
    
    # --- The lines that caused the error now have their variable defined ---
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    labels_path = EMBEDDINGS_DIR / "labels.npy"
    manifest_path = EMBEDDINGS_DIR / "manifest.csv"
    
    np.save(embeddings_path, embeddings_array)
    np.save(labels_path, labels_array)
    
    manifest_df = pd.DataFrame(manifest_data)
    manifest_df.to_csv(manifest_path, index=False)
    
    logging.info("--- Embedding Generation Complete ---")
    logging.info(f"Saved embeddings array of shape {embeddings_array.shape} to: {embeddings_path}")
    logging.info(f"Saved labels array of shape {labels_array.shape} to: {labels_path}")
    logging.info(f"Saved manifest CSV with {len(manifest_df)} entries to: {manifest_path}")

if __name__ == '__main__':
    generate_embeddings()