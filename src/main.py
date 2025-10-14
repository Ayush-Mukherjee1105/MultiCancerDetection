import argparse
import os
from src.unzip_dataset import unzip_dataset
from src.embeddings import run_embedding_extraction
from src.train import train_from_embeddings
from src.evaluate import evaluate_models
from src.config import OUTPUT_DIR

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--unzip', action='store_true', help='Unzip dataset')
    p.add_argument('--extract', action='store_true', help='Extract embeddings')
    p.add_argument('--train', action='store_true', help='Train models')
    p.add_argument('--evaluate', action='store_true', help='Evaluate models')
    p.add_argument('--run_all', action='store_true', help='Run unzip -> extract -> train -> evaluate')
    return p.parse_args()

def run_all():
    print("Unzipping dataset...")
    unzip_dataset()
    print("Extracting embeddings...")
    emb_file = run_embedding_extraction()
    print("Training models...")
    train_from_embeddings(emb_file)
    print("Evaluating models...")
    evaluate_models()
    print("All steps completed.")
# ...existing code...

if __name__ == '__main__':
    args = parse_args()
    if args.run_all:
        run_all()
    else:
        if args.unzip:
            unzip_dataset()
        if args.extract:
            run_embedding_extraction()
        if args.train:
            emb = os.path.join(OUTPUT_DIR, 'embeddings.joblib')
            train_from_embeddings(emb)
        if args.evaluate:
            evaluate_models()
