import os, sys
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

print("✅ Launcher active: calling src.main")

from src.main import parse_args, run_all, unzip_dataset, run_embedding_extraction, train_from_embeddings, evaluate_models
from src.config import OUTPUT_DIR

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
