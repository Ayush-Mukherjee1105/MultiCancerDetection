# src/train.py

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True) # Create models directory

# --- Main Training Logic ---

def train_ensemble_models():
    """
    Loads embeddings, trains a stacking ensemble model, and saves it to disk.
    """
    logging.info("--- Starting Model Training ---")

    # 1. Define file paths and check if they exist
    embeddings_path = EMBEDDINGS_DIR / "embeddings.npy"
    labels_path = EMBEDDINGS_DIR / "labels.npy"

    if not embeddings_path.exists() or not labels_path.exists():
        logging.error(f"Error: Embeddings file ({embeddings_path}) or labels file ({labels_path}) not found.")
        logging.error("Please run the 'embeddings.py' script successfully before running training.")
        return

    # 2. Load the pre-generated embeddings and labels
    X = np.load(embeddings_path)
    y_raw = np.load(labels_path)
    
    logging.info(f"Loaded embeddings with shape: {X.shape}")
    logging.info(f"Loaded labels with shape: {y_raw.shape}")

    # 3. Encode labels from strings (e.g., 'benign') to integers (e.g., 0)
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    joblib.dump(le, MODELS_DIR / "label_encoder.joblib")
    logging.info(f"Labels encoded into {len(le.classes_)} classes.")
    logging.info(f"Class names: {list(le.classes_)}")

    # 4. Split data into training and testing sets (stratified to maintain class proportions)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    logging.info(f"Data split into {len(X_train)} training and {len(X_test)} testing samples.")

    # 5. Handle class imbalance with SMOTE on the training data only
    logging.info("Applying SMOTE to handle class imbalance on the training set...")
    # n_jobs=-1 uses all available CPU cores
    smote = SMOTE(random_state=42, n_jobs=-1)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    logging.info(f"Training data resampled from {len(X_train)} to {len(X_train_resampled)} samples.")

    # 6. Define the base models for the ensemble
    estimators = [
        ('rf', RandomForestClassifier(n_estimators=150, random_state=42, n_jobs=-1, max_depth=15)),
        ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42, n_jobs=-1, n_estimators=200, learning_rate=0.1)),
    ]

    # 7. Create and train the Stacking Classifier
    # The final estimator learns to combine the predictions of the base models
    stacking_classifier = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(max_iter=1000, C=0.5),
        cv=5, # 5-fold cross-validation on the base models
        n_jobs=-1,
        passthrough=True # Allows the final estimator to also see the original data
    )
    
    logging.info("--- Training StackingEnsemble ---")
    stacking_classifier.fit(X_train_resampled, y_train_resampled)
    model_path = MODELS_DIR / "stackingensemble_model.joblib"
    joblib.dump(stacking_classifier, model_path)
    logging.info(f"Saved trained StackingEnsemble model to {model_path}")

    # 8. Save the test set for consistent evaluation
    np.save(EMBEDDINGS_DIR / "X_test.npy", X_test)
    np.save(EMBEDDINGS_DIR / "y_test.npy", y_test)
    logging.info("Test data saved for evaluation.")
    
    logging.info("--- Model training complete! ---")

if __name__ == '__main__':
    train_ensemble_models()