import numpy as np
import pandas as pd
from pathlib import Path
import logging
import joblib
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support
)
import seaborn as sns
import matplotlib.pyplot as plt

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# --- Main Evaluation Logic ---

def evaluate_model():
    """
    Loads the test data and the trained stacking model, evaluates its performance,
    and saves the results and visualizations.
    """
    logging.info("--- Starting Model Evaluation ---")

    try:
        X_test = np.load(EMBEDDINGS_DIR / "X_test.npy")
        y_test = np.load(EMBEDDINGS_DIR / "y_test.npy")
        le = joblib.load(MODELS_DIR / "label_encoder.joblib")
        model = joblib.load(MODELS_DIR / "stackingensemble_model.joblib")
    except FileNotFoundError as e:
        logging.error(f"Required file not found: {e}. Please run train.py successfully first.")
        return

    logging.info(f"Loaded test data with shape: {X_test.shape}")
    class_names = le.classes_

    logging.info("Making predictions on the test set...")
    y_pred = model.predict(X_test)

    logging.info("Generating classification report...")
    report_str = classification_report(y_test, y_pred, target_names=class_names)
    print("\n" + "="*60)
    print("      Classification Report for Stacking Ensemble")
    print("="*60)
    print(report_str)
    print("="*60 + "\n")
    
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    summary = {
        "Model": "Stacking Ensemble",
        "Accuracy": accuracy,
        "Precision (Weighted)": precision,
        "Recall (Weighted)": recall,
        "F1-Score (Weighted)": f1
    }
    
    logging.info("--- Overall Performance Summary ---")
    # --- THIS IS THE CORRECTED BLOCK ---
    for key, value in summary.items():
        if isinstance(value, str):
            logging.info(f"{key}: {value}")
        else:
            logging.info(f"{key}: {value:.4f}")
    # --- END OF CORRECTION ---

    summary_df = pd.DataFrame([summary])
    summary_path = OUTPUTS_DIR / "evaluation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    logging.info(f"Evaluation summary saved to {summary_path}")

    logging.info("Generating confusion matrix...")
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(16, 14))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix for Stacking Ensemble', fontsize=18)
    plt.ylabel('True Label', fontsize=14)
    plt.xlabel('Predicted Label', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    cm_path = OUTPUTS_DIR / "stacking_ensemble_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    logging.info(f"Saved confusion matrix to {cm_path}")
    
    logging.info("--- Evaluation Complete ---")

if __name__ == '__main__':
    evaluate_model()