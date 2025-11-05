# app.py

import streamlit as st
from PIL import Image
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import shap
import matplotlib.pyplot as plt
import random
import sys

# --- App Configuration ---
st.set_page_config(
    page_title="AI Medical Image Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- USER-FRIENDLY LABEL MAPPING ---
# This dictionary translates the model's technical labels into plain English.
LABEL_MAPPING = {
    'adipose': 'Adipose Tissue (Colon, Fatty)',
    'basophil': 'Basophil (Blood Cell)',
    'benign': 'Benign Tumor (Non-Cancerous)',
    'complex': 'Complex Colon Tissue',
    'debris': 'Debris (Colon)',
    'empty': 'Empty Slide Area',
    'erythroblast': 'Erythroblast (Immature Red Blood Cell)',
    'glioma': 'Glioma Tumor (Brain)',
    'lympho': 'Lymphocytes (Colon)',
    'malignant': 'Malignant Tumor (Cancerous)',
    'meningioma': 'Meningioma Tumor (Brain)',
    'monocyte': 'Monocyte (Blood Cell)',
    'mucosa': 'Healthy Colon Lining',
    'myeloblast': 'Myeloblast (Immature White Blood Cell)',
    'normal': 'Normal Breast Tissue',
    'notumor': 'No Tumor Detected (Brain)',
    'pituitary': 'Pituitary Tumor (Brain)',
    'seg_neutrophil': 'Neutrophil (Blood Cell)',
    'stroma': 'Stroma (Connective Tissue, Colon)',
    'tumor': 'Tumor Tissue (Colon)'
}

# --- Asset Loading ---
@st.cache_resource
def load_assets():
    """Loads all necessary models, data, and the SHAP explainer."""
    base_dir = Path(__file__).resolve().parent
    models_dir = base_dir / "models"
    data_dir = base_dir / "data" / "clean"
    
    try:
        model = joblib.load(models_dir / "stackingensemble_model.joblib")
        le = joblib.load(models_dir / "label_encoder.joblib")
        sys.path.append(str(base_dir / "src"))
        from transformer_model import VisionTransformerExtractor
        extractor = VisionTransformerExtractor()
        base_model = model.named_estimators_['xgb']
        explainer = shap.TreeExplainer(base_model)
        image_paths = [p for p in list(data_dir.rglob("*.*")) if p.parent.name not in ["test", "10"]]
    except Exception as e:
        st.error(f"Error loading application assets: {e}")
        return None, None, None, None, None
        
    return model, le, extractor, explainer, image_paths

# --- Main App Interface ---
st.title("AI Medical Image Analyzer")
st.write("This tool uses a deep learning model to analyze medical images. Upload an image or use a random sample to get a prediction about the tissue type.")

model, le, extractor, explainer, image_paths = load_assets()

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Controls")
    uploaded_file = st.file_uploader("Upload Your Image", type=["png", "jpg", "jpeg"])
    if st.button("Use a Random Sample Image", use_container_width=True):
        if image_paths:
            st.session_state.image_to_process = random.choice(image_paths)
            if 'uploaded_file' in st.session_state: del st.session_state['uploaded_file']
        else:
            st.warning("No clean images found to test with.")

# --- Analysis Section ---
image_path = None
if uploaded_file:
    image_path = uploaded_file
    if 'image_to_process' in st.session_state: del st.session_state['image_to_process']
elif 'image_to_process' in st.session_state:
    image_path = st.session_state.image_to_process

if model and image_path:
    col1, col2 = st.columns([2, 3])
    
    try:
        image = Image.open(image_path)
        with col1:
            st.header("Image Preview")
            st.image(image, caption="Image for Analysis", use_container_width=True)

        with col2:
            st.header("AI Analysis")
            with st.spinner("Analyzing..."):
                # 1. Process Image
                temp_path = Path("temp_image.png")
                image.convert("RGB").save(temp_path)
                embedding = extractor.get_embedding(str(temp_path))
                embedding_reshaped = embedding.reshape(1, -1)
                
                # 2. Predict
                prediction_encoded = model.predict(embedding_reshaped)[0]
                prediction_proba = model.predict_proba(embedding_reshaped)
                predicted_class_technical = le.inverse_transform([prediction_encoded])[0]
                confidence = prediction_proba[0][prediction_encoded]
                
                # --- TRANSLATE TO USER-FRIENDLY LABEL ---
                display_class = LABEL_MAPPING.get(predicted_class_technical, predicted_class_technical.title())
                st.success(f"**Prediction:** {display_class}")
                st.metric("Confidence", f"{confidence:.2%}")

                # 3. Explain (SHAP)
                st.subheader("Prediction Explanation")
                shap_explanation = explainer(embedding_reshaped)
                shap_values_for_class = shap_explanation.values[0, :, prediction_encoded]
                base_value_for_class = shap_explanation.base_values[0, prediction_encoded]
                
                force_plot = shap.force_plot(base_value_for_class, shap_values_for_class, embedding_reshaped[0], matplotlib=True, show=False, text_rotation=15)
                st.pyplot(force_plot, bbox_inches='tight', use_container_width=True)

                with st.expander("How to read this plot?"):
                    st.write("""
                    This plot shows the internal reasoning of the AI model.
                    - **Base Value:** The model's average prediction for this category.
                    - **Features in Red:** Abstract features learned from the image that **pushed the prediction higher** (confirming the result).
                    - **Features in Blue:** Abstract features that **pushed the prediction lower** (contradicting the result).
                    The final prediction is the sum of all these feature influences.
                    """)
                
                # 4. Show Full Probabilities with user-friendly names
                st.subheader("All Class Probabilities")
                proba_df = pd.DataFrame({'Technical Label': le.classes_, 'Probability': prediction_proba[0]})
                # --- TRANSLATE ALL LABELS IN THE TABLE ---
                proba_df['Class'] = proba_df['Technical Label'].map(LABEL_MAPPING)
                proba_df = proba_df[['Class', 'Probability']].sort_values('Probability', ascending=False)
                st.dataframe(proba_df, use_container_width=True)
                
                if temp_path.exists(): temp_path.unlink()
                if 'image_to_process' in st.session_state: del st.session_state['image_to_process']

    except Exception as e:
        st.error(f"An error occurred during analysis: {e}")

st.markdown("---")
st.warning("**Disclaimer:** This tool is for educational and research purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.", icon="⚠️")