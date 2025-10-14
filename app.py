# app.py -- Streamlit front-end for the Lung Cancer Ensemble classifier
# Put this file in your project root (same folder that contains `src/`, `outputs/`, run.py, archive.zip)
# Run with: streamlit run app.py

import os
import io
import joblib
import numpy as np
from PIL import Image
import streamlit as st

# Load config fallback (so we reuse same OUTPUT_DIR from project)
try:
    from src.config import OUTPUT_DIR, EXTRACT_DIR
except Exception:
    # fallback - assume outputs folder next to this file
    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(ROOT, "outputs")
    EXTRACT_DIR = os.path.join(ROOT, "data", "lung_images")

# Try importing TF Keras; if not available use histogram fallback
TF_AVAILABLE = True
try:
    from tensorflow.keras.preprocessing import image as kimage
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
except Exception:
    TF_AVAILABLE = False

# UI: Title
st.set_page_config(page_title="Lung Cancer Ensemble Classifier", layout="wide")
st.title("Lung Cancer Ensemble Classifier — Frontend")

# Sidebar: model selection and options
st.sidebar.header("Model selection & options")
models_present = []
for fn in os.listdir(OUTPUT_DIR) if os.path.exists(OUTPUT_DIR) else []:
    if fn.endswith("_model.joblib"):
        models_present.append(fn.replace("_model.joblib", ""))

if not models_present:
    st.sidebar.error("No trained models found in outputs/. Run training pipeline first.")
selected_models = st.sidebar.multiselect("Select model(s) to use (one or many):", options=sorted(models_present),
                                         default=["stack"] if "stack" in models_present else sorted(models_present)[:1])

soft_vote_weights = {}
if st.sidebar.checkbox("Enable Soft-Voting (weighted average of probabilities)", value=True):
    st.sidebar.markdown("If multiple base models selected, you can set weights.")
    for m in selected_models:
        w = st.sidebar.number_input(f"Weight for {m}", min_value=0.0, max_value=100.0, value=1.0, step=0.5)
        soft_vote_weights[m] = float(w)

show_shap = st.sidebar.checkbox("Show SHAP explanation (tree models only)", value=False)
sample_button = st.sidebar.button("Use random sample from dataset")

# Load label encoder & model objects
@st.cache_resource(show_spinner=False)
def load_models(output_dir):
    models = {}
    for fname in os.listdir(output_dir):
        if fname.endswith("_model.joblib"):
            name = fname.replace("_model.joblib", "")
            try:
                models[name] = joblib.load(os.path.join(output_dir, fname))
            except Exception as e:
                st.sidebar.warning(f"Could not load {fname}: {e}")
    # label encoder
    le = None
    le_path = os.path.join(output_dir, "label_encoder.joblib")
    if os.path.exists(le_path):
        le = joblib.load(le_path)
    return models, le

models_dict, label_encoder = load_models(OUTPUT_DIR)

# Load ResNet model lazily
resnet_model = None
if TF_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def get_resnet():
        return ResNet50(weights='imagenet', include_top=False, pooling='avg')
    resnet_model = get_resnet()

# Image loader + preprocess -> embedding (same logic as training)
def preprocess_image_to_embedding(pil_img):
    pil_img = pil_img.convert("RGB")
    img = pil_img.resize((224,224))
    arr = np.array(img).astype("float32")
    if TF_AVAILABLE:
        x = np.expand_dims(arr, axis=0)
        x = preprocess_input(x)
        feat = resnet_model.predict(x, verbose=0)  # shape (1, 2048)
        return feat.flatten()
    else:
        # grayscale histogram fallback
        img_gray = pil_img.convert("L").resize((128,128))
        arr = np.array(img_gray).flatten()
        hist, _ = np.histogram(arr, bins=32, range=(0,255))
        hist = hist / (hist.sum()+1e-9)
        return hist

# Image upload + sample selection
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Input")
    uploaded = st.file_uploader("Upload an image (jpg/png) or choose a sample", type=["jpg","jpeg","png"])
    if sample_button:
        # pick a random image from extracted dataset
        import random
        all_images = []
        if os.path.exists(EXTRACT_DIR):
            for root, _, files in os.walk(EXTRACT_DIR):
                for f in files:
                    if f.lower().endswith((".jpg",".jpeg",".png",".tif",".bmp")):
                        all_images.append(os.path.join(root,f))
        if all_images:
            chosen = random.choice(all_images)
            uploaded = open(chosen, "rb")
            st.success(f"Selected random sample: {chosen}")
        else:
            st.warning("No images found in dataset folder.")
    if uploaded:
        try:
            bytes_data = uploaded.read()
            pil = Image.open(io.BytesIO(bytes_data))
            st.image(pil, caption="Uploaded image", use_column_width=True)
        except Exception as e:
            st.error("Could not read uploaded file: " + str(e))
            pil = None
    else:
        pil = None

with col2:
    st.subheader("Selected models")
    if not selected_models:
        st.info("No models selected. Choose models from the sidebar.")
    else:
        st.write("Models to run:", selected_models)
        st.markdown("**Label encoder classes:** " + (", ".join(label_encoder.classes_) if label_encoder is not None else "unknown"))

# Run prediction button
if st.button("Run prediction"):
    if pil is None:
        st.error("Upload an image first (or press 'Use random sample').")
    elif not selected_models:
        st.error("Select at least one model in the sidebar.")
    else:
        X_emb = preprocess_image_to_embedding(pil).reshape(1,-1)
        # Collect per-model probabilities
        per_model_probs = {}
        per_model_preds = {}
        for mname in selected_models:
            if mname not in models_dict:
                st.warning(f"Model {mname} not found in outputs/; skipping.")
                continue
            model = models_dict[mname]
            # some models may not implement predict_proba (rare). handle gracefully.
            try:
                probs = model.predict_proba(X_emb)
            except Exception:
                # fallback: use decision_function + softmax or direct predict
                try:
                    dec = model.decision_function(X_emb)
                    # if multiclass, apply softmax
                    def softmax(a):
                        e = np.exp(a - np.max(a, axis=1, keepdims=True))
                        return e / e.sum(axis=1, keepdims=True)
                    probs = softmax(dec)
                except Exception:
                    pred = model.predict(X_emb)
                    probs = np.zeros((1, len(label_encoder.classes_))) if label_encoder is not None else np.zeros((1,1))
                    try:
                        probs[0, pred[0]] = 1.0
                    except Exception:
                        pass
            per_model_probs[mname] = probs.flatten()
            per_model_preds[mname] = np.argmax(probs, axis=1)[0]

        # Display numeric probabilities
        st.subheader("Per-model probabilities (numeric)")
        for mname, probs in per_model_probs.items():
            st.write(f"**{mname}** : ", {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))})

        # FORMULAS and calculation display
        st.subheader("Formulas & calculation steps")
        # Majority voting formula (hard voting)
        st.latex(r""" \hat{y}_{majority} = \mathrm{mode}\{\hat{y}_1, \hat{y}_2, ..., \hat{y}_M\} """)
        # Soft-voting formula (weighted average)
        st.latex(r""" P(y=c) = \frac{\sum_{m=1}^M w_m P_m(y=c)}{\sum_{m=1}^M w_m} """)
        # Show numeric example for soft voting if multiple models selected
        if len(per_model_probs) >= 2:
            st.markdown("**Soft-voting numeric example (per-class):**")
            # compute weighted average
            total_w = sum(soft_vote_weights.get(m,1.0) for m in per_model_probs.keys())
            class_names = list(label_encoder.classes_) if label_encoder is not None else [str(i) for i in range(len(next(iter(per_model_probs.values()))))]
            # compute weighted probs
            weighted = np.zeros(len(class_names), dtype=float)
            rows = []
            for mname, probs in per_model_probs.items():
                w = soft_vote_weights.get(mname, 1.0)
                rows.append((mname, w, probs))
                weighted += w * np.array(probs)
            weighted = weighted / (total_w + 1e-12)
            # show rows
            for (mname, w, probs) in rows:
                st.write(f"{mname} (weight={w}):")
                st.write({class_names[i]: float(probs[i]) for i in range(len(probs))})
            st.write("Weighted average (normalized):")
            st.write({class_names[i]: float(weighted[i]) for i in range(len(weighted))})
            pred_class = class_names[int(np.argmax(weighted))]
            st.success(f"Soft-voting predicted class: **{pred_class}** (prob {float(np.max(weighted)):.4f})")

        # Majority vote numeric
        st.markdown("**Majority vote (hard voting) numeric example:**")
        votes = [per_model_preds[m] for m in per_model_preds.keys()]
        # map to class names
        try:
            vote_names = [label_encoder.classes_[v] for v in votes]
        except Exception:
            vote_names = votes
        from collections import Counter
        vc = Counter(vote_names)
        st.write("Votes:", vc)
        top = vc.most_common(1)[0]
        st.success(f"Majority-vote predicted class: **{top[0]}** (count {top[1]})")

        # Stacking meta-estimator demonstration
        if "stack" in models_dict and "stack" in selected_models:
            st.markdown("**Stacking (meta-estimator) calculation**")
            stack_model = models_dict["stack"]
            # Build meta-features: join each base model probability vector (same approach used in training)
            # For simplicity, use probs from selected base models other than 'stack'
            base_models = [m for m in selected_models if m != "stack"]
            if len(base_models) == 0:
                st.info("No base models selected for stacking demonstration (select base models too).")
            else:
                meta = []
                for b in base_models:
                    meta.extend(per_model_probs[b])
                meta = np.array(meta).reshape(1,-1)
                # run meta prediction (if meta estimator has predict_proba)
                try:
                    meta_probs = stack_model.final_estimator_.predict_proba(meta)
                    st.write("Meta-estimator (final) probabilities:", {label_encoder.classes_[i]: float(meta_probs[0][i]) for i in range(meta_probs.shape[1])})
                    st.success(f"Stack predicted: {label_encoder.classes_[int(np.argmax(meta_probs))]}")
                except Exception as e:
                    st.info("Could not compute meta-estimator probabilities (meta estimator may be wrapped).")
                    try:
                        meta_pred = stack_model.final_estimator_.predict(meta)
                        st.write("Meta-estimator prediction:", meta_pred)
                    except Exception as e2:
                        st.write("Stack meta-eval failed:", e2)

        # Optionally SHAP (may be slow)
        if show_shap:
            try:
                import shap
                st.subheader("SHAP explanation (approx)")
                # pick first model that is tree-based
                tree_model = None
                for nm in selected_models:
                    m = models_dict.get(nm)
                    if m is None: continue
                    # heuristic: tree-based models have feature_importances_
                    if hasattr(m, "feature_importances_") or m.__class__.__name__.lower().startswith("xgb"):
                        tree_model = (nm, m)
                        break
                if tree_model is None:
                    st.info("No tree-based model selected (RandomForest/GradientBoost/XGBoost recommended) for SHAP.")
                else:
                    nm, m = tree_model
                    explainer = shap.TreeExplainer(m)
                    shap_values = explainer.shap_values(X_emb)
                    st.write(f"Showing SHAP (model: {nm}) — summary plot below (may take a moment).")
                    # produce a plot and show
                    import matplotlib.pyplot as plt
                    fig = plt.figure(figsize=(6,4))
                    try:
                        shap.summary_plot(shap_values, X_emb, show=False)
                    except Exception:
                        # alternate for multiclass
                        shap.summary_plot(shap_values, X_emb, class_names=label_encoder.classes_, show=False)
                    st.pyplot(fig)
            except Exception as e:
                st.error("SHAP failed (maybe not installed or model incompatible): " + str(e))

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for the Lung Cancer Ensemble project. Ensure outputs/ contains trained models.")
