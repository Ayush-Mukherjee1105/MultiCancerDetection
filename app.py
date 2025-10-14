# app.py - Streamlit front-end for Lung Cancer Ensemble classifier
# Place this file in project root (same folder as src/, outputs/, archive.zip)
# Run: streamlit run app.py

import os
import io
import random
import joblib
import numpy as np
from PIL import Image
import streamlit as st

# Try to import project config
try:
    from src.config import OUTPUT_DIR, EXTRACT_DIR
except Exception:
    ROOT = os.path.dirname(os.path.abspath(__file__))
    OUTPUT_DIR = os.path.join(ROOT, "outputs")
    EXTRACT_DIR = os.path.join(ROOT, "data", "lung_images")

# Try TF Keras for ResNet embeddings, else fallback to histogram features
TF_AVAILABLE = True
try:
    from tensorflow.keras.preprocessing import image as kimage
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
except Exception:
    TF_AVAILABLE = False

st.set_page_config(page_title="Lung Cancer Ensemble Classifier", layout="wide")
st.title("Lung Cancer Ensemble Classifier — Frontend")

# Sidebar controls
st.sidebar.header("Model selection & options")
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR, exist_ok=True)

models_present = []
for fn in os.listdir(OUTPUT_DIR):
    if fn.endswith("_model.joblib"):
        models_present.append(fn.replace("_model.joblib", ""))

if not models_present:
    st.sidebar.error("No trained models found in outputs/. Run the training pipeline first.")
selected_models = st.sidebar.multiselect(
    "Select model(s) to use (one or many):",
    options=sorted(models_present),
    default=["stack"] if "stack" in models_present else (sorted(models_present)[:1] if models_present else [])
)

use_soft_vote = st.sidebar.checkbox("Enable Soft-Voting (weighted average of probabilities)", value=True)
soft_vote_weights = {}
if use_soft_vote:
    st.sidebar.markdown("If multiple base models selected, set weights:")
    for m in selected_models:
        soft_vote_weights[m] = float(st.sidebar.number_input(f"Weight for {m}", min_value=0.0, max_value=100.0, value=1.0, step=0.5))

show_shap = st.sidebar.checkbox("Show SHAP explanation (tree models only)", value=False)
sample_button = st.sidebar.button("Use random sample from dataset")

# Load label encoder and models (cached)
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
    le_path = os.path.join(output_dir, "label_encoder.joblib")
    le = joblib.load(le_path) if os.path.exists(le_path) else None
    return models, le

models_dict, label_encoder = load_models(OUTPUT_DIR)

# Load ResNet lazily (cached)
resnet_model = None
if TF_AVAILABLE:
    @st.cache_resource(show_spinner=False)
    def get_resnet():
        return ResNet50(weights='imagenet', include_top=False, pooling='avg')
    resnet_model = get_resnet()

# Utilities
def preprocess_image_to_embedding(pil_img):
    pil_img = pil_img.convert("RGB")
    if TF_AVAILABLE and resnet_model is not None:
        img = pil_img.resize((224,224))
        arr = np.array(img).astype("float32")
        x = np.expand_dims(arr, axis=0)
        x = preprocess_input(x)
        feat = resnet_model.predict(x, verbose=0)
        return feat.flatten()
    else:
        # histogram fallback
        img_gray = pil_img.convert("L").resize((128,128))
        arr = np.array(img_gray).flatten()
        hist, _ = np.histogram(arr, bins=32, range=(0,255))
        hist = hist / (hist.sum()+1e-9)
        return hist

# Uploader + Random sample UI (robust using session_state)
col1, col2 = st.columns([1,2])
with col1:
    st.subheader("Input")
    uploaded_file = st.file_uploader("Upload an image (jpg/png) or choose a sample", type=["jpg","jpeg","png"])

    if sample_button:
        # choose random image from EXTRACT_DIR and store path in session_state
        all_images = []
        if os.path.exists(EXTRACT_DIR):
            for root, _, files in os.walk(EXTRACT_DIR):
                for f in files:
                    if f.lower().endswith((".jpg", ".jpeg", ".png", ".tif", ".bmp")):
                        all_images.append(os.path.join(root, f))
        if all_images:
            chosen = random.choice(all_images)
            st.session_state['sample_path'] = chosen
            st.success(f"Selected random sample: {chosen}")
        else:
            st.warning("No images found in dataset folder.")
            if 'sample_path' in st.session_state:
                del st.session_state['sample_path']

    # Determine source: uploaded file takes precedence
    pil = None
    if uploaded_file is not None:
        try:
            bytes_data = uploaded_file.read()
            pil = Image.open(io.BytesIO(bytes_data))
            st.image(pil, caption="Uploaded image", use_container_width=True)
        except Exception as e:
            st.error("Could not read uploaded file: " + str(e))
            pil = None
    elif 'sample_path' in st.session_state:
        sample_path = st.session_state['sample_path']
        try:
            pil = Image.open(sample_path)
            st.image(pil, caption=f"Sample image: {os.path.basename(sample_path)}", use_container_width=True)
        except Exception as e:
            st.error("Could not open sample image: " + str(e))
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

# Prediction & calculations
X_emb = None  # will hold embedding for SHAP block
if st.button("Run prediction"):
    if pil is None:
        st.error("Upload an image first (or press 'Use random sample').")
    elif not selected_models:
        st.error("Select at least one model in the sidebar.")
    else:
        # compute embedding
        with st.spinner("Computing embedding..."):
            X_emb = preprocess_image_to_embedding(pil).reshape(1,-1)

        per_model_probs = {}
        per_model_preds = {}

        for mname in selected_models:
            if mname not in models_dict:
                st.warning(f"Model {mname} not found in outputs/; skipping.")
                continue
            model = models_dict[mname]
            # try predict_proba, fallback to decision_function or predict
            probs = None
            try:
                probs = model.predict_proba(X_emb)
            except Exception:
                try:
                    dec = model.decision_function(X_emb)
                    # softmax for multiclass
                    if dec.ndim == 1:
                        # binary or single-dim decision
                        probs = np.vstack([1 - 1/(1+np.exp(dec)), 1/(1+np.exp(dec))]).T
                    else:
                        e = np.exp(dec - np.max(dec, axis=1, keepdims=True))
                        probs = e / e.sum(axis=1, keepdims=True)
                except Exception:
                    pred = model.predict(X_emb)
                    probs = np.zeros((1, len(label_encoder.classes_))) if label_encoder is not None else np.zeros((1,1))
                    try:
                        probs[0, pred[0]] = 1.0
                    except Exception:
                        pass
            per_model_probs[mname] = probs.flatten()
            per_model_preds[mname] = int(np.argmax(probs, axis=1)[0])

        # Show per-model numeric probabilities
        st.subheader("Per-model probabilities (numeric)")
        for mname, probs in per_model_probs.items():
            st.write(f"**{mname}** : ", {label_encoder.classes_[i]: float(probs[i]) for i in range(len(probs))})

        # Formulas display
        st.subheader("Formulas & calculation steps")
        st.latex(r"\hat{y}_{majority} = \mathrm{mode}\{\hat{y}_1, \hat{y}_2, ..., \hat{y}_M\}")
        st.latex(r"P(y=c) = \frac{\sum_{m=1}^M w_m P_m(y=c)}{\sum_{m=1}^M w_m}")

        # Soft-voting numeric example
        if len(per_model_probs) >= 2 and use_soft_vote:
            st.markdown("**Soft-voting numeric example (per-class):**")
            total_w = sum(soft_vote_weights.get(m, 1.0) for m in per_model_probs.keys())
            class_names = list(label_encoder.classes_) if label_encoder is not None else [str(i) for i in range(len(next(iter(per_model_probs.values()))))]
            weighted = np.zeros(len(class_names), dtype=float)
            rows = []
            for mname, probs in per_model_probs.items():
                w = soft_vote_weights.get(mname, 1.0)
                rows.append((mname, w, probs))
                weighted += w * np.array(probs)
            weighted = weighted / (total_w + 1e-12)
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
        try:
            vote_names = [label_encoder.classes_[v] for v in votes]
        except Exception:
            vote_names = votes
        from collections import Counter
        vc = Counter(vote_names)
        st.write("Votes:", vc)
        top = vc.most_common(1)[0]
        st.success(f"Majority-vote predicted class: **{top[0]}** (count {top[1]})")

        # Stacking demonstration (if stack selected)
        if "stack" in models_dict and "stack" in selected_models:
            st.markdown("**Stacking (meta-estimator) calculation**")
            stack_model = models_dict["stack"]
            base_models = [m for m in selected_models if m != "stack"]
            if len(base_models) == 0:
                st.info("No base models selected for stacking demonstration (select base models too).")
            else:
                meta = []
                for b in base_models:
                    meta.extend(per_model_probs[b])
                meta = np.array(meta).reshape(1, -1)
                try:
                    # try to use final_estimator directly if accessible
                    meta_probs = stack_model.final_estimator_.predict_proba(meta)
                    st.write("Meta-estimator (final) probabilities:", {label_encoder.classes_[i]: float(meta_probs[0][i]) for i in range(meta_probs.shape[1])})
                    st.success(f"Stack predicted: {label_encoder.classes_[int(np.argmax(meta_probs))]}")
                except Exception:
                    try:
                        meta_pred = stack_model.final_estimator_.predict(meta)
                        st.write("Meta-estimator prediction:", meta_pred)
                    except Exception as e:
                        st.write("Stack meta-eval failed:", e)

        # ---------- SHAP explanation (robust, full block) ----------
        if show_shap:
            try:
                import shap
                import matplotlib.pyplot as plt
                st.subheader("SHAP explanation (single-sample & global importance)")

                background_path = os.path.join(OUTPUT_DIR, "data_splits.joblib")
                if not os.path.exists(background_path):
                    st.info("No data_splits.joblib in outputs/ — save data_splits.joblib during training to enable SHAP.")
                else:
                    splits = joblib.load(background_path)
                    X_bg = splits.get('X_train', None)
                    if X_bg is None:
                        X_bg = splits.get('X_test', None)
                    if X_bg is None:
                        st.info("data_splits.joblib doesn't contain X_train/X_test arrays.")
                    else:
                        bg_size = min(100, int(X_bg.shape[0]))
                        rng = np.random.default_rng(0)
                        idx = rng.choice(X_bg.shape[0], size=bg_size, replace=False)
                        background = X_bg[idx]

                        # pick a tree-based model or stack final estimator
                        tree_model = None
                        tree_name = None
                        for nm in selected_models:
                            m = models_dict.get(nm)
                            if m is None:
                                continue
                            if hasattr(m, "feature_importances_") or m.__class__.__name__.lower().startswith("xgb"):
                                tree_model = m
                                tree_name = nm
                                break
                        if tree_model is None and "stack" in models_dict:
                            try:
                                cand = models_dict["stack"].final_estimator_
                                if hasattr(cand, "feature_importances_") or cand.__class__.__name__.lower().startswith("xgb"):
                                    tree_model = cand
                                    tree_name = "stack.final_estimator"
                            except Exception:
                                pass

                        if tree_model is None:
                            st.info("No tree-based model found among selected models. SHAP TreeExplainer requires tree models (RF/GBM/XGBoost).")
                        else:
                            st.write(f"Explaining model: **{tree_name or 'unknown'}** (background size={background.shape[0]})")
                            try:
                                X_sample = np.asarray(X_emb)
                                if X_sample.ndim == 1:
                                    X_sample = X_sample.reshape(1, -1)

                                # Build explainer (prefer TreeExplainer)
                                try:
                                    explainer = shap.TreeExplainer(tree_model, data=background, feature_perturbation="interventional")
                                except Exception:
                                    explainer = shap.Explainer(tree_model, background)

                                shap_values = explainer.shap_values(X_sample)

                                # debug info (safe)
                                def safe_info(obj):
                                    try:
                                        a = np.asarray(obj)
                                        return f"type={type(obj)}, np_shape={a.shape}, dtype={a.dtype if hasattr(a,'dtype') else 'n/a'}"
                                    except Exception:
                                        return f"type={type(obj)}, repr={repr(obj)[:200]}"
                                expected_value = getattr(explainer, "expected_value", None)
                                st.write("SHAP DEBUG: explainer.expected_value ->", safe_info(expected_value))
                                st.write("SHAP DEBUG: shap_values ->", safe_info(shap_values))

                                # Determine predicted class index (robust)
                                class_idx = 0
                                try:
                                    if hasattr(tree_model, "predict_proba"):
                                        p = tree_model.predict_proba(X_sample)[0]
                                        class_idx = int(np.argmax(p))
                                    else:
                                        found = None
                                        for nm_tmp, mt in models_dict.items():
                                            if hasattr(mt, "predict_proba"):
                                                found = mt.predict_proba(X_sample)[0]
                                                break
                                        if found is not None:
                                            class_idx = int(np.argmax(found))
                                except Exception:
                                    class_idx = 0

                                # Normalize shap_values into vals (1d array for single sample)
                                vals = None
                                global_mean_abs = None
                                arr = shap_values
                                if isinstance(arr, list):
                                    sel = min(class_idx, len(arr)-1)
                                    sv = np.asarray(arr[sel])
                                    if sv.ndim == 2:
                                        vals = sv[0]
                                    else:
                                        vals = sv.reshape(sv.shape[0], -1)[0]
                                    mean_abs_per_class = []
                                    for item in arr:
                                        a = np.abs(np.asarray(item))
                                        if a.ndim == 1:
                                            mean_abs_per_class.append(a)
                                        else:
                                            mean_abs_per_class.append(a.mean(axis=0))
                                    global_mean_abs = np.mean(mean_abs_per_class, axis=0)
                                else:
                                    arr_np = np.asarray(arr)
                                    if arr_np.ndim == 2:
                                        vals = arr_np[0]
                                        global_mean_abs = np.abs(arr_np).mean(axis=0)
                                    elif arr_np.ndim == 3:
                                        n_classes = arr_np.shape[0]
                                        use_class = min(class_idx, n_classes - 1)
                                        vals = arr_np[use_class, 0, :]
                                        global_mean_abs = np.mean(np.abs(arr_np), axis=(0,1))
                                    else:
                                        flat = np.abs(arr_np).reshape(-1, arr_np.shape[-1])
                                        vals = flat[0]
                                        global_mean_abs = np.abs(flat).mean(axis=0)

                                vals = np.asarray(vals)
                                global_mean_abs = np.asarray(global_mean_abs)

                                # Waterfall (try new API) -> fallback -> bar
                                plotted = False
                                try:
                                    base_value = 0.0
                                    ev = getattr(explainer, "expected_value", None)
                                    if ev is not None:
                                        ev_arr = np.asarray(ev)
                                        if ev_arr.ndim == 0:
                                            base_value = float(ev_arr)
                                        else:
                                            pick = class_idx if class_idx < ev_arr.size else 0
                                            base_value = float(ev_arr.flat[pick])
                                except Exception:
                                    base_value = 0.0

                                try:
                                    expl = shap.Explanation(values=vals, base_values=base_value, data=X_sample[0])
                                    plt.figure(figsize=(6,4))
                                    shap.plots.waterfall(expl, show=False)
                                    st.pyplot(plt.gcf())
                                    plotted = True
                                except Exception:
                                    try:
                                        plt.figure(figsize=(6,4))
                                        shap.plots._waterfall.waterfall_legacy(base_value, vals, feature_names=None, show=False)
                                        st.pyplot(plt.gcf())
                                        plotted = True
                                    except Exception:
                                        plotted = False

                                if not plotted:
                                    st.warning("Could not produce waterfall; showing bar of |SHAP| for features (single sample).")
                                    abs_vals = np.abs(vals)
                                    topk = min(30, abs_vals.size)
                                    idxs = np.argsort(abs_vals)[-topk:][::-1]
                                    feat_names = [f"f{i}" for i in range(len(abs_vals))]
                                    fig_bar = plt.figure(figsize=(6, max(2, topk*0.15)))
                                    plt.barh(range(len(idxs))[::-1], abs_vals[idxs])
                                    plt.yticks(range(len(idxs)), [feat_names[i] for i in idxs])
                                    plt.xlabel("|SHAP value| (single sample)")
                                    st.pyplot(fig_bar)

                                if global_mean_abs is not None:
                                    st.subheader("Global importance (mean |SHAP| across background)")
                                    topk = min(20, global_mean_abs.size)
                                    idxs = np.argsort(global_mean_abs)[-topk:][::-1]
                                    feat_names = [f"f{i}" for i in range(len(global_mean_abs))]
                                    fig_glob = plt.figure(figsize=(6, max(2, topk*0.15)))
                                    plt.barh(range(len(idxs))[::-1], global_mean_abs[idxs])
                                    plt.yticks(range(len(idxs)), [feat_names[i] for i in idxs])
                                    plt.xlabel("mean |SHAP value|")
                                    st.pyplot(fig_glob)

                            except Exception as e_inner:
                                import traceback
                                st.error("SHAP computation failed (inner). See debug below.")
                                st.text(traceback.format_exc())
            except Exception as e_outer:
                import traceback
                st.error("SHAP construction/execution failed.")
                st.text(traceback.format_exc())
# End SHAP block

st.sidebar.markdown("---")
st.sidebar.markdown("Developed for the Lung Cancer Ensemble project. Ensure outputs/ contains trained models.")
