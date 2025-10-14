import os, joblib, numpy as np
from tqdm import tqdm
from src.utils import find_images, images_by_class
from src.config import EXTRACT_DIR, OUTPUT_DIR, MAX_PER_CLASS, BATCH_SIZE
import cv2

try:
    from tensorflow.keras.preprocessing import image
    from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
    TF_AVAILABLE = True
except Exception:
    TF_AVAILABLE = False

def gather_sample_paths(base_dir, max_per_class=MAX_PER_CLASS):
    all_images = find_images(base_dir)
    by_cls = images_by_class(all_images)
    paths, labels = [], []
    for cls, imgs in by_cls.items():
        take = min(len(imgs), max_per_class)
        sel = imgs[:take]
        paths += sel
        labels += [cls]*len(sel)
    return paths, labels

def extract_resnet_embeddings(paths, batch_size=BATCH_SIZE):
    model = ResNet50(weights='imagenet', include_top=False, pooling='avg')
    feats = []
    for i in tqdm(range(0, len(paths), batch_size), desc='ResNet embeddings'):
        batch = paths[i:i+batch_size]
        arr = []
        for p in batch:
            try:
                img = image.load_img(p, target_size=(224,224))
                x = image.img_to_array(img)
                if x.shape[-1] == 1:
                    x = np.repeat(x, 3, axis=-1)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
                arr.append(x[0])
            except Exception:
                arr.append(np.zeros((224,224,3), dtype=np.float32))
        arr = np.stack(arr, axis=0)
        feat = model.predict(arr, verbose=0)
        feats.append(feat)
    X = np.vstack(feats)
    return X

def extract_histogram_features(paths):
    feats = []
    for p in tqdm(paths, desc='histograms'):
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None:
            feats.append([0]*32)
            continue
        img = cv2.resize(img, (128,128))
        h = cv2.calcHist([img],[0],None,[32],[0,256]).flatten()
        h = h / (h.sum()+1e-9)
        feats.append(h.tolist())
    return np.array(feats)

def run_embedding_extraction(extract_dir=EXTRACT_DIR, output_dir=OUTPUT_DIR):
    os.makedirs(output_dir, exist_ok=True)
    paths, labels = gather_sample_paths(extract_dir)
    print('Collected', len(paths), 'images for embedding extraction.')
    if TF_AVAILABLE:
        print('TensorFlow available — extracting ResNet50 embeddings.')
        X = extract_resnet_embeddings(paths)
    else:
        print('TensorFlow not available — extracting histogram features (fallback).')
        X = extract_histogram_features(paths)
    joblib.dump({'paths': paths, 'labels': labels, 'X': X}, os.path.join(output_dir, 'embeddings.joblib'))
    print('Saved embeddings.joblib to', output_dir)
    return os.path.join(output_dir, 'embeddings.joblib')

if __name__ == '__main__':
    run_embedding_extraction()
