import os, joblib, json
from src.config import OUTPUT_DIR
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def evaluate_models(output_dir=OUTPUT_DIR):
    splits = joblib.load(os.path.join(output_dir, 'data_splits.joblib'))
    X_test = splits['X_test']; y_test = splits['y_test']
    # find model files
    results = {}
    for fname in os.listdir(output_dir):
        if fname.endswith('_model.joblib'):
            name = fname.replace('_model.joblib','')
            m = joblib.load(os.path.join(output_dir, fname))
            y_pred = m.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            cr = classification_report(y_test, y_pred, output_dict=True)
            cm = confusion_matrix(y_test, y_pred).tolist()
            results[name] = {'accuracy': acc, 'classification_report': cr, 'confusion_matrix': cm}
            print(f'{name} accuracy: {acc:.4f}')
    with open(os.path.join(output_dir, 'eval_summary.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print('Evaluation saved to eval_summary.json')
    return results

if __name__ == '__main__':
    evaluate_models()
