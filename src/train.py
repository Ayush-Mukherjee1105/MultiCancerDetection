import os, joblib
from src.config import OUTPUT_DIR, RANDOM_STATE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def train_from_embeddings(emb_file=None, output_dir=OUTPUT_DIR):
    data = joblib.load(emb_file)
    X = data['X']; labels = data['labels']
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder(); y = le.fit_transform(labels)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE)

    models = {}

    print('Training RandomForest...')
    models['rf'] = RandomForestClassifier(n_estimators=150, random_state=RANDOM_STATE, n_jobs=-1)
    models['rf'].fit(X_train, y_train)

    print('Training AdaBoost...')
    models['ada'] = AdaBoostClassifier(n_estimators=150, random_state=RANDOM_STATE)
    models['ada'].fit(X_train, y_train)

    print('Training GradientBoost...')
    models['gb'] = GradientBoostingClassifier(n_estimators=150, random_state=RANDOM_STATE)
    models['gb'].fit(X_train, y_train)

    print('Training XGBoost...')
    models['xgb'] = XGBClassifier(n_estimators=150, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE)
    models['xgb'].fit(X_train, y_train)

    # stacking
    estimators = [('rf', models['rf']), ('ada', models['ada']), ('gb', models['gb'])]
    stack = StackingClassifier(estimators=estimators,
                               final_estimator=XGBClassifier(n_estimators=300, use_label_encoder=False, eval_metric='mlogloss', random_state=RANDOM_STATE),
                               n_jobs=-1)
    print('Training stacking classifier...')
    stack.fit(X_train, y_train)
    models['stack'] = stack

    # Save models and encoder
    os.makedirs(output_dir, exist_ok=True)
    for name,m in models.items():
        joblib.dump(m, os.path.join(output_dir, f'{name}_model.joblib'))
    joblib.dump(le, os.path.join(output_dir, 'label_encoder.joblib'))
    joblib.dump({'X_train': X_train, 'X_test': X_test, 'y_train': y_train, 'y_test': y_test}, os.path.join(output_dir, 'data_splits.joblib'))

    summary = {}
    for name,m in models.items():
        acc = accuracy_score(y_test, m.predict(X_test))
        summary[name] = float(acc)
        print(f'{name} accuracy: {acc:.4f}')
    joblib.dump(summary, os.path.join(output_dir, 'train_summary.joblib'))
    return summary

if __name__ == '__main__':
    emb = os.path.join(OUTPUT_DIR, 'embeddings.joblib')
    if not os.path.exists(emb):
        raise FileNotFoundError('embeddings.joblib not found. Run embedding extraction first.')
    train_from_embeddings(emb)
