from sklearn.metrics import f1_score, recall_score, roc_auc_score
import os
import joblib

def evaluate_model(model, X, y):
    preds = model.predict(X)
    print("F1:", f1_score(y, preds))
    print("Recall:", recall_score(y, preds))
    print("ROC AUC:", roc_auc_score(y, model.predict_proba(X)[:, 1]))

def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)  # Ensure directory exists
    joblib.dump(model, path)