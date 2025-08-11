import os
import json
import shap
import joblib
import logging
import matplotlib.pyplot as plt

from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# -----------------------------
# Logging Setup
# -----------------------------
def setup_logger():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = f"outputs/logs/train_{timestamp}.log"
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        filename=log_path,
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter("%(levelname)s | %(message)s")
    console.setFormatter(formatter)
    logging.getLogger().addHandler(console)

# -----------------------------
# Model Factory
# -----------------------------
def get_model(name, params):
    if name == "logistic_regression":
        return LogisticRegression(**params)
    elif name == "random_forest":
        return RandomForestClassifier(**params)
    elif name == "xgboost":
        return xgb.XGBClassifier(**params)
    else:
        raise ValueError(f"Unknown model: {name}")

# -----------------------------
# Evaluation Function
# -----------------------------
def evaluate_model(model, X_train, y_train, X_val, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "Accuracy": accuracy_score(y_val, y_pred),
        "Precision": precision_score(y_val, y_pred),
        "Recall": recall_score(y_val, y_pred),
        "F1 Score": f1_score(y_val, y_pred),
        "ROC AUC": roc_auc_score(y_val, y_proba) if y_proba is not None else "N/A"
    }

    return model, metrics

# -----------------------------
# Save Metrics
# -----------------------------
def save_metrics(metrics, model_name, output_dir="outputs/metrics"):
    os.makedirs(output_dir, exist_ok=True)
    metrics_path = os.path.join(output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"📊 Saved metrics: {metrics_path}")

# -----------------------------
# SHAP Explainability
# -----------------------------
def explain_model(model, X_val, model_name, output_dir="outputs/shap"):
    os.makedirs(output_dir, exist_ok=True)

    try:
        if model_name == "logistic_regression":
            explainer = shap.LinearExplainer(model, X_val, feature_perturbation="interventional")
        elif model_name == "random_forest":
            explainer = shap.TreeExplainer(model)
        elif model_name == "xgboost":
            explainer = shap.TreeExplainer(model)
        else:
            logging.warning(f"No SHAP explainer for {model_name}")
            return

        shap_values = explainer(X_val)

        # Summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_val, show=False)
        summary_path = os.path.join(output_dir, f"{model_name}_summary.png")
        plt.savefig(summary_path)
        plt.close()
        logging.info(f"📈 Saved SHAP summary plot: {summary_path}")

        # Force plot for first instance (updated API)
        try:
            force_plot_path = os.path.join(output_dir, f"{model_name}_force.png")
            if hasattr(shap_values, 'values'):
                # For newer SHAP versions
                shap.plots.force(explainer.expected_value, shap_values.values[0], show=False)
            else:
                # For older SHAP versions
                shap.plots.force(explainer.expected_value, shap_values[0], show=False)
            plt.savefig(force_plot_path)
            plt.close()
            logging.info(f"📌 Saved SHAP force plot: {force_plot_path}")
        except Exception as e:
            # Try alternative approach for random forest
            try:
                if model_name == "random_forest":
                    shap.plots.force(explainer.expected_value[0], shap_values[0], show=False)
                    plt.savefig(force_plot_path)
                    plt.close()
                    logging.info(f"📌 Saved SHAP force plot: {force_plot_path}")
                else:
                    logging.warning(f"Force plot failed for {model_name}: {e}")
            except Exception as e2:
                logging.warning(f"Force plot failed for {model_name}: {e2}")

    except Exception as e:
        logging.error(f"SHAP explainability failed for {model_name}: {e}")

# -----------------------------
# Model Ranking
# -----------------------------
def rank_models(all_metrics, key="F1 Score"):
    ranked = sorted(all_metrics.items(), key=lambda x: x[1].get(key, 0), reverse=True)
    logging.info(f"\n🏆 Model ranking by {key}:")
    for name, metrics in ranked:
        score = metrics[key]
        logging.info(f"{name}: {score:.4f}" if isinstance(score, float) else f"{name}: {score}")

# -----------------------------
# Main Runner
# -----------------------------
def run_all_models(X_train, y_train, X_val, y_val):
    setup_logger()

    model_configs = {
        "logistic_regression": {"solver": "liblinear", "random_state": 42},
        "random_forest": {"n_estimators": 100, "max_depth": 5, "random_state": 42},
        "xgboost": {
            "n_estimators": 100,
            "max_depth": 5,
            "eval_metric": "logloss",
            "random_state": 42
        }
    }

    results_dir = "outputs/models"
    os.makedirs(results_dir, exist_ok=True)

    all_metrics = {}

    for name, params in model_configs.items():
        logging.info(f"\n🔍 Training and evaluating: {name}")
        model = get_model(name, params)
        trained_model, metrics = evaluate_model(model, X_train, y_train, X_val, y_val)

        for metric_name, value in metrics.items():
            msg = f"{metric_name}: {value:.4f}" if isinstance(value, float) else f"{metric_name}: {value}"
            logging.info(msg)

        # Save model
        model_path = os.path.join(results_dir, f"{name}.joblib")
        joblib.dump(trained_model, model_path)
        logging.info(f"💾 Saved model: {model_path}")

        # Save metrics
        save_metrics(metrics, name)
        all_metrics[name] = metrics

        # SHAP explainability
        explain_model(trained_model, X_val, name)

    # Rank models
    rank_models(all_metrics)

# -----------------------------
# Example Usage
# -----------------------------
if __name__ == "__main__":
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    run_all_models(X_train, y_train, X_val, y_val)