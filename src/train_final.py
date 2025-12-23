# Import libraries
import os
import json
import joblib
import argparse
import hashlib
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import tempfile
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    RocCurveDisplay,
    ConfusionMatrixDisplay
)
from lightgbm import LGBMClassifier
import mlflow
import mlflow.sklearn
from src.data_loader import load_data
from src.preprocessing import build_preprocessor


# Project Root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Threshold optimization helper
def find_best_threshold(y_true, y_prob):
    thresholds = np.linspace(0.01, 0.99, 99)
    best_t = 0.5
    best_f1 = 0.0

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_t = t

    return best_t, best_f1


# Training function including evaluation and MLflow logging
def run_training(data_path, register_name, threshold_config, retrain_reason="manual"):
    data_path = os.path.join(PROJECT_ROOT, data_path)
    threshold_config = os.path.join(PROJECT_ROOT, threshold_config)

    # 1. Data loading
    X, y = load_data(data_path)

    # 2. Load threshold config
    if os.path.exists(threshold_config):
        with open(threshold_config, "r") as f:
            config = json.load(f)
        threshold = config.get("threshold", 0.5)
    else:
        threshold = 0.5

    # 3. Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 4. Preprocessing
    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # 5. Model training
    lgbm = LGBMClassifier(
        n_estimators=4235,
        num_leaves=5,
        min_child_samples=52,
        learning_rate=0.05635706130547884,
        colsample_bytree=1.0,
        reg_alpha=0.4049511136971195,
        reg_lambda=0.06829159402829672,
        max_bin=(2 ** 9) - 1,
        random_state=42,
        n_jobs=-1
    )
    lgbm.fit(X_train_prep, y_train)

    # 6. Threshold recalibration
    y_pred_proba = lgbm.predict_proba(X_test_prep)[:, 1]
    best_threshold, best_f1 = find_best_threshold(y_test, y_pred_proba)

    print(f"\nOptimal threshold found: {best_threshold:.4f} (F1={best_f1:.4f})")

    # Use recalibrated threshold
    threshold = best_threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # 7. Metric evaluation
    metrics = {
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
    }

    print("\n *** Evaluation Metrics ***")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # 8. MLflow logging
    mlflow.set_tracking_uri(f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}")
    mlflow.set_experiment("fraud_detection_experiment")

    with mlflow.start_run(run_name="lgbm_training"):
        # Log metrics
        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        # Log parameters
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_estimators", lgbm.n_estimators)
        mlflow.log_param("num_leaves", lgbm.num_leaves)
        mlflow.log_param("learning_rate", lgbm.learning_rate)

        # Log dataset version (hash)
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        mlflow.set_tag("data_version", f"{datetime.now().strftime('%Y-%m-%d')}_{data_hash[:8]}")

        # Log retrain reason
        mlflow.set_tag("retrain_reason", retrain_reason)
        mlflow.set_tag("month", datetime.now().strftime("%Y-%m"))

        # ROC curve
        RocCurveDisplay.from_predictions(y_test, y_pred_proba)
        plt.title("ROC Curve")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close()

        # Confusion matrix
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize="true")
        plt.title("Confusion Matrix")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close()

        # Classification report
        report_text = classification_report(y_test, y_pred)
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".txt") as f:
            f.write(report_text)
            temp_path = f.name
        mlflow.log_artifact(temp_path, artifact_path="reports")

        # Log model
        mlflow.sklearn.log_model(lgbm, register_name)

    # 9. Save model + preprocessor + threshold locally
    models_dir = os.path.join(PROJECT_ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, f"{register_name}.joblib")
    artifact = {"preprocessor": preprocessor, "model": lgbm, "threshold": threshold}
    joblib.dump(artifact, model_path)

    # 10. Save updated threshold to config file
    with open(threshold_config, "w") as f:
        json.dump({"threshold": float(threshold)}, f)


# Script execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LGBM fraud detection model")
    parser.add_argument("--data", type=str, required=True, help="Path to dataset CSV (relative to repo root)")
    parser.add_argument("--register-name", type=str, required=True, help="MLflow model registry name")
    parser.add_argument("--threshold-config", type=str, default="config/threshold.json", help="Path to threshold config JSON (relative to repo root)")
    parser.add_argument("--retrain-reason", type=str, default="manual", help="Reason for retrain (monthly|drift|manual)")
    args = parser.parse_args()

    run_training(
        data_path=args.data,
        register_name=args.register_name,
        threshold_config=args.threshold_config,
        retrain_reason=args.retrain_reason
    )
