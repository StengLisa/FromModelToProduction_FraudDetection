# Script for training the final LGBM model based on the AutoML results + logging the model details

import os
import json
import joblib
import numpy as np
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
from src.data_loader import load_data
from src.preprocessing import build_preprocessor

import mlflow
import mlflow.sklearn

import hashlib, pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import tempfile

def run_training():
    # 1. Load data & config
    X, y = load_data()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, "..", "config", "threshold.json")

    with open(config_path, "r") as f:
        config = json.load(f)

    threshold = config["threshold"]

    # 2. Split (fixed random_state for reproducibility)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # 3. Preprocess
    preprocessor = build_preprocessor()
    X_train_prep = preprocessor.fit_transform(X_train)
    X_test_prep = preprocessor.transform(X_test)

    # 4. Train fixed LightGBM model with AutoML-discovered config
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

    # 5. Evaluate
    y_pred_proba = lgbm.predict_proba(X_test_prep)[:, 1]
    y_pred = (y_pred_proba > threshold).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print("\n=== Evaluation Metrics ===")
    print("ROC AUC:", roc_auc)
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))

    # --- MLflow logging ---
    mlflow.set_tracking_uri("file:/Users/LisaSteng/ModelToProduction_FraudDetection/mlruns")
    mlflow.set_experiment("fraud_detection_experiment")

    with mlflow.start_run(run_name="final_lgbm_model"):
        
        # Log metrics
        mlflow.log_metric("roc_auc", roc_auc)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        # Log parameters (important for reproducibility)
        mlflow.log_param("threshold", threshold)
        mlflow.log_param("n_estimators", lgbm.n_estimators)
        mlflow.log_param("num_leaves", lgbm.num_leaves)
        mlflow.log_param("learning_rate", lgbm.learning_rate)

        # Log dataset version (hash)
        data_path = pathlib.Path("../data/fraud_transactions.csv")
        with open(data_path, "rb") as f:
            data_hash = hashlib.md5(f.read()).hexdigest()
        mlflow.set_tag("data_version", f"{datetime.now().strftime('%Y-%m-%d')}_{data_hash[:8]}")

        # Log ROC curve plot
        RocCurveDisplay.from_predictions(y_test, y_pred_proba)
        plt.title("ROC Curve")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close()
            
        # Log confusion matrix plot
        ConfusionMatrixDisplay.from_predictions(y_test, y_pred, normalize='true')
        plt.title("Confusion Matrix")
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            plt.savefig(tmp.name)
            mlflow.log_artifact(tmp.name, artifact_path="plots")
        plt.close()

        # Log classification report (same pattern as plots)
        report_text = classification_report(y_test, y_pred)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as tmp:
            tmp.write(report_text)
            tmp.flush()              # ensure buffer is written
            os.fsync(tmp.fileno())   # force flush to disk
            mlflow.log_artifact(tmp.name, artifact_path="reports")


        # Log model artifact
        mlflow.sklearn.log_model(lgbm, "model")

    # 6. Ensure models/ folder exists (sibling to src/)
    project_root = os.path.dirname(os.path.abspath(__file__))  # points to src/
    models_dir = os.path.join(project_root, "..", "models")
    os.makedirs(models_dir, exist_ok=True)

    # 7. Save both together (preprocessor + fixed model)
    model_path = os.path.join(models_dir, "fraud_model_final.joblib")
    artifact = {"preprocessor": preprocessor, "model": lgbm, "threshold": threshold}
    joblib.dump(artifact, model_path)


if __name__ == "__main__":
    run_training()
