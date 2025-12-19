# Import libraries
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)
from flaml import AutoML
from src.data_loader import load_data
from src.preprocessing import build_preprocessor
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, average_precision_score

# 1. Data loading
X, y = load_data()

# 2. Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, test_size=0.2, random_state=42
)

# 3. Preprocessing
preprocessor = build_preprocessor()
X_train_prep = preprocessor.fit_transform(X_train)
X_test_prep = preprocessor.transform(X_test)

# 4. AutoML
automl = AutoML()
automl.fit(
    X_train=X_train_prep,
    y_train=y_train,
    task="classification",
    metric="roc_auc",
    time_budget=300,
    seed=42
)

# 5. Metric evaluation
y_pred_proba = automl.predict_proba(X_test_prep)[:, 1]
y_pred = (y_pred_proba > 0.5).astype(int)

print("\n*** Evaluation Metrics ***")
print("ROC AUC:", roc_auc_score(y_test, y_pred_proba))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1 Score:", f1_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


# 6. Ensure models/folder exists
project_root = os.path.dirname(os.path.abspath(__file__))  
models_dir = os.path.join(project_root, "..", "models")
os.makedirs(models_dir, exist_ok=True)

# 7. Save model
model_path = os.path.join(models_dir, "fraud_model.joblib")
joblib.dump((preprocessor, automl), model_path)

# 8. Inspect AutoML results
print("\n*** AutoML Summary ***")
print(f"Best estimator: {automl.best_estimator}")
print("Best config:")
for k, v in automl.best_config.items():
    print(f"  {k}: {v}")
print(f"Best loss: {automl.best_loss:.6f}")

best_result = automl.best_result
print("\nBest result details:")
print(f"  Validation loss: {best_result['val_loss']:.6f}")
print(f"  Training iterations: {best_result['training_iteration']}")
print(f"  Total time (s): {best_result['time_total_s']:.2f}")
print(f"  Prediction time: {best_result['pred_time']:.6e}")
