import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify, Response
import requests
import mlflow
from mlflow.tracking import MlflowClient

# Project root 
#PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# MLflow tracking URI setup
#mlflow.set_tracking_uri(f"file:{os.path.join(PROJECT_ROOT, 'mlruns')}")

#current_dir = os.path.dirname(os.path.abspath(__file__))
#project_root = os.path.dirname(current_dir)
#tracking_dir = os.path.join(project_root, "mlruns")

#mlflow.set_tracking_uri(f"file:{tracking_dir}")

current_dir = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(current_dir) 

tracking_dir = os.path.join(PROJECT_ROOT, "mlruns")
mlflow.set_tracking_uri(f"file:{tracking_dir}")

# Load model & preprocessor
model_path = os.path.join(PROJECT_ROOT, "models", "fraud_model.joblib")
artifact = joblib.load(model_path)
preprocessor = artifact["preprocessor"]
model = artifact["model"]
threshold = artifact.get("threshold", 0.5)


# Authorization setup
users = {
    "admin_user": {"password": "admin_pass", "role": "admin"},
    "stakeholder_user": {"password": "stakeholder_pass", "role": "stakeholder"},
    "scientist_user": {"password": "scientist_pass", "role": "data_scientist"},
}

role_permissions = {
    "admin": {"read", "write", "delete"},
    "data_scientist": {"read", "write"},
    "stakeholder": {"read"},
}

def authenticate():
    auth = request.authorization
    if not auth or auth.username not in users:
        return None
    user = users[auth.username]
    if user["password"] != auth.password:
        return None
    return {"username": auth.username, "role": user["role"]}

def require_permission(user, permission):
    if permission not in role_permissions[user["role"]]:
        return jsonify({
            "error": f"Access denied: {user['role']} lacks {permission} permission"
        }), 403
    return None


# Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Welcome to the Fraud Detection API with RBAC",
        "available_endpoints": [
            "/health",
            "/predict",
            "/delete-run/<run_id>",
            "/mlflow-ui"
        ]
    })

@app.route("/health", methods=["GET"])
def health():
    user = authenticate()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    perm_check = require_permission(user, "read")
    if perm_check: return perm_check
    return jsonify({"status": "ok", "threshold": threshold, "user": user})

@app.route("/predict", methods=["POST"])
def predict():
    user = authenticate()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    perm_check = require_permission(user, "write")
    if perm_check: return perm_check

    try:
        data = request.get_json(force=True)
        df = pd.DataFrame([data]) if isinstance(data, dict) else pd.DataFrame(data)
        X_trans = preprocessor.transform(df)
        proba = model.predict_proba(X_trans)[:, 1]
        preds = (proba > threshold).astype(int)

        results = [
            {"fraud_probability": float(p), "fraud_decision": int(d)}
            for p, d in zip(proba, preds)
        ]
        return jsonify({"results": results, "user": user})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/delete-run/<run_id>", methods=["DELETE"])
def delete_run(run_id):
    user = authenticate()
    if not user:
        return jsonify({"error": "Unauthorized"}), 401
    perm_check = require_permission(user, "delete")
    if perm_check: return perm_check

    try:
        client = MlflowClient()
        client.delete_run(run_id)
        return jsonify({"message": f"Run {run_id} deleted by {user['username']}"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#@app.route("/mlflow-ui", methods=["GET", "POST"])
#def mlflow_ui():
    #user = authenticate()
    #if not user:
        #return jsonify({"error": "Unauthorized"}), 401

    #if user["role"] == "stakeholder" and request.method != "GET":
        #return jsonify({"error": "Stakeholders have read-only access"}), 403

    #if user["role"] == "data_scientist" and request.method not in ["GET", "POST"]:
        #return jsonify({"error": "Data scientists cannot delete"}), 403

    #try:
        #mlflow_url = f"http://127.0.0.1:5000{request.full_path}"
        #resp = requests.request(
            #method=request.method,
            #url=mlflow_url,
            #headers={k: v for k, v in request.headers if k != "Host"},
            #data=request.get_data(),
            #cookies=request.cookies,
            #allow_redirects=False,
            #timeout=10,
        #)
        #return Response(resp.content, status=resp.status_code, content_type=resp.headers.get("Content-Type"))
    #except Exception as e:
        #return jsonify({"error": f"MLflow UI not available: {str(e)}"}), 503

# Script execution
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)
