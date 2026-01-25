# From Model To Production - Fraud Detection

This case study aims to design and implement an automatic fraud detection system that can be seamlessly integrated into an existing environment and retrained on a regular basis.

## Data Source
The data originates from a static .csv file which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/vardhansiramdasu/fraudulent-transactions-prediction?resource=download). 
Basic exploratory data analysis was performed to understand the data source and fraud characteristics.

## Modeling
FLAML AutoML was used for modeling to benchmark different algorithms. LightGBM was identified as the best performing model based on F1-Score. The preprocessing pipeline (standardization for numeric features and one-hot encoding for categorical features), together with the trained model and the fraud-probability threshold, are packaged as an artifact for later deployment via the API.

## Monitoring
MLflow is used for monitoring model-centric reliability. All artifacts, metrics, parameters, and tags are stored locally and can be inspected through the MLflow UI.

## API and RBAC
Using Flask, three API endpoints were set up (`GET /health`, `POST /predict`, and `DELETE /delete-run`). Role-based access control is implemented with three fixed roles: stakeholder (read-only), scientist (read and write), and admin (full access).

## Drift Simulation & Detection
Drift simulation is based on three components: Fraud Prevalence Drift, Fraud Feature Drift, and Fraud Type Drift. Drift is computed by (1) taking the relative difference for each numeric feature between the baseline month (Month 0, before any simulation is applied) and the current month, and then (2) averaging those differences to produce a single drift score. If the drift score exceeds the drift threshold, the `retrain_flag` is set to `true`, triggering model retraining.

## Automation / MLOps
GitHub Actions is used as an MLOps platform to automate the entire process end-to-end. In addition to a scheduled monthly forced retraining, workflows can be triggered manually for either a drift-based monthly training or a manual full-year simulation.
