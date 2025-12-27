# Import libraries
import os
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data loading function
def load_data(path=None):

    # Detect if running inside GitHub Actions
    running_in_github = os.getenv("GITHUB_ACTIONS") == "true"

    if path is None:
        if running_in_github:
            # Use the small dataset
            path = os.path.join(PROJECT_ROOT, "data", "fraud_transactions_small.csv")
        else:
            # Use the full dataset
            path = os.path.join(PROJECT_ROOT, "data", "fraud_transactions.csv")

    # Load dataset
    df = pd.read_csv(path)

    # Target variable
    y = df["isFraud"].astype(int)

    # Features (drop identifiers and flags)
    X = df.drop(columns=["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")

    return X, y
