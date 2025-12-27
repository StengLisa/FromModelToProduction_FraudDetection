# Import libraries
import os
import pandas as pd

# Project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Define data loading function
def load_data(path=None):
    
    # Default path
    if path is None:
        path = os.path.join(PROJECT_ROOT, "data", "fraud_transactions.csv")

    # Load dataset
    df = pd.read_csv(path)

    # Target variable
    y = df["isFraud"].astype(int)

    # Features (drop identifiers and flags)
    X = df.drop(columns=["isFraud", "nameOrig", "nameDest", "isFlaggedFraud"], errors="ignore")

    return X, y
