# Import libraries
import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from src.train_final import run_training

BASELINE_STATS = Path("artifacts/baseline_stats.parquet")
DRIFT_LOG = Path("artifacts/drift_log.csv")


def compute_stats(df):
    return df.mean(numeric_only=True)


def compute_drift(baseline, current):
    eps = 1e-6
    relative = (baseline - current).abs() / (baseline.abs() + eps)
    return relative.mean()


# Drift simulation
def simulate_monthly_data(df, month):

    df = df.copy()
    rng = np.random.default_rng(seed=month)

    # 1. Fraud Prevalence Drift
    base_rate = df["isFraud"].mean()
    target_rate = min(base_rate + 0.0002 * month, 0.015)  

    current_rate = df["isFraud"].mean()

    if current_rate < target_rate:
        fraud = df[df["isFraud"] == 1]
        needed = int((target_rate * len(df)) - len(fraud))

        if needed > 0 and len(fraud) > 0:
            extra = fraud.sample(needed, replace=True, random_state=month)

            # Noise reduced again: 0.003 → 0.001
            noise_scale = 0.003
            for col in ["amount", "oldbalanceOrg", "newbalanceOrig",
                        "oldbalanceDest", "newbalanceDest"]:
                if col in extra.columns:
                    extra[col] *= rng.normal(1.0, noise_scale, size=len(extra))

            df = pd.concat([df, extra], ignore_index=True)


    # 2. Fraud Feature Drift
    fraud_mask = df["isFraud"] == 1

    if "amount" in df.columns:
        df.loc[fraud_mask, "amount"] *= rng.normal(
            1 + 0.0001 * month, 0.01, size=fraud_mask.sum()
        )

    balance_cols = [
        "oldbalanceOrg", "newbalanceOrig",
        "oldbalanceDest", "newbalanceDest"
    ]
    for col in balance_cols:
        if col in df.columns:
            df.loc[fraud_mask, col] *= rng.normal(
                1 + 0.0005 * month, 0.008, size=fraud_mask.sum()
            )

    
    # 3. Fraud type drift
    if "type" in df.columns:
        fraud_types = ["TRANSFER", "CASH_OUT"]
        df.loc[fraud_mask, "type"] = rng.choice(
            fraud_types, size=fraud_mask.sum(), p=[0.6, 0.4]
        )

    return df


# Main pipeline

def main(month, drift_threshold):
    Path("artifacts").mkdir(exist_ok=True)

    # Load data
    base_df = pd.read_csv("data/fraud_transactions_small.csv")

    if month == 0:
        # Compute baseline stats on features
        stats = compute_stats(base_df.drop(columns=["isFraud"]))
        stats.to_frame("mean").to_parquet(BASELINE_STATS)

        # Train baseline model
        run_training(
            data_path="data/fraud_transactions_small.csv",
            register_name="fraud_model",
            threshold_config="config/threshold.json",
            retrain_reason="baseline"
        )
        return

    # Simulate new month
    df_new = simulate_monthly_data(base_df, month)

    # Compute drift
    current_stats = compute_stats(df_new.drop(columns=["isFraud"]))
    baseline_stats = pd.read_parquet(BASELINE_STATS)["mean"]
    drift_value = compute_drift(baseline_stats, current_stats)

    print(f"Month {month}: drift = {drift_value}")

    retrain = drift_value > drift_threshold

    # Log drift
    if DRIFT_LOG.exists():
        drift_log = pd.read_csv(DRIFT_LOG)
    else:
        drift_log = pd.DataFrame(columns=["month", "drift", "retrained"])

    drift_log.loc[len(drift_log)] = [month, drift_value, retrain]
    drift_log.to_csv(DRIFT_LOG, index=False)

    if retrain:
        print("Retraining due to drift")
        temp_path = f"data/simulated_month_{month}.csv"
        df_new.to_csv(temp_path, index=False)

        run_training(
            data_path=temp_path,
            register_name="fraud_model",
            threshold_config="config/threshold.json",
            retrain_reason="drift"
        )

        # Update baseline stats after retraining
        new_stats = compute_stats(df_new.drop(columns=["isFraud"]))
        new_stats.to_frame("mean").to_parquet(BASELINE_STATS)
        print(f"Updated baseline stats after retraining in month {month}")

        # Write retrain flag 
        RETRAIN_FLAG.write_text("true")

    else:
        print("No retraining needed")
        RETRAIN_FLAG.write_text("false")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, required=True)

    # Default threshold for drift
    parser.add_argument("--drift-threshold", type=float, default=0.05)

    args = parser.parse_args()
    main(args.month, args.drift_threshold)
