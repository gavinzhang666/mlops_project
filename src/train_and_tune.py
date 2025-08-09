import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import subprocess
import tarfile
import argparse

STAGED_DATA_PATH = "data/staged/data.csv"
MODEL_DIR = "model"
MODEL_PKL_PATH = "models/model.pkl"
MODEL_JOBLIB_PATH = os.path.join(MODEL_DIR, "model.joblib")


def main(sample_size=None):
    os.makedirs("models", exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(STAGED_DATA_PATH)

    # If sample_size provided, take a subset
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"[INFO] Using sample size: {df.shape}")

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter grid search
    param_grid = {
        "n_estimators": [100, 200] if not sample_size else [50],
        "max_depth": [None, 10, 20] if not sample_size else [10],
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(X_train, y_train)

    print(f"[INFO] Best parameters: {clf.best_params_}")

    # Save as pkl
    joblib.dump(clf.best_estimator_, MODEL_PKL_PATH)
    print(f"[INFO] Model saved to {MODEL_PKL_PATH}")

    # Save as joblib for SageMaker
    joblib.dump(clf.best_estimator_, MODEL_JOBLIB_PATH)
    print(f"[INFO] Model saved to {MODEL_JOBLIB_PATH}")

    # Create model.tar.gz
    tar_path = "model.tar.gz"
    with tarfile.open(tar_path, "w:gz") as tar:
        tar.add(MODEL_DIR, arcname=".")
    print(f"[INFO] Model archive created at {tar_path}")

    # Upload to S3
    s3_uri = "s3://mlops-credit-yuhui/model/model.tar.gz"
    subprocess.run(["aws", "s3", "cp", tar_path, s3_uri], check=True)
    print(f"[INFO] Model uploaded to {s3_uri}")


# test trigger for
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of rows to sample for quick tests",
    )
    args = parser.parse_args()
    main(sample_size=args.sample)
