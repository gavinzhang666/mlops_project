import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib

STAGED_DATA_PATH = "data/staged/data.csv"
MODEL_PATH = "models/model.pkl"


def main():
    # Load data
    df = pd.read_csv(STAGED_DATA_PATH)
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # One-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Load model
    model = joblib.load(MODEL_PATH)
    y_pred = model.predict(X)
    y_pred_proba = model.predict_proba(X)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_pred_proba)

    print(f"[INFO] Accuracy: {acc:.4f}")
    print(f"[INFO] ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
