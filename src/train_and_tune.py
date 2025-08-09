import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

STAGED_DATA_PATH = "data/staged/data.csv"
MODEL_PATH = "models/model.pkl"

def main():
    os.makedirs("models", exist_ok=True)
    
    # Load data
    df = pd.read_csv(STAGED_DATA_PATH)
    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    # One-hot encoding for categorical variables
    X = pd.get_dummies(X, drop_first=True)

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Hyperparameter grid search
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10, 20]
    }
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3)
    clf.fit(X_train, y_train)

    print(f"[INFO] Best parameters: {clf.best_params_}")

    # Save best model
    joblib.dump(clf.best_estimator_, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
