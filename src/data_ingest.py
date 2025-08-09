import pandas as pd
import os

RAW_DATA_PATH = "data/raw/credit_risk_dataset.csv"
STAGED_DATA_PATH = "data/staged/data.csv"

def main():
    # Ensure staged directory exists
    os.makedirs(os.path.dirname(STAGED_DATA_PATH), exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Loaded raw data shape: {df.shape}")

    # Save to staged directory
    df.to_csv(STAGED_DATA_PATH, index=False)
    print(f"[INFO] Saved staged data to {STAGED_DATA_PATH}")

if __name__ == "__main__":
    main()
