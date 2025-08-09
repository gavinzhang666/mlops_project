import pandas as pd
import os
import argparse

RAW_DATA_PATH = "data/raw/credit_risk_dataset.csv"
STAGED_DATA_PATH = "data/staged/data.csv"

def main(sample_size=None):
    # Ensure staged directory exists
    os.makedirs(os.path.dirname(STAGED_DATA_PATH), exist_ok=True)
    
    # Load raw data
    df = pd.read_csv(RAW_DATA_PATH)
    print(f"[INFO] Loaded raw data shape: {df.shape}")

    # If sample_size provided, take a small random subset
    if sample_size:
        df = df.sample(n=min(sample_size, len(df)), random_state=42)
        print(f"[INFO] Using sample size: {df.shape}")

    # Save to staged directory
    df.to_csv(STAGED_DATA_PATH, index=False)
    print(f"[INFO] Saved staged data to {STAGED_DATA_PATH}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=int, default=None, help="Number of rows to sample for quick tests")
    args = parser.parse_args()
    main(sample_size=args.sample)
