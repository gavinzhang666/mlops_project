import pandas as pd

STAGED_DATA_PATH = "data/staged/data.csv"

def main():
    df = pd.read_csv(STAGED_DATA_PATH)
    print("[INFO] Data validation report:")
    
    # Check missing values
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0])

    # Data types
    print("\n[INFO] Data types:")
    print(df.dtypes)

    # Basic statistics
    print("\n[INFO] Basic statistics:")
    print(df.describe())

if __name__ == "__main__":
    main()
