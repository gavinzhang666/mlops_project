import boto3
import json
import pandas as pd

# AWS SageMaker runtime client
runtime_client = boto3.client("sagemaker-runtime", region_name="us-east-1")

# Your SageMaker endpoint name
endpoint_name = "credit-risk-endpoint-20250809-174309"

# Load staged data
df = pd.read_csv("data/staged/data.csv")

# Drop target column to get features
X = df.drop(columns=["loan_status"])

# One-hot encode categorical variables (same as training)
X = pd.get_dummies(X, drop_first=True)

# Prepare payload in SageMaker expected format
payload = {"instances": X.values.tolist()}  # Must be list of lists

# Call SageMaker endpoint
response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType="application/json",
    Body=json.dumps(payload)
)

# Parse predictions
result = json.loads(response["Body"].read().decode())

# Add predictions to DataFrame
df["predicted_loan_status"] = result["predictions"]

# Save to CSV
df.to_csv("predictions.csv", index=False)

print("[INFO] Predictions saved to predictions.csv")
