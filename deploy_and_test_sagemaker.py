import boto3
from datetime import datetime

# AWS SageMaker client
sm_client = boto3.client("sagemaker", region_name="us-east-1")

# Generate unique names with timestamp
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
model_name = f"credit-risk-model-{timestamp}"
endpoint_config_name = f"credit-risk-config-{timestamp}"
endpoint_name = f"credit-risk-endpoint-{timestamp}"

# ECR image URI
image_uri = "778086627816.dkr.ecr.us-east-1.amazonaws.com/credit-risk-flask:latest"

# SageMaker execution role ARN
role_arn = "arn:aws:iam::778086627816:role/SageMakerExecutionRole"

# S3 path to model
model_s3_path = "s3://mlops-credit-yuhui/model/model.tar.gz"

def create_and_deploy():
    print(f"[INFO] Model: {model_name}")
    print(f"[INFO] Endpoint Config: {endpoint_config_name}")
    print(f"[INFO] Endpoint: {endpoint_name}")

    # 1. Create model (with S3 model location)
    print("[INFO] Creating model...")
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            "Image": image_uri,
            "ModelDataUrl": model_s3_path  # <-- Tells SageMaker where to get model.joblib
        },
        ExecutionRoleArn=role_arn
    )

    # 2. Create endpoint config
    print("[INFO] Creating endpoint config...")
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": "ml.t2.medium"
            }
        ]
    )

    # 3. Create endpoint
    print("[INFO] Creating endpoint (may take several minutes)...")
    sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )

    # 4. Wait until endpoint is ready
    print("[INFO] Waiting for endpoint to be InService...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)
    print(f"[INFO] Endpoint {endpoint_name} is now InService.")

if __name__ == "__main__":
    create_and_deploy()
