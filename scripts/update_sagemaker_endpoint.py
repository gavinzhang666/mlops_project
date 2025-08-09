import boto3
import time
import os

# AWS settings
AWS_REGION = "us-east-1"  # Change to your region
S3_MODEL_PATH = "s3://mlops-credit-yuhui/model/model.tar.gz"  # Model S3 path
MODEL_NAME = "credit-risk-model"
ENDPOINT_NAME = "credit-risk-endpoint"
INSTANCE_TYPE = "ml.m5.large"

def update_sagemaker_model():
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

    # Create or update the model
    print(f"[INFO] Creating/Updating model: {MODEL_NAME}")
    container = {
        "Image": "<your_ecr_repo_uri>:latest",  # Replace with your ECR image URI
        "ModelDataUrl": S3_MODEL_PATH
    }
    try:
        sm_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer=container,
            ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]  # From GitHub Secrets
        )
    except sm_client.exceptions.ResourceInUse:
        # If model exists, delete and recreate
        sm_client.delete_model(ModelName=MODEL_NAME)
        time.sleep(5)
        sm_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer=container,
            ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]
        )

    # Create new endpoint config
    endpoint_config_name = f"{ENDPOINT_NAME}-{int(time.time())}"
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": MODEL_NAME,
                "InitialInstanceCount": 1,
                "InstanceType": INSTANCE_TYPE
            }
        ]
    )

    # Update or create endpoint
    try:
        sm_client.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )
    except sm_client.exceptions.ResourceNotFound:
        sm_client.create_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name
        )

    print("[INFO] SageMaker endpoint update started.")

if __name__ == "__main__":
    update_sagemaker_model()
