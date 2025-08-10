import boto3
import time
import os

# AWS settings
AWS_REGION = "us-east-1"  # Change if needed
S3_MODEL_PATH = "s3://mlops-credit-yuhui/model/model.tar.gz"
MODEL_NAME = "credit-risk-model"
ENDPOINT_NAME = "credit-risk-endpoint"
INSTANCE_TYPE = "ml.m5.large"

def update_sagemaker_model():
    sm_client = boto3.client("sagemaker", region_name=AWS_REGION)

    # Read image URI from env (passed in CI/CD)
    image_uri = os.getenv("ECR_IMAGE_URI")
    if not image_uri:
        raise ValueError("ECR_IMAGE_URI environment variable not set.")

    container = {
        "Image": image_uri,
        "ModelDataUrl": S3_MODEL_PATH
    }

    # Create or recreate the model
    print(f"[INFO] Creating or updating model: {MODEL_NAME}")
    try:
        sm_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer=container,
            ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]
        )
    except sm_client.exceptions.ResourceInUse:
        sm_client.delete_model(ModelName=MODEL_NAME)
        time.sleep(5)
        sm_client.create_model(
            ModelName=MODEL_NAME,
            PrimaryContainer=container,
            ExecutionRoleArn=os.environ["SAGEMAKER_ROLE_ARN"]
        )

    # Create new endpoint config with timestamp
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

    # Update endpoint if exists, else create new
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

    print("[INFO] SageMaker endpoint update/creation started.")

if __name__ == "__main__":
    update_sagemaker_model()
