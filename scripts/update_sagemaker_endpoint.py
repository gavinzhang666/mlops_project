import os
import time
import boto3
from botocore.exceptions import ClientError

# Config (overridable via env)
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
ECR_IMAGE_URI = os.environ["ECR_IMAGE_URI"]  # must be set in CI secrets/env
S3_MODEL_PATH = os.getenv("S3_MODEL_PATH", "s3://mlops-credit-yuhui/model/model.tar.gz")
ENDPOINT_NAME = os.getenv("ENDPOINT_NAME", "credit-risk-endpoint")
INSTANCE_TYPE = os.getenv("SAGEMAKER_INSTANCE_TYPE", "ml.t2.medium")
ROLE_ARN = os.environ["SAGEMAKER_ROLE_ARN"]  # must be set in CI secrets/env

sm = boto3.client("sagemaker", region_name=AWS_REGION)


def ensure_model():
    """Create a new model with a unique name to avoid name conflicts."""
    ts = int(time.time())
    model_name = f"credit-risk-model-{ts}"

    container = {
        "Image": ECR_IMAGE_URI,
        "ModelDataUrl": S3_MODEL_PATH,
        # Optionally set env for container here:
        # "Environment": {"SAGEMAKER_REGION": AWS_REGION}
    }

    print(f"[INFO] Creating model: {model_name}")
    sm.create_model(
        ModelName=model_name,
        PrimaryContainer=container,
        ExecutionRoleArn=ROLE_ARN,
    )
    return model_name


def create_endpoint_config(model_name: str) -> str:
    """Create a fresh endpoint config that points to the new model."""
    ts = int(time.time())
    endpoint_config_name = f"{ENDPOINT_NAME}-config-{ts}"

    print(f"[INFO] Creating endpoint config: {endpoint_config_name}")
    sm.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "VariantName": "AllTraffic",
                "ModelName": model_name,
                "InitialInstanceCount": 1,
                "InstanceType": INSTANCE_TYPE,
            }
        ],
    )
    return endpoint_config_name


def upsert_endpoint(endpoint_config_name: str):
    """Update endpoint if exists; otherwise create it."""
    try:
        sm.describe_endpoint(EndpointName=ENDPOINT_NAME)
        print(f"[INFO] Updating endpoint: {ENDPOINT_NAME}")
        sm.update_endpoint(
            EndpointName=ENDPOINT_NAME,
            EndpointConfigName=endpoint_config_name,
        )
    except ClientError as e:
        if e.response["Error"]["Code"] == "ValidationException":
            # Treat as not found
            print(f"[INFO] Creating endpoint: {ENDPOINT_NAME}")
            sm.create_endpoint(
                EndpointName=ENDPOINT_NAME,
                EndpointConfigName=endpoint_config_name,
            )
        else:
            raise

    print("[INFO] Endpoint update/creation request submitted (asynchronous).")


def main():
    print("[INFO] Starting SageMaker deployment.")
    print(f"[INFO] Using image: {ECR_IMAGE_URI}")
    print(f"[INFO] Using model data: {S3_MODEL_PATH}")

    model_name = ensure_model()
    endpoint_config_name = create_endpoint_config(model_name)
    upsert_endpoint(endpoint_config_name)

    print("[INFO] Done. Endpoint will transition to InService asynchronously.")


if __name__ == "__main__":
    main()
