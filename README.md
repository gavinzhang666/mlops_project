# Credit Risk Prediction MLOps Pipeline

## Project Overview

### Topic & Data Acquisition
This project focuses on building an end-to-end **MLOps pipeline** for predicting credit risk using machine learning.  
The goal is to automate the process of data ingestion, model training, version control, deployment, and monitoring.

We selected the **Credit Risk Dataset** from Kaggle:  
[https://www.kaggle.com/datasets/laotse/credit-risk-dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset)  

- **Reason for Choosing This Dataset**:  
  Credit risk prediction is a common and high-impact use case in financial services. It helps banks, lenders, and fintech companies assess the likelihood of a borrower defaulting on a loan, thereby reducing financial losses.
  
- **Dataset Format**:  
  The dataset is provided as a CSV file (`credit_risk_dataset.csv`). It contains information about borrower profiles and their loan repayment outcomes.

- **Schema**:
  - `person_age` — Age of the individual
  - `person_income` — Annual income of the individual
  - `person_emp_length` — Employment length (years)
  - `loan_amnt` — Loan amount requested
  - `loan_int_rate` — Interest rate of the loan
  - `loan_status` — Target variable (0 = good loan, 1 = default)
  - Additional demographic and credit history features

- **Privacy Considerations**:  
  This dataset is anonymized and does not contain personally identifiable information (PII). However, in real-world use, strict compliance with **GDPR** or **CCPA** should be enforced, ensuring secure data storage and access controls.

---

## Setup & Run Instructions

### 1. Local Setup

#### Prerequisites
- Python 3.10+
- AWS CLI configured with credentials
- DVC (Data Version Control)
- Docker (for container builds)
- Git

#### Install Dependencies
```bash
pip install -r requirements.txt
pip install dvc[s3] boto3 mlflow
```

#### Pull Data from DVC Remote
```bash
dvc pull data/raw/credit_risk_dataset.csv
```

#### Run Local Training
```bash
python src/train_and_tune.py
```

This will produce `model.tar.gz` which can be deployed to AWS SageMaker.

---

### 2. Run in CI (GitHub Actions)

The CI pipeline runs on every push to `main` or `feature/*` branches:
1. Checkout repository
2. Install dependencies
3. Lint code with `flake8`
4. Run unit tests with `pytest`
5. Train a sample model for sanity check

You can view the pipeline in `.github/workflows/ci_cd.yml`.

---

## Deployment Steps

### Step 1: Build & Push Docker Image
```bash
aws ecr get-login-password --region us-east-1   | docker login --username AWS --password-stdin <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com

docker build -t <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-flask:latest .
docker push <AWS_ACCOUNT_ID>.dkr.ecr.us-east-1.amazonaws.com/credit-risk-flask:latest
```

### Step 2: Upload Model to S3
```bash
aws s3 cp model.tar.gz s3://mlops-credit-yuhui/model/model.tar.gz
```

### Step 3: Update SageMaker Endpoint
```bash
python scripts/update_sagemaker_endpoint.py
```

This script will:
- Create a new model in SageMaker (with timestamp)
- Create a new endpoint configuration
- Deploy or update the endpoint

---

## Governance & Incident Response

- **Model Versioning**: MLflow is used to track model versions in the registry.
- **Approval Workflow**: Only approved models in MLflow can be deployed to production.
- **Incident Playbook**:
  1. Detect performance degradation or drift via monitoring tools.
  2. Roll back to a previous model version from MLflow registry.
  3. Trigger retraining pipeline with updated data.

---

## Repository Structure
```
.
├── data/                # DVC-tracked datasets
├── src/                 # Source code for data processing, training, and serving
├── scripts/             # Deployment scripts
├── .github/workflows/   # CI/CD configurations
├── requirements.txt     # Python dependencies
└── README.md            # Project documentation
```

---

## Notes
- Ensure AWS resources are cleaned up after testing to avoid charges.
- Check SageMaker endpoints, S3 buckets, and CloudWatch logs regularly.
