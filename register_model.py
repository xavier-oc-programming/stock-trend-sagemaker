# Requires:
# - AWS account with SageMaker access enabled
# - IAM role with AmazonSageMakerFullAccess and AmazonS3FullAccess
# - S3 bucket created: xavier-sagemaker-stock-trend (or update S3_BUCKET constant)
# - AWS credentials configured: aws configure
# - Region: us-east-1
# SageMaker free tier: 250 hours of t2.medium for training (first 2 months)
# Real-time endpoint: ml.t2.medium ~$0.065/hour — DELETE after demo to avoid charges
# Run delete_endpoint.py after recording the live URL for the portfolio

# The SageMaker Model Registry maintains a versioned catalogue of
# trained models with their artefact locations, inference specs,
# and approval status. This is the production equivalent of the
# manual model_registry.csv used in telco-churn-predictor —
# the same pattern (version every training run, log metrics,
# approve before deployment) implemented as a managed AWS service.

import json
from datetime import datetime

import boto3
import sagemaker

from config import (
    ENDPOINT_INSTANCE,
    MODEL_APPROVAL_STATUS,
    MODEL_DIR,
    MODEL_PACKAGE_GROUP,
    REGION,
)

MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=== register_model.py ===")

    # ── Load training job metadata ─────────────────────────────────────────────
    metadata_path = str(MODEL_DIR / 'training_job_metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)
    model_data_s3 = metadata['model_data_s3']
    print(f"\nModel artefact: {model_data_s3}")

    sm_client = boto3.client('sagemaker', region_name=REGION)

    # ── Create model package group (idempotent) ────────────────────────────────
    print(f"\nEnsuring model package group '{MODEL_PACKAGE_GROUP}' exists...")
    try:
        sm_client.create_model_package_group(
            ModelPackageGroupName=MODEL_PACKAGE_GROUP,
            ModelPackageGroupDescription='Stock trend LSTM models — SageMaker MLOps',
        )
        print("  Created new group.")
    except sm_client.exceptions.ClientError:
        # Group already exists — expected on re-runs
        print("  Group already exists.")

    # ── Retrieve the TF inference container image URI ─────────────────────────
    inference_image = sagemaker.image_uris.retrieve(
        'tensorflow',
        REGION,
        version='2.16',
        image_scope='inference',
        instance_type=ENDPOINT_INSTANCE,
    )
    print(f"  Inference image: {inference_image}")

    # ── Register model package ─────────────────────────────────────────────────
    timestamp = datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')
    print(f"\nRegistering model package (status: {MODEL_APPROVAL_STATUS})...")

    response = sm_client.create_model_package(
        ModelPackageGroupName=MODEL_PACKAGE_GROUP,
        ModelPackageDescription=f'Stock trend LSTM — trained {timestamp}',
        InferenceSpecification={
            'Containers': [{
                'Image': inference_image,
                'ModelDataUrl': model_data_s3,
                'Environment': {
                    'SAGEMAKER_TFS_DEFAULT_MODEL_NAME': 'model',
                },
            }],
            'SupportedContentTypes': ['application/json'],
            'SupportedResponseMIMETypes': ['application/json'],
            'SupportedRealtimeInferenceInstanceTypes': [ENDPOINT_INSTANCE],
        },
        ModelApprovalStatus=MODEL_APPROVAL_STATUS,
    )

    model_package_arn = response['ModelPackageArn']
    print(f"\n  Model package ARN: {model_package_arn}")

    # ── Persist ARN for deploy_endpoint.py ────────────────────────────────────
    arn_path = str(MODEL_DIR / 'model_package_arn.txt')
    with open(arn_path, 'w') as f:
        f.write(model_package_arn)
    print(f"  ARN saved to {arn_path}")
    print("\n  Run deploy_endpoint.py next.")


if __name__ == '__main__':
    main()
