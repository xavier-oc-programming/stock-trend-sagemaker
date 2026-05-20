# Requires:
# - AWS account with SageMaker access enabled
# - IAM role with AmazonSageMakerFullAccess and AmazonS3FullAccess
# - S3 bucket created: xavier-sagemaker-stock-trend (or update S3_BUCKET constant)
# - AWS credentials configured: aws configure
# - Region: us-east-1
# SageMaker free tier: 250 hours of t2.medium for training (first 2 months)
# Real-time endpoint: ml.t2.medium ~$0.065/hour — DELETE after demo to avoid charges
# Run delete_endpoint.py after recording the live URL for the portfolio

import json
from datetime import datetime

import boto3
import numpy as np
import sagemaker
from sagemaker import ModelPackage

from config import (
    ENDPOINT_INSTANCE,
    ENDPOINT_NAME,
    MODEL_DIR,
    REGION,
    ROLE_NAME,
    SEQUENCE_LENGTH,
)

MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=== deploy_endpoint.py ===")

    # ── Load model package ARN ─────────────────────────────────────────────────
    arn_path = str(MODEL_DIR / 'model_package_arn.txt')
    with open(arn_path) as f:
        model_package_arn = f.read().strip()
    print(f"\nModel package ARN: {model_package_arn}")

    # ── Resolve IAM role ARN ───────────────────────────────────────────────────
    iam = boto3.client('iam', region_name=REGION)
    role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']

    # ── Load sequence config for feature count ─────────────────────────────────
    with open(str(MODEL_DIR / 'sequence_config.json')) as f:
        seq_config = json.load(f)
    n_features = seq_config['n_features']

    # ── Create ModelPackage and deploy ─────────────────────────────────────────
    print(f"\nDeploying to endpoint '{ENDPOINT_NAME}' on {ENDPOINT_INSTANCE}...")
    print("(wait=True — this typically takes 5-10 minutes)\n")

    sm_session = sagemaker.Session(
        boto_session=boto3.Session(region_name=REGION)
    )

    model = ModelPackage(
        role=role_arn,
        model_package_arn=model_package_arn,
        sagemaker_session=sm_session,
    )

    predictor = model.deploy(
        initial_instance_count=1,
        instance_type=ENDPOINT_INSTANCE,
        endpoint_name=ENDPOINT_NAME,
        wait=True,
    )

    print(f"\nEndpoint '{ENDPOINT_NAME}' is active.")

    # ── Smoke test ─────────────────────────────────────────────────────────────
    print("\nRunning endpoint smoke test...")
    dummy_input = np.zeros((1, SEQUENCE_LENGTH, n_features)).tolist()
    response = predictor.predict({'instances': dummy_input})
    print(f"  Smoke test response: {response}")

    # ── Save endpoint metadata ─────────────────────────────────────────────────
    deployed_at = datetime.utcnow().isoformat() + 'Z'
    endpoint_metadata = {
        'endpoint_name': ENDPOINT_NAME,
        'endpoint_url': (
            f'https://runtime.sagemaker.{REGION}.amazonaws.com'
            f'/endpoints/{ENDPOINT_NAME}/invocations'
        ),
        'instance_type': ENDPOINT_INSTANCE,
        'deployed_at': deployed_at,
        'model_package_arn': model_package_arn,
        'n_features': n_features,
    }
    metadata_path = str(MODEL_DIR / 'endpoint_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(endpoint_metadata, f, indent=2)
    print(f"\n  Metadata saved to {metadata_path}")

    print(f"\nEndpoint deployed: {ENDPOINT_NAME}")
    print("IMPORTANT: Delete this endpoint when done to avoid charges.")
    print("  Run: python delete_endpoint.py")
    print(f"  Estimated cost: ~$0.065/hour while running ({ENDPOINT_INSTANCE})")


if __name__ == '__main__':
    main()
