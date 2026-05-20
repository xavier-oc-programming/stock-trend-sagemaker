# Requires:
# - AWS account with SageMaker access enabled
# - IAM role with AmazonSageMakerFullAccess and AmazonS3FullAccess
# - S3 bucket created: xavier-sagemaker-stock-trend (or update S3_BUCKET constant)
# - AWS credentials configured: aws configure
# - Region: us-east-1
# SageMaker free tier: 250 hours of t2.medium for training (first 2 months)
# Real-time endpoint: ml.t2.medium ~$0.065/hour — DELETE after demo to avoid charges
# Run delete_endpoint.py after recording the live URL for the portfolio

# wait=True blocks this script until the training job completes and prints
# live CloudWatch logs to stdout. If the job fails, check the SageMaker
# console: AWS Console → SageMaker → Training → Training jobs → <job name>
# The "Failure reason" field and the CloudWatch log group
# /aws/sagemaker/TrainingJobs will show the full traceback.
# To monitor without blocking, set wait=False and poll with:
#   sm_client.describe_training_job(TrainingJobName=job_name)

import json
from datetime import datetime

import boto3
import sagemaker
from sagemaker.tensorflow import TensorFlow

from config import (
    MODEL_DIR,
    REGION,
    ROLE_NAME,
    S3_BUCKET,
    S3_PREFIX,
    TRAINING_INSTANCE,
)

MODEL_DIR.mkdir(exist_ok=True)


def main() -> None:
    print("=== run_training_job.py ===")

    # ── Resolve IAM role ARN ───────────────────────────────────────────────────
    print(f"\nResolving IAM role: {ROLE_NAME}...")
    iam = boto3.client('iam', region_name=REGION)
    role_arn = iam.get_role(RoleName=ROLE_NAME)['Role']['Arn']
    print(f"  Role ARN: {role_arn}")

    # ── Create TensorFlow estimator ────────────────────────────────────────────
    estimator = TensorFlow(
        entry_point='train_sagemaker.py',
        source_dir='training',
        role=role_arn,
        instance_count=1,
        instance_type=TRAINING_INSTANCE,
        framework_version='2.15',
        py_version='py311',
        hyperparameters={
            'epochs': 50,
            'batch-size': 32,
            'lstm-units-1': 64,
            'lstm-units-2': 32,
            'dropout-rate': 0.2,
            'learning-rate': 0.001,
        },
        output_path=f's3://{S3_BUCKET}/{S3_PREFIX}/model-artifacts/',
        base_job_name='stock-trend-lstm',
        sagemaker_session=sagemaker.Session(boto_session=boto3.Session(region_name=REGION)),
    )

    # ── Data channels — S3 URIs written by prepare_data.py ────────────────────
    train_data = f's3://{S3_BUCKET}/{S3_PREFIX}/data/train/'
    test_data  = f's3://{S3_BUCKET}/{S3_PREFIX}/data/test/'
    print(f"\nTrain channel: {train_data}")
    print(f"Test channel:  {test_data}")

    # ── Launch training job ────────────────────────────────────────────────────
    start_time = datetime.utcnow()
    print(f"\nLaunching training job on {TRAINING_INSTANCE}...")
    print("(wait=True — live logs will stream below)\n")

    estimator.fit(
        {'train': train_data, 'test': test_data},
        wait=True,
        logs='All',
    )

    end_time = datetime.utcnow()
    duration_seconds = (end_time - start_time).seconds
    job_name = estimator.latest_training_job.name

    print(f"\n=== Training job complete ===")
    print(f"  Job name:          {job_name}")
    print(f"  Model artefact:    {estimator.model_data}")
    print(f"  Duration:          {duration_seconds // 60}m {duration_seconds % 60}s")

    # ── Persist metadata for downstream scripts ────────────────────────────────
    metadata = {
        'job_name': job_name,
        'model_data_s3': estimator.model_data,
        'training_instance': TRAINING_INSTANCE,
        'hyperparameters': estimator.hyperparameters(),
        'completed_at': end_time.isoformat() + 'Z',
        'duration_seconds': duration_seconds,
    }
    metadata_path = str(MODEL_DIR / 'training_job_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n  Metadata saved to {metadata_path}")
    print("  Run register_model.py next.")


if __name__ == '__main__':
    main()
