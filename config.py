# Requires:
# - AWS account with SageMaker access enabled
# - IAM role with AmazonSageMakerFullAccess and AmazonS3FullAccess
# - S3 bucket created: xavier-sagemaker-stock-trend (or update S3_BUCKET constant)
# - AWS credentials configured: aws configure
# - Region: us-east-1
# SageMaker free tier: 250 hours of t2.medium for training (first 2 months)
# Real-time endpoint: ml.t2.medium ~$0.065/hour — DELETE after demo to avoid charges
# Run delete_endpoint.py after recording the live URL for the portfolio

from pathlib import Path

# ── S3 / AWS ──────────────────────────────────────────────────────────────────
S3_BUCKET = 'xavier-sagemaker-stock-trend'
S3_PREFIX = 'stock-trend'
REGION = 'us-east-1'
ROLE_NAME = 'SageMakerExecutionRole'  # IAM role name — update if different

# ── Data constants — identical to stock-trend-lstm ────────────────────────────
TICKERS = ['AAPL', 'MSFT', 'GOOGL']
SEQUENCE_LENGTH = 60
PREDICTION_HORIZON = 5
TRAIN_SPLIT = 0.8
RANDOM_STATE = 42

# ── SageMaker training job ────────────────────────────────────────────────────
# ml.m5.xlarge is cost-effective for TF training jobs.
# ml.p2.xlarge (GPU) is faster but ~10x more expensive — not needed
# for this sequence length and dataset size.
TRAINING_INSTANCE = 'ml.m5.xlarge'

# ── SageMaker endpoint ────────────────────────────────────────────────────────
# ml.t2.medium is the cheapest real-time endpoint instance.
# DELETE the endpoint after recording the demo URL — it bills by the hour.
ENDPOINT_INSTANCE = 'ml.t2.medium'
ENDPOINT_NAME = 'stock-trend-endpoint'

# ── Model Registry ────────────────────────────────────────────────────────────
MODEL_PACKAGE_GROUP = 'stock-trend-models'
MODEL_APPROVAL_STATUS = 'Approved'

# ── Local directories ─────────────────────────────────────────────────────────
MODEL_DIR = Path('models')
PLOTS_DIR = Path('plots')
DATA_DIR = Path('data')

# ── Amazon Bedrock ────────────────────────────────────────────────────────────
BEDROCK_MODEL_ID = 'us.anthropic.claude-haiku-4-5-20251001-v1:0'
BEDROCK_REGION = 'us-east-1'
