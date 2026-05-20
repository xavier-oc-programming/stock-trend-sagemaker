# delete_endpoint.py
# Run this script after recording the live demo URL for the portfolio.
# SageMaker real-time endpoints bill by the hour (~$0.065/hr for ml.t2.medium).
# Leaving an endpoint running unused will incur charges.
# This script deletes both the endpoint and its configuration.

# Requires:
# - AWS credentials configured: aws configure
# - Region: us-east-1

import json
from datetime import datetime

import boto3

from config import ENDPOINT_NAME, MODEL_DIR, REGION


def main() -> None:
    print("=== delete_endpoint.py ===")

    metadata_path = str(MODEL_DIR / 'endpoint_metadata.json')
    with open(metadata_path) as f:
        metadata = json.load(f)

    endpoint_name = metadata['endpoint_name']
    sm_client = boto3.client('sagemaker', region_name=REGION)

    # ── Delete endpoint ────────────────────────────────────────────────────────
    print(f"\nDeleting endpoint: {endpoint_name}...")
    sm_client.delete_endpoint(EndpointName=endpoint_name)
    print("  Endpoint deleted.")

    # ── Delete endpoint configuration ──────────────────────────────────────────
    # The endpoint config is a separate resource that persists after deletion.
    # It is safe to delete — re-deploying will create a new config.
    try:
        sm_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
        print("  Endpoint configuration deleted.")
    except sm_client.exceptions.ClientError as e:
        # Config name may differ from endpoint name in edge cases — safe to skip
        print(f"  Note: could not delete endpoint config — {e}")

    # ── Update metadata with deletion timestamp ────────────────────────────────
    metadata['deleted_at'] = datetime.utcnow().isoformat() + 'Z'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nEndpoint '{endpoint_name}' deleted. No further charges.")


if __name__ == '__main__':
    main()
