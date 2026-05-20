# bedrock_comparator.py
# Compares SageMaker-hosted LSTM predictions against Amazon Bedrock (Claude Haiku)
# zero-shot pattern interpretation on 30 test sequences.
# Evaluates whether a prompted LLM matches a trained model on the same sequences.
# Part of the four-project cross-task Bedrock benchmark:
#   tabular classification (churn), short text (spam),
#   long text (sentiment), time series (trend ← this project).

# Requires:
# - AWS credentials configured: aws configure
# - Bedrock access enabled in us-east-1
# - models/scaler.pkl and models/feature_names.pkl (from prepare_data.py)
# - data/X_test.npy and data/y_test.npy (from prepare_data.py)
# - LSTM predictions available via models/model_metrics.json

import json
import pickle
import random

import boto3
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from config import (
    BEDROCK_MODEL_ID,
    BEDROCK_REGION,
    DATA_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    RANDOM_STATE,
    SEQUENCE_LENGTH,
)

random.seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

N_SEQUENCES = 30  # number of test sequences to evaluate


def load_test_data() -> tuple[np.ndarray, np.ndarray, list[str]]:
    X_test = np.load(str(DATA_DIR / 'X_test.npy'))
    y_test = np.load(str(DATA_DIR / 'y_test.npy'))
    with open(str(MODEL_DIR / 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    return X_test, y_test, feature_names


def format_sequence_for_bedrock(sequence: np.ndarray, feature_names: list[str]) -> str:
    """Summarise the last 5 days of key indicators as a compact text prompt."""
    # sequence shape: (SEQUENCE_LENGTH, n_features)
    last_5 = sequence[-5:]
    feat_idx = {name: i for i, name in enumerate(feature_names)}

    lines = ["Recent 5-day snapshot (most recent last):"]
    for day_idx, row in enumerate(last_5):
        close_i = feat_idx.get('Close', 3)
        rsi_i   = feat_idx.get('RSI', 12)
        macd_i  = feat_idx.get('MACD', 9)
        ret_i   = feat_idx.get('Return_1d', 18)

        lines.append(
            f"  Day -{4 - day_idx}: Close={row[close_i]:.2f}, "
            f"RSI={row[rsi_i]:.1f}, MACD={row[macd_i]:.4f}, "
            f"1d_return={row[ret_i]:.4f}"
        )
    return '\n'.join(lines)


def query_bedrock(client, prompt_text: str) -> dict:
    """Query Bedrock Claude Haiku and return parsed JSON."""
    system_prompt = (
        "You are a technical analysis assistant. "
        "Given a 5-day snapshot of stock price indicators, predict whether "
        "the stock will trend UP or DOWN over the next 5 trading days. "
        "Respond ONLY with valid JSON in the format: "
        '{"direction": "up" or "down", "confidence": 0.0-1.0, "reasoning": "one sentence"}'
    )

    body = json.dumps({
        'anthropic_version': 'bedrock-2023-05-31',
        'max_tokens': 200,
        'system': system_prompt,
        'messages': [{'role': 'user', 'content': prompt_text}],
    })

    response = client.invoke_model(
        modelId=BEDROCK_MODEL_ID,
        body=body,
        contentType='application/json',
        accept='application/json',
    )

    result = json.loads(response['body'].read())
    raw_text = result['content'][0]['text']

    # Strip markdown code fences if present
    clean_text = raw_text.strip()
    if clean_text.startswith('```'):
        clean_text = clean_text.split('```')[1]
        if clean_text.startswith('json'):
            clean_text = clean_text[4:]

    parsed = json.loads(clean_text.strip())
    return parsed


def main() -> None:
    print("=== bedrock_comparator.py ===")
    print(f"Model: {BEDROCK_MODEL_ID}")
    print(f"Evaluating {N_SEQUENCES} test sequences...\n")

    # ── Load data ──────────────────────────────────────────────────────────────
    X_test, y_test, feature_names = load_test_data()
    print(f"Test set: {X_test.shape[0]} sequences")

    # Sample N_SEQUENCES random indices
    indices = random.sample(range(len(X_test)), min(N_SEQUENCES, len(X_test)))
    X_sample = X_test[indices]
    y_sample = y_test[indices]

    # ── Bedrock inference ──────────────────────────────────────────────────────
    bedrock = boto3.client('bedrock-runtime', region_name=BEDROCK_REGION)
    bedrock_preds = []
    bedrock_confs = []
    results_rows = []

    for i, (seq, true_label) in enumerate(zip(X_sample, y_sample)):
        print(f"  [{i+1:02d}/{N_SEQUENCES}] ", end='', flush=True)
        prompt = format_sequence_for_bedrock(seq, feature_names)

        try:
            parsed = query_bedrock(bedrock, prompt)
            direction = parsed.get('direction', 'down').lower()
            confidence = float(parsed.get('confidence', 0.5))
            reasoning = parsed.get('reasoning', '')

            pred_label = 1 if direction == 'up' else 0
            correct = int(pred_label == int(true_label))
            print(f"{'✓' if correct else '✗'}  direction={direction}  conf={confidence:.2f}")

        except Exception as e:
            print(f"ERROR: {e}")
            pred_label = 0
            confidence = 0.5
            reasoning = 'parse error'
            correct = int(pred_label == int(true_label))

        bedrock_preds.append(pred_label)
        bedrock_confs.append(confidence)
        results_rows.append({
            'index': indices[i],
            'true_label': int(true_label),
            'bedrock_pred': pred_label,
            'bedrock_conf': round(confidence, 4),
            'correct': correct,
        })

    # ── Metrics ────────────────────────────────────────────────────────────────
    bedrock_acc = accuracy_score(y_sample, bedrock_preds)
    bedrock_f1  = f1_score(y_sample, bedrock_preds, zero_division=0)
    try:
        bedrock_auc = roc_auc_score(y_sample, bedrock_confs)
    except Exception:
        bedrock_auc = float('nan')

    # Load LSTM metrics for comparison
    lstm_metrics = {}
    metrics_path = MODEL_DIR / 'model_metrics.json'
    if metrics_path.exists():
        with open(metrics_path) as f:
            lstm_metrics = json.load(f)

    print(f"\n{'='*52}")
    print(f"{'Metric':<20} {'SageMaker LSTM':>15} {'Bedrock Haiku':>15}")
    print(f"{'-'*52}")
    print(f"{'Accuracy':<20} {lstm_metrics.get('accuracy', 'TBD'):>15} {bedrock_acc:>15.4f}")
    print(f"{'F1 Score':<20} {lstm_metrics.get('f1', 'TBD'):>15} {bedrock_f1:>15.4f}")
    print(f"{'ROC-AUC':<20} {lstm_metrics.get('roc_auc', 'TBD'):>15} {bedrock_auc:>15.4f}")
    print(f"{'='*52}")
    print(f"\nBedrock evaluated {N_SEQUENCES} sequences via zero-shot prompting.")
    print("LSTM evaluated on the full test set (SageMaker Training Job output).")
    print("Bedrock uses only the last 5 days of indicators; LSTM uses all 60.")

    # ── Save results ───────────────────────────────────────────────────────────
    results_df = pd.DataFrame(results_rows)
    sample_path = str(MODEL_DIR / 'comparison_sample.csv')
    results_df.to_csv(sample_path, index=False)
    print(f"\nSample results saved to {sample_path}")

    summary = {
        'n_sequences': N_SEQUENCES,
        'bedrock_model': BEDROCK_MODEL_ID,
        'bedrock_accuracy': round(bedrock_acc, 4),
        'bedrock_f1': round(bedrock_f1, 4),
        'bedrock_roc_auc': round(bedrock_auc, 4) if not np.isnan(bedrock_auc) else None,
        'lstm_accuracy': lstm_metrics.get('accuracy'),
        'lstm_f1': lstm_metrics.get('f1'),
        'lstm_roc_auc': lstm_metrics.get('roc_auc'),
    }
    with open(str(MODEL_DIR / 'bedrock_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print("Summary saved to models/bedrock_results.json")


if __name__ == '__main__':
    main()
