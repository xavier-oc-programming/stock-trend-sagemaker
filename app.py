# Pattern detection only — not financial advice. Past patterns do not
# guarantee future results.
#
# This Flask app does NO local ML inference.
# All predictions are served by the SageMaker Real-Time Endpoint via boto3.
# Requires endpoint_metadata.json to be present in models/.

import json
import pickle
from datetime import datetime
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import yfinance as yf
from flask import Flask, jsonify, render_template, request

from config import ENDPOINT_NAME, MODEL_DIR, REGION, SEQUENCE_LENGTH
from sequence_builder import add_technical_indicators

app = Flask(__name__)

# ── Load artefacts at startup ─────────────────────────────────────────────────
MODEL_LOADED = True
scaler = None
feature_names = None
sequence_config = None
endpoint_metadata = None
model_metrics = None

_load_errors: list[str] = []

for name, path in [
    ('scaler',            MODEL_DIR / 'scaler.pkl'),
    ('feature_names',     MODEL_DIR / 'feature_names.pkl'),
    ('sequence_config',   MODEL_DIR / 'sequence_config.json'),
    ('endpoint_metadata', MODEL_DIR / 'endpoint_metadata.json'),
]:
    if not path.exists():
        _load_errors.append(str(path))
        MODEL_LOADED = False

if MODEL_LOADED:
    with open(MODEL_DIR / 'scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open(MODEL_DIR / 'feature_names.pkl', 'rb') as f:
        feature_names = pickle.load(f)
    with open(MODEL_DIR / 'sequence_config.json') as f:
        sequence_config = json.load(f)
    with open(MODEL_DIR / 'endpoint_metadata.json') as f:
        endpoint_metadata = json.load(f)
    _active_endpoint = endpoint_metadata.get('endpoint_name', ENDPOINT_NAME)
else:
    _active_endpoint = ENDPOINT_NAME

metrics_path = MODEL_DIR / 'model_metrics.json'
if metrics_path.exists():
    with open(metrics_path) as f:
        model_metrics = json.load(f)

if _load_errors:
    print(f"WARNING: Missing model files: {_load_errors}")
    print("Start the SageMaker pipeline first, then re-launch app.py.")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _confidence_label(prob: float) -> str:
    if prob >= 0.70 or prob <= 0.30:
        return 'high'
    if prob >= 0.60 or prob <= 0.40:
        return 'medium'
    return 'low'


def _ma_crossover_signal(df: pd.DataFrame) -> dict:
    """Return MA crossover signal from the most recent row in df."""
    latest = df.iloc[-1]
    sma20 = float(latest.get('SMA_20', np.nan))
    sma50 = float(latest.get('SMA_50', np.nan))

    if np.isnan(sma20) or np.isnan(sma50):
        return {'prediction': 'unknown', 'sma20': None, 'sma50': None, 'signal': 'unknown'}

    if sma20 > sma50:
        signal = 'golden cross'
        prediction = 'bullish'
    elif sma20 < sma50:
        signal = 'death cross'
        prediction = 'bearish'
    else:
        signal = 'neutral'
        prediction = 'neutral'

    return {
        'prediction': prediction,
        'sma20': round(sma20, 4),
        'sma50': round(sma50, 4),
        'signal': signal,
    }


def _call_sagemaker_endpoint(sequence: np.ndarray) -> float:
    """Invoke the SageMaker endpoint and return the raw sigmoid probability."""
    runtime = boto3.client('sagemaker-runtime', region_name=REGION)
    payload = json.dumps({'instances': sequence.tolist()})
    response = runtime.invoke_endpoint(
        EndpointName=_active_endpoint,
        ContentType='application/json',
        Body=payload,
    )
    result = json.loads(response['Body'].read())
    # TF Serving returns {'predictions': [[prob]]}
    probability = float(result['predictions'][0][0])
    return probability


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if not MODEL_LOADED:
        return jsonify({
            'error': 'Model artefacts not found. Run the SageMaker pipeline first.',
            'missing_files': _load_errors,
        }), 503

    data = request.get_json(force=True)
    ticker = data.get('ticker', '').upper().strip()
    period = data.get('period', '1y')

    if not ticker:
        return jsonify({'error': 'ticker is required'}), 400

    # ── Download fresh data ────────────────────────────────────────────────────
    try:
        df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty or len(df) < SEQUENCE_LENGTH + 60:
            return jsonify({'error': f'Insufficient data for {ticker}'}), 400
    except Exception as e:
        return jsonify({'error': f'yfinance error: {str(e)}'}), 400

    # ── Feature engineering ────────────────────────────────────────────────────
    try:
        df = add_technical_indicators(df)
    except Exception as e:
        return jsonify({'error': f'Feature engineering failed: {str(e)}'}), 400

    if len(df) < SEQUENCE_LENGTH:
        return jsonify({'error': 'Not enough data after indicator calculation'}), 400

    # Actual date range used for this prediction — the last SEQUENCE_LENGTH rows
    sequence_start_date = df.index[-SEQUENCE_LENGTH].strftime('%d %b %Y')
    sequence_end_date   = df.index[-1].strftime('%d %b %Y')
    as_of_date          = df.index[-1].strftime('%Y-%m-%d')

    # Model training date — loaded for transparency in the response
    trained_at_display = 'Unknown'
    try:
        with open(MODEL_DIR / 'training_job_metadata.json') as f:
            _tj = json.load(f)
            _raw = _tj.get('completed_at', '')
            if _raw and _raw != 'TBD':
                try:
                    trained_at_display = datetime.fromisoformat(_raw.replace('Z', '')).strftime('%d %b %Y')
                except Exception:
                    trained_at_display = _raw
    except Exception:
        pass

    # ── MA crossover signal (local — no endpoint needed) ──────────────────────
    ma_signal = _ma_crossover_signal(df)

    # ── Build input sequence ───────────────────────────────────────────────────
    X_raw = df[feature_names].values[-SEQUENCE_LENGTH:]        # shape (60, n_features)
    X_scaled = scaler.transform(X_raw)                         # same scaler fitted on train
    sequence = X_scaled[np.newaxis, :, :]                      # shape (1, 60, n_features)

    # ── Call SageMaker endpoint ────────────────────────────────────────────────
    try:
        probability = _call_sagemaker_endpoint(sequence)
    except Exception as e:
        return jsonify({'error': f'SageMaker endpoint error: {str(e)}'}), 503

    lstm_prediction = 'bullish' if probability >= 0.5 else 'bearish'
    lstm_agreement = lstm_prediction == ma_signal.get('prediction', '')
    inference_source = 'SageMaker Real-Time Endpoint'

    # ── Recent price window for chart ─────────────────────────────────────────
    recent = df.tail(30)
    recent_prices = [round(float(p), 2) for p in recent['Close'].values]
    recent_dates  = [str(d.date()) for d in recent.index]

    temporal_summary = (
        f"Pattern based on {SEQUENCE_LENGTH} trading days ending {sequence_end_date} "
        f"· Model trained {trained_at_display} · Live SageMaker endpoint"
    )

    return jsonify({
        'ticker': ticker,
        'as_of_date': as_of_date,
        'sequence_days': SEQUENCE_LENGTH,
        'sequence_start_date': sequence_start_date,
        'sequence_end_date': sequence_end_date,
        'model_trained_at': trained_at_display,
        'lstm': {
            'prediction': lstm_prediction,
            'probability': round(probability, 4),
            'confidence': _confidence_label(probability),
            'served_by': inference_source,
        },
        'ma_crossover': ma_signal,
        'agreement': lstm_agreement,
        'recent_prices': recent_prices,
        'recent_dates': recent_dates,
        'temporal_summary': temporal_summary,
        'disclaimer': 'Pattern detection only. Not financial advice.',
    })


@app.route('/api/model-info')
def model_info():
    info: dict = {'served_by': 'Amazon SageMaker Real-Time Endpoint'}
    if model_metrics:
        info.update(model_metrics)
    if endpoint_metadata:
        info.update(endpoint_metadata)

    # Add training date in both raw and display format
    try:
        with open(MODEL_DIR / 'training_job_metadata.json') as f:
            _tj = json.load(f)
            _raw = _tj.get('completed_at', '')
            info['model_trained_at'] = _raw
            if _raw and _raw != 'TBD':
                try:
                    info['model_trained_at_display'] = datetime.fromisoformat(
                        _raw.replace('Z', '')
                    ).strftime('%d %b %Y')
                except Exception:
                    info['model_trained_at_display'] = _raw
            else:
                info['model_trained_at_display'] = 'Unknown'
    except Exception:
        info['model_trained_at'] = None
        info['model_trained_at_display'] = 'Unknown'

    return jsonify(info)


@app.route('/api/pipeline-status')
def pipeline_status():
    # Training job metadata
    tj_meta = None
    tj_path = MODEL_DIR / 'training_job_metadata.json'
    if tj_path.exists():
        with open(tj_path) as f:
            tj_raw = json.load(f)
        tj_meta = {
            'name': tj_raw.get('job_name'),
            'completed_at': tj_raw.get('completed_at'),
        }

    # Endpoint active check via SageMaker API
    endpoint_active = False
    active_endpoint_name = None
    try:
        sm = boto3.client('sagemaker', region_name=REGION)
        resp = sm.describe_endpoint(EndpointName=_active_endpoint)
        endpoint_active = resp['EndpointStatus'] == 'InService'
        active_endpoint_name = _active_endpoint if endpoint_active else None
    except Exception:
        pass

    return jsonify({
        'data_prepared': (MODEL_DIR / 'sequence_config.json').exists(),
        'training_job': tj_meta,
        'model_registered': (MODEL_DIR / 'model_package_arn.txt').exists(),
        'endpoint_active': endpoint_active,
        'endpoint_name': active_endpoint_name,
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
