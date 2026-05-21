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
import pickle
from datetime import datetime
from pathlib import Path

import boto3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yfinance as yf

from config import (
    DATA_DIR,
    MODEL_DIR,
    PLOTS_DIR,
    PREDICTION_HORIZON,
    RANDOM_STATE,
    REGION,
    S3_BUCKET,
    S3_PREFIX,
    SEQUENCE_LENGTH,
    TICKERS,
    TRAIN_SPLIT,
)
from sequence_builder import (
    add_technical_indicators,
    build_sequences,
    create_labels,
    scale_features,
)

# ── Setup ─────────────────────────────────────────────────────────────────────
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)

s3_client = boto3.client('s3', region_name=REGION)


def download_ticker_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    print(f"  Downloading {ticker} ({period})...")
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    # yfinance 1.x returns MultiIndex columns even for single tickers — flatten to simple names
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['Ticker'] = ticker
    return df


def upload_to_s3(local_path: str, s3_key: str) -> str:
    s3_client.upload_file(local_path, S3_BUCKET, s3_key)
    uri = f's3://{S3_BUCKET}/{s3_key}'
    print(f"  Uploaded → {uri}")
    return uri


# ── 01 Price history ──────────────────────────────────────────────────────────
def plot_price_history(frames: dict[str, pd.DataFrame]) -> None:
    fig, axes = plt.subplots(len(frames), 1, figsize=(14, 4 * len(frames)))
    for ax, (ticker, df) in zip(axes, frames.items()):
        ax.plot(df.index, df['Close'], linewidth=1)
        ax.set_title(f'{ticker} — Closing Price (5 years)', fontsize=12)
        ax.set_ylabel('Price (USD)')
        ax.grid(alpha=0.3)
    plt.tight_layout()
    path = str(PLOTS_DIR / '01_price_history.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 02 Technical indicators ───────────────────────────────────────────────────
def plot_technical_indicators(df: pd.DataFrame, ticker: str = 'AAPL') -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 10))

    axes[0].plot(df.index, df['Close'], label='Close', linewidth=1)
    axes[0].plot(df.index, df['SMA_20'], label='SMA 20', linewidth=1)
    axes[0].plot(df.index, df['SMA_50'], label='SMA 50', linewidth=1)
    axes[0].set_title(f'{ticker} — Price & Moving Averages')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(df.index, df['RSI'], linewidth=1, color='purple')
    axes[1].axhline(70, color='red', linestyle='--', linewidth=0.8, label='Overbought 70')
    axes[1].axhline(30, color='green', linestyle='--', linewidth=0.8, label='Oversold 30')
    axes[1].set_title('RSI (14)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    axes[2].plot(df.index, df['MACD'], label='MACD', linewidth=1)
    axes[2].plot(df.index, df['MACD_Signal'], label='Signal', linewidth=1)
    axes[2].bar(df.index, df['MACD_Hist'], alpha=0.3, label='Histogram')
    axes[2].set_title('MACD')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    plt.tight_layout()
    path = str(PLOTS_DIR / '02_technical_indicators.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 03 Label distribution ─────────────────────────────────────────────────────
def plot_label_distribution(y_train: np.ndarray, y_test: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (split, y) in zip(axes, [('Train', y_train), ('Test', y_test)]):
        counts = pd.Series(y).value_counts().sort_index()
        ax.bar(['Bearish (0)', 'Bullish (1)'], counts.values, color=['#e74c3c', '#2ecc71'])
        ax.set_title(f'Label Distribution — {split}')
        ax.set_ylabel('Count')
        for i, v in enumerate(counts.values):
            ax.text(i, v + 5, str(v), ha='center', fontsize=9)
    plt.tight_layout()
    path = str(PLOTS_DIR / '03_label_distribution.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── 04 Feature correlation ────────────────────────────────────────────────────
def plot_feature_correlation(X_df: pd.DataFrame) -> None:
    corr = X_df.corr()
    plt.figure(figsize=(14, 10))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='RdBu_r', center=0,
                linewidths=0.3, vmin=-1, vmax=1)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    path = str(PLOTS_DIR / '04_feature_correlation.png')
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved {path}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    print("=== prepare_data.py ===")

    # ── Download ──────────────────────────────────────────────────────────────
    print("\n[1/6] Downloading data...")
    raw_frames: dict[str, pd.DataFrame] = {}
    enriched_frames: list[pd.DataFrame] = []

    for ticker in TICKERS:
        df = download_ticker_data(ticker)
        raw_frames[ticker] = df
        df = add_technical_indicators(df)
        df = create_labels(df, horizon=PREDICTION_HORIZON)
        df['Ticker'] = ticker
        enriched_frames.append(df)

    # ── EDA plots 01 & 02 ────────────────────────────────────────────────────
    print("\n[2/6] Generating EDA plots...")
    plot_price_history(raw_frames)
    plot_technical_indicators(enriched_frames[0], ticker=TICKERS[0])

    # ── Combine and split ─────────────────────────────────────────────────────
    print("\n[3/6] Combining and splitting...")
    combined = pd.concat(enriched_frames).sort_index()

    feature_cols = [
        'Open', 'High', 'Low', 'Close', 'Volume',
        'SMA_20', 'SMA_50', 'EMA_12', 'EMA_26',
        'MACD', 'MACD_Signal', 'MACD_Hist',
        'RSI', 'BB_Upper', 'BB_Lower', 'BB_Width', 'BB_Position',
        'Volume_SMA_20', 'Volume_Ratio',
        'Return_1d', 'Return_5d', 'Return_20d',
        'ATR_14', 'Price_to_SMA20', 'Price_to_SMA50',
    ]

    X_all = combined[feature_cols].values
    y_all = combined['Label'].values
    dates = combined.index

    split_idx = int(len(X_all) * TRAIN_SPLIT)
    X_train_raw = X_all[:split_idx]
    X_test_raw = X_all[split_idx:]
    y_train_raw = y_all[:split_idx]
    y_test_raw = y_all[split_idx:]
    train_dates = dates[:split_idx]
    test_dates = dates[split_idx:]

    print(f"  Train rows: {len(X_train_raw)}, Test rows: {len(X_test_raw)}")

    # ── Scale ─────────────────────────────────────────────────────────────────
    print("\n[4/6] Scaling features...")
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train_raw, X_test_raw)

    # ── Plots 03 & 04 ─────────────────────────────────────────────────────────
    plot_label_distribution(y_train_raw, y_test_raw)
    plot_feature_correlation(pd.DataFrame(X_train_raw, columns=feature_cols))

    # ── Build sequences ───────────────────────────────────────────────────────
    print("\n[5/6] Building sequences...")
    X_train, y_train = build_sequences(X_train_scaled, y_train_raw)
    X_test, y_test = build_sequences(X_test_scaled, y_test_raw)

    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"  X_test:  {X_test.shape},  y_test:  {y_test.shape}")

    # ── Save locally ──────────────────────────────────────────────────────────
    np.save(str(DATA_DIR / 'X_train.npy'), X_train)
    np.save(str(DATA_DIR / 'y_train.npy'), y_train)
    np.save(str(DATA_DIR / 'X_test.npy'), X_test)
    np.save(str(DATA_DIR / 'y_test.npy'), y_test)

    # Scaler params for sequence_config.json
    scaler_params = {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist(),
        'var_': scaler.var_.tolist(),
    }

    sequence_config = {
        'SEQUENCE_LENGTH': SEQUENCE_LENGTH,
        'PREDICTION_HORIZON': PREDICTION_HORIZON,
        'TICKERS': TICKERS,
        'feature_names': feature_cols,
        'n_features': len(feature_cols),
        'scaler_params': scaler_params,
        'train_date_start': str(train_dates[0].date()),
        'train_date_end': str(train_dates[-1].date()),
        'test_date_start': str(test_dates[0].date()),
        'test_date_end': str(test_dates[-1].date()),
    }

    config_path = str(DATA_DIR / 'sequence_config.json')
    with open(config_path, 'w') as f:
        json.dump(sequence_config, f, indent=2)

    # Save scaler and feature names to models/
    with open(str(MODEL_DIR / 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(str(MODEL_DIR / 'feature_names.pkl'), 'wb') as f:
        pickle.dump(feature_cols, f)
    # Copy sequence_config to models/ for app.py startup
    with open(str(MODEL_DIR / 'sequence_config.json'), 'w') as f:
        json.dump(sequence_config, f, indent=2)

    print("  Saved local artefacts to data/ and models/")

    # ── Upload to S3 ──────────────────────────────────────────────────────────
    print("\n[6/6] Uploading to S3...")
    uploads = [
        (str(DATA_DIR / 'X_train.npy'), f'{S3_PREFIX}/data/train/X_train.npy'),
        (str(DATA_DIR / 'y_train.npy'), f'{S3_PREFIX}/data/train/y_train.npy'),
        (str(DATA_DIR / 'X_test.npy'),  f'{S3_PREFIX}/data/test/X_test.npy'),
        (str(DATA_DIR / 'y_test.npy'),  f'{S3_PREFIX}/data/test/y_test.npy'),
        (config_path,                    f'{S3_PREFIX}/data/sequence_config.json'),
        (config_path,                    f'{S3_PREFIX}/data/train/sequence_config.json'),
    ]
    for local, key in uploads:
        upload_to_s3(local, key)

    # Upload plots
    for plot_file in PLOTS_DIR.glob('*.png'):
        upload_to_s3(str(plot_file), f'{S3_PREFIX}/plots/{plot_file.name}')

    print("\n=== prepare_data.py complete ===")
    print(f"S3 train channel: s3://{S3_BUCKET}/{S3_PREFIX}/data/train/")
    print(f"S3 test channel:  s3://{S3_BUCKET}/{S3_PREFIX}/data/test/")


if __name__ == '__main__':
    main()
