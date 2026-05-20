# sequence_builder.py
# Feature engineering and sequence construction — identical to
# stock-trend-lstm. Separated into a standalone module so it can
# be used both locally (for data prep) and inside the SageMaker
# training container (uploaded as a dependency with the training script).

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import SEQUENCE_LENGTH, PREDICTION_HORIZON


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators to OHLCV dataframe. Returns df with NaNs dropped."""
    df = df.copy()

    # Moving averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI (14-period)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = rolling_mean + 2 * rolling_std
    df['BB_Lower'] = rolling_mean - 2 * rolling_std
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Width'] + 1e-10)

    # Volume indicators
    df['Volume_SMA_20'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_SMA_20'] + 1e-10)

    # Price momentum
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_20d'] = df['Close'].pct_change(20)

    # Average True Range (volatility)
    high_low = df['High'] - df['Low']
    high_close = (df['High'] - df['Close'].shift()).abs()
    low_close = (df['Low'] - df['Close'].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['ATR_14'] = true_range.rolling(window=14).mean()

    # Price relative to moving averages — normalised
    df['Price_to_SMA20'] = df['Close'] / (df['SMA_20'] + 1e-10)
    df['Price_to_SMA50'] = df['Close'] / (df['SMA_50'] + 1e-10)

    df.dropna(inplace=True)
    return df


def create_labels(df: pd.DataFrame, horizon: int = PREDICTION_HORIZON) -> pd.DataFrame:
    """Create binary trend labels: 1 if Close rises over next `horizon` days, else 0."""
    df = df.copy()
    # future_return > 0 → uptrend (1), else downtrend (0)
    df['Future_Return'] = df['Close'].shift(-horizon) / df['Close'] - 1
    df['Label'] = (df['Future_Return'] > 0).astype(int)
    df.dropna(inplace=True)
    return df


def build_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a window of `sequence_length` over X to create 3-D sequences."""
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i: i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, transform both splits. Returns scaled arrays + fitted scaler."""
    n_train, n_test = X_train.shape[0], X_test.shape[0]
    n_features = X_train.shape[1]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(n_train, -1)
    X_test_scaled = scaler.transform(X_test.reshape(-1, n_features)).reshape(n_test, -1)
    return X_train_scaled, X_test_scaled, scaler
