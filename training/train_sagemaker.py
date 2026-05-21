# train_sagemaker.py
# Runs inside SageMaker's managed TensorFlow training container.
# Key differences from a local training script:
# 1. Data is read from environment variable paths, not local paths
# 2. Model must be saved to SM_MODEL_DIR in SavedModel format
# 3. Hyperparameters arrive via argparse, not hardcoded constants
# 4. stdout is captured as CloudWatch training logs
# 5. This script has no __main__ guard — SageMaker calls it directly

# Requires:
# - AWS account with SageMaker access enabled
# - IAM role with AmazonSageMakerFullAccess and AmazonS3FullAccess
# - S3 bucket created: xavier-sagemaker-stock-trend (or update S3_BUCKET constant)
# - AWS credentials configured: aws configure
# - Region: us-east-1

import argparse
import json
import os
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, BatchNormalization, Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# ── SageMaker directory paths (set by the training container) ─────────────────
SM_CHANNEL_TRAIN = os.environ['SM_CHANNEL_TRAIN']
SM_CHANNEL_TEST = os.environ['SM_CHANNEL_TEST']
SM_MODEL_DIR = os.environ['SM_MODEL_DIR']
SM_OUTPUT_DATA_DIR = os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data')

# ── Hyperparameters via argparse ──────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--lstm-units-1', type=int, default=64)
parser.add_argument('--lstm-units-2', type=int, default=32)
parser.add_argument('--dropout-rate', type=float, default=0.2)
parser.add_argument('--learning-rate', type=float, default=0.001)
parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
args, _ = parser.parse_known_args()

print("=== SageMaker Training Job ===")
print(f"Hyperparameters: {vars(args)}")

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"\nLoading data from {SM_CHANNEL_TRAIN} and {SM_CHANNEL_TEST}...")
X_train = np.load(os.path.join(SM_CHANNEL_TRAIN, 'X_train.npy'))
y_train = np.load(os.path.join(SM_CHANNEL_TRAIN, 'y_train.npy'))
X_test  = np.load(os.path.join(SM_CHANNEL_TEST,  'X_test.npy'))
y_test  = np.load(os.path.join(SM_CHANNEL_TEST,  'y_test.npy'))

with open(os.path.join(SM_CHANNEL_TRAIN, 'sequence_config.json')) as f:
    sequence_config = json.load(f)

n_features = X_train.shape[2]
sequence_length = X_train.shape[1]

print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
print(f"X_test:  {X_test.shape},  y_test:  {y_test.shape}")
print(f"Features: {n_features}, Sequence length: {sequence_length}")

# ── Build LSTM model ─────────────────────────────────────────────────────────
# Architecture identical to stock-trend-lstm; hyperparameter values arrive
# from the training job definition via argparse above.
model = Sequential([
    LSTM(args.lstm_units_1, return_sequences=True,
         input_shape=(sequence_length, n_features)),
    BatchNormalization(),
    Dropout(args.dropout_rate),

    LSTM(args.lstm_units_2, return_sequences=False),
    BatchNormalization(),
    Dropout(args.dropout_rate),

    Dense(16, activation='relu'),
    Dropout(args.dropout_rate / 2),

    Dense(1, activation='sigmoid'),
])

model.compile(
    optimizer=Adam(learning_rate=args.learning_rate),
    loss='binary_crossentropy',
    metrics=['accuracy', keras.metrics.AUC(name='auc')],
)
model.summary()

# ── Callbacks ─────────────────────────────────────────────────────────────────
callbacks = [
    EarlyStopping(monitor='val_auc', patience=10, mode='max',
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5,
                      min_lr=1e-6, verbose=1),
]

# ── Train ─────────────────────────────────────────────────────────────────────
print("\nTraining...")
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=args.epochs,
    batch_size=args.batch_size,
    callbacks=callbacks,
    verbose=2,  # one line per epoch — concise CloudWatch logs
)

# ── Evaluate ──────────────────────────────────────────────────────────────────
print("\nEvaluating...")
y_prob = model.predict(X_test, verbose=0).flatten()
y_pred = (y_prob >= 0.5).astype(int)

metrics = {
    'accuracy':  float(accuracy_score(y_test, y_pred)),
    'precision': float(precision_score(y_test, y_pred, zero_division=0)),
    'recall':    float(recall_score(y_test, y_pred, zero_division=0)),
    'f1':        float(f1_score(y_test, y_pred, zero_division=0)),
    'roc_auc':   float(roc_auc_score(y_test, y_prob)),
    'epochs_trained': len(history.history['loss']),
    'hyperparameters': vars(args),
    'sequence_config': sequence_config,
}

# Print all metrics — SageMaker captures stdout as training logs
print("\n=== Evaluation Metrics ===")
for k, v in metrics.items():
    if isinstance(v, float):
        print(f"  {k}: {v:.4f}")
    else:
        print(f"  {k}: {v}")

# ── Save model ────────────────────────────────────────────────────────────────
# SageMaker TensorFlow serving expects the model at SM_MODEL_DIR/1/
# in SavedModel format. The '1' is the version number used by TF Serving.
saved_model_path = os.path.join(SM_MODEL_DIR, '1')
print(f"\nSaving model to {saved_model_path}")
model.save(saved_model_path)
print("Model saved.")

# ── Save metrics ──────────────────────────────────────────────────────────────
Path(SM_OUTPUT_DATA_DIR).mkdir(parents=True, exist_ok=True)
metrics_path = os.path.join(SM_OUTPUT_DATA_DIR, 'metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)
print(f"Metrics saved to {metrics_path}")

print("\n=== Training complete ===")
