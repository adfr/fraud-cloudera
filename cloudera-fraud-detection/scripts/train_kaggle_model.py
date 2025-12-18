#!/usr/bin/env python3
"""
Train LightGBM model on Kaggle Credit Card Fraud dataset.
Optimized for the PCA-transformed features in the Kaggle dataset.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, precision_recall_curve,
    classification_report, confusion_matrix, f1_score
)
from sklearn.preprocessing import StandardScaler
import joblib
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


def load_data(data_dir: str = "data"):
    """Load the Kaggle Credit Card Fraud dataset."""

    # Try pre-split data first
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Loading pre-split data...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        return train_df, test_df

    # Otherwise load full dataset
    data_path = os.path.join(data_dir, "creditcard_fraud.csv")
    if not os.path.exists(data_path):
        data_path = os.path.join(data_dir, "creditcard.csv")

    if not os.path.exists(data_path):
        print("ERROR: Dataset not found!")
        print("Please run: python scripts/download_kaggle_data.py")
        raise FileNotFoundError("Dataset not found")

    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)

    # Rename Class to is_fraud if needed
    if 'Class' in df.columns:
        df = df.rename(columns={'Class': 'is_fraud'})

    # Temporal split (last 20% as test)
    df = df.sort_values('Time').reset_index(drop=True)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    return train_df, test_df


def prepare_features(train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Prepare features for training."""

    # Define feature columns
    # PCA features (V1-V28) are already normalized
    pca_features = [f'V{i}' for i in range(1, 29)]

    # Check what features are available
    available_features = list(train_df.columns)

    # Build feature list based on what's available
    feature_cols = pca_features.copy()

    # Add Amount features if available
    if 'Amount_log' in available_features:
        feature_cols.append('Amount_log')
    elif 'Amount' in available_features:
        # Create log amount
        train_df['Amount_log'] = np.log1p(train_df['Amount'])
        test_df['Amount_log'] = np.log1p(test_df['Amount'])
        feature_cols.append('Amount_log')

    if 'Amount_scaled' in available_features:
        feature_cols.append('Amount_scaled')

    # Add time features if available
    if 'hour_sin' in available_features:
        feature_cols.extend(['hour_sin', 'hour_cos'])
    elif 'Time' in available_features:
        # Create hour features
        train_df['hour'] = (train_df['Time'] % 86400) / 3600
        test_df['hour'] = (test_df['Time'] % 86400) / 3600
        train_df['hour_sin'] = np.sin(2 * np.pi * train_df['hour'] / 24)
        train_df['hour_cos'] = np.cos(2 * np.pi * train_df['hour'] / 24)
        test_df['hour_sin'] = np.sin(2 * np.pi * test_df['hour'] / 24)
        test_df['hour_cos'] = np.cos(2 * np.pi * test_df['hour'] / 24)
        feature_cols.extend(['hour_sin', 'hour_cos'])

    # Add interaction features if available
    for feat in ['V1_Amount', 'V2_Amount', 'V4_Amount', 'is_high_amount']:
        if feat in available_features:
            feature_cols.append(feat)

    # Ensure all features exist
    feature_cols = [f for f in feature_cols if f in train_df.columns]

    print(f"\nUsing {len(feature_cols)} features")

    X_train = train_df[feature_cols]
    y_train = train_df['is_fraud']
    X_test = test_df[feature_cols]
    y_test = test_df['is_fraud']

    return X_train, y_train, X_test, y_test, feature_cols


def train_lightgbm(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM model with optimized parameters."""

    # Calculate class weight
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    print(f"\nClass distribution:")
    print(f"  Non-fraud: {n_neg:,}")
    print(f"  Fraud: {n_pos:,}")
    print(f"  Scale pos weight: {scale_pos_weight:.2f}")

    # LightGBM parameters optimized for fraud detection
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss', 'average_precision'],
        'boosting_type': 'gbdt',
        'num_leaves': 63,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 8,
        'min_child_samples': 100,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1,
        'n_jobs': -1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'min_split_gain': 0.01
    }

    # Create datasets
    train_data = lgb.Dataset(
        X_train, label=y_train,
        feature_name=feature_names
    )
    val_data = lgb.Dataset(
        X_val, label=y_val,
        reference=train_data,
        feature_name=feature_names
    )

    print("\nTraining LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=2000,
        callbacks=[
            lgb.early_stopping(100),
            lgb.log_evaluation(100)
        ]
    )

    return model


def evaluate_model(model, X, y, dataset_name, threshold=None):
    """Comprehensive model evaluation."""

    predictions = model.predict(X)
    auc_roc = roc_auc_score(y, predictions)
    auc_pr = average_precision_score(y, predictions)

    # Find optimal threshold
    precision, recall, thresholds = precision_recall_curve(y, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find threshold with best F1 (minimum 50% recall)
    valid_idx = np.where(recall >= 0.5)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(f1_scores[valid_idx])]
        optimal_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    else:
        optimal_threshold = thresholds[np.argmax(f1_scores)]

    if threshold is None:
        threshold = optimal_threshold

    y_pred = (predictions > threshold).astype(int)

    # Metrics
    cm = confusion_matrix(y, y_pred)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*50}")
    print(f"{dataset_name} Results (threshold={threshold:.4f})")
    print(f"{'='*50}")
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR:  {auc_pr:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  TN: {tn:,}  FP: {fp:,}")
    print(f"  FN: {fn:,}  TP: {tp:,}")
    print(f"\nMetrics:")
    print(f"  Precision: {tp/(tp+fp):.4f}" if (tp+fp) > 0 else "  Precision: N/A")
    print(f"  Recall:    {tp/(tp+fn):.4f}" if (tp+fn) > 0 else "  Recall: N/A")
    print(f"  F1 Score:  {f1_score(y, y_pred):.4f}")

    return {
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'optimal_threshold': optimal_threshold,
        'precision': tp/(tp+fp) if (tp+fp) > 0 else 0,
        'recall': tp/(tp+fn) if (tp+fn) > 0 else 0,
        'f1': f1_score(y, y_pred),
        'confusion_matrix': cm.tolist()
    }


def save_model(model, feature_names, metrics, output_dir="models"):
    """Save trained model and metadata."""

    os.makedirs(output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(output_dir, "fraud_detection_lgb.pkl")
    joblib.dump(model, model_path)
    print(f"\nModel saved to: {model_path}")

    # Save metadata
    metadata = {
        'model_type': 'LightGBM',
        'dataset': 'Kaggle Credit Card Fraud',
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'optimal_threshold': metrics['test']['optimal_threshold'],
        'performance': {
            'train': metrics['train'],
            'val': metrics['val'],
            'test': metrics['test']
        },
        'model_params': {
            'num_leaves': 63,
            'learning_rate': 0.01,
            'max_depth': 8,
            'num_boost_round': model.best_iteration
        }
    }

    metadata_path = os.path.join(output_dir, "model_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")

    # Save feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    importance_path = os.path.join(output_dir, "feature_importance.csv")
    importance_df.to_csv(importance_path, index=False)
    print(f"Feature importance saved to: {importance_path}")

    print("\nTop 10 Important Features:")
    print(importance_df.head(10).to_string(index=False))

    return model_path, metadata


def main():
    print("="*60)
    print("  Fraud Detection Model Training (Kaggle Dataset)")
    print("="*60)

    # Load data
    print("\n1. Loading Data...")
    train_df, test_df = load_data()
    print(f"   Train: {len(train_df):,} rows ({train_df['is_fraud'].sum():,} fraud)")
    print(f"   Test:  {len(test_df):,} rows ({test_df['is_fraud'].sum():,} fraud)")

    # Prepare features
    print("\n2. Preparing Features...")
    X_train_full, y_train_full, X_test, y_test, feature_names = prepare_features(train_df, test_df)

    # Split train into train/val
    print("\n3. Creating Validation Split...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=0.15,
        random_state=42,
        stratify=y_train_full
    )
    print(f"   Train: {len(X_train):,} ({y_train.sum():,} fraud)")
    print(f"   Val:   {len(X_val):,} ({y_val.sum():,} fraud)")
    print(f"   Test:  {len(X_test):,} ({y_test.sum():,} fraud)")

    # Train model
    print("\n4. Training Model...")
    model = train_lightgbm(X_train, y_train, X_val, y_val, feature_names)

    # Evaluate
    print("\n5. Evaluating Model...")
    metrics = {
        'train': evaluate_model(model, X_train, y_train, "Training Set"),
        'val': evaluate_model(model, X_val, y_val, "Validation Set"),
        'test': evaluate_model(model, X_test, y_test, "Test Set")
    }

    # Save
    print("\n6. Saving Model...")
    model_path, metadata = save_model(model, feature_names, metrics)

    # Summary
    print("\n" + "="*60)
    print("  Training Complete!")
    print("="*60)
    print(f"\nPerformance Summary:")
    print(f"  Train AUC-ROC: {metrics['train']['auc_roc']:.4f}")
    print(f"  Val AUC-ROC:   {metrics['val']['auc_roc']:.4f}")
    print(f"  Test AUC-ROC:  {metrics['test']['auc_roc']:.4f}")
    print(f"  Test AUC-PR:   {metrics['test']['auc_pr']:.4f}")
    print(f"  Test F1:       {metrics['test']['f1']:.4f}")
    print(f"\nOptimal Threshold: {metrics['test']['optimal_threshold']:.4f}")
    print(f"\nModel saved to: {model_path}")


if __name__ == "__main__":
    main()
