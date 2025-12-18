#!/usr/bin/env python3
"""
Train LightGBM model for credit card fraud detection.
Uses the new feature pipeline with batch and real-time features.
"""

import os
os.environ['MPLBACKEND'] = 'Agg'  # Set before importing matplotlib

import sys
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, precision_recall_curve, classification_report,
    precision_score, recall_score, f1_score, confusion_matrix
)
import joblib
import json
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.features.feature_pipeline import FeaturePipeline
from scripts.generate_training_data import FraudDataGenerator


def train_lightgbm_model(X_train, y_train, X_val, y_val, feature_names):
    """Train LightGBM model with optimized hyperparameters"""

    # Calculate scale_pos_weight for extreme imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    print(f"Class balance - Non-fraud: {n_neg}, Fraud: {n_pos}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")

    # LightGBM parameters for imbalanced classification
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.01,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': 7,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True,
        'min_split_gain': 0.001,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'n_jobs': -1
    }

    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_names)

    # Train model
    print("\nTraining LightGBM model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        num_boost_round=1000,
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)]
    )

    return model


def evaluate_model(model, X, y, dataset_name):
    """Evaluate model performance"""
    predictions = model.predict(X, num_iteration=model.best_iteration)
    auc_score = roc_auc_score(y, predictions)

    # Find optimal threshold using validation set
    precision, recall, thresholds = precision_recall_curve(y, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)

    # Find best threshold with minimum recall of 0.5 for fraud detection
    valid_indices = np.where(recall >= 0.5)[0]
    if len(valid_indices) > 0:
        best_valid_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        optimal_threshold = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 0.5
    else:
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

    # Binary predictions with optimal threshold
    y_pred = (predictions > optimal_threshold).astype(int)

    print(f"\n{dataset_name} Performance:")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")

    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Normal', 'Fraud']))

    return auc_score, optimal_threshold, predictions


def main():
    print("=" * 60)
    print("Fraud Detection Model Training (v2)")
    print("=" * 60)

    # Configuration
    data_path = 'data/fraud_training_data.csv'
    use_existing_data = os.path.exists(data_path)

    # Step 1: Generate or load training data
    print("\n1. Preparing Training Data...")

    if not use_existing_data:
        print("Generating synthetic training data...")
        generator = FraudDataGenerator(seed=42)
        df = generator.generate_dataset(
            n_users=500,
            days=365,
            output_path=data_path
        )
    else:
        print(f"Loading existing data from {data_path}...")
        df = pd.read_csv(data_path)
        print(f"Loaded {len(df)} transactions")

    # Step 2: Initialize and fit feature pipeline
    print("\n2. Feature Engineering...")

    pipeline = FeaturePipeline(window_size=50)
    pipeline.fit(df)

    # Transform data
    features_df = pipeline.transform_batch(df, include_target=True)
    feature_names = pipeline.get_feature_names()

    print(f"\nFeature set: {len(feature_names)} features")

    # Step 3: Prepare train/val/test splits
    print("\n3. Preparing Data Splits...")

    X = features_df[feature_names]
    y = features_df['is_fraud']

    # First split: separate test set (20%)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Second split: train and validation (80% of remaining)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.2, random_state=42, stratify=y_temp
    )

    print(f"Train set: {len(X_train):,} samples ({y_train.sum():,} fraud)")
    print(f"Validation set: {len(X_val):,} samples ({y_val.sum():,} fraud)")
    print(f"Test set: {len(X_test):,} samples ({y_test.sum():,} fraud)")

    # Step 4: Handle class imbalance with undersampling
    print("\n4. Handling Class Imbalance...")

    fraud_indices = np.where(y_train == 1)[0]
    non_fraud_indices = np.where(y_train == 0)[0]

    # Undersample non-fraud to get 5:1 ratio
    n_fraud = len(fraud_indices)
    n_non_fraud_sample = min(n_fraud * 5, len(non_fraud_indices))

    np.random.seed(42)
    sampled_non_fraud_indices = np.random.choice(non_fraud_indices, n_non_fraud_sample, replace=False)
    balanced_indices = np.concatenate([fraud_indices, sampled_non_fraud_indices])
    np.random.shuffle(balanced_indices)

    X_train_balanced = X_train.iloc[balanced_indices]
    y_train_balanced = y_train.iloc[balanced_indices]

    print(f"Balanced training: {len(X_train_balanced):,} samples ({y_train_balanced.sum():,} fraud)")

    # Step 5: Train model
    print("\n5. Training Model...")

    model = train_lightgbm_model(
        X_train_balanced.values,
        y_train_balanced.values,
        X_val.values,
        y_val.values,
        feature_names
    )

    # Step 6: Evaluate on all sets
    print("\n6. Model Evaluation...")

    train_auc, _, _ = evaluate_model(model, X_train.values, y_train.values, "Training Set")
    val_auc, optimal_threshold, _ = evaluate_model(model, X_val.values, y_val.values, "Validation Set")
    test_auc, _, test_predictions = evaluate_model(model, X_test.values, y_test.values, "Test Set")

    # Step 7: Save model and artifacts
    print("\n7. Saving Model Artifacts...")

    os.makedirs('models', exist_ok=True)

    # Save model
    model_path = 'models/fraud_detection_lgb.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    # Save metadata
    metadata = {
        'model_type': 'LightGBM',
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'optimal_threshold': float(optimal_threshold),
        'performance': {
            'train_auc': float(train_auc),
            'val_auc': float(val_auc),
            'test_auc': float(test_auc)
        },
        'data_info': {
            'total_samples': len(df),
            'fraud_rate': float(y.mean()),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        },
        'hyperparameters': {
            'num_leaves': 31,
            'learning_rate': 0.01,
            'max_depth': 7,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8
        }
    }

    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Metadata saved to models/model_metadata.json")

    # Save feature pipeline
    pipeline.save('models/feature_pipeline')
    print("Feature pipeline saved to models/feature_pipeline/")

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)

    importance_df.to_csv('models/feature_importance.csv', index=False)

    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"\nModel Performance Summary:")
    print(f"  Train AUC: {train_auc:.4f}")
    print(f"  Validation AUC: {val_auc:.4f}")
    print(f"  Test AUC: {test_auc:.4f}")
    print(f"  Optimal Threshold: {optimal_threshold:.4f}")
    print(f"\nArtifacts saved:")
    print(f"  - models/fraud_detection_lgb.pkl")
    print(f"  - models/model_metadata.json")
    print(f"  - models/feature_pipeline/")
    print(f"  - models/feature_importance.csv")


if __name__ == "__main__":
    main()
