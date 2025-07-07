#!/usr/bin/env python3
"""
Train LightGBM model for credit card fraud detection.
"""
import os
os.environ['MPLBACKEND'] = 'Agg'  # Set before importing matplotlib

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import joblib
import json
from datetime import datetime

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with basic hyperparameters"""
    
    # Calculate scale_pos_weight for extreme imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos
    
    print(f"Class balance - Non-fraud: {n_neg}, Fraud: {n_pos}")
    print(f"Scale pos weight: {scale_pos_weight:.2f}")
    
    # LightGBM parameters for imbalanced classification
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 15,  # Reduced to prevent overfitting
        'learning_rate': 0.01,  # Lower learning rate
        'feature_fraction': 0.7,
        'bagging_fraction': 0.7,
        'bagging_freq': 5,
        'max_depth': 5,  # Limit depth to prevent overfitting
        'min_child_samples': 50,  # Increased to prevent overfitting
        'scale_pos_weight': scale_pos_weight,  # Weight positive class
        'random_state': 42,
        'verbose': -1,
        'force_row_wise': True,
        'min_split_gain': 0.01,  # Prevent overfitting
        'reg_alpha': 0.1,  # L1 regularization
        'reg_lambda': 0.1  # L2 regularization
    }
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train model
    print("Training LightGBM model...")
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
        # Among valid indices, choose the one with best F1
        best_valid_idx = valid_indices[np.argmax(f1_scores[valid_indices])]
        optimal_threshold = thresholds[best_valid_idx] if best_valid_idx < len(thresholds) else 0.5
    else:
        # If no threshold gives 50% recall, use the one with best F1
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    # Binary predictions with optimal threshold
    y_pred = (predictions > optimal_threshold).astype(int)
    
    print(f"\n{dataset_name} Performance:")
    print(f"AUC-ROC Score: {auc_score:.4f}")
    print(f"Optimal Threshold: {optimal_threshold:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, y_pred))
    
    return auc_score, optimal_threshold, predictions

def main():
    print("Loading processed transaction data...")
    
    # Load processed data
    data_path = 'data/processed_transactions.csv'
    if not os.path.exists(data_path):
        print("Error: Processed data not found. Please run feature_engineering.py first.")
        return
    
    df = pd.read_csv(data_path, parse_dates=['datetime'])
    
    # Load feature columns
    with open('data/feature_columns.txt', 'r') as f:
        feature_cols = [line.strip() for line in f]
    
    print(f"Loaded {len(df)} transactions with {len(feature_cols)} features")
    
    # Split data by time - use data before last year for training
    last_year_start = df['datetime'].max() - pd.DateOffset(years=1)
    
    train_df = df[df['datetime'] < last_year_start]
    test_df = df[df['datetime'] >= last_year_start]
    
    print(f"\nTrain set: {len(train_df)} transactions (before {last_year_start.date()})")
    print(f"Test set (last year): {len(test_df)} transactions")
    
    # Prepare features and target
    X_train_full = train_df[feature_cols]
    y_train_full = train_df['is_fraud']
    
    X_test = test_df[feature_cols]
    y_test = test_df['is_fraud']
    
    # Split training data for validation - ensure we have enough fraud cases
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    # Manual undersampling to avoid dependency issues
    # Get indices of fraud and non-fraud samples
    fraud_indices = np.where(y_train == 1)[0]
    non_fraud_indices = np.where(y_train == 0)[0]
    
    # Undersample non-fraud to get 10:1 ratio
    n_fraud = len(fraud_indices)
    n_non_fraud_sample = min(n_fraud * 10, len(non_fraud_indices))
    
    # Random sample from non-fraud
    np.random.seed(42)
    sampled_non_fraud_indices = np.random.choice(non_fraud_indices, n_non_fraud_sample, replace=False)
    
    # Combine indices
    balanced_indices = np.concatenate([fraud_indices, sampled_non_fraud_indices])
    np.random.shuffle(balanced_indices)
    
    # Create balanced dataset
    X_train_balanced = X_train.iloc[balanced_indices]
    y_train_balanced = y_train.iloc[balanced_indices]
    
    print(f"\nOriginal training samples: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"Balanced training samples: {len(X_train_balanced)} ({y_train_balanced.sum()} fraud)")
    print(f"Validation samples: {len(X_val)} ({y_val.sum()} fraud)")
    
    # Train model with balanced data
    model = train_lightgbm_model(X_train_balanced, y_train_balanced, X_val, y_val)
    
    # Evaluate on different sets (use original unbalanced data for evaluation)
    train_auc, _, _ = evaluate_model(model, X_train, y_train, "Training Set (Original)")
    val_auc, optimal_threshold, _ = evaluate_model(model, X_val, y_val, "Validation Set")
    test_auc, _, test_predictions = evaluate_model(model, X_test, y_test, "Test Set (Last Year)")
    
    # Save model and metadata
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_path = 'models/fraud_detection_lgb.pkl'
    joblib.dump(model, model_path)
    print(f"\nModel saved to {model_path}")
    
    # Save metadata
    metadata = {
        'train_date': datetime.now().isoformat(),
        'features': feature_cols,
        'optimal_threshold': optimal_threshold,
        'performance': {
            'train_auc': train_auc,
            'val_auc': val_auc,
            'test_auc': test_auc
        },
        'data_split': {
            'train_end_date': last_year_start.isoformat(),
            'train_samples': len(X_train_full),
            'test_samples': len(X_test)
        }
    }
    
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("Model metadata saved to models/model_metadata.json")
    
    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importance(importance_type='gain')
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))
    
    importance_df.to_csv('models/feature_importance.csv', index=False)

if __name__ == "__main__":
    main()