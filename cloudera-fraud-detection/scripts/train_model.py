#!/usr/bin/env python3
"""
Train LightGBM model for credit card fraud detection.
"""
import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, classification_report
import joblib
import os
import json
from datetime import datetime

def train_lightgbm_model(X_train, y_train, X_val, y_val):
    """Train LightGBM model with basic hyperparameters"""
    
    # LightGBM parameters for imbalanced classification
    params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'max_depth': -1,
        'min_child_samples': 20,
        'is_unbalance': True,  # Handle class imbalance
        'random_state': 42,
        'verbose': -1
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
        num_boost_round=300,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )
    
    return model

def evaluate_model(model, X, y, dataset_name):
    """Evaluate model performance"""
    predictions = model.predict(X, num_iteration=model.best_iteration)
    auc_score = roc_auc_score(y, predictions)
    
    # Find optimal threshold using validation set
    precision, recall, thresholds = precision_recall_curve(y, predictions)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
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
    
    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full, 
        test_size=0.2, 
        random_state=42, 
        stratify=y_train_full
    )
    
    print(f"\nTraining samples: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"Validation samples: {len(X_val)} ({y_val.sum()} fraud)")
    
    # Train model
    model = train_lightgbm_model(X_train, y_train, X_val, y_val)
    
    # Evaluate on different sets
    train_auc, _, _ = evaluate_model(model, X_train, y_train, "Training Set")
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