#!/usr/bin/env python3
"""
Test script for training a simple model without complex dependencies
Uses basic algorithms to test the pipeline
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

def train_simple_model(X_train, y_train, X_test, y_test):
    """Train a simple logistic regression model"""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
    
    print("Training Logistic Regression model...")
    
    # Train model with class balancing
    model = LogisticRegression(
        class_weight='balanced',
        random_state=42,
        max_iter=1000
    )
    
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0)
    }
    
    return model, metrics

def test_simple_model_training():
    """Test simple model training pipeline"""
    print("Testing simple model training...")
    
    # Load engineered features
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    test_file = os.path.join(data_dir, 'test_processed_transactions.csv')
    
    if not os.path.exists(test_file):
        print("No test data found. Please run test_feature_engineering.py first.")
        return False
    
    df = pd.read_csv(test_file)
    print(f"Loaded data: {df.shape}")
    
    try:
        # Check if sklearn is available
        import sklearn
        print(f"Using scikit-learn version: {sklearn.__version__}")
    except ImportError:
        print("Scikit-learn not available. Installing...")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "scikit-learn==1.3.0"], 
                      capture_output=True)
        import sklearn
    
    # Feature columns
    feature_cols = [
        'Amount_clean', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'is_online', 'is_chip', 'is_swipe', 'mcc_encoded', 'state_encoded',
        'is_online_state', 'time_since_last', 'amount_mean_5', 'amount_std_5',
        'amount_mean_10', 'amount_std_10', 'amount_mean_30', 'amount_std_30',
        'amount_deviation', 'amount_zscore', 'trans_count_day'
    ]
    
    # Filter available features
    available_features = [col for col in feature_cols if col in df.columns]
    print(f"Using {len(available_features)} features")
    
    # Prepare data
    X = df[available_features]
    y = df['is_fraud']
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training set: {X_train.shape}, fraud rate: {y_train.mean():.2%}")
    print(f"Test set: {X_test.shape}, fraud rate: {y_test.mean():.2%}")
    
    # Train model
    model, metrics = train_simple_model(X_train, y_train, X_test, y_test)
    
    print("\nModel Performance:")
    print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-Score: {metrics['f1']:.4f}")
    
    # Save model
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, 'simple_test_model.pkl')
    joblib.dump(model, model_path)
    print(f"\nSaved model to: {model_path}")
    
    # Save metadata
    metadata = {
        'trained_at': datetime.now().isoformat(),
        'model_type': 'LogisticRegression',
        'n_features': len(available_features),
        'feature_names': available_features,
        'metrics': metrics,
        'training_samples': len(X_train),
        'test_samples': len(X_test)
    }
    
    metadata_path = os.path.join(model_dir, 'simple_test_metadata.pkl')
    joblib.dump(metadata, metadata_path)
    print(f"Saved metadata to: {metadata_path}")
    
    return True

def test_simple_scoring():
    """Test scoring with the simple model"""
    print("\n" + "="*50)
    print("Testing Simple Model Scoring")
    print("="*50)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'simple_test_model.pkl')
    metadata_path = os.path.join(model_dir, 'simple_test_metadata.pkl')
    
    if not os.path.exists(model_path):
        print("No simple model found.")
        return False
    
    # Load model and metadata
    model = joblib.load(model_path)
    metadata = joblib.load(metadata_path)
    
    # Create test transaction
    print("Scoring sample transactions...")
    
    # Sample transaction data
    sample_data = pd.DataFrame({
        'Amount_clean': [250.0, 25.0, 2500.0],
        'hour': [2, 14, 23],
        'day_of_week': [6, 2, 0],
        'is_weekend': [1, 0, 0],
        'is_online': [1, 0, 1]
    })
    
    # Add missing features as zeros
    for feature in metadata['feature_names']:
        if feature not in sample_data.columns:
            sample_data[feature] = 0
    
    # Score transactions
    X_sample = sample_data[metadata['feature_names']]
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)[:, 1]
    
    print("\nSample Scoring Results:")
    for i in range(len(sample_data)):
        status = "FRAUD" if predictions[i] == 1 else "Normal"
        print(f"Transaction {i+1}: {status} (probability: {probabilities[i]:.3f})")
    
    return True

if __name__ == "__main__":
    print("="*60)
    print("Simple Model Test")
    print("="*60)
    
    success = test_simple_model_training()
    
    if success:
        print("\n✓ Simple model training test passed!")
        scoring_success = test_simple_scoring()
        if scoring_success:
            print("\n✓ Simple model scoring test passed!")
            print("\n🎉 All simple model tests completed successfully!")
        else:
            print("\n✗ Simple model scoring test failed!")
    else:
        print("\n✗ Simple model training test failed!")
        sys.exit(1)