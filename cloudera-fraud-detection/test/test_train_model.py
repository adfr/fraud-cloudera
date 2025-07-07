#!/usr/bin/env python3
"""
Test script for train_model.py
Tests the model training pipeline with engineered features
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from train_model import train_lightgbm_model, evaluate_model

def test_model_training():
    """Test the model training pipeline"""
    print("Loading engineered features...")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    
    # Try to load test data or real data
    test_file = os.path.join(data_dir, 'test_processed_transactions.csv')
    real_file = os.path.join(data_dir, 'processed_transactions.csv')
    
    if os.path.exists(test_file):
        df = pd.read_csv(test_file)
        print(f"Loaded test data from: {test_file}")
    elif os.path.exists(real_file):
        df = pd.read_csv(real_file)
        print(f"Loaded real data from: {real_file}")
    else:
        print("No processed data found. Please run test_feature_engineering.py first.")
        return False
    
    print(f"Data shape: {df.shape}")
    print(f"Fraud rate: {(df['Is Fraud?'] == 'Yes').mean():.2%}\n")
    
    try:
        # Prepare data for training
        print("Preparing data for training...")
        
        # Load feature columns (should be created by feature engineering)
        feature_cols = [
            'Amount_clean', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
            'is_online', 'is_chip', 'is_swipe', 'mcc_encoded', 'state_encoded',
            'is_online_state', 'time_since_last', 'amount_mean_5', 'amount_std_5',
            'amount_mean_10', 'amount_std_10', 'amount_mean_30', 'amount_std_30',
            'amount_deviation', 'amount_zscore', 'trans_count_day'
        ]
        
        # Filter features that exist in the dataset
        available_features = [col for col in feature_cols if col in df.columns]
        print(f"Available features: {len(available_features)} out of {len(feature_cols)}")
        
        # Prepare features and target
        X = df[available_features]
        y = df['is_fraud'] if 'is_fraud' in df.columns else (df['Is Fraud?'] == 'Yes').astype(int)
        
        # Split data
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Further split training for validation
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
        )
        
        print(f"Training set: {X_train_split.shape}")
        print(f"Validation set: {X_val.shape}")
        print(f"Test set: {X_test.shape}")
        print(f"Features used: {len(available_features)}")
        print(f"Training fraud rate: {y_train_split.mean():.2%}")
        print(f"Test fraud rate: {y_test.mean():.2%}\n")
        
        # Train model
        print("Training LightGBM model...")
        model = train_lightgbm_model(X_train_split, y_train_split, X_val, y_val)
        
        print("\nModel training completed!")
        
        # Evaluate model on test set
        print("Evaluating model on test set...")
        test_metrics = evaluate_model(model, X_test, y_test, "Test")
        
        # Show feature importance
        feature_importance = pd.DataFrame({
            'feature': available_features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        print("\nTop 10 Most Important Features:")
        for idx, row in feature_importance.iterrows():
            print(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Save test model
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
        os.makedirs(model_dir, exist_ok=True)
        
        test_model_path = os.path.join(model_dir, 'test_fraud_model.pkl')
        joblib.dump(model, test_model_path)
        print(f"\nSaved test model to: {test_model_path}")
        
        # Save test metadata
        metadata = {
            'trained_at': datetime.now().isoformat(),
            'n_features': len(available_features),
            'feature_names': available_features,
            'metrics': test_metrics,
            'training_samples': len(X_train_split),
            'test_samples': len(X_test),
            'model_type': 'LightGBM'
        }
        
        metadata_path = os.path.join(model_dir, 'test_model_metadata.pkl')
        joblib.dump(metadata, metadata_path)
        print(f"Saved test metadata to: {metadata_path}")
        
        return True
        
    except Exception as e:
        print(f"\nError during model training: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_predictions():
    """Test making predictions with the trained model"""
    print("\n" + "="*60)
    print("Testing Model Predictions")
    print("="*60)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'test_fraud_model.pkl')
    
    if not os.path.exists(model_path):
        print("No test model found. Training phase must complete first.")
        return False
    
    try:
        # Load model
        model = joblib.load(model_path)
        metadata = joblib.load(os.path.join(model_dir, 'test_model_metadata.pkl'))
        
        print(f"Loaded model with {metadata['n_features']} features")
        
        # Create sample data for prediction
        n_samples = 10
        X_sample = np.random.randn(n_samples, metadata['n_features'])
        
        # Make predictions
        predictions = model.predict(X_sample)
        probabilities = model.predict_proba(X_sample)[:, 1]
        
        print(f"\nSample predictions ({n_samples} transactions):")
        for i in range(n_samples):
            fraud_status = "FRAUD" if predictions[i] == 1 else "Normal"
            print(f"  Transaction {i+1}: {fraud_status} (probability: {probabilities[i]:.3f})")
        
        return True
        
    except Exception as e:
        print(f"\nError during prediction test: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Model Training Test")
    print("="*60)
    
    # Test training
    success = test_model_training()
    
    if success:
        print("\n✓ Model training test passed!")
        
        # Test predictions
        pred_success = test_model_predictions()
        if pred_success:
            print("\n✓ Model prediction test passed!")
        else:
            print("\n✗ Model prediction test failed!")
    else:
        print("\n✗ Model training test failed!")
        sys.exit(1)