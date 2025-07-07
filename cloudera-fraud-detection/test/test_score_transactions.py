#!/usr/bin/env python3
"""
Test script for score_transactions.py
Tests the transaction scoring pipeline with trained model
"""
import sys
import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from score_transactions import score_transactions, generate_alerts

def test_transaction_scoring():
    """Test the transaction scoring pipeline"""
    print("Testing transaction scoring pipeline...")
    
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    
    # Check for required files
    test_data_file = os.path.join(data_dir, 'test_processed_transactions.csv')
    real_data_file = os.path.join(data_dir, 'processed_transactions.csv')
    test_model_file = os.path.join(model_dir, 'test_fraud_model.pkl')
    real_model_file = os.path.join(model_dir, 'fraud_model.pkl')
    
    # Load data
    if os.path.exists(test_data_file):
        df = pd.read_csv(test_data_file)
        print(f"Loaded test data from: {test_data_file}")
    elif os.path.exists(real_data_file):
        df = pd.read_csv(real_data_file)
        print(f"Loaded real data from: {real_data_file}")
    else:
        print("No processed data found. Please run test_feature_engineering.py first.")
        return False
    
    # Load model
    if os.path.exists(test_model_file):
        model = joblib.load(test_model_file)
        metadata = joblib.load(os.path.join(model_dir, 'test_model_metadata.pkl'))
        print(f"Loaded test model")
    elif os.path.exists(real_model_file):
        model = joblib.load(real_model_file)
        metadata = joblib.load(os.path.join(model_dir, 'model_metadata.pkl'))
        print(f"Loaded real model")
    else:
        print("No model found. Please run test_train_model.py first.")
        return False
    
    print(f"\nData shape: {df.shape}")
    print(f"Model features: {metadata['n_features']}")
    
    try:
        # Score transactions
        print("\nScoring transactions...")
        df_scored = score_transactions(df, model, metadata['feature_names'], threshold=0.5)
        
        print(f"Scored {len(df_scored)} transactions")
        print(f"New columns added: fraud_probability, fraud_prediction, risk_level")
        
        # Show score distribution
        print("\nFraud Probability Distribution:")
        print(df_scored['fraud_probability'].describe())
        
        print("\nRisk Level Distribution:")
        print(df_scored['risk_level'].value_counts().sort_index())
        
        # Generate alerts
        print("\nGenerating fraud alerts...")
        alerts = generate_alerts(df_scored)
        
        print(f"Generated {len(alerts)} alerts")
        
        if len(alerts) > 0:
            print("\nSample Alerts (top 5):")
            for idx, alert in alerts.head(5).iterrows():
                print(f"\n  Alert {idx + 1}:")
                print(f"    User: {alert['User']}, Card: {alert['Card']}")
                print(f"    Amount: {alert['Amount']}")
                print(f"    Merchant: {alert['Merchant Name']}")
                print(f"    Risk: {alert['risk_level']} (score: {alert['fraud_probability']:.3f})")
                print(f"    Actual fraud: {alert['Is Fraud?']}")
        
        # Save test outputs
        output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
        os.makedirs(output_dir, exist_ok=True)
        
        # Save scored transactions
        scored_file = os.path.join(output_dir, 'test_scored_transactions.csv')
        df_scored.to_csv(scored_file, index=False)
        print(f"\nSaved scored transactions to: {scored_file}")
        
        # Save alerts
        if len(alerts) > 0:
            alerts_file = os.path.join(output_dir, 'test_fraud_alerts.csv')
            alerts.to_csv(alerts_file, index=False)
            print(f"Saved alerts to: {alerts_file}")
        
        # Generate summary report
        print("\n" + "="*60)
        print("Scoring Summary Report")
        print("="*60)
        
        # Compare predictions with actual fraud
        if 'Is Fraud?' in df_scored.columns:
            from sklearn.metrics import classification_report, confusion_matrix
            
            y_true = (df_scored['Is Fraud?'] == 'Yes').astype(int)
            y_pred = df_scored['fraud_prediction']
            
            print("\nClassification Report:")
            print(classification_report(y_true, y_pred, target_names=['Normal', 'Fraud']))
            
            print("\nConfusion Matrix:")
            cm = confusion_matrix(y_true, y_pred)
            print("              Predicted")
            print("              Normal  Fraud")
            print(f"Actual Normal  {cm[0,0]:5d}  {cm[0,1]:5d}")
            print(f"       Fraud   {cm[1,0]:5d}  {cm[1,1]:5d}")
            
            # Alert effectiveness
            if len(alerts) > 0:
                alert_precision = (alerts['Is Fraud?'] == 'Yes').mean()
                total_frauds = (df_scored['Is Fraud?'] == 'Yes').sum()
                frauds_caught = (alerts['Is Fraud?'] == 'Yes').sum()
                alert_recall = frauds_caught / total_frauds if total_frauds > 0 else 0
                
                print(f"\nAlert Effectiveness:")
                print(f"  Alert Precision: {alert_precision:.1%} of alerts are actual fraud")
                print(f"  Alert Recall: {alert_recall:.1%} of frauds generate alerts")
        
        return True
        
    except Exception as e:
        print(f"\nError during transaction scoring: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_real_time_scoring():
    """Test scoring individual transactions in real-time"""
    print("\n" + "="*60)
    print("Testing Real-Time Scoring")
    print("="*60)
    
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'test_fraud_model.pkl')
    
    if not os.path.exists(model_path):
        print("No test model found. Please run test_train_model.py first.")
        return False
    
    try:
        # Load model
        model = joblib.load(model_path)
        metadata = joblib.load(os.path.join(model_dir, 'test_model_metadata.pkl'))
        
        # Simulate real-time transaction
        print("Simulating real-time transaction scoring...")
        
        # Create a single transaction with suspicious patterns
        suspicious_transaction = pd.DataFrame({
            'hour': [3],  # Early morning
            'day_of_week': [6],  # Weekend
            'is_weekend': [1],
            'Amount_clean': [2500],  # Large amount
            'is_online': [1],  # Online transaction
            'time_since_last': [0.1],  # Very recent after last transaction
            'amount_mean_5': [50],  # Usually small transactions
            'amount_std_5': [10],
            'transaction_count_5': [5],
            'amount_diff_from_mean': [2450]  # Much larger than usual
        })
        
        # Add remaining features as zeros (simplified)
        for feature in metadata['feature_names']:
            if feature not in suspicious_transaction.columns:
                suspicious_transaction[feature] = 0
        
        # Score transaction
        X = suspicious_transaction[metadata['feature_names']]
        score = model.predict_proba(X)[0, 1]
        prediction = model.predict(X)[0]
        
        print(f"\nSuspicious Transaction Score: {score:.3f}")
        print(f"Prediction: {'FRAUD' if prediction == 1 else 'Normal'}")
        print(f"Risk Level: {'Very High' if score > 0.8 else 'High' if score > 0.6 else 'Medium' if score > 0.4 else 'Low'}")
        
        return True
        
    except Exception as e:
        print(f"\nError during real-time scoring test: {str(e)}")
        return False

if __name__ == "__main__":
    print("="*60)
    print("Transaction Scoring Test")
    print("="*60)
    
    # Test batch scoring
    success = test_transaction_scoring()
    
    if success:
        print("\n✓ Batch scoring test passed!")
        
        # Test real-time scoring
        rt_success = test_real_time_scoring()
        if rt_success:
            print("\n✓ Real-time scoring test passed!")
        else:
            print("\n✗ Real-time scoring test failed!")
    else:
        print("\n✗ Batch scoring test failed!")
        sys.exit(1)