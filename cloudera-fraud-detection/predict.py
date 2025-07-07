#!/usr/bin/env python3
"""
CML Model prediction function for fraud detection
This file is used by Cloudera ML's native model deployment system
"""
import os
import json
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# Global variables to cache model and metadata
model = None
metadata = None
feature_names = None

def load_model():
    """Load the trained model and metadata"""
    global model, metadata, feature_names
    
    if model is None:
        try:
            # Try to load production model first
            model_path = "fraud-cloudera/cloudera-fraud-detection/models/fraud_detection_lgb.pkl"
            metadata_path = "fraud-cloudera/cloudera-fraud-detection/models/model_metadata.json"
            
            if not os.path.exists(model_path):
                # Fallback to test model
                model_path = "fraud-cloudera/cloudera-fraud-detection/models/test_fraud_model.pkl"
                metadata_path = "fraud-cloudera/cloudera-fraud-detection/models/test_model_metadata.pkl"
            
            if not os.path.exists(model_path):
                raise FileNotFoundError("No trained model found. Please train a model first.")
            
            print(f"Loading model from: {model_path}")
            model = joblib.load(model_path)
            
            # Load metadata
            if metadata_path.endswith('.json'):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                feature_names = metadata.get('features', [])
            else:
                metadata = joblib.load(metadata_path)
                feature_names = metadata.get('feature_names', [])
            
            print(f"Model loaded successfully with {len(feature_names)} features")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def validate_and_prepare_input(input_data):
    """Validate and prepare input data for prediction"""
    
    # Expected feature names for the fraud detection model
    expected_features = [
        'Amount_clean', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'is_online', 'is_chip', 'is_swipe', 'mcc_encoded', 'state_encoded',
        'is_online_state', 'time_since_last', 'amount_mean_5', 'amount_std_5',
        'amount_mean_10', 'amount_std_10', 'amount_mean_30', 'amount_std_30',
        'amount_deviation', 'amount_zscore', 'trans_count_day'
    ]
    
    # Use model's feature names if available, otherwise use expected features
    features_to_use = feature_names if feature_names else expected_features
    
    # Prepare features dictionary
    features = {}
    
    # Map input data to model features with defaults
    for feature in features_to_use:
        if feature in input_data:
            features[feature] = input_data[feature]
        else:
            # Set sensible defaults for missing features
            if 'amount' in feature.lower():
                features[feature] = 0.0
            elif 'count' in feature.lower():
                features[feature] = 1
            elif 'is_' in feature:
                features[feature] = 0
            elif feature in ['hour', 'day_of_week', 'day_of_month']:
                features[feature] = 0
            else:
                features[feature] = 0.0
    
    # Convert to DataFrame
    df = pd.DataFrame([features])
    
    # Ensure features are in the correct order
    df = df.reindex(columns=features_to_use, fill_value=0.0)
    
    return df

def calculate_risk_level(probability):
    """Calculate risk level based on fraud probability"""
    if probability >= 0.9:
        return "Very High"
    elif probability >= 0.6:
        return "High"
    elif probability >= 0.3:
        return "Medium"
    else:
        return "Low"

def generate_explanation(input_data, probability):
    """Generate explanation for predictions"""
    explanations = []
    
    # Check amount patterns
    amount = input_data.get('Amount_clean', 0)
    if amount > 1000:
        explanations.append(f"Large transaction amount: ${amount:.2f}")
    
    # Check time patterns
    hour = input_data.get('hour', 12)
    if hour < 6 or hour > 22:
        explanations.append(f"Unusual transaction time: {hour:02d}:00")
    
    # Check weekend pattern
    if input_data.get('is_weekend', 0) == 1:
        explanations.append("Weekend transaction")
    
    # Check online transaction
    if input_data.get('is_online', 0) == 1:
        explanations.append("Online transaction")
    
    # Check amount deviation
    zscore = input_data.get('amount_zscore', 0)
    if abs(zscore) > 2:
        explanations.append(f"Amount deviates from user pattern (z-score: {zscore:.1f})")
    
    # Check transaction frequency
    daily_count = input_data.get('trans_count_day', 1)
    if daily_count > 10:
        explanations.append(f"High daily transaction frequency: {daily_count}")
    
    return explanations if explanations else ["Multiple risk factors detected"]

def predict(request):
    """
    Main prediction function called by CML
    
    Args:
        request: Dictionary containing transaction features
        
    Returns:
        Dictionary containing fraud prediction results
    """
    
    try:
        # Load model if not already loaded
        load_model()
        
        # Validate and prepare input
        input_df = validate_and_prepare_input(request)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # For models with predict_proba (like LightGBM)
            probabilities = model.predict_proba(input_df)
            if probabilities.shape[1] > 1:
                fraud_probability = float(probabilities[0, 1])
            else:
                fraud_probability = float(probabilities[0, 0])
        else:
            # For models without predict_proba
            fraud_probability = float(model.predict(input_df)[0])
        
        # Use optimal threshold if available
        threshold = metadata.get('optimal_threshold', 0.5) if metadata else 0.5
        fraud_prediction = int(fraud_probability > threshold)
        
        # Calculate risk level
        risk_level = calculate_risk_level(fraud_probability)
        
        # Prepare response
        response = {
            "fraud_probability": fraud_probability,
            "fraud_prediction": fraud_prediction,
            "fraud_label": "FRAUD" if fraud_prediction == 1 else "NORMAL",
            "risk_level": risk_level,
            "threshold_used": threshold,
            "prediction_timestamp": datetime.now().isoformat(),
            "confidence": "high" if fraud_probability > 0.8 or fraud_probability < 0.2 else "medium"
        }
        
        # Add model metadata if available
        if metadata:
            response["model_version"] = metadata.get('train_date', 'unknown')
            response["model_type"] = metadata.get('model_type', 'unknown')
        
        # Add explanation for high-risk transactions
        if fraud_probability > 0.6:
            response["explanation"] = generate_explanation(request, fraud_probability)
        
        # Add feature importance for debugging (top 5)
        if hasattr(model, 'feature_importances_') and feature_names:
            importance_data = list(zip(feature_names, model.feature_importances_))
            importance_data.sort(key=lambda x: x[1], reverse=True)
            response["top_features"] = [
                {"feature": name, "importance": float(imp)} 
                for name, imp in importance_data[:5]
            ]
        
        return response
        
    except Exception as e:
        # Return error response that CML can handle
        error_response = {
            "error": str(e),
            "fraud_probability": None,
            "fraud_prediction": None,
            "fraud_label": "ERROR",
            "risk_level": "UNKNOWN",
            "prediction_timestamp": datetime.now().isoformat(),
            "success": False
        }
        print(f"Prediction error: {str(e)}")
        return error_response

# For local testing
if __name__ == "__main__":
    print("Testing CML fraud detection prediction function...")
    
    # Test cases
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "Amount_clean": 45.50,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_online": 0,
                "is_chip": 1,
                "amount_mean_30": 50.0,
                "amount_zscore": -0.3,
                "trans_count_day": 3
            }
        },
        {
            "name": "Suspicious Transaction", 
            "data": {
                "Amount_clean": 2500.0,
                "hour": 3,
                "day_of_week": 6,
                "is_weekend": 1,
                "is_online": 1,
                "amount_mean_30": 48.0,
                "amount_zscore": 163.5,
                "trans_count_day": 15
            }
        },
        {
            "name": "Minimal Data Transaction",
            "data": {
                "Amount_clean": 100.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"\n{'='*50}")
        print(f"Testing: {test_case['name']}")
        print(f"{'='*50}")
        
        print("Input:")
        print(json.dumps(test_case['data'], indent=2))
        
        result = predict(test_case['data'])
        
        print("\nOutput:")
        print(json.dumps(result, indent=2))
        
        if result.get('fraud_label') != 'ERROR':
            print(f"\nSummary: {result['fraud_label']} "
                  f"(Risk: {result['risk_level']}, "
                  f"Probability: {result['fraud_probability']:.3f})")
    
    print(f"\n{'='*50}")
    print("Local testing completed!")
    print(f"{'='*50}")