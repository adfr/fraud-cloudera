#!/usr/bin/env python3
"""
Model serving script for fraud detection model in Cloudera ML
Handles prediction requests via REST API
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
            print(f"Expected features: {feature_names}")
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise

def validate_input(request_data):
    """Validate and prepare input data for prediction"""
    
    if not isinstance(request_data, dict):
        raise ValueError("Request must be a JSON object")
    
    # Extract features from request
    features = {}
    
    # Map request fields to model features
    for feature in feature_names:
        if feature in request_data:
            features[feature] = request_data[feature]
        else:
            # Set default values for missing features
            features[feature] = 0.0
    
    # Convert to DataFrame for prediction
    df = pd.DataFrame([features])
    
    # Ensure all features are present and in correct order
    df = df.reindex(columns=feature_names, fill_value=0.0)
    
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
        input_data = validate_input(request)
        
        # Make prediction
        fraud_probability = model.predict_proba(input_data)[0, 1]
        fraud_prediction = int(fraud_probability > metadata.get('optimal_threshold', 0.5))
        risk_level = calculate_risk_level(fraud_probability)
        
        # Prepare response
        response = {
            "fraud_probability": float(fraud_probability),
            "fraud_prediction": fraud_prediction,
            "fraud_label": "FRAUD" if fraud_prediction == 1 else "NORMAL",
            "risk_level": risk_level,
            "threshold_used": metadata.get('optimal_threshold', 0.5),
            "model_version": metadata.get('train_date', 'unknown'),
            "prediction_timestamp": datetime.now().isoformat(),
            "confidence": "high" if fraud_probability > 0.8 or fraud_probability < 0.2 else "medium"
        }
        
        # Add explanation for high-risk transactions
        if fraud_probability > 0.6:
            response["explanation"] = generate_explanation(input_data, fraud_probability)
        
        return response
        
    except Exception as e:
        # Return error response
        error_response = {
            "error": str(e),
            "fraud_probability": None,
            "fraud_prediction": None,
            "fraud_label": "ERROR",
            "risk_level": "UNKNOWN",
            "prediction_timestamp": datetime.now().isoformat()
        }
        return error_response

def generate_explanation(input_data, probability):
    """Generate explanation for high-risk predictions"""
    
    explanations = []
    
    # Check amount patterns
    amount = input_data['Amount_clean'].iloc[0]
    if amount > 1000:
        explanations.append(f"Large transaction amount: ${amount:.2f}")
    
    # Check time patterns
    hour = input_data['hour'].iloc[0]
    if hour < 6 or hour > 22:
        explanations.append(f"Unusual transaction time: {hour:02d}:00")
    
    # Check weekend pattern
    if input_data['is_weekend'].iloc[0] == 1:
        explanations.append("Weekend transaction")
    
    # Check online transaction
    if input_data['is_online'].iloc[0] == 1:
        explanations.append("Online transaction")
    
    # Check amount deviation
    if 'amount_zscore' in input_data.columns:
        zscore = input_data['amount_zscore'].iloc[0]
        if abs(zscore) > 2:
            explanations.append(f"Amount significantly different from user pattern (z-score: {zscore:.1f})")
    
    # Check transaction frequency
    if 'trans_count_day' in input_data.columns:
        daily_count = input_data['trans_count_day'].iloc[0]
        if daily_count > 10:
            explanations.append(f"High transaction frequency today: {daily_count} transactions")
    
    return explanations if explanations else ["Multiple risk factors detected"]

def health_check():
    """Health check endpoint for the model service"""
    try:
        load_model()
        return {
            "status": "healthy",
            "model_loaded": model is not None,
            "features_count": len(feature_names) if feature_names else 0,
            "model_type": type(model).__name__ if model else "unknown",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# For local testing
if __name__ == "__main__":
    # Test the prediction function
    print("Testing fraud detection model serving...")
    
    # Sample test request
    test_request = {
        "Amount_clean": 2500.0,
        "hour": 3,
        "day_of_week": 6,
        "day_of_month": 15,
        "is_weekend": 1,
        "is_online": 1,
        "is_chip": 0,
        "is_swipe": 0,
        "mcc_encoded": 5,
        "state_encoded": 10,
        "is_online_state": 1,
        "time_since_last": 0.1,
        "amount_mean_5": 50.0,
        "amount_std_5": 10.0,
        "amount_mean_10": 45.0,
        "amount_std_10": 12.0,
        "amount_mean_30": 48.0,
        "amount_std_30": 15.0,
        "amount_deviation": 2452.0,
        "amount_zscore": 163.5,
        "trans_count_day": 15
    }
    
    print("\nTest request:")
    print(json.dumps(test_request, indent=2))
    
    print("\nPrediction result:")
    result = predict(test_request)
    print(json.dumps(result, indent=2))
    
    print("\nHealth check:")
    health = health_check()
    print(json.dumps(health, indent=2))