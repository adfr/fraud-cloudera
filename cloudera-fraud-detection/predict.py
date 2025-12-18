#!/usr/bin/env python3
"""
CML Model prediction function for fraud detection with transaction rating.
This file is used by Cloudera ML's native model deployment system.
"""

import os
import sys
import json
from datetime import datetime

# Auto-install dependencies if missing (for CML model deployment)
def ensure_dependencies():
    """Install required packages if not available"""
    required = ['joblib', 'pandas', 'numpy', 'lightgbm', 'scikit-learn']
    missing = []
    for pkg in required:
        try:
            __import__(pkg.replace('-', '_'))
        except ImportError:
            missing.append(pkg)

    if missing:
        import subprocess
        print(f"Installing missing dependencies: {missing}")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)

ensure_dependencies()

import joblib
import pandas as pd
import numpy as np

# CML model decorator for PBJ runtimes
try:
    import cml.models_v1 as models
    CML_AVAILABLE = True
except ImportError:
    CML_AVAILABLE = False

# Global variables to cache model and metadata
model = None
metadata = None
feature_names = None
rating_engine = None


def load_model():
    """Load the trained model and metadata"""
    global model, metadata, feature_names

    if model is None:
        try:
            # Get the directory where this script lives
            script_dir = os.path.dirname(os.path.abspath(__file__))

            # Try to load production model first
            model_path = os.path.join(script_dir, "models", "fraud_detection_lgb.pkl")
            metadata_path = os.path.join(script_dir, "models", "model_metadata.json")

            if not os.path.exists(model_path):
                # Fallback to test model
                model_path = os.path.join(script_dir, "models", "test_fraud_model.pkl")
                metadata_path = os.path.join(script_dir, "models", "test_model_metadata.pkl")

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


def get_rating_engine():
    """Get or create the rating engine"""
    global rating_engine
    if rating_engine is None:
        from scripts.transaction_rating import TransactionRatingEngine
        rating_engine = TransactionRatingEngine()
    return rating_engine


def validate_and_prepare_input(input_data):
    """Validate and prepare input data for prediction"""

    # Full feature list for enhanced model
    expected_features = [
        # Transaction attributes
        'Amount_clean', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'is_online', 'is_chip', 'is_swipe', 'is_online_state',

        # Velocity features
        'time_since_last', 'trans_count_1h', 'trans_count_24h',
        'amount_sum_1h', 'amount_sum_24h', 'trans_count_day',

        # Rolling features
        'amount_mean_5', 'amount_std_5', 'amount_mean_10', 'amount_std_10',

        # Deviation features
        'amount_deviation', 'amount_zscore', 'is_different_state', 'is_new_merchant',

        # Batch features (user profile)
        'user_avg_amount_30d', 'user_std_amount_30d',
        'user_avg_amount_90d', 'user_std_amount_90d',
        'user_avg_daily_transactions', 'user_median_amount',
        'user_max_amount_ever', 'user_pref_mcc_count',
        'user_online_ratio', 'user_chip_ratio',

        # Merchant features
        'merchant_avg_amount', 'merchant_fraud_rate', 'merchant_transaction_count',

        # Category features
        'mcc_avg_amount', 'mcc_fraud_rate', 'mcc_encoded',
        'state_fraud_rate', 'state_encoded',
    ]

    # Backward compatibility - use model's feature names if available
    features_to_use = feature_names if feature_names else expected_features

    # Prepare features dictionary
    features = {}

    # Extract raw transaction fields for rating
    raw_transaction = {}
    for key in ['User', 'Card', 'Year', 'Month', 'Day', 'Time', 'Amount',
                'Use Chip', 'Merchant Name', 'Merchant State', 'MCC', 'transaction_id']:
        if key in input_data:
            raw_transaction[key] = input_data[key]

    # Parse amount if provided as string
    if 'Amount' in input_data and isinstance(input_data['Amount'], str):
        input_data['Amount_clean'] = float(
            input_data['Amount'].replace('$', '').replace(',', '')
        )

    # Parse time features if Time is provided
    if 'Time' in input_data:
        time_str = input_data['Time']
        hour = int(time_str.split(':')[0])
        input_data['hour'] = hour

        # Compute day features if date is provided
        if all(k in input_data for k in ['Year', 'Month', 'Day']):
            try:
                dt = datetime(input_data['Year'], input_data['Month'], input_data['Day'])
                input_data['day_of_week'] = dt.weekday()
                input_data['day_of_month'] = input_data['Day']
                input_data['is_weekend'] = 1 if dt.weekday() >= 5 else 0
            except:
                pass

    # Parse transaction type
    if 'Use Chip' in input_data:
        trans_type = input_data['Use Chip']
        input_data['is_online'] = 1 if trans_type == 'Online Transaction' else 0
        input_data['is_chip'] = 1 if trans_type == 'Chip Transaction' else 0
        input_data['is_swipe'] = 1 if trans_type == 'Swipe Transaction' else 0

    # Parse state features
    if 'Merchant State' in input_data:
        state = input_data['Merchant State']
        input_data['is_online_state'] = 1 if not state else 0

    # Map input data to model features with defaults
    for feature in features_to_use:
        if feature in input_data:
            features[feature] = input_data[feature]
        else:
            # Set sensible defaults for missing features
            if 'amount' in feature.lower():
                features[feature] = input_data.get('Amount_clean', 0.0)
            elif 'count' in feature.lower():
                features[feature] = 1
            elif feature.startswith('is_'):
                features[feature] = 0
            elif feature in ['hour', 'day_of_week', 'day_of_month']:
                features[feature] = 0
            elif 'std' in feature.lower():
                features[feature] = 0.0
            elif 'ratio' in feature.lower():
                features[feature] = 0.5
            elif 'rate' in feature.lower():
                features[feature] = 0.01
            else:
                features[feature] = 0.0

    # Convert to DataFrame
    df = pd.DataFrame([features])

    # Ensure features are in the correct order
    df = df.reindex(columns=features_to_use, fill_value=0.0)

    return df, raw_transaction


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

    # Check velocity
    count_1h = input_data.get('trans_count_1h', 0)
    if count_1h > 3:
        explanations.append(f"High velocity: {count_1h} transactions in last hour")

    return explanations if explanations else ["Multiple risk factors detected"]


# Apply CML decorator if available (required for PBJ runtimes)
def _predict_impl(request):
    """
    Main prediction function called by CML

    Args:
        request: Dictionary containing transaction features

    Returns:
        Dictionary containing fraud prediction results with rating
    """

    try:
        # Load model if not already loaded
        load_model()

        # Validate and prepare input
        input_df, raw_transaction = validate_and_prepare_input(request)

        # Make prediction
        if hasattr(model, 'predict_proba'):
            probabilities = model.predict_proba(input_df)
            if probabilities.shape[1] > 1:
                fraud_probability = float(probabilities[0, 1])
            else:
                fraud_probability = float(probabilities[0, 0])
        else:
            fraud_probability = float(model.predict(input_df)[0])

        # Use optimal threshold if available
        threshold = metadata.get('optimal_threshold', 0.5) if metadata else 0.5
        fraud_prediction = int(fraud_probability > threshold)

        # Calculate risk level
        risk_level = calculate_risk_level(fraud_probability)

        # Calculate transaction rating
        try:
            engine = get_rating_engine()
            rating_result = engine.rate_transaction(
                raw_transaction if raw_transaction else request,
                fraud_probability,
                user_profile={
                    'user_avg_amount_30d': request.get('user_avg_amount_30d', request.get('Amount_clean', 0)),
                    'user_std_amount_30d': request.get('user_std_amount_30d', 0),
                    'user_home_state': request.get('user_home_state', ''),
                    'user_online_ratio': request.get('user_online_ratio', 0.5),
                },
                historical_stats={
                    'trans_count_1h': request.get('trans_count_1h', 0),
                    'trans_count_24h': request.get('trans_count_24h', 0),
                    'time_since_last': request.get('time_since_last', 24),
                }
            )
            transaction_rating = rating_result.get('transaction_rating', 'N/A')
            rating_score = rating_result.get('rating_score', 0)
            recommendation = rating_result.get('recommendation', {})
        except Exception as e:
            print(f"Rating calculation error: {e}")
            transaction_rating = 'N/A'
            rating_score = 0
            recommendation = {}

        # Prepare response
        response = {
            "fraud_probability": fraud_probability,
            "fraud_prediction": fraud_prediction,
            "fraud_label": "FRAUD" if fraud_prediction == 1 else "NORMAL",
            "risk_level": risk_level,
            "threshold_used": threshold,
            "prediction_timestamp": datetime.now().isoformat(),
            "confidence": "high" if fraud_probability > 0.8 or fraud_probability < 0.2 else "medium",

            # Transaction rating
            "transaction_rating": transaction_rating,
            "rating_score": rating_score,
            "recommendation": recommendation.get('action', 'APPROVE') if recommendation else 'APPROVE',
            "recommendation_message": recommendation.get('message', '') if recommendation else '',
            "should_approve": recommendation.get('should_approve', True) if recommendation else True,
            "requires_review": recommendation.get('requires_review', False) if recommendation else False
        }

        # Add model metadata if available
        if metadata:
            response["model_version"] = metadata.get('train_date', 'unknown')
            response["model_type"] = "LightGBM"

        # Add explanation for high-risk transactions
        if fraud_probability > 0.6:
            response["explanation"] = generate_explanation(request, fraud_probability)
            if recommendation:
                response["risk_factors"] = recommendation.get('primary_risk_factors', [])

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
            "transaction_rating": "ERROR",
            "prediction_timestamp": datetime.now().isoformat(),
            "success": False
        }
        print(f"Prediction error: {str(e)}")
        return error_response


# Create the decorated predict function for CML
if CML_AVAILABLE:
    @models.cml_model
    def predict(request):
        """CML model endpoint with decorator"""
        return _predict_impl(request)
else:
    # Fallback for local testing without CML
    predict = _predict_impl


# For local testing
if __name__ == "__main__":
    print("Testing CML fraud detection prediction function with rating...")

    # Test cases
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "User": 1,
                "Card": 0,
                "Year": 2024,
                "Month": 1,
                "Day": 15,
                "Time": "14:30",
                "Amount": "$45.50",
                "Use Chip": "Chip Transaction",
                "Merchant Name": "GROCERY_STORE",
                "Merchant State": "CA",
                "MCC": 5411,
                "amount_mean_30": 50.0,
                "amount_zscore": -0.3,
                "trans_count_day": 3
            }
        },
        {
            "name": "Suspicious Transaction",
            "data": {
                "User": 1,
                "Card": 0,
                "Year": 2024,
                "Month": 1,
                "Day": 15,
                "Time": "03:15",
                "Amount": "$2500.0",
                "Use Chip": "Online Transaction",
                "Merchant Name": "ELECTRONICS_STORE",
                "Merchant State": "",
                "MCC": 5732,
                "amount_mean_30": 48.0,
                "amount_zscore": 51.0,
                "trans_count_day": 15,
                "trans_count_1h": 5
            }
        },
        {
            "name": "Minimal Data Transaction",
            "data": {
                "Amount": "$100.0"
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n{'='*60}")
        print(f"Testing: {test_case['name']}")
        print('='*60)

        print("Input:")
        print(json.dumps(test_case['data'], indent=2))

        result = predict(test_case['data'])

        print("\nOutput:")
        print(json.dumps(result, indent=2))

        if result.get('fraud_label') != 'ERROR':
            print(f"\nSummary:")
            print(f"  Label: {result['fraud_label']}")
            print(f"  Risk Level: {result['risk_level']}")
            print(f"  Fraud Probability: {result['fraud_probability']:.3f}")
            print(f"  Transaction Rating: {result['transaction_rating']}")
            print(f"  Recommendation: {result['recommendation']}")

    print(f"\n{'='*60}")
    print("Local testing completed!")
    print('='*60)
