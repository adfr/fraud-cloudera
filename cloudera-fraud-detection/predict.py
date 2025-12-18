#!/usr/bin/env python3
"""
CML Model prediction function for fraud detection.
Minimal version for CML deployment.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import cml.models_v1 as models

# Global model cache
_model = None
_metadata = None

def _load_model():
    """Load the trained model"""
    global _model, _metadata
    if _model is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, "models", "fraud_detection_lgb.pkl")
        metadata_path = os.path.join(script_dir, "models", "model_metadata.json")

        if os.path.exists(model_path):
            _model = joblib.load(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model not found at {model_path}")

        if os.path.exists(metadata_path):
            with open(metadata_path) as f:
                _metadata = json.load(f)
    return _model, _metadata


@models.cml_model
def predict(args):
    """
    CML model prediction function.

    Args:
        args: Dictionary with transaction features

    Returns:
        Dictionary with prediction results
    """
    # Handle None input
    if args is None:
        args = {}

    try:
        model, metadata = _load_model()

        if model is None:
            return {
                "error": "Model not loaded",
                "fraud_probability": 0.5,
                "fraud_prediction": 0,
                "success": False
            }

        # Get feature names from metadata
        feature_names = metadata.get('features', []) if metadata else []

        # Build feature vector with defaults
        features = {}
        for feat in feature_names:
            features[feat] = args.get(feat, 0.0)

        # Create DataFrame for prediction
        df = pd.DataFrame([features])

        # Make prediction
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(df)
            fraud_prob = float(proba[0, 1]) if proba.shape[1] > 1 else float(proba[0, 0])
        else:
            fraud_prob = float(model.predict(df)[0])

        threshold = metadata.get('optimal_threshold', 0.5) if metadata else 0.5
        fraud_pred = int(fraud_prob > threshold)

        return {
            "fraud_probability": fraud_prob,
            "fraud_prediction": fraud_pred,
            "fraud_label": "FRAUD" if fraud_pred == 1 else "NORMAL",
            "threshold": threshold,
            "success": True
        }

    except Exception as e:
        return {
            "error": str(e),
            "fraud_probability": None,
            "fraud_prediction": None,
            "success": False
        }
