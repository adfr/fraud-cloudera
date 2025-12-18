import os
import json
import joblib
import pandas as pd
import cml.models_v1 as models

# Global cache
_model = None
_metadata = None

def _load():
    global _model, _metadata
    if _model is None:
        d = os.path.dirname(os.path.abspath(__file__))
        mp = os.path.join(d, "models", "fraud_detection_lgb.pkl")
        meta_p = os.path.join(d, "models", "model_metadata.json")
        if os.path.exists(mp):
            _model = joblib.load(mp)
        if os.path.exists(meta_p):
            with open(meta_p) as f:
                _metadata = json.load(f)
    return _model, _metadata

@models.cml_model
def predict(args):
    if args is None:
        args = {}

    model, meta = _load()

    if model is None:
        return {"error": "Model not found", "fraud_probability": 0.5}

    # Get features
    feats = meta.get("features", []) if meta else []
    row = {f: args.get(f, 0.0) for f in feats}
    df = pd.DataFrame([row])

    # Predict
    prob = float(model.predict_proba(df)[0, 1])
    thresh = meta.get("optimal_threshold", 0.5) if meta else 0.5

    return {
        "fraud_probability": prob,
        "fraud_prediction": int(prob > thresh),
        "fraud_label": "FRAUD" if prob > thresh else "NORMAL"
    }
