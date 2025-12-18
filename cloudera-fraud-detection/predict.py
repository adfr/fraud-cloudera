import os
import json
import cml.models_v1 as models

# Global cache
_model = None
_metadata = None

def _load():
    global _model, _metadata
    if _model is None:
        try:
            import joblib
            # Use getcwd() - in CML PBJ, cwd is /home/cdsw/ (project root)
            d = os.path.join(os.getcwd(), "cloudera-fraud-detection")
            mp = os.path.join(d, "models", "fraud_detection_lgb.pkl")
            meta_p = os.path.join(d, "models", "model_metadata.json")
            if os.path.exists(mp):
                _model = joblib.load(mp)
            if os.path.exists(meta_p):
                with open(meta_p) as f:
                    _metadata = json.load(f)
        except Exception as e:
            return None, None, str(e)
    return _model, _metadata, None

@models.cml_model
def predict(args):
    try:
        if args is None:
            args = {}

        result = _load()
        model, meta = result[0], result[1]
        err = result[2] if len(result) > 2 else None

        if err:
            return {"error": err}

        if model is None:
            d = os.path.join(os.getcwd(), "cloudera-fraud-detection")
            mp = os.path.join(d, "models", "fraud_detection_lgb.pkl")
            return {"error": f"Model not found at {mp}", "exists": os.path.exists(mp)}

        import pandas as pd
        feats = meta.get("features", []) if meta else []
        row = {f: args.get(f, 0.0) for f in feats}
        df = pd.DataFrame([row])

        prob = float(model.predict_proba(df)[0, 1])
        thresh = meta.get("optimal_threshold", 0.5) if meta else 0.5

        return {
            "fraud_probability": prob,
            "fraud_prediction": int(prob > thresh),
            "fraud_label": "FRAUD" if prob > thresh else "NORMAL"
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}
