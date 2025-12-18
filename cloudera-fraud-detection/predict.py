import os
import json
import cml.models_v1 as models

# Global cache
_model = None
_metadata = None

def _find_model():
    """Try multiple possible paths for the model file"""
    possible_paths = [
        "/home/cdsw/cloudera-fraud-detection/models",  # Standard CML path
        "/home/cdsw/models",  # Model at project root
        os.path.join(os.getcwd(), "cloudera-fraud-detection", "models"),
        os.path.join(os.getcwd(), "models"),
    ]

    for base in possible_paths:
        mp = os.path.join(base, "fraud_detection_lgb.pkl")
        if os.path.exists(mp):
            return base
    return None

def _load():
    global _model, _metadata
    if _model is None:
        try:
            import joblib
            base = _find_model()
            if base is None:
                return None, None, None

            mp = os.path.join(base, "fraud_detection_lgb.pkl")
            meta_p = os.path.join(base, "model_metadata.json")

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
            # Return diagnostic info - list actual directory contents
            diag = {
                "error": "Model not found",
                "cwd": os.getcwd(),
                "cwd_contents": os.listdir(os.getcwd()) if os.path.exists(os.getcwd()) else "N/A",
            }
            # Check cloudera-fraud-detection dir
            cfd = "/home/cdsw/cloudera-fraud-detection"
            if os.path.exists(cfd):
                diag["cfd_exists"] = True
                diag["cfd_contents"] = os.listdir(cfd)
                models_dir = os.path.join(cfd, "models")
                if os.path.exists(models_dir):
                    diag["models_dir_contents"] = os.listdir(models_dir)
                else:
                    diag["models_dir_exists"] = False
            else:
                diag["cfd_exists"] = False
            return diag

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
