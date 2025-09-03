
from fastapi import FastAPI
import joblib, pandas as pd
from pathlib import Path

app = FastAPI(title="DDL ML API")

MODEL_PATH = Path("models/rf.joblib")
FEATS_PATH = Path("models/features.csv")

@app.on_event("startup")
def load_model():
    global model, features
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    features = pd.read_csv(FEATS_PATH, header=None)[0].tolist() if FEATS_PATH.exists() else []

@app.get("/health")
def health():
    return {"model_loaded": MODEL_PATH.exists(), "n_features": len(features)}

@app.post("/predict")
def predict(payload: dict):
    if not MODEL_PATH.exists():
        return {"error": "Model not trained"}
    X = pd.DataFrame([payload])
    # align columns
    for col in features:
        if col not in X.columns:
            X[col] = 0
    X = X[features]
    proba = float(model.predict_proba(X)[:,1][0])
    return {"yhat_proba": proba}
