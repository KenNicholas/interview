import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any

app = FastAPI(title="Research Analyst Model Inference API")

# Load model/pipeline saat startup
MODEL_PATH = "best_pipeline.joblib"
try:
    model_pipeline = joblib.load(MODEL_PATH)
except Exception as e:
    model_pipeline = None

class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]

@app.get("/")
def home():
    return {"status": "online", "message": "Model Inference API Ready"}

@app.post("/predict")
def predict(payload: PredictRequest):
    if not model_pipeline:
        raise HTTPException(status_code=500, detail="Model file not loaded.")
    
    try:
        input_df = pd.DataFrame(payload.data)
        predictions = model_pipeline.predict(input_df)
        
        # Probabilitas jika didukung
        try:
            probabilities = model_pipeline.predict_proba(input_df).tolist()
        except:
            probabilities = None
            
        return {
            "predictions": predictions.tolist(),
            "probabilities": probabilities
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# run via terminal: uvicorn 03_fastapi_app:app --reload