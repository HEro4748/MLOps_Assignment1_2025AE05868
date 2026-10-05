"""FastAPI web service for Heart Disease Prediction."""
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Heart Disease Prediction API",
    description="Production-ready MLOps API for predicting heart disease risk.",
    version="1.0.0"
)

# Load the production model pipeline at startup
model_path = "models/production_model.joblib"
try:
    model = joblib.load(model_path)
except Exception as e:
    model = None

# Define input schema matching UCI dataset features
class PatientFeatures(BaseModel):
    age: float = Field(..., example=55.0)
    sex: float = Field(..., example=1.0)
    cp: float = Field(..., example=1.0)
    trestbps: float = Field(..., example=130.0)
    chol: float = Field(..., example=250.0)
    fbs: float = Field(..., example=0.0)
    restecg: float = Field(..., example=1.0)
    thalach: float = Field(..., example=150.0)
    exang: float = Field(..., example=0.0)
    oldpeak: float = Field(..., example=1.0)
    slope: float = Field(..., example=2.0)
    ca: float = Field(..., example=0.0)
    thal: float = Field(..., example=3.0)

@app.get("/health")
def health_check():
    """Health check endpoint to verify service and model status."""
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

@app.post("/predict")
def predict_heart_disease(features: PatientFeatures):
    """Accept patient clinical metrics and return heart disease prediction & confidence."""
    if model is None:
        raise HTTPException(status_code=500, detail="Model artifact not loaded on server.")
    
    try:
        # Convert incoming Pydantic payload to DataFrame matching training features
        input_data = pd.DataFrame([features.dict()])
        
        # Predict class and probability
        prediction = int(model.predict(input_data)[0])
        probability = float(model.predict_proba(input_data)[0][1])
        
        return {
            "heart_disease_prediction": prediction,
            "confidence_score": round(probability, 4),
            "risk_status": "High Risk" if prediction == 1 else "Low Risk"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))