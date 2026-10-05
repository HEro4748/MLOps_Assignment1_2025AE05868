"""Unit tests for heart disease pipeline and model."""
import os
import joblib
import pandas as pd
import pytest

def test_processed_data_exists():
    """Verify that the processed data file exists and is not empty."""
    assert os.path.exists("data/processed_heart.csv"), "Processed data file is missing!"
    df = pd.read_csv("data/processed_heart.csv")
    assert not df.empty, "Processed dataset is empty!"
    assert "target" in df.columns, "Target column missing from dataset!"

def test_production_model_exists():
    """Verify that the production model artifact exists."""
    assert os.path.exists("models/production_model.joblib"), "Production model artifact missing!"

def test_model_prediction_output():
    """Verify that the packaged model outputs correct predictions and probabilities."""
    model_path = "models/production_model.joblib"
    if not os.path.exists(model_path):
        pytest.skip("Model artifact not found, skipping prediction test.")
        
    model = joblib.load(model_path)
    
    # Load a small sample from processed data
    df = pd.read_csv("data/processed_heart.csv")
    X_sample = df.drop(columns=["target"]).head(5)
    
    predictions = model.predict(X_sample)
    probabilities = model.predict_proba(X_sample)
    
    # Assertions
    assert len(predictions) == 5, "Prediction count mismatch!"
    assert all(p in [0, 1] for p in predictions), "Predictions must be binary (0 or 1)!"
    assert probabilities.shape == (5, 2), "Probability output shape should be (n_samples, 2)!"