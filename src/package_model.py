"""Model Packaging and Reproducibility script."""
import pandas as pd
import numpy as np
import joblib
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

def package_and_save_model():
    print("Training final model for packaging...")
    data_path = "data/processed_heart.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please run src/eda.py first!")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Build complete pipeline (Scaler + Random Forest)
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    
    pipeline.fit(X_train, y_train)
    
    os.makedirs("models", exist_ok=True)
    model_path = "models/production_model.joblib"
    joblib.dump(pipeline, model_path)
    print(f"[+] Production model package saved to {model_path}")
    
    # Verify artifact by reloading and predicting on a sample
    print("\nVerifying reproducibility via test inference...")
    loaded_model = joblib.load(model_path)
    sample_input = X_test.iloc[[0]]
    prediction = loaded_model.predict(sample_input)[0]
    probability = loaded_model.predict_proba(sample_input)[0][1]
    
    print(f"Sample Input Features:\n{sample_input.to_dict(orient='records')[0]}")
    print(f"Prediction Result -> Class: {int(prediction)}, Confidence: {probability:.4f}")

if __name__ == "__main__":
    package_and_save_model()    