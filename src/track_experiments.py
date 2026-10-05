"""MLflow Experiment Tracking for Heart Disease Classification."""
import pandas as pd
import numpy as np
import os
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

def run_experiment():
    # 1. Setup MLflow tracking URI
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("heart-disease-experiment")
    
    # 2. Load processed data
    data_path = "data/processed_heart.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please run src/eda.py first!")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define models to track
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(random_state=42)
    }
    
    print("--- Starting MLflow Experiment Tracking ---")
    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            # Create pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('classifier', model)
            ])
            
            # Fit model
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_pred = pipeline.predict(X_test)
            y_proba = pipeline.predict_proba(X_test)[:, 1]
            
            # Metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            roc_auc = roc_auc_score(y_test, y_proba)
            
            # Log parameters and metrics to MLflow
            mlflow.log_param("model_name", name)
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)
            mlflow.log_metric("roc_auc", roc_auc)
            
            # Log model artifact with trusted types enabled for tree-based models
            if name == "RandomForest":
                mlflow.sklearn.log_model(pipeline, "model", skops_trusted_types=["sklearn.tree._tree.Tree"])
            else:
                mlflow.sklearn.log_model(pipeline, "model")
            
            print(f"[+] Logged {name} run to MLflow | ROC-AUC: {roc_auc:.4f}, Accuracy: {acc:.4f}")

    print("\nExperiment tracking complete! You can inspect runs using `mlflow ui`.")

if __name__ == "__main__":
    run_experiment()