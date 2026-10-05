"""Feature Engineering and Model Training for Heart Disease Prediction."""
import pandas as pd
import numpy as np
import os
import joblib

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

def train_models():
    # 1. Load processed data
    data_path = "data/processed_heart.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found. Please run src/eda.py first!")
        
    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target']
    
    # 2. Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 3. Define models with preprocessing pipelines
    models = {
        "Logistic Regression": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=42))
        ]),
        "Random Forest": Pipeline([
            ('scaler', StandardScaler()),
            ('classifier', RandomForestClassifier(random_state=42))
        ])
    }
    
    os.makedirs("models", exist_ok=True)
    best_model = None
    best_score = 0
    best_name = ""
    
    print("--- Model Training & Cross-Validation Evaluation ---")
    for name, pipeline in models.items():
        # Perform 5-fold cross-validation
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='roc_auc')
        print(f"\n{name}:")
        print(f"  CV ROC-AUC Scores: {np.round(cv_scores, 4)}")
        print(f"  Mean CV ROC-AUC:   {cv_scores.mean():.4f}")
        
        # Fit on full training data
        pipeline.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"  Test Accuracy:  {acc:.4f}")
        print(f"  Test Precision: {prec:.4f}")
        print(f"  Test Recall:    {rec:.4f}")
        print(f"  Test ROC-AUC:   {roc_auc:.4f}")
        
        # Track best model based on test ROC-AUC
        if roc_auc > best_score:
            best_score = roc_auc
            best_model = pipeline
            best_name = name

    # Save the best performing model
    model_output_path = "models/best_model.joblib"
    joblib.dump(best_model, model_output_path)
    print(f"\n[+] Best Model ({best_name}) saved successfully to {model_output_path} with ROC-AUC: {best_score:.4f}!")

if __name__ == "__main__":
    train_models()