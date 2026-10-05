"""Data Acquisition and EDA script for UCI Heart Disease Dataset."""
import pandas as pd
import numpy as np
import os

from ucimlrepo import fetch_ucirepo

def load_and_preprocess_data():
    print("Fetching UCI Heart Disease dataset...")
    # Fetch dataset (ID 45 is the Heart Disease dataset)
    heart_disease = fetch_ucirepo(id=45)
    
    # Extract features and target
    X = heart_disease.data.features
    y = heart_disease.data.targets
    
    # Combine into a single dataframe for EDA/cleaning
    df = pd.concat([X, y], axis=1)
    
    print(f"Initial dataset shape: {df.shape}")
    
    # Target column in UCI Heart Disease is usually 'num' (0 = no disease, >0 = disease present)
    # Let's convert it to a binary classification target (0: No disease, 1: Disease present)
    target_col = df.columns[-1]
    print(f"Original target column: {target_col}")
    df['target'] = (df[target_col] > 0).astype(int)
    df = df.drop(columns=[target_col])
    
    # Handle missing values (UCI dataset uses NaN or '?' for missing)
    df.replace('?', np.nan, inplace=True)
    df = df.astype(float)
    
    # Fill missing values with median for numerical columns
    df.fillna(df.median(), inplace=True)
    
    print("\n--- Class Balance ---")
    print(df['target'].value_counts(normalize=True))
    
    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)
    
    # Save processed data
    output_path = "data/processed_heart.csv"
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data successfully saved to {output_path}!")

if __name__ == "__main__":
    load_and_preprocess_data()