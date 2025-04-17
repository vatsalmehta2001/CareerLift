import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def test_data_loader():
    """Test data loading"""
    print("\n=== Testing Data Loading ===")
    
    try:
        # Load the dataset
        df = pd.read_csv('data/adult.csv')
        print(f"Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Display first few rows
        print("\nFirst 5 rows:")
        print(df.head())
        
        # Display column names
        print("\nColumn names:")
        for col in df.columns:
            print(f"- {col}")
        
        # Check for missing values
        print("\nMissing values:")
        print(df.isnull().sum().sum())
        
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def test_simple_model(df):
    """Test a simple model on the data"""
    print("\n=== Testing Simple Model ===")
    
    try:
        # Clean the data (simplified)
        df_clean = df.copy()
        
        # Replace '?' with NaN
        df_clean.replace('?', np.nan, inplace=True)
        
        # Drop rows with missing values for simplicity
        df_clean = df_clean.dropna()
        
        # Convert target to binary
        if 'income' in df_clean.columns:
            df_clean['income'] = df_clean['income'].map({'>50K': 1, '<=50K': 0})
        
        # Select features and target
        numeric_features = df_clean.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if 'income' in numeric_features:
            numeric_features.remove('income')
        
        X = df_clean[numeric_features]
        y = df_clean['income']
        
        print(f"Features used: {numeric_features}")
        print(f"Data shape: {X.shape}")
        
        # Create and train a simple model
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X, y)
        
        # Evaluate on training data (for simplicity)
        accuracy = model.score(X, y)
        print(f"Training accuracy: {accuracy:.4f}")
        
        return model, X, y, numeric_features
    except Exception as e:
        print(f"Error training model: {e}")
        return None, None, None, None

def test_prediction(model, df, numeric_features):
    """Test prediction for a sample person"""
    print("\n=== Testing Prediction ===")
    
    try:
        # Create a sample person (using only numeric features for simplicity)
        sample = {
            'age': 35,
            'fnlwgt': 200000,
            'education_num': 10,
            'capital_gain': 0,
            'capital_loss': 0,
            'hours_per_week': 40
        }
        
        # Create DataFrame from sample
        sample_df = pd.DataFrame([sample])
        
        # Ensure all features are present
        for feature in numeric_features:
            if feature not in sample_df.columns:
                sample_df[feature] = 0
        
        # Keep only the features used by the model
        sample_df = sample_df[numeric_features]
        
        # Make prediction
        prediction = model.predict(sample_df)[0]
        probability = model.predict_proba(sample_df)[0, 1]
        
        print(f"Sample person: {sample}")
        print(f"Prediction: {'Income >50K' if prediction == 1 else 'Income <=50K'}")
        print(f"Probability of income >50K: {probability:.2%}")
        
        return sample, prediction, probability
    except Exception as e:
        print(f"Error making prediction: {e}")
        return None, None, None

def main():
    """Main test function"""
    print("=== Career Path Optimizer - Test Run ===")
    
    # Check if data file exists
    if not os.path.exists('data/adult.csv'):
        print("Error: data/adult.csv not found")
        return
    
    # Test data loading
    df = test_data_loader()
    if df is None:
        return
    
    # Test simple model
    model, X, y, numeric_features = test_simple_model(df)
    if model is None:
        return
    
    # Test prediction
    sample, prediction, probability = test_prediction(model, df, numeric_features)
    if sample is None:
        return
    
    print("\n=== Test Run Completed Successfully ===")

if __name__ == "__main__":
    main() 