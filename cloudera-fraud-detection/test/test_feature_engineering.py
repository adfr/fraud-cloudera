#!/usr/bin/env python3
"""
Test script for feature_engineering.py
Tests the feature engineering pipeline with sample data
"""
import sys
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Add the scripts directory to the Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from feature_engineering import engineer_features

def create_sample_data():
    """Create sample credit card transaction data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate dates over the past year
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365)
    dates = pd.date_range(start=start_date, end=end_date, periods=n_samples)
    
    # Create sample data
    data = {
        'User': np.random.randint(1, 100, n_samples),
        'Card': np.random.randint(1, 5, n_samples),
        'Year': dates.year,
        'Month': dates.month,
        'Day': dates.day,
        'Time': [f"{np.random.randint(0, 24):02d}:{np.random.randint(0, 60):02d}" for _ in range(n_samples)],
        'Amount': ['$' + f"{np.random.uniform(1, 500):.2f}" for _ in range(n_samples)],
        'Use Chip': np.random.choice(['Chip Transaction', 'Swipe Transaction', 'Online Transaction'], n_samples),
        'Merchant Name': [f"Merchant_{i}" for i in np.random.randint(1, 50, n_samples)],
        'Merchant City': [f"City_{i}" for i in np.random.randint(1, 20, n_samples)],
        'Merchant State': np.random.choice(['CA', 'NY', 'TX', 'FL', 'IL', ''], n_samples, p=[0.2, 0.15, 0.15, 0.1, 0.1, 0.3]),
        'Zip': np.random.randint(10000, 99999, n_samples),
        'MCC': np.random.choice([5411, 5812, 5814, 4111, 5541], n_samples),
        'Errors?': np.random.choice(['', 'Insufficient Balance', 'Bad PIN'], n_samples, p=[0.95, 0.03, 0.02]),
        'Is Fraud?': np.random.choice(['No', 'Yes'], n_samples, p=[0.98, 0.02])
    }
    
    return pd.DataFrame(data)

def test_feature_engineering():
    """Test the feature engineering pipeline"""
    print("Creating sample data...")
    df = create_sample_data()
    print(f"Created {len(df)} sample transactions")
    print(f"Columns: {df.columns.tolist()}")
    print(f"Fraud rate: {(df['Is Fraud?'] == 'Yes').mean():.2%}\n")
    
    # Create data directory if it doesn't exist
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    # Save sample data
    sample_file = os.path.join(data_dir, 'sample_transactions.csv')
    df.to_csv(sample_file, index=False)
    print(f"Saved sample data to: {sample_file}")
    
    print("\nTesting feature engineering...")
    try:
        # Run feature engineering
        df_features, feature_cols = engineer_features(df.copy())
        
        print("\nFeature engineering completed successfully!")
        print(f"Original columns: {len(df.columns)}")
        print(f"Engineered columns: {len(df_features.columns)}")
        
        # Show new features
        original_cols = set(df.columns)
        new_cols = set(df_features.columns) - original_cols
        print(f"\nNew features created ({len(new_cols)}):")
        for col in sorted(new_cols):
            print(f"  - {col}")
        
        print(f"\nFeature columns for modeling ({len(feature_cols)}):")
        for col in feature_cols:
            print(f"  - {col}")
        
        # Check for missing values
        missing_counts = df_features.isnull().sum()
        if missing_counts.any():
            print("\nColumns with missing values:")
            for col, count in missing_counts[missing_counts > 0].items():
                print(f"  - {col}: {count} ({count/len(df_features):.1%})")
        else:
            print("\nNo missing values found in engineered features")
        
        # Save engineered features
        output_file = os.path.join(data_dir, 'test_processed_transactions.csv')
        df_features.to_csv(output_file, index=False)
        print(f"\nSaved engineered features to: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"\nError during feature engineering: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("="*60)
    print("Feature Engineering Test")
    print("="*60)
    
    success = test_feature_engineering()
    
    if success:
        print("\n✓ Feature engineering test passed!")
    else:
        print("\n✗ Feature engineering test failed!")
        sys.exit(1)