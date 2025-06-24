#!/usr/bin/env python3
"""
Feature engineering for credit card fraud detection.
Prepares data for model training.
"""
import pandas as pd
import numpy as np
from datetime import datetime
import os

def engineer_features(df):
    """Create features for fraud detection model"""
    print("Starting feature engineering...")
    
    # Convert date/time columns to datetime
    df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
        hour=df['Time'].str.split(':').str[0].astype(int),
        minute=df['Time'].str.split(':').str[1].astype(int)
    ))
    
    # Extract time-based features
    df['hour'] = df['datetime'].dt.hour
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['day_of_month'] = df['datetime'].dt.day
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Clean amount column
    df['Amount_clean'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)
    
    # Transaction type features
    df['is_online'] = (df['Use Chip'] == 'Online Transaction').astype(int)
    df['is_chip'] = (df['Use Chip'] == 'Chip Transaction').astype(int)
    df['is_swipe'] = (df['Use Chip'] == 'Swipe Transaction').astype(int)
    
    # MCC (Merchant Category Code) features
    df['mcc_encoded'] = pd.Categorical(df['MCC']).codes
    
    # State features
    df['is_online_state'] = (df['Merchant State'] == '').astype(int)
    df['state_encoded'] = pd.Categorical(df['Merchant State'].fillna('ONLINE')).codes
    
    # User transaction patterns
    df = df.sort_values(['User', 'Card', 'datetime'])
    
    # Calculate time since last transaction for each user/card
    df['time_since_last'] = df.groupby(['User', 'Card'])['datetime'].diff().dt.total_seconds() / 3600  # in hours
    df['time_since_last'] = df['time_since_last'].fillna(0)
    
    # Rolling statistics per user/card (last 30 transactions)
    for window in [5, 10, 30]:
        df[f'amount_mean_{window}'] = df.groupby(['User', 'Card'])['Amount_clean'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'amount_std_{window}'] = df.groupby(['User', 'Card'])['Amount_clean'].transform(
            lambda x: x.rolling(window, min_periods=1).std()
        ).fillna(0)
    
    # Amount deviation from user's average
    df['amount_deviation'] = df['Amount_clean'] - df['amount_mean_30']
    df['amount_zscore'] = df['amount_deviation'] / (df['amount_std_30'] + 1e-5)
    
    # Transaction frequency features
    df['trans_count_day'] = df.groupby(['User', 'Card', df['datetime'].dt.date]).cumcount() + 1
    
    # Convert target variable
    df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)
    
    # Select features for modeling
    feature_cols = [
        'Amount_clean', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'is_online', 'is_chip', 'is_swipe', 'mcc_encoded', 'state_encoded',
        'is_online_state', 'time_since_last', 'amount_mean_5', 'amount_std_5',
        'amount_mean_10', 'amount_std_10', 'amount_mean_30', 'amount_std_30',
        'amount_deviation', 'amount_zscore', 'trans_count_day'
    ]
    
    return df, feature_cols

def main():
    print("Loading credit card transaction data...")
    
    # Read the CSV file
    csv_path = '../card_transaction.v1.csv'
    if not os.path.exists(csv_path):
        csv_path = 'card_transaction.v1.csv'
    
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} transactions")
    
    # Engineer features
    df_features, feature_cols = engineer_features(df)
    
    # Save processed data
    output_path = 'data/processed_transactions.csv'
    os.makedirs('data', exist_ok=True)
    
    # Save full dataset with features
    df_features.to_csv(output_path, index=False)
    print(f"Saved processed data to {output_path}")
    
    # Save feature list
    with open('data/feature_columns.txt', 'w') as f:
        f.write('\n'.join(feature_cols))
    print(f"Saved {len(feature_cols)} feature names to data/feature_columns.txt")
    
    # Print data statistics
    print("\nData Statistics:")
    print(f"Total transactions: {len(df_features)}")
    print(f"Fraud transactions: {df_features['is_fraud'].sum()} ({df_features['is_fraud'].mean()*100:.2f}%)")
    print(f"Date range: {df_features['datetime'].min()} to {df_features['datetime'].max()}")

if __name__ == "__main__":
    main()