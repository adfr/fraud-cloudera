#!/usr/bin/env python3
"""
Feature Pipeline for Fraud Detection
Combines batch and real-time features for model training and inference.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os

from .batch_features import BatchFeatureEngineer
from .realtime_features import RealTimeFeatureEngineer


class FeaturePipeline:
    """
    Unified feature pipeline that combines:
    - Batch features (pre-computed from historical data)
    - Real-time features (computed at transaction time)

    This pipeline is used for both:
    1. Training: Process historical data to create training features
    2. Inference: Process individual transactions in real-time
    """

    # All features used by the model
    ALL_FEATURES = [
        # Transaction attributes (real-time)
        'Amount_clean',
        'hour',
        'day_of_week',
        'day_of_month',
        'is_weekend',
        'is_online',
        'is_chip',
        'is_swipe',
        'is_online_state',

        # Velocity features (real-time)
        'time_since_last',
        'trans_count_1h',
        'trans_count_24h',
        'amount_sum_1h',
        'amount_sum_24h',
        'trans_count_day',

        # Rolling features (real-time)
        'amount_mean_5',
        'amount_std_5',
        'amount_mean_10',
        'amount_std_10',

        # Deviation features (real-time + batch)
        'amount_deviation',
        'amount_zscore',
        'is_different_state',
        'is_new_merchant',

        # User profile features (batch)
        'user_avg_amount_30d',
        'user_std_amount_30d',
        'user_avg_amount_90d',
        'user_std_amount_90d',
        'user_avg_daily_transactions',
        'user_median_amount',
        'user_max_amount_ever',
        'user_pref_mcc_count',
        'user_online_ratio',
        'user_chip_ratio',

        # Merchant features (batch)
        'merchant_avg_amount',
        'merchant_fraud_rate',
        'merchant_transaction_count',

        # MCC features (batch)
        'mcc_avg_amount',
        'mcc_fraud_rate',
        'mcc_encoded',

        # State features (batch)
        'state_fraud_rate',
        'state_encoded',
    ]

    def __init__(self, window_size=50):
        """
        Args:
            window_size: Number of recent transactions to keep for real-time features
        """
        self.batch_engineer = BatchFeatureEngineer()
        self.realtime_engineer = RealTimeFeatureEngineer(window_size=window_size)
        self.is_fitted = False

    def fit(self, df):
        """
        Fit the pipeline on historical data.
        This computes batch features and initializes real-time state.

        Args:
            df: DataFrame with historical transactions
        """
        print("Fitting feature pipeline...")

        # Fit batch features
        self.batch_engineer.fit(df)

        # Initialize real-time history from data
        self.realtime_engineer.load_history_from_dataframe(df)

        self.is_fitted = True
        print("Feature pipeline fitted successfully")

        return self

    def transform_batch(self, df, include_target=True):
        """
        Transform a DataFrame of transactions for training.
        Processes all transactions and computes all features.

        Args:
            df: DataFrame with transactions
            include_target: Whether to include the target variable

        Returns:
            DataFrame with all features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        print(f"Transforming {len(df)} transactions...")

        # Ensure datetime
        if 'datetime' not in df.columns:
            df = df.copy()
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
                hour=df['Time'].str.split(':').str[0].astype(int),
                minute=df['Time'].str.split(':').str[1].astype(int)
            ))

        # Sort by time
        df = df.sort_values('datetime').reset_index(drop=True)

        # Create a temporary real-time engineer for batch processing
        temp_realtime = RealTimeFeatureEngineer(window_size=self.realtime_engineer.window_size)

        # Process each transaction
        all_features = []
        for idx, row in df.iterrows():
            transaction = row.to_dict()

            # Get batch features
            batch_features = self.batch_engineer.transform(transaction)

            # Get real-time features
            realtime_features = temp_realtime.transform(transaction, batch_features)

            # Combine features
            combined = {**batch_features, **realtime_features}

            # Update history for next transaction
            temp_realtime.update_history(transaction, row['User'], row['Card'])

            all_features.append(combined)

            if (idx + 1) % 10000 == 0:
                print(f"  Processed {idx + 1}/{len(df)} transactions")

        # Create features DataFrame
        features_df = pd.DataFrame(all_features)

        # Ensure all features are present
        for feature in self.ALL_FEATURES:
            if feature not in features_df.columns:
                features_df[feature] = 0

        # Reorder columns
        features_df = features_df[self.ALL_FEATURES]

        # Add target if requested
        if include_target:
            features_df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int).values

        print(f"Transformation complete. Features: {len(self.ALL_FEATURES)}")

        return features_df

    def transform_single(self, transaction, update_history=True):
        """
        Transform a single transaction for real-time inference.

        Args:
            transaction: dict with transaction details
            update_history: Whether to update real-time history after processing

        Returns:
            dict of all features
        """
        if not self.is_fitted:
            raise RuntimeError("Pipeline must be fitted before transform")

        # Get batch features
        batch_features = self.batch_engineer.transform(transaction)

        # Get real-time features
        realtime_features = self.realtime_engineer.transform(transaction, batch_features)

        # Combine features
        combined = {**batch_features, **realtime_features}

        # Update history if requested
        if update_history:
            user_id = transaction.get('User', 0)
            card_id = transaction.get('Card', 0)
            self.realtime_engineer.update_history(transaction, user_id, card_id)

        # Ensure all features are present
        for feature in self.ALL_FEATURES:
            if feature not in combined:
                combined[feature] = 0

        return combined

    def get_feature_vector(self, transaction):
        """
        Get a feature vector suitable for model prediction.

        Args:
            transaction: dict with transaction details

        Returns:
            numpy array of features in correct order
        """
        features = self.transform_single(transaction)
        return np.array([features[f] for f in self.ALL_FEATURES])

    def save(self, path):
        """
        Save the fitted pipeline to disk.

        Args:
            path: Directory path to save pipeline artifacts
        """
        os.makedirs(path, exist_ok=True)

        # Save batch features
        self.batch_engineer.save(os.path.join(path, 'batch_features.json'))

        # Save real-time state
        rt_state = self.realtime_engineer.get_state()
        with open(os.path.join(path, 'realtime_state.json'), 'w') as f:
            json.dump(rt_state, f)

        # Save metadata
        metadata = {
            'features': self.ALL_FEATURES,
            'window_size': self.realtime_engineer.window_size,
            'is_fitted': self.is_fitted,
            'saved_at': datetime.now().isoformat()
        }
        with open(os.path.join(path, 'pipeline_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Pipeline saved to {path}")

    def load(self, path):
        """
        Load a fitted pipeline from disk.

        Args:
            path: Directory path containing pipeline artifacts
        """
        # Load batch features
        self.batch_engineer.load(os.path.join(path, 'batch_features.json'))

        # Load real-time state
        with open(os.path.join(path, 'realtime_state.json'), 'r') as f:
            rt_state = json.load(f)
        self.realtime_engineer.load_state(rt_state)

        # Load metadata
        with open(os.path.join(path, 'pipeline_metadata.json'), 'r') as f:
            metadata = json.load(f)

        self.is_fitted = metadata['is_fitted']
        print(f"Pipeline loaded from {path}")

        return self

    @classmethod
    def get_feature_names(cls):
        """Get list of all feature names"""
        return cls.ALL_FEATURES.copy()


def main():
    """Example usage of the feature pipeline"""

    # Create sample data
    print("Creating sample data...")
    data = {
        'User': [1, 1, 1, 2, 2],
        'Card': [0, 0, 0, 0, 0],
        'Year': [2024, 2024, 2024, 2024, 2024],
        'Month': [1, 1, 1, 1, 1],
        'Day': [1, 1, 2, 1, 2],
        'Time': ['10:00', '14:30', '09:15', '11:00', '16:00'],
        'Amount': ['$50.00', '$125.50', '$75.00', '$200.00', '$45.00'],
        'Use Chip': ['Chip Transaction', 'Online Transaction', 'Chip Transaction',
                     'Swipe Transaction', 'Chip Transaction'],
        'Merchant Name': ['STORE_A', 'ONLINE_B', 'STORE_A', 'STORE_C', 'STORE_D'],
        'Merchant City': ['NYC', '', 'NYC', 'LA', 'LA'],
        'Merchant State': ['NY', '', 'NY', 'CA', 'CA'],
        'Zip': ['10001', '', '10001', '90001', '90002'],
        'MCC': [5411, 5732, 5411, 5812, 5411],
        'Errors?': ['', '', '', '', ''],
        'Is Fraud?': ['No', 'No', 'No', 'No', 'Yes']
    }

    df = pd.DataFrame(data)

    # Create and fit pipeline
    print("\nFitting feature pipeline...")
    pipeline = FeaturePipeline(window_size=10)
    pipeline.fit(df)

    # Transform batch
    print("\nTransforming batch...")
    features_df = pipeline.transform_batch(df)
    print(f"\nFeature columns: {list(features_df.columns)}")
    print(f"\nFeature values:\n{features_df}")

    # Transform single transaction
    print("\n\nTransforming single transaction...")
    new_transaction = {
        'User': 1,
        'Card': 0,
        'Year': 2024,
        'Month': 1,
        'Day': 3,
        'Time': '11:30',
        'Amount': '$500.00',
        'Use Chip': 'Online Transaction',
        'Merchant Name': 'NEW_MERCHANT',
        'Merchant City': '',
        'Merchant State': '',
        'Zip': '',
        'MCC': 5944,
    }

    features = pipeline.transform_single(new_transaction)
    print(f"\nFeatures for new transaction:")
    for key, value in features.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
