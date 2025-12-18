#!/usr/bin/env python3
"""
Batch Feature Engineering for Fraud Detection
These features are pre-computed from historical data and stored for lookup.
They represent static or slowly-changing characteristics.
"""

import pandas as pd
import numpy as np
from datetime import datetime
import json
import os


class BatchFeatureEngineer:
    """
    Batch features are pre-computed from historical transaction data.
    These features capture:
    - User spending profiles (historical averages)
    - Merchant risk profiles
    - Geographic patterns
    - Time-based patterns
    """

    BATCH_FEATURES = [
        # User profile features
        'user_avg_amount_30d',
        'user_std_amount_30d',
        'user_avg_amount_90d',
        'user_std_amount_90d',
        'user_avg_daily_transactions',
        'user_median_amount',
        'user_max_amount_ever',

        # User preference features
        'user_pref_mcc_count',
        'user_online_ratio',
        'user_chip_ratio',
        'user_home_state',

        # Merchant features
        'merchant_avg_amount',
        'merchant_fraud_rate',
        'merchant_transaction_count',

        # MCC features
        'mcc_avg_amount',
        'mcc_fraud_rate',
        'mcc_encoded',

        # State features
        'state_fraud_rate',
        'state_encoded',
    ]

    def __init__(self):
        self.user_profiles = {}
        self.merchant_profiles = {}
        self.mcc_profiles = {}
        self.state_profiles = {}
        self.is_fitted = False

    def fit(self, df):
        """
        Compute batch features from historical data.
        This is typically run offline/batch process.
        """
        print("Computing batch features from historical data...")

        # Ensure datetime column exists
        if 'datetime' not in df.columns:
            df = self._add_datetime(df.copy())

        # Clean amount
        if 'Amount_clean' not in df.columns:
            df['Amount_clean'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)

        # Compute user profiles
        self._compute_user_profiles(df)

        # Compute merchant profiles
        self._compute_merchant_profiles(df)

        # Compute MCC profiles
        self._compute_mcc_profiles(df)

        # Compute state profiles
        self._compute_state_profiles(df)

        self.is_fitted = True
        print(f"Batch feature computation complete")
        print(f"  User profiles: {len(self.user_profiles)}")
        print(f"  Merchant profiles: {len(self.merchant_profiles)}")
        print(f"  MCC profiles: {len(self.mcc_profiles)}")
        print(f"  State profiles: {len(self.state_profiles)}")

        return self

    def _add_datetime(self, df):
        """Add datetime column from components"""
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
            hour=df['Time'].str.split(':').str[0].astype(int),
            minute=df['Time'].str.split(':').str[1].astype(int)
        ))
        return df

    def _compute_user_profiles(self, df):
        """Compute user-level batch features"""
        for user_id in df['User'].unique():
            user_df = df[df['User'] == user_id]

            # Get last 90 days of data for this user
            max_date = user_df['datetime'].max()
            last_30d = user_df[user_df['datetime'] >= max_date - pd.Timedelta(days=30)]
            last_90d = user_df[user_df['datetime'] >= max_date - pd.Timedelta(days=90)]

            # Find most common state (home state)
            states = user_df[user_df['Merchant State'] != '']['Merchant State']
            home_state = states.mode()[0] if len(states) > 0 else 'UNKNOWN'

            self.user_profiles[user_id] = {
                'user_avg_amount_30d': last_30d['Amount_clean'].mean() if len(last_30d) > 0 else 0,
                'user_std_amount_30d': last_30d['Amount_clean'].std() if len(last_30d) > 1 else 0,
                'user_avg_amount_90d': last_90d['Amount_clean'].mean() if len(last_90d) > 0 else 0,
                'user_std_amount_90d': last_90d['Amount_clean'].std() if len(last_90d) > 1 else 0,
                'user_avg_daily_transactions': len(user_df) / max(1, (user_df['datetime'].max() - user_df['datetime'].min()).days),
                'user_median_amount': user_df['Amount_clean'].median(),
                'user_max_amount_ever': user_df['Amount_clean'].max(),
                'user_pref_mcc_count': user_df['MCC'].nunique(),
                'user_online_ratio': (user_df['Use Chip'] == 'Online Transaction').mean(),
                'user_chip_ratio': (user_df['Use Chip'] == 'Chip Transaction').mean(),
                'user_home_state': home_state,
            }

    def _compute_merchant_profiles(self, df):
        """Compute merchant-level batch features"""
        # Convert fraud indicator
        df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)

        for merchant in df['Merchant Name'].unique():
            merchant_df = df[df['Merchant Name'] == merchant]

            self.merchant_profiles[merchant] = {
                'merchant_avg_amount': merchant_df['Amount_clean'].mean(),
                'merchant_fraud_rate': merchant_df['is_fraud'].mean(),
                'merchant_transaction_count': len(merchant_df),
            }

    def _compute_mcc_profiles(self, df):
        """Compute MCC-level batch features"""
        df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)

        for mcc in df['MCC'].unique():
            mcc_df = df[df['MCC'] == mcc]

            self.mcc_profiles[mcc] = {
                'mcc_avg_amount': mcc_df['Amount_clean'].mean(),
                'mcc_fraud_rate': mcc_df['is_fraud'].mean(),
                'mcc_encoded': pd.Categorical([mcc], categories=df['MCC'].unique()).codes[0],
            }

    def _compute_state_profiles(self, df):
        """Compute state-level batch features"""
        df['is_fraud'] = (df['Is Fraud?'] == 'Yes').astype(int)

        all_states = df['Merchant State'].fillna('ONLINE').unique()

        for state in all_states:
            state_df = df[df['Merchant State'].fillna('ONLINE') == state]

            self.state_profiles[state] = {
                'state_fraud_rate': state_df['is_fraud'].mean() if len(state_df) > 0 else 0,
                'state_encoded': pd.Categorical([state], categories=all_states).codes[0],
            }

    def transform(self, transaction):
        """
        Look up batch features for a single transaction.
        This is used during real-time scoring.

        Args:
            transaction: dict with User, Merchant Name, MCC, Merchant State

        Returns:
            dict of batch features
        """
        if not self.is_fitted:
            raise RuntimeError("BatchFeatureEngineer must be fitted before transform")

        user_id = transaction.get('User')
        merchant = transaction.get('Merchant Name', 'UNKNOWN')
        mcc = transaction.get('MCC')
        state = transaction.get('Merchant State', 'ONLINE') or 'ONLINE'

        features = {}

        # User features
        user_profile = self.user_profiles.get(user_id, {})
        features['user_avg_amount_30d'] = user_profile.get('user_avg_amount_30d', 0)
        features['user_std_amount_30d'] = user_profile.get('user_std_amount_30d', 0)
        features['user_avg_amount_90d'] = user_profile.get('user_avg_amount_90d', 0)
        features['user_std_amount_90d'] = user_profile.get('user_std_amount_90d', 0)
        features['user_avg_daily_transactions'] = user_profile.get('user_avg_daily_transactions', 1)
        features['user_median_amount'] = user_profile.get('user_median_amount', 0)
        features['user_max_amount_ever'] = user_profile.get('user_max_amount_ever', 0)
        features['user_pref_mcc_count'] = user_profile.get('user_pref_mcc_count', 1)
        features['user_online_ratio'] = user_profile.get('user_online_ratio', 0.5)
        features['user_chip_ratio'] = user_profile.get('user_chip_ratio', 0.5)

        # Merchant features
        merchant_profile = self.merchant_profiles.get(merchant, {})
        features['merchant_avg_amount'] = merchant_profile.get('merchant_avg_amount', 50)
        features['merchant_fraud_rate'] = merchant_profile.get('merchant_fraud_rate', 0.01)
        features['merchant_transaction_count'] = merchant_profile.get('merchant_transaction_count', 1)

        # MCC features
        mcc_profile = self.mcc_profiles.get(mcc, {})
        features['mcc_avg_amount'] = mcc_profile.get('mcc_avg_amount', 50)
        features['mcc_fraud_rate'] = mcc_profile.get('mcc_fraud_rate', 0.01)
        features['mcc_encoded'] = mcc_profile.get('mcc_encoded', 0)

        # State features
        state_profile = self.state_profiles.get(state, {})
        features['state_fraud_rate'] = state_profile.get('state_fraud_rate', 0.01)
        features['state_encoded'] = state_profile.get('state_encoded', 0)

        return features

    def save(self, path):
        """Save batch features to disk"""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        data = {
            'user_profiles': self.user_profiles,
            'merchant_profiles': self.merchant_profiles,
            'mcc_profiles': {str(k): v for k, v in self.mcc_profiles.items()},
            'state_profiles': self.state_profiles,
            'is_fitted': self.is_fitted,
            'batch_features': self.BATCH_FEATURES,
            'saved_at': datetime.now().isoformat()
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)

        print(f"Batch features saved to {path}")

    def load(self, path):
        """Load batch features from disk"""
        with open(path, 'r') as f:
            data = json.load(f)

        self.user_profiles = {int(k): v for k, v in data['user_profiles'].items()}
        self.merchant_profiles = data['merchant_profiles']
        self.mcc_profiles = {int(k) if k.isdigit() else k: v for k, v in data['mcc_profiles'].items()}
        self.state_profiles = data['state_profiles']
        self.is_fitted = data['is_fitted']

        print(f"Batch features loaded from {path}")
        return self
