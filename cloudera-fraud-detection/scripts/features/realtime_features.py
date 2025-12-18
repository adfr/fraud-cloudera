#!/usr/bin/env python3
"""
Real-Time Feature Engineering for Fraud Detection
These features are computed on-the-fly during transaction processing.
They require access to recent transaction history (in-memory or cache).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import json


class RealTimeFeatureEngineer:
    """
    Real-time features are computed dynamically at transaction time.
    These features capture:
    - Transaction-level attributes
    - Recent transaction patterns (last N transactions)
    - Velocity features (transactions per time window)
    - Deviation from historical patterns
    """

    REALTIME_FEATURES = [
        # Transaction attributes
        'Amount_clean',
        'hour',
        'day_of_week',
        'day_of_month',
        'is_weekend',
        'is_online',
        'is_chip',
        'is_swipe',
        'is_online_state',

        # Velocity features (real-time aggregations)
        'time_since_last',
        'trans_count_1h',
        'trans_count_24h',
        'amount_sum_1h',
        'amount_sum_24h',
        'trans_count_day',

        # Rolling features (recent N transactions)
        'amount_mean_5',
        'amount_std_5',
        'amount_mean_10',
        'amount_std_10',

        # Deviation features
        'amount_deviation',
        'amount_zscore',
        'is_different_state',
        'is_new_merchant',
    ]

    def __init__(self, window_size=50):
        """
        Args:
            window_size: Number of recent transactions to keep per user
        """
        self.window_size = window_size
        # In-memory transaction history per user/card
        self.user_history = defaultdict(list)
        self.user_merchants = defaultdict(set)
        self.user_states = defaultdict(set)

    def compute_transaction_features(self, transaction):
        """
        Compute basic transaction-level features.
        These don't require historical data.
        """
        features = {}

        # Amount - clean and extract
        amount = transaction.get('Amount', '$0')
        if isinstance(amount, str):
            amount = float(amount.replace('$', '').replace(',', ''))
        features['Amount_clean'] = amount

        # Time features
        time_str = transaction.get('Time', '12:00')
        hour = int(time_str.split(':')[0])
        features['hour'] = hour

        # Date features
        year = transaction.get('Year', datetime.now().year)
        month = transaction.get('Month', datetime.now().month)
        day = transaction.get('Day', datetime.now().day)

        try:
            dt = datetime(year, month, day)
            features['day_of_week'] = dt.weekday()
            features['day_of_month'] = day
            features['is_weekend'] = 1 if dt.weekday() >= 5 else 0
        except:
            features['day_of_week'] = 0
            features['day_of_month'] = day
            features['is_weekend'] = 0

        # Transaction type
        use_chip = transaction.get('Use Chip', '')
        features['is_online'] = 1 if use_chip == 'Online Transaction' else 0
        features['is_chip'] = 1 if use_chip == 'Chip Transaction' else 0
        features['is_swipe'] = 1 if use_chip == 'Swipe Transaction' else 0

        # State features
        state = transaction.get('Merchant State', '')
        features['is_online_state'] = 1 if state == '' else 0

        return features

    def compute_velocity_features(self, transaction, user_id, card_id):
        """
        Compute velocity/aggregation features in real-time.
        These require access to recent transaction history.
        """
        features = {}
        key = (user_id, card_id)

        # Get current transaction time
        year = transaction.get('Year', datetime.now().year)
        month = transaction.get('Month', datetime.now().month)
        day = transaction.get('Day', datetime.now().day)
        time_str = transaction.get('Time', '12:00')
        hour, minute = map(int, time_str.split(':'))

        current_time = datetime(year, month, day, hour, minute)

        # Get historical transactions for this user/card
        history = self.user_history.get(key, [])

        # Time since last transaction
        if history:
            last_time = history[-1]['datetime']
            time_diff = (current_time - last_time).total_seconds() / 3600  # hours
            features['time_since_last'] = max(0, time_diff)
        else:
            features['time_since_last'] = 0

        # Transactions in last 1 hour
        one_hour_ago = current_time - timedelta(hours=1)
        trans_1h = [t for t in history if t['datetime'] >= one_hour_ago]
        features['trans_count_1h'] = len(trans_1h)
        features['amount_sum_1h'] = sum(t['amount'] for t in trans_1h)

        # Transactions in last 24 hours
        one_day_ago = current_time - timedelta(hours=24)
        trans_24h = [t for t in history if t['datetime'] >= one_day_ago]
        features['trans_count_24h'] = len(trans_24h)
        features['amount_sum_24h'] = sum(t['amount'] for t in trans_24h)

        # Transactions today
        today_start = current_time.replace(hour=0, minute=0, second=0)
        trans_today = [t for t in history if t['datetime'] >= today_start]
        features['trans_count_day'] = len(trans_today) + 1  # including current

        return features

    def compute_rolling_features(self, transaction, user_id, card_id):
        """
        Compute rolling window features from recent transactions.
        """
        features = {}
        key = (user_id, card_id)

        # Get historical transactions for this user/card
        history = self.user_history.get(key, [])
        amounts = [t['amount'] for t in history]

        # Current amount
        amount = transaction.get('Amount', '$0')
        if isinstance(amount, str):
            amount = float(amount.replace('$', '').replace(',', ''))

        # Rolling mean/std for last 5 transactions
        last_5 = amounts[-5:] if len(amounts) >= 5 else amounts
        if last_5:
            features['amount_mean_5'] = np.mean(last_5)
            features['amount_std_5'] = np.std(last_5) if len(last_5) > 1 else 0
        else:
            features['amount_mean_5'] = amount
            features['amount_std_5'] = 0

        # Rolling mean/std for last 10 transactions
        last_10 = amounts[-10:] if len(amounts) >= 10 else amounts
        if last_10:
            features['amount_mean_10'] = np.mean(last_10)
            features['amount_std_10'] = np.std(last_10) if len(last_10) > 1 else 0
        else:
            features['amount_mean_10'] = amount
            features['amount_std_10'] = 0

        return features

    def compute_deviation_features(self, transaction, user_id, card_id, batch_features=None):
        """
        Compute deviation from historical patterns.
        """
        features = {}
        key = (user_id, card_id)

        # Current amount
        amount = transaction.get('Amount', '$0')
        if isinstance(amount, str):
            amount = float(amount.replace('$', '').replace(',', ''))

        # Get historical mean/std (from batch features or history)
        if batch_features:
            avg = batch_features.get('user_avg_amount_30d', amount)
            std = batch_features.get('user_std_amount_30d', 1)
        else:
            history = self.user_history.get(key, [])
            amounts = [t['amount'] for t in history]
            avg = np.mean(amounts) if amounts else amount
            std = np.std(amounts) if len(amounts) > 1 else 1

        # Deviation features
        features['amount_deviation'] = amount - avg
        features['amount_zscore'] = (amount - avg) / (std + 1e-5)

        # Geographic deviation
        current_state = transaction.get('Merchant State', '')
        user_home = batch_features.get('user_home_state', 'UNKNOWN') if batch_features else 'UNKNOWN'
        features['is_different_state'] = 1 if current_state and current_state != user_home else 0

        # New merchant check
        merchant = transaction.get('Merchant Name', '')
        features['is_new_merchant'] = 1 if merchant not in self.user_merchants.get(key, set()) else 0

        return features

    def update_history(self, transaction, user_id, card_id):
        """
        Update the transaction history after processing.
        Call this after the transaction is processed.
        """
        key = (user_id, card_id)

        # Parse transaction details
        year = transaction.get('Year', datetime.now().year)
        month = transaction.get('Month', datetime.now().month)
        day = transaction.get('Day', datetime.now().day)
        time_str = transaction.get('Time', '12:00')
        hour, minute = map(int, time_str.split(':'))

        amount = transaction.get('Amount', '$0')
        if isinstance(amount, str):
            amount = float(amount.replace('$', '').replace(',', ''))

        trans_record = {
            'datetime': datetime(year, month, day, hour, minute),
            'amount': amount,
            'merchant': transaction.get('Merchant Name', ''),
            'state': transaction.get('Merchant State', ''),
            'mcc': transaction.get('MCC', 0),
        }

        # Add to history
        self.user_history[key].append(trans_record)

        # Keep only recent transactions
        if len(self.user_history[key]) > self.window_size:
            self.user_history[key] = self.user_history[key][-self.window_size:]

        # Update merchant set
        self.user_merchants[key].add(trans_record['merchant'])

        # Update state set
        if trans_record['state']:
            self.user_states[key].add(trans_record['state'])

    def transform(self, transaction, batch_features=None):
        """
        Compute all real-time features for a transaction.

        Args:
            transaction: dict with transaction details
            batch_features: dict of pre-computed batch features (optional)

        Returns:
            dict of real-time features
        """
        user_id = transaction.get('User', 0)
        card_id = transaction.get('Card', 0)

        features = {}

        # Transaction-level features
        features.update(self.compute_transaction_features(transaction))

        # Velocity features
        features.update(self.compute_velocity_features(transaction, user_id, card_id))

        # Rolling features
        features.update(self.compute_rolling_features(transaction, user_id, card_id))

        # Deviation features
        features.update(self.compute_deviation_features(transaction, user_id, card_id, batch_features))

        return features

    def load_history_from_dataframe(self, df):
        """
        Load transaction history from a DataFrame.
        Useful for initializing real-time features from historical data.
        """
        print(f"Loading {len(df)} historical transactions...")

        # Sort by time
        if 'datetime' not in df.columns:
            df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
                hour=df['Time'].str.split(':').str[0].astype(int),
                minute=df['Time'].str.split(':').str[1].astype(int)
            ))

        df = df.sort_values('datetime')

        # Clean amount
        if 'Amount_clean' not in df.columns:
            df['Amount_clean'] = df['Amount'].str.replace('$', '').str.replace(',', '').astype(float)

        # Load each transaction
        for _, row in df.iterrows():
            key = (row['User'], row['Card'])

            trans_record = {
                'datetime': row['datetime'],
                'amount': row['Amount_clean'],
                'merchant': row['Merchant Name'],
                'state': row['Merchant State'] if pd.notna(row['Merchant State']) else '',
                'mcc': row['MCC'],
            }

            self.user_history[key].append(trans_record)
            self.user_merchants[key].add(row['Merchant Name'])
            if trans_record['state']:
                self.user_states[key].add(trans_record['state'])

        # Trim to window size
        for key in self.user_history:
            if len(self.user_history[key]) > self.window_size:
                self.user_history[key] = self.user_history[key][-self.window_size:]

        print(f"Loaded history for {len(self.user_history)} user/card combinations")

    def get_state(self):
        """Get serializable state for persistence"""
        state = {
            'user_history': {},
            'user_merchants': {},
            'user_states': {},
        }

        for key, history in self.user_history.items():
            str_key = f"{key[0]}_{key[1]}"
            state['user_history'][str_key] = [
                {**t, 'datetime': t['datetime'].isoformat()}
                for t in history
            ]

        for key, merchants in self.user_merchants.items():
            str_key = f"{key[0]}_{key[1]}"
            state['user_merchants'][str_key] = list(merchants)

        for key, states in self.user_states.items():
            str_key = f"{key[0]}_{key[1]}"
            state['user_states'][str_key] = list(states)

        return state

    def load_state(self, state):
        """Load state from serialized format"""
        for str_key, history in state.get('user_history', {}).items():
            parts = str_key.split('_')
            key = (int(parts[0]), int(parts[1]))
            self.user_history[key] = [
                {**t, 'datetime': datetime.fromisoformat(t['datetime'])}
                for t in history
            ]

        for str_key, merchants in state.get('user_merchants', {}).items():
            parts = str_key.split('_')
            key = (int(parts[0]), int(parts[1]))
            self.user_merchants[key] = set(merchants)

        for str_key, states in state.get('user_states', {}).items():
            parts = str_key.split('_')
            key = (int(parts[0]), int(parts[1]))
            self.user_states[key] = set(states)
