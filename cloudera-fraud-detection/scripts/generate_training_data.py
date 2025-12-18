#!/usr/bin/env python3
"""
Training Data Generator for Fraud Detection
Generates synthetic credit card transactions with realistic fraud patterns.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os
import json

class FraudDataGenerator:
    """Generate synthetic transaction data with fraud scenarios"""

    def __init__(self, seed=42):
        np.random.seed(seed)
        random.seed(seed)

        # Configuration
        self.fraud_rate = 0.02  # 2% fraud rate

        # Merchant Category Codes (MCC) - realistic categories
        self.mcc_categories = {
            5411: "Grocery Stores",
            5541: "Gas Stations",
            5812: "Restaurants",
            5912: "Drug Stores",
            5311: "Department Stores",
            5999: "Misc Retail",
            7011: "Hotels",
            4111: "Transportation",
            5732: "Electronics",
            5944: "Jewelry",
            5691: "Clothing",
            7832: "Entertainment",
            4814: "Telecom",
            5300: "Wholesale Clubs",
            6011: "ATM Cash Advance"
        }

        # US States
        self.states = [
            'CA', 'TX', 'FL', 'NY', 'PA', 'IL', 'OH', 'GA', 'NC', 'MI',
            'NJ', 'VA', 'WA', 'AZ', 'MA', 'TN', 'IN', 'MO', 'MD', 'WI',
            'CO', 'MN', 'SC', 'AL', 'LA', 'KY', 'OR', 'OK', 'CT', 'UT',
            'NV', 'AR', 'MS', 'KS', 'NM', 'NE', 'WV', 'ID', 'HI', 'NH',
            'ME', 'MT', 'RI', 'DE', 'SD', 'ND', 'AK', 'VT', 'WY', 'DC'
        ]

        # Transaction types
        self.transaction_types = ['Chip Transaction', 'Swipe Transaction', 'Online Transaction']

    def generate_user_profile(self, user_id):
        """Generate a user spending profile"""
        profile = {
            'user_id': user_id,
            'home_state': np.random.choice(self.states),
            'avg_transaction': np.random.lognormal(3.5, 0.8),  # ~$33 median
            'std_transaction': np.random.lognormal(2.5, 0.5),
            'preferred_mccs': np.random.choice(list(self.mcc_categories.keys()),
                                               size=np.random.randint(3, 8), replace=False),
            'active_hours': (np.random.randint(7, 12), np.random.randint(18, 23)),
            'transactions_per_day': np.random.poisson(3) + 1,
            'online_preference': np.random.beta(2, 5),  # Most users prefer in-person
        }
        return profile

    def generate_normal_transaction(self, user_profile, card_id, date):
        """Generate a normal (non-fraudulent) transaction"""

        # Time - usually during active hours
        active_start, active_end = user_profile['active_hours']
        if np.random.random() < 0.85:  # 85% during active hours
            hour = np.random.randint(active_start, active_end + 1)
        else:
            hour = np.random.randint(0, 24)
        minute = np.random.randint(0, 60)

        # Amount - follows user's spending pattern
        amount = max(1.0, np.random.normal(user_profile['avg_transaction'],
                                           user_profile['std_transaction']))

        # MCC - usually from preferred categories
        if np.random.random() < 0.8:
            mcc = np.random.choice(user_profile['preferred_mccs'])
        else:
            mcc = np.random.choice(list(self.mcc_categories.keys()))

        # Transaction type
        if np.random.random() < user_profile['online_preference']:
            trans_type = 'Online Transaction'
            state = ''
        else:
            trans_type = np.random.choice(['Chip Transaction', 'Swipe Transaction'],
                                          p=[0.7, 0.3])
            # Usually home state or nearby
            if np.random.random() < 0.85:
                state = user_profile['home_state']
            else:
                state = np.random.choice(self.states)

        return {
            'User': user_profile['user_id'],
            'Card': card_id,
            'Year': date.year,
            'Month': date.month,
            'Day': date.day,
            'Time': f"{hour:02d}:{minute:02d}",
            'Amount': f"${amount:.2f}",
            'Use Chip': trans_type,
            'Merchant Name': f"MERCHANT_{mcc}_{np.random.randint(1000, 9999)}",
            'Merchant City': f"CITY_{np.random.randint(1, 100)}",
            'Merchant State': state,
            'Zip': f"{np.random.randint(10000, 99999)}" if state else "",
            'MCC': mcc,
            'Errors?': '',
            'Is Fraud?': 'No'
        }

    def generate_fraud_transaction(self, user_profile, card_id, date, fraud_type='random'):
        """Generate a fraudulent transaction with specific patterns"""

        fraud_types = ['high_amount', 'unusual_time', 'geographic', 'rapid_succession',
                       'unusual_merchant', 'online_burst']

        if fraud_type == 'random':
            fraud_type = np.random.choice(fraud_types)

        # Start with normal transaction and modify based on fraud type
        trans = self.generate_normal_transaction(user_profile, card_id, date)
        trans['Is Fraud?'] = 'Yes'

        if fraud_type == 'high_amount':
            # Unusually high transaction amount
            amount = user_profile['avg_transaction'] * np.random.uniform(5, 20)
            trans['Amount'] = f"${amount:.2f}"
            trans['MCC'] = np.random.choice([5944, 5732, 5691])  # Jewelry, Electronics, Clothing

        elif fraud_type == 'unusual_time':
            # Transaction at unusual hour
            hour = np.random.choice([2, 3, 4, 5])  # Late night
            trans['Time'] = f"{hour:02d}:{np.random.randint(0, 60):02d}"

        elif fraud_type == 'geographic':
            # Transaction far from home
            other_states = [s for s in self.states if s != user_profile['home_state']]
            trans['Merchant State'] = np.random.choice(other_states)
            trans['Use Chip'] = 'Swipe Transaction'  # Often card is cloned

        elif fraud_type == 'rapid_succession':
            # Higher amount than normal
            amount = user_profile['avg_transaction'] * np.random.uniform(2, 5)
            trans['Amount'] = f"${amount:.2f}"

        elif fraud_type == 'unusual_merchant':
            # Purchase at unusual merchant category
            unusual_mccs = [mcc for mcc in self.mcc_categories.keys()
                          if mcc not in user_profile['preferred_mccs']]
            trans['MCC'] = np.random.choice(unusual_mccs)
            amount = user_profile['avg_transaction'] * np.random.uniform(3, 8)
            trans['Amount'] = f"${amount:.2f}"

        elif fraud_type == 'online_burst':
            # Online transaction burst (card compromise)
            trans['Use Chip'] = 'Online Transaction'
            trans['Merchant State'] = ''
            amount = user_profile['avg_transaction'] * np.random.uniform(2, 10)
            trans['Amount'] = f"${amount:.2f}"
            trans['MCC'] = np.random.choice([5732, 5944, 5691, 5999])

        return trans

    def generate_dataset(self, n_users=1000, days=365, output_path=None):
        """Generate a complete dataset with realistic fraud patterns"""

        print(f"Generating fraud detection training data...")
        print(f"  Users: {n_users}")
        print(f"  Days: {days}")
        print(f"  Target fraud rate: {self.fraud_rate * 100:.1f}%")

        transactions = []
        start_date = datetime(2024, 1, 1)

        # Generate user profiles
        user_profiles = {}
        for user_id in range(n_users):
            user_profiles[user_id] = self.generate_user_profile(user_id)
            # Each user has 1-3 cards
            user_profiles[user_id]['cards'] = list(range(np.random.randint(1, 4)))

        # Generate transactions for each day
        for day_offset in range(days):
            current_date = start_date + timedelta(days=day_offset)

            for user_id, profile in user_profiles.items():
                # Number of transactions for this user today
                n_transactions = np.random.poisson(profile['transactions_per_day'])

                for _ in range(n_transactions):
                    card_id = np.random.choice(profile['cards'])

                    # Decide if fraud
                    if np.random.random() < self.fraud_rate:
                        trans = self.generate_fraud_transaction(profile, card_id, current_date)
                    else:
                        trans = self.generate_normal_transaction(profile, card_id, current_date)

                    transactions.append(trans)

        # Create DataFrame
        df = pd.DataFrame(transactions)

        # Sort by user, card, and datetime
        df['datetime'] = pd.to_datetime(df[['Year', 'Month', 'Day']].assign(
            hour=df['Time'].str.split(':').str[0].astype(int),
            minute=df['Time'].str.split(':').str[1].astype(int)
        ))
        df = df.sort_values(['User', 'Card', 'datetime']).reset_index(drop=True)
        df = df.drop('datetime', axis=1)

        # Add rapid succession fraud (multiple transactions in short time)
        df = self._add_rapid_succession_fraud(df, user_profiles)

        # Statistics
        n_fraud = (df['Is Fraud?'] == 'Yes').sum()
        actual_fraud_rate = n_fraud / len(df)

        print(f"\nDataset Statistics:")
        print(f"  Total transactions: {len(df):,}")
        print(f"  Fraud transactions: {n_fraud:,} ({actual_fraud_rate*100:.2f}%)")
        print(f"  Normal transactions: {len(df) - n_fraud:,}")

        # Save if path provided
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"\nSaved to: {output_path}")

            # Save metadata
            metadata = {
                'generation_date': datetime.now().isoformat(),
                'n_users': n_users,
                'days': days,
                'total_transactions': len(df),
                'fraud_transactions': int(n_fraud),
                'fraud_rate': actual_fraud_rate,
                'date_range': {
                    'start': str(start_date.date()),
                    'end': str((start_date + timedelta(days=days-1)).date())
                }
            }
            metadata_path = output_path.replace('.csv', '_metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f"Metadata saved to: {metadata_path}")

        return df

    def _add_rapid_succession_fraud(self, df, user_profiles):
        """Add rapid succession fraud patterns (multiple transactions in short period)"""

        # For some fraud transactions, add a burst of related transactions
        fraud_indices = df[df['Is Fraud?'] == 'Yes'].index.tolist()

        burst_transactions = []
        for idx in fraud_indices[:int(len(fraud_indices) * 0.1)]:  # 10% get bursts
            base_trans = df.loc[idx]
            user_id = base_trans['User']

            # Add 2-5 more fraudulent transactions
            n_burst = np.random.randint(2, 6)
            for i in range(n_burst):
                burst_trans = base_trans.copy()

                # Slight time variation
                hour = int(base_trans['Time'].split(':')[0])
                minute = int(base_trans['Time'].split(':')[1]) + i + 1
                if minute >= 60:
                    hour = (hour + 1) % 24
                    minute = minute % 60
                burst_trans['Time'] = f"{hour:02d}:{minute:02d}"

                # Different amount
                amount = float(base_trans['Amount'].replace('$', '').replace(',', ''))
                amount = amount * np.random.uniform(0.5, 1.5)
                burst_trans['Amount'] = f"${amount:.2f}"

                burst_transactions.append(burst_trans)

        if burst_transactions:
            burst_df = pd.DataFrame(burst_transactions)
            df = pd.concat([df, burst_df], ignore_index=True)

        return df


def main():
    """Generate training dataset"""
    generator = FraudDataGenerator(seed=42)

    # Generate dataset
    output_path = 'data/fraud_training_data.csv'
    df = generator.generate_dataset(
        n_users=500,      # 500 users
        days=365,         # 1 year of data
        output_path=output_path
    )

    print("\nSample transactions:")
    print(df.head(10).to_string())

    print("\nFraud distribution by month:")
    df['Month_Year'] = df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2)
    monthly_fraud = df.groupby('Month_Year')['Is Fraud?'].apply(
        lambda x: (x == 'Yes').sum()
    )
    print(monthly_fraud)


if __name__ == "__main__":
    main()
