#!/usr/bin/env python3
"""
SDV-based Realistic Transaction Generator

Uses Synthetic Data Vault (SDV) to learn from the Kaggle Credit Card Fraud dataset
and generate statistically realistic transactions for real-time fraud detection evaluation.

Features:
- Learns statistical properties from real fraud data
- Generates realistic transactions preserving correlations
- Supports conditional generation (fraud vs legitimate)
- Real-time streaming mode for evaluation
- Integration with ML model endpoints

Usage:
    # First, train the synthesizer on Kaggle data
    python scripts/sdv_transaction_generator.py train --data data/creditcard_fraud.csv

    # Generate transactions
    python scripts/sdv_transaction_generator.py generate --count 100

    # Generate with specific fraud rate
    python scripts/sdv_transaction_generator.py generate --count 100 --fraud-rate 0.05

    # Stream transactions to ML model
    python scripts/sdv_transaction_generator.py stream --model-endpoint http://localhost:8080/predict

Requirements:
    pip install sdv pandas numpy
"""

import os
import sys
import json
import time
import pickle
import argparse
import warnings
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# SDV imports
try:
    from sdv.single_table import GaussianCopulaSynthesizer, CTGANSynthesizer
    from sdv.metadata import SingleTableMetadata
    SDV_AVAILABLE = True
except ImportError:
    SDV_AVAILABLE = False
    print("Warning: SDV not installed. Install with: pip install sdv")


MODEL_DIR = "models"
SYNTHESIZER_PATH = os.path.join(MODEL_DIR, "sdv_synthesizer.pkl")
METADATA_PATH = os.path.join(MODEL_DIR, "sdv_metadata.json")


class SDVTransactionGenerator:
    """
    Generate realistic credit card transactions using SDV.

    Learns statistical distributions and correlations from real fraud data
    to generate synthetic transactions that maintain realistic properties.
    """

    def __init__(self, synthesizer_path: str = SYNTHESIZER_PATH):
        self.synthesizer_path = synthesizer_path
        self.synthesizer = None
        self.metadata = None
        self.feature_columns = None
        self.fraud_synthesizer = None  # Separate model for fraud transactions
        self.legitimate_synthesizer = None  # Separate model for legitimate

    def train(self,
              data_path: str,
              model_type: str = 'gaussian_copula',
              epochs: int = 300,
              separate_fraud_model: bool = True) -> None:
        """
        Train the SDV synthesizer on the Kaggle dataset.

        Args:
            data_path: Path to creditcard_fraud.csv
            model_type: 'gaussian_copula' (fast) or 'ctgan' (better quality)
            epochs: Training epochs for CTGAN
            separate_fraud_model: Train separate models for fraud/legitimate
        """
        if not SDV_AVAILABLE:
            raise ImportError("SDV not installed. Run: pip install sdv")

        print(f"Loading data from: {data_path}")
        df = pd.read_csv(data_path)

        # Select features for synthesis
        # Use PCA features (V1-V28), Amount, and time features
        pca_cols = [f'V{i}' for i in range(1, 29)]
        feature_cols = pca_cols + ['Amount', 'Amount_log', 'hour', 'is_fraud']

        # Filter to available columns
        feature_cols = [c for c in feature_cols if c in df.columns]
        self.feature_columns = feature_cols

        df_train = df[feature_cols].copy()

        print(f"Training data: {len(df_train):,} rows, {len(feature_cols)} features")
        print(f"Fraud rate: {df_train['is_fraud'].mean()*100:.3f}%")

        # Create metadata
        print("\nCreating SDV metadata...")
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(df_train)

        # Set is_fraud as categorical
        metadata.update_column('is_fraud', sdtype='categorical')

        self.metadata = metadata

        if separate_fraud_model:
            print("\nTraining separate models for fraud and legitimate transactions...")

            # Split data
            df_fraud = df_train[df_train['is_fraud'] == 1].drop('is_fraud', axis=1)
            df_legit = df_train[df_train['is_fraud'] == 0].drop('is_fraud', axis=1)

            # Create metadata without is_fraud
            meta_no_fraud = SingleTableMetadata()
            meta_no_fraud.detect_from_dataframe(df_fraud)

            # Train fraud model
            print(f"\n  Training fraud model on {len(df_fraud):,} transactions...")
            if model_type == 'ctgan':
                self.fraud_synthesizer = CTGANSynthesizer(
                    meta_no_fraud,
                    epochs=epochs,
                    verbose=True
                )
            else:
                self.fraud_synthesizer = GaussianCopulaSynthesizer(meta_no_fraud)
            self.fraud_synthesizer.fit(df_fraud)

            # Train legitimate model (sample to speed up)
            legit_sample = df_legit.sample(n=min(50000, len(df_legit)), random_state=42)
            print(f"\n  Training legitimate model on {len(legit_sample):,} transactions...")
            if model_type == 'ctgan':
                self.legitimate_synthesizer = CTGANSynthesizer(
                    meta_no_fraud,
                    epochs=epochs // 2,  # Less epochs needed
                    verbose=True
                )
            else:
                self.legitimate_synthesizer = GaussianCopulaSynthesizer(meta_no_fraud)
            self.legitimate_synthesizer.fit(legit_sample)

        else:
            # Train single model
            print(f"\nTraining {model_type} synthesizer...")
            if model_type == 'ctgan':
                self.synthesizer = CTGANSynthesizer(
                    metadata,
                    epochs=epochs,
                    verbose=True
                )
            else:
                self.synthesizer = GaussianCopulaSynthesizer(metadata)

            self.synthesizer.fit(df_train)

        # Save models
        self._save_models()
        print("\nTraining complete!")

    def _save_models(self) -> None:
        """Save trained synthesizers to disk."""
        os.makedirs(MODEL_DIR, exist_ok=True)

        save_data = {
            'synthesizer': self.synthesizer,
            'fraud_synthesizer': self.fraud_synthesizer,
            'legitimate_synthesizer': self.legitimate_synthesizer,
            'feature_columns': self.feature_columns,
            'metadata': self.metadata
        }

        with open(self.synthesizer_path, 'wb') as f:
            pickle.dump(save_data, f)

        print(f"Saved synthesizer to: {self.synthesizer_path}")

        # Save metadata separately for inspection
        meta_info = {
            'feature_columns': self.feature_columns,
            'saved_at': datetime.now().isoformat(),
            'has_separate_models': self.fraud_synthesizer is not None
        }
        with open(METADATA_PATH, 'w') as f:
            json.dump(meta_info, f, indent=2)

    def load(self) -> bool:
        """Load trained synthesizer from disk."""
        if not os.path.exists(self.synthesizer_path):
            print(f"No saved model found at: {self.synthesizer_path}")
            print("Train first with: python scripts/sdv_transaction_generator.py train")
            return False

        print(f"Loading synthesizer from: {self.synthesizer_path}")
        with open(self.synthesizer_path, 'rb') as f:
            save_data = pickle.load(f)

        self.synthesizer = save_data.get('synthesizer')
        self.fraud_synthesizer = save_data.get('fraud_synthesizer')
        self.legitimate_synthesizer = save_data.get('legitimate_synthesizer')
        self.feature_columns = save_data.get('feature_columns')
        self.metadata = save_data.get('metadata')

        print("Synthesizer loaded successfully!")
        return True

    def generate(self,
                 n_samples: int = 100,
                 fraud_rate: float = None,
                 fraud_only: bool = False,
                 legitimate_only: bool = False) -> pd.DataFrame:
        """
        Generate synthetic transactions.

        Args:
            n_samples: Number of transactions to generate
            fraud_rate: Desired fraud rate (0.0-1.0). If None, uses natural rate
            fraud_only: Generate only fraud transactions
            legitimate_only: Generate only legitimate transactions

        Returns:
            DataFrame of synthetic transactions
        """
        if self.fraud_synthesizer is not None and self.legitimate_synthesizer is not None:
            # Use separate models
            return self._generate_separate(n_samples, fraud_rate, fraud_only, legitimate_only)
        elif self.synthesizer is not None:
            # Use combined model
            return self._generate_combined(n_samples)
        else:
            raise RuntimeError("No synthesizer loaded. Call load() or train() first.")

    def _generate_separate(self,
                           n_samples: int,
                           fraud_rate: float = None,
                           fraud_only: bool = False,
                           legitimate_only: bool = False) -> pd.DataFrame:
        """Generate using separate fraud/legitimate models."""
        if fraud_only:
            df = self.fraud_synthesizer.sample(n_samples)
            df['is_fraud'] = 1
        elif legitimate_only:
            df = self.legitimate_synthesizer.sample(n_samples)
            df['is_fraud'] = 0
        else:
            # Determine fraud count
            if fraud_rate is None:
                fraud_rate = 0.00172  # Natural rate from Kaggle dataset

            n_fraud = int(n_samples * fraud_rate)
            n_legit = n_samples - n_fraud

            # Generate from each model
            df_fraud = self.fraud_synthesizer.sample(max(1, n_fraud))
            df_fraud['is_fraud'] = 1

            df_legit = self.legitimate_synthesizer.sample(n_legit)
            df_legit['is_fraud'] = 0

            # Combine and shuffle
            df = pd.concat([df_fraud, df_legit], ignore_index=True)
            df = df.sample(frac=1).reset_index(drop=True)

        # Add transaction metadata
        df = self._add_transaction_metadata(df)

        return df

    def _generate_combined(self, n_samples: int) -> pd.DataFrame:
        """Generate using combined model."""
        df = self.synthesizer.sample(n_samples)
        df = self._add_transaction_metadata(df)
        return df

    def _add_transaction_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add transaction IDs and timestamps."""
        now = datetime.now()

        # Add transaction IDs
        df['transaction_id'] = [
            f"TXN_{now.strftime('%Y%m%d%H%M%S')}_{i:06d}"
            for i in range(len(df))
        ]

        # Add timestamps
        df['timestamp'] = now.isoformat()

        # Ensure Amount is positive
        if 'Amount' in df.columns:
            df['Amount'] = df['Amount'].abs()

        # Recalculate Amount_log if needed
        if 'Amount_log' in df.columns and 'Amount' in df.columns:
            df['Amount_log'] = np.log1p(df['Amount'])

        return df

    def generate_single(self, is_fraud: bool = False) -> Dict:
        """
        Generate a single transaction as a dictionary.

        Args:
            is_fraud: Whether to generate a fraud transaction

        Returns:
            Transaction dictionary
        """
        df = self.generate(
            n_samples=1,
            fraud_only=is_fraud,
            legitimate_only=not is_fraud
        )

        return df.iloc[0].to_dict()

    def stream_transactions(self,
                           model_endpoint: str = None,
                           nifi_endpoint: str = None,
                           rate: float = 1.0,
                           fraud_rate: float = 0.01,
                           duration: int = None,
                           verbose: bool = True) -> None:
        """
        Stream generated transactions to model/NiFi endpoint.

        Args:
            model_endpoint: ML model API endpoint
            nifi_endpoint: NiFi HTTP input endpoint
            rate: Transactions per second
            fraud_rate: Rate of fraudulent transactions
            duration: Duration in seconds (None = infinite)
            verbose: Print transaction details
        """
        import requests

        print(f"\nStreaming transactions at {rate} TPS")
        print(f"Fraud rate: {fraud_rate*100:.2f}%")
        if model_endpoint:
            print(f"Model endpoint: {model_endpoint}")
        if nifi_endpoint:
            print(f"NiFi endpoint: {nifi_endpoint}")
        print("\nPress Ctrl+C to stop\n")

        start_time = time.time()
        count = 0
        fraud_count = 0
        delay = 1.0 / rate

        try:
            while True:
                # Check duration
                if duration and (time.time() - start_time) > duration:
                    break

                # Generate transaction
                is_fraud = np.random.random() < fraud_rate
                trans = self.generate_single(is_fraud=is_fraud)

                count += 1
                if is_fraud:
                    fraud_count += 1

                # Send to endpoints
                result = None
                if model_endpoint:
                    try:
                        response = requests.post(
                            model_endpoint,
                            json=self._prepare_model_input(trans),
                            headers={'Content-Type': 'application/json'},
                            timeout=10
                        )
                        result = response.json() if response.ok else None
                    except Exception as e:
                        if verbose:
                            print(f"Model error: {e}")

                if nifi_endpoint:
                    try:
                        requests.post(
                            nifi_endpoint,
                            json=trans,
                            headers={'Content-Type': 'application/json'},
                            timeout=10
                        )
                    except Exception as e:
                        if verbose:
                            print(f"NiFi error: {e}")

                # Display
                if verbose:
                    fraud_label = "FRAUD" if is_fraud else "LEGIT"
                    amount = trans.get('Amount', 0)

                    if result:
                        prob = result.get('fraud_probability', 0)
                        rating = result.get('rating', 'N/A')
                        print(f"[{count:5d}] {fraud_label:5s} ${amount:>10.2f} "
                              f"-> Prob: {prob:.4f} Rating: {rating}")
                    else:
                        print(f"[{count:5d}] {fraud_label:5s} ${amount:>10.2f}")

                time.sleep(delay)

        except KeyboardInterrupt:
            print("\n\nStopped by user")

        # Summary
        elapsed = time.time() - start_time
        print(f"\n{'='*50}")
        print(f"Summary:")
        print(f"  Transactions: {count}")
        print(f"  Fraud: {fraud_count} ({fraud_count/count*100:.2f}%)")
        print(f"  Duration: {elapsed:.1f}s")
        print(f"  Actual TPS: {count/elapsed:.2f}")

    def _prepare_model_input(self, trans: Dict) -> Dict:
        """Prepare transaction for ML model input."""
        # Extract PCA features and Amount
        model_input = {}

        for key, value in trans.items():
            if key.startswith('V') or key in ['Amount', 'Amount_log', 'hour']:
                if isinstance(value, (int, float)) and not np.isnan(value):
                    model_input[key] = float(value)

        return model_input


def evaluate_synthesizer(generator: SDVTransactionGenerator,
                         original_data_path: str,
                         n_samples: int = 1000) -> Dict:
    """
    Evaluate quality of synthetic data vs original.

    Returns metrics comparing distributions.
    """
    print("\nEvaluating synthesizer quality...")

    # Load original
    df_original = pd.read_csv(original_data_path)

    # Generate synthetic
    df_synthetic = generator.generate(n_samples=n_samples)

    # Compare statistics
    metrics = {}

    # Compare Amount distribution
    if 'Amount' in df_original.columns and 'Amount' in df_synthetic.columns:
        metrics['amount'] = {
            'original_mean': float(df_original['Amount'].mean()),
            'synthetic_mean': float(df_synthetic['Amount'].mean()),
            'original_std': float(df_original['Amount'].std()),
            'synthetic_std': float(df_synthetic['Amount'].std()),
        }

    # Compare V1-V5 distributions (sample)
    for v in ['V1', 'V2', 'V3', 'V4', 'V5']:
        if v in df_original.columns and v in df_synthetic.columns:
            metrics[v] = {
                'original_mean': float(df_original[v].mean()),
                'synthetic_mean': float(df_synthetic[v].mean()),
                'mean_diff': abs(df_original[v].mean() - df_synthetic[v].mean())
            }

    # Fraud rate comparison
    if 'is_fraud' in df_original.columns and 'is_fraud' in df_synthetic.columns:
        metrics['fraud_rate'] = {
            'original': float(df_original['is_fraud'].mean()),
            'synthetic': float(df_synthetic['is_fraud'].mean())
        }

    print("\nEvaluation Results:")
    print(json.dumps(metrics, indent=2))

    return metrics


def main():
    parser = argparse.ArgumentParser(
        description='SDV-based Realistic Transaction Generator'
    )

    subparsers = parser.add_subparsers(dest='command', help='Commands')

    # Train command
    train_parser = subparsers.add_parser('train', help='Train the synthesizer')
    train_parser.add_argument('--data', '-d', type=str, default='data/creditcard_fraud.csv',
                             help='Path to training data')
    train_parser.add_argument('--model-type', '-m', type=str,
                             choices=['gaussian_copula', 'ctgan'],
                             default='gaussian_copula',
                             help='SDV model type')
    train_parser.add_argument('--epochs', '-e', type=int, default=300,
                             help='Training epochs for CTGAN')
    train_parser.add_argument('--single-model', action='store_true',
                             help='Train single model instead of separate fraud/legit')

    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate transactions')
    gen_parser.add_argument('--count', '-c', type=int, default=100,
                           help='Number of transactions')
    gen_parser.add_argument('--fraud-rate', '-f', type=float, default=None,
                           help='Fraud rate (0.0-1.0)')
    gen_parser.add_argument('--fraud-only', action='store_true',
                           help='Generate only fraud transactions')
    gen_parser.add_argument('--legit-only', action='store_true',
                           help='Generate only legitimate transactions')
    gen_parser.add_argument('--output', '-o', type=str,
                           help='Output CSV file')

    # Stream command
    stream_parser = subparsers.add_parser('stream', help='Stream transactions')
    stream_parser.add_argument('--model-endpoint', '-m', type=str,
                              help='ML model API endpoint')
    stream_parser.add_argument('--nifi-endpoint', '-n', type=str,
                              help='NiFi HTTP input endpoint')
    stream_parser.add_argument('--rate', '-r', type=float, default=1.0,
                              help='Transactions per second')
    stream_parser.add_argument('--fraud-rate', '-f', type=float, default=0.01,
                              help='Fraud rate for streaming')
    stream_parser.add_argument('--duration', '-d', type=int, default=None,
                              help='Duration in seconds')
    stream_parser.add_argument('--quiet', '-q', action='store_true',
                              help='Reduce output verbosity')

    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate synthesizer')
    eval_parser.add_argument('--data', '-d', type=str, default='data/creditcard_fraud.csv',
                            help='Path to original data for comparison')
    eval_parser.add_argument('--samples', '-s', type=int, default=1000,
                            help='Number of samples to generate')

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    generator = SDVTransactionGenerator()

    if args.command == 'train':
        print("="*60)
        print("  SDV Transaction Generator - Training")
        print("="*60)

        if not SDV_AVAILABLE:
            print("\nERROR: SDV not installed.")
            print("Install with: pip install sdv")
            sys.exit(1)

        generator.train(
            data_path=args.data,
            model_type=args.model_type,
            epochs=args.epochs,
            separate_fraud_model=not args.single_model
        )

    elif args.command == 'generate':
        print("="*60)
        print("  SDV Transaction Generator - Generate")
        print("="*60)

        if not generator.load():
            sys.exit(1)

        df = generator.generate(
            n_samples=args.count,
            fraud_rate=args.fraud_rate,
            fraud_only=args.fraud_only,
            legitimate_only=args.legit_only
        )

        print(f"\nGenerated {len(df)} transactions")
        print(f"Fraud: {df['is_fraud'].sum()} ({df['is_fraud'].mean()*100:.2f}%)")

        # Show sample
        print("\nSample transactions:")
        display_cols = ['transaction_id', 'Amount', 'is_fraud']
        if 'V1' in df.columns:
            display_cols.insert(2, 'V1')
        print(df[display_cols].head(10).to_string())

        if args.output:
            df.to_csv(args.output, index=False)
            print(f"\nSaved to: {args.output}")

    elif args.command == 'stream':
        print("="*60)
        print("  SDV Transaction Generator - Streaming")
        print("="*60)

        if not generator.load():
            sys.exit(1)

        generator.stream_transactions(
            model_endpoint=args.model_endpoint,
            nifi_endpoint=args.nifi_endpoint,
            rate=args.rate,
            fraud_rate=args.fraud_rate,
            duration=args.duration,
            verbose=not args.quiet
        )

    elif args.command == 'evaluate':
        print("="*60)
        print("  SDV Transaction Generator - Evaluation")
        print("="*60)

        if not generator.load():
            sys.exit(1)

        evaluate_synthesizer(generator, args.data, args.samples)


if __name__ == "__main__":
    main()
