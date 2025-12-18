#!/usr/bin/env python3
"""
Kaggle Credit Card Fraud Dataset Downloader and Processor
Downloads the famous Credit Card Fraud Detection dataset from Kaggle.

Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- 284,807 transactions
- 492 frauds (0.172% fraud rate)
- Features V1-V28 (PCA transformed), Time, Amount
- Highly imbalanced dataset

Requirements:
    pip install kaggle pyyaml

Setup (YAML config - recommended):
    1. Create Kaggle account at https://www.kaggle.com
    2. Go to Account Settings -> API -> Create New Token
    3. Create config/kaggle.yaml with your credentials:

       username: your_kaggle_username
       key: your_kaggle_api_key

    4. chmod 600 config/kaggle.yaml

Usage:
    python scripts/download_kaggle_data.py

    # With custom config path:
    python scripts/download_kaggle_data.py --config path/to/kaggle.yaml

    # Or with manual download:
    python scripts/download_kaggle_data.py --manual path/to/creditcard.csv
"""

import os
import sys
import subprocess
import json


def load_kaggle_credentials_early():
    """
    Load Kaggle credentials BEFORE importing kaggle package.
    The kaggle package auto-authenticates on import, so we must set
    environment variables first.

    Supports both JSON and YAML formats.
    """
    # Get the script's directory and project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)

    # Search paths - supports BOTH json and yaml
    search_paths = [
        # Project root - JSON
        os.path.join(project_dir, "kaggle.json"),
        # Project root - YAML
        os.path.join(project_dir, "kaggle.yaml"),
        os.path.join(project_dir, "kaggle.yml"),
        # Cloudera CML location
        os.path.expanduser("~/.config/kaggle/kaggle.json"),
        # Standard Kaggle location
        os.path.expanduser("~/.kaggle/kaggle.json"),
        # Config subfolder - JSON and YAML
        os.path.join(project_dir, "config", "kaggle.json"),
        os.path.join(project_dir, "config", "kaggle.yaml"),
        os.path.join(project_dir, "config", "kaggle.yml"),
    ]

    for path in search_paths:
        if os.path.exists(path):
            try:
                with open(path, 'r') as f:
                    content = f.read()

                # Parse based on extension
                if path.endswith('.json'):
                    config = json.loads(content)
                else:
                    # YAML - simple key: value parsing without importing yaml
                    config = {}
                    for line in content.split('\n'):
                        line = line.strip()
                        if ':' in line and not line.startswith('#'):
                            key, val = line.split(':', 1)
                            config[key.strip()] = val.strip()

                username = config.get('username')
                key = config.get('key')
                if username and key:
                    os.environ['KAGGLE_USERNAME'] = username
                    os.environ['KAGGLE_KEY'] = key
                    print(f"Loaded Kaggle credentials from: {path}")
                    return True
            except Exception as e:
                print(f"Warning: Failed to load {path}: {e}")

    # No credentials found - print helpful message
    print("WARNING: No kaggle credentials found. Searched:")
    for path in search_paths:
        exists = "EXISTS" if os.path.exists(path) else "not found"
        print(f"  - {path} ({exists})")
    print(f"\nCreate one of these files with your Kaggle credentials:")
    print(f"  JSON: {os.path.join(project_dir, 'kaggle.json')}")
    print(f"  YAML: {os.path.join(project_dir, 'kaggle.yaml')}")
    return False


def check_environment():
    """Check if running in proper environment and setup if needed."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(script_dir)
    venv_dir = os.path.join(project_dir, "venv")

    # Check if we're in a virtual environment
    in_venv = hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)

    if not in_venv and os.path.exists(venv_dir):
        print("Virtual environment exists but not activated.")
        print(f"Please activate it first:")
        print(f"  source {venv_dir}/bin/activate")
        print(f"  python scripts/download_kaggle_data.py")
        sys.exit(1)

    # Auto-install required packages (but DON'T import kaggle yet - it auto-authenticates!)
    required_packages = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'yaml': 'pyyaml',
        'sklearn': 'scikit-learn',
    }

    missing_packages = []
    for module, package in required_packages.items():
        try:
            __import__(module)
        except ImportError:
            missing_packages.append(package)

    # Check kaggle separately without importing
    try:
        import importlib.util
        if importlib.util.find_spec('kaggle') is None:
            missing_packages.append('kaggle')
    except:
        missing_packages.append('kaggle')

    if missing_packages:
        print(f"Installing missing packages: {', '.join(missing_packages)}")
        for package in missing_packages:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
        print("Packages installed successfully.\n")


# Load credentials FIRST (before kaggle import)
load_kaggle_credentials_early()

# Run environment check
check_environment()

import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json
import zipfile
import shutil
import yaml

YAML_AVAILABLE = True


DATASET_NAME = "mlg-ulb/creditcardfraud"
OUTPUT_DIR = "data"
OUTPUT_FILE = "creditcard_fraud.csv"

# Config file search paths (JSON and YAML supported)
DEFAULT_CONFIG_PATHS = [
    # Project root (simplest - just drop kaggle.json in the project folder)
    "kaggle.json",
    # Standard Kaggle JSON location
    os.path.expanduser("~/.kaggle/kaggle.json"),
    # Config subfolder
    "config/kaggle.json",
    "config/kaggle.yaml",
]


def load_kaggle_credentials(config_path: str = None) -> bool:
    """
    Load Kaggle credentials from config file (JSON or YAML).

    Supports:
    - Standard Kaggle JSON: ~/.kaggle/kaggle.json
    - Project YAML: config/kaggle.yaml

    Sets KAGGLE_USERNAME and KAGGLE_KEY environment variables
    which the Kaggle API will use for authentication.

    Args:
        config_path: Path to config file. If None, searches default locations.

    Returns:
        True if credentials were loaded successfully
    """
    # Find config file
    if config_path:
        paths_to_check = [config_path]
    else:
        paths_to_check = DEFAULT_CONFIG_PATHS

    config_file = None
    for path in paths_to_check:
        if os.path.exists(path):
            config_file = path
            break

    if not config_file:
        print("No Kaggle config found. Searched locations:")
        for path in paths_to_check:
            print(f"  - {path}")
        print("\nOption 1: Use standard Kaggle setup:")
        print("  kaggle.com -> Settings -> API -> Create New Token")
        print("  This downloads kaggle.json to ~/.kaggle/")
        print("\nOption 2: Create config/kaggle.yaml with:")
        print("  username: your_kaggle_username")
        print("  key: your_kaggle_api_key")
        print("\nFalling back to standard Kaggle authentication...")
        return False

    try:
        print(f"Loading Kaggle credentials from: {config_file}")
        with open(config_file, 'r') as f:
            # Detect format by extension
            if config_file.endswith('.json'):
                config = json.load(f)
            else:
                config = yaml.safe_load(f)

        username = config.get('username')
        key = config.get('key')

        if not username or not key:
            print(f"Error: Config must contain 'username' and 'key' fields")
            print(f"Found keys: {list(config.keys())}")
            return False

        # Set environment variables for Kaggle API
        os.environ['KAGGLE_USERNAME'] = username
        os.environ['KAGGLE_KEY'] = key

        print(f"Loaded credentials for user: {username}")
        return True

    except json.JSONDecodeError as e:
        print(f"Error parsing JSON config: {e}")
        return False
    except yaml.YAMLError as e:
        print(f"Error parsing YAML config: {e}")
        return False
    except Exception as e:
        print(f"Error loading config: {e}")
        return False


def download_from_kaggle(output_dir: str, config_path: str = None) -> str:
    """
    Download the Credit Card Fraud dataset from Kaggle.

    Args:
        output_dir: Directory to save the dataset
        config_path: Path to YAML config file with Kaggle credentials

    Returns:
        Path to the downloaded CSV file
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Install with: pip install kaggle")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        sys.exit(1)

    # Load credentials from YAML config
    load_kaggle_credentials(config_path)

    print(f"\nDownloading dataset: {DATASET_NAME}")
    print("This may take a few minutes...")

    try:
        api = KaggleApi()
        api.authenticate()

        # Download dataset
        api.dataset_download_files(
            DATASET_NAME,
            path=output_dir,
            unzip=True
        )

        csv_path = os.path.join(output_dir, "creditcard.csv")

        if os.path.exists(csv_path):
            print(f"Downloaded to: {csv_path}")
            return csv_path
        else:
            raise FileNotFoundError("Download completed but CSV not found")

    except Exception as e:
        print(f"ERROR: Failed to download from Kaggle: {e}")
        print("\nPlease download manually:")
        print("1. Go to https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("2. Click 'Download' button")
        print("3. Extract creditcard.csv")
        print(f"4. Run: python {__file__} --manual path/to/creditcard.csv")
        sys.exit(1)


def load_and_process_data(csv_path: str) -> pd.DataFrame:
    """
    Load and process the Kaggle Credit Card Fraud dataset.

    Args:
        csv_path: Path to creditcard.csv

    Returns:
        Processed DataFrame
    """
    print(f"\nLoading data from: {csv_path}")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df):,} transactions")

    # Dataset info
    print("\nDataset Statistics:")
    print(f"  Total transactions: {len(df):,}")
    print(f"  Fraud transactions: {df['Class'].sum():,}")
    print(f"  Fraud rate: {df['Class'].mean()*100:.3f}%")
    print(f"  Features: {len(df.columns)}")

    # Rename columns for consistency
    df = df.rename(columns={'Class': 'is_fraud'})

    # Add derived features
    print("\nAdding derived features...")

    # Convert Time (seconds from first transaction) to hour of day (cyclic)
    # Assuming Time wraps every 24 hours (86400 seconds)
    df['hour'] = (df['Time'] % 86400) / 3600
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

    # Day indicator (assuming 2 days in dataset based on Time range)
    df['day'] = (df['Time'] // 86400).astype(int)

    # Amount features
    df['Amount_log'] = np.log1p(df['Amount'])
    df['Amount_scaled'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()

    # Amount percentiles
    df['Amount_percentile'] = df['Amount'].rank(pct=True)

    # High amount indicator
    df['is_high_amount'] = (df['Amount'] > df['Amount'].quantile(0.95)).astype(int)

    # Interaction features (selected V features with Amount)
    df['V1_Amount'] = df['V1'] * df['Amount_log']
    df['V2_Amount'] = df['V2'] * df['Amount_log']
    df['V4_Amount'] = df['V4'] * df['Amount_log']

    print(f"  Added {len(df.columns) - 31} new features")
    print(f"  Total features: {len(df.columns)}")

    return df


def create_train_test_split(df: pd.DataFrame, output_dir: str):
    """
    Create train/test split preserving temporal order.

    Args:
        df: Processed DataFrame
        output_dir: Directory to save splits
    """
    from sklearn.model_selection import train_test_split

    print("\nCreating train/test splits...")

    # Sort by time to maintain temporal order
    df = df.sort_values('Time').reset_index(drop=True)

    # Use last 20% as test (temporal split)
    split_idx = int(len(df) * 0.8)

    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    print(f"  Train set: {len(train_df):,} ({train_df['is_fraud'].sum():,} fraud)")
    print(f"  Test set: {len(test_df):,} ({test_df['is_fraud'].sum():,} fraud)")

    # Save splits
    train_path = os.path.join(output_dir, "train.csv")
    test_path = os.path.join(output_dir, "test.csv")

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\n  Saved: {train_path}")
    print(f"  Saved: {test_path}")

    return train_df, test_df


def create_metadata(df: pd.DataFrame, output_dir: str):
    """
    Create dataset metadata file.
    """
    # Feature groups
    pca_features = [f'V{i}' for i in range(1, 29)]
    time_features = ['Time', 'hour', 'hour_sin', 'hour_cos', 'day']
    amount_features = ['Amount', 'Amount_log', 'Amount_scaled', 'Amount_percentile', 'is_high_amount']
    interaction_features = ['V1_Amount', 'V2_Amount', 'V4_Amount']

    # All features for model (excluding target and raw Time)
    model_features = pca_features + ['Amount_log', 'Amount_scaled', 'hour_sin', 'hour_cos',
                                      'is_high_amount'] + interaction_features

    metadata = {
        "dataset": "Kaggle Credit Card Fraud Detection",
        "source": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
        "description": "Transactions made by European cardholders in September 2013",
        "downloaded_at": datetime.now().isoformat(),
        "statistics": {
            "total_transactions": len(df),
            "fraud_transactions": int(df['is_fraud'].sum()),
            "fraud_rate": float(df['is_fraud'].mean()),
            "amount_min": float(df['Amount'].min()),
            "amount_max": float(df['Amount'].max()),
            "amount_mean": float(df['Amount'].mean()),
            "time_span_hours": float(df['Time'].max() / 3600)
        },
        "features": {
            "pca_features": pca_features,
            "time_features": time_features,
            "amount_features": amount_features,
            "interaction_features": interaction_features,
            "model_features": model_features,
            "target": "is_fraud"
        },
        "notes": [
            "V1-V28 are PCA-transformed features (original features anonymized)",
            "Time is seconds elapsed from first transaction",
            "Amount is transaction amount",
            "Class (renamed to is_fraud) is 1 for fraud, 0 for legitimate"
        ]
    }

    metadata_path = os.path.join(output_dir, "dataset_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Saved metadata: {metadata_path}")

    # Also save feature list
    features_path = os.path.join(output_dir, "model_features.txt")
    with open(features_path, 'w') as f:
        f.write('\n'.join(model_features))

    print(f"  Saved feature list: {features_path}")

    return metadata


def main():
    parser = argparse.ArgumentParser(
        description='Download and process Kaggle Credit Card Fraud dataset'
    )
    parser.add_argument(
        '--manual', '-m',
        type=str,
        help='Path to manually downloaded creditcard.csv'
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to Kaggle YAML config file (default: config/kaggle.yaml)'
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=OUTPUT_DIR,
        help=f'Output directory (default: {OUTPUT_DIR})'
    )
    parser.add_argument(
        '--skip-split',
        action='store_true',
        help='Skip creating train/test split'
    )

    args = parser.parse_args()

    print("="*60)
    print("  Kaggle Credit Card Fraud Dataset")
    print("="*60)

    # Create output directory
    os.makedirs(args.output, exist_ok=True)

    # Get CSV path
    if args.manual:
        if not os.path.exists(args.manual):
            print(f"ERROR: File not found: {args.manual}")
            sys.exit(1)
        csv_path = args.manual
    else:
        csv_path = download_from_kaggle(args.output, config_path=args.config)

    # Load and process
    df = load_and_process_data(csv_path)

    # Save processed data
    processed_path = os.path.join(args.output, OUTPUT_FILE)
    df.to_csv(processed_path, index=False)
    print(f"\nSaved processed data: {processed_path}")

    # Create train/test split
    if not args.skip_split:
        create_train_test_split(df, args.output)

    # Create metadata
    create_metadata(df, args.output)

    print("\n" + "="*60)
    print("  Download and processing complete!")
    print("="*60)
    print(f"\nNext steps:")
    print(f"  1. Train model: python scripts/train_kaggle_model.py")
    print(f"  2. Or use notebook for exploration")


if __name__ == "__main__":
    main()
