#!/usr/bin/env python3
"""
End-to-End Demo for Fraud Detection System
Demonstrates the complete workflow:
1. Generate training data
2. Train LightGBM model with batch + real-time features
3. Test the model locally
4. Simulate NiFi transaction flow
5. Rate transactions

Usage:
    python demo_fraud_detection.py [--mode full|quick|test]
"""

import os
import sys
import json
import argparse
import time
from datetime import datetime

# Add scripts to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def print_step(step_num, description):
    """Print a step indicator"""
    print(f"\n[Step {step_num}] {description}")
    print("-" * 50)


def demo_data_generation(n_users=100, days=90):
    """Demo: Generate synthetic training data"""
    print_step(1, "Generating Synthetic Training Data")

    from scripts.generate_training_data import FraudDataGenerator

    generator = FraudDataGenerator(seed=42)
    df = generator.generate_dataset(
        n_users=n_users,
        days=days,
        output_path='data/demo_training_data.csv'
    )

    print(f"\nSample transactions:")
    print(df.head(5)[['User', 'Card', 'Time', 'Amount', 'Use Chip', 'MCC', 'Is Fraud?']].to_string(index=False))

    return df


def demo_feature_engineering(df):
    """Demo: Feature engineering with batch and real-time features"""
    print_step(2, "Feature Engineering (Batch + Real-time)")

    from scripts.features.feature_pipeline import FeaturePipeline

    pipeline = FeaturePipeline(window_size=30)

    print("Fitting pipeline on historical data...")
    pipeline.fit(df)

    print("\nBatch Features (pre-computed):")
    print("  - User spending profiles (avg, std, median)")
    print("  - Merchant risk profiles")
    print("  - MCC and State profiles")

    print("\nReal-time Features (computed at transaction time):")
    print("  - Transaction velocity (1h, 24h)")
    print("  - Rolling amount statistics")
    print("  - Deviation from historical patterns")

    # Transform a sample of data
    print("\nTransforming sample data...")
    sample_df = df.head(1000)
    features_df = pipeline.transform_batch(sample_df, include_target=True)

    print(f"\nTotal features: {len(pipeline.get_feature_names())}")
    print(f"Sample features:")
    feature_sample = features_df.iloc[0]
    for col in list(features_df.columns)[:10]:
        print(f"  {col}: {feature_sample[col]:.4f}")

    return pipeline, features_df


def demo_model_training(features_df, pipeline):
    """Demo: Train LightGBM model"""
    print_step(3, "Training LightGBM Model")

    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_auc_score, classification_report
    import numpy as np
    import joblib

    feature_names = pipeline.get_feature_names()
    X = features_df[feature_names]
    y = features_df['is_fraud']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training samples: {len(X_train)} ({y_train.sum()} fraud)")
    print(f"Test samples: {len(X_test)} ({y_test.sum()} fraud)")

    # Handle class imbalance
    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1

    # Train model
    params = {
        'objective': 'binary',
        'metric': ['auc', 'binary_logloss'],
        'boosting_type': 'gbdt',
        'num_leaves': 15,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'max_depth': 5,
        'min_child_samples': 20,
        'scale_pos_weight': scale_pos_weight,
        'random_state': 42,
        'verbose': -1
    }

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    print("\nTraining model...")
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        valid_names=['test'],
        num_boost_round=200,
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
    )

    # Evaluate
    predictions = model.predict(X_test)
    auc_score = roc_auc_score(y_test, predictions)
    y_pred = (predictions > 0.5).astype(int)

    print(f"\nTest AUC-ROC: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal', 'Fraud']))

    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/demo_fraud_model.pkl')

    # Save metadata
    metadata = {
        'train_date': datetime.now().isoformat(),
        'features': feature_names,
        'optimal_threshold': 0.5,
        'performance': {'test_auc': float(auc_score)}
    }
    with open('models/demo_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\nModel saved to models/demo_fraud_model.pkl")

    return model


def demo_transaction_rating():
    """Demo: Transaction rating system"""
    print_step(4, "Transaction Rating System")

    from scripts.transaction_rating import TransactionRatingEngine

    engine = TransactionRatingEngine()

    # Test transactions
    test_cases = [
        {
            'name': 'Normal Purchase',
            'transaction': {
                'User': 1, 'Card': 0,
                'Year': 2024, 'Month': 12, 'Day': 18,
                'Time': '14:30',
                'Amount': '$45.00',
                'Use Chip': 'Chip Transaction',
                'Merchant State': 'CA',
                'MCC': 5411
            },
            'fraud_probability': 0.02
        },
        {
            'name': 'Late Night ATM',
            'transaction': {
                'User': 1, 'Card': 0,
                'Year': 2024, 'Month': 12, 'Day': 18,
                'Time': '03:15',
                'Amount': '$500.00',
                'Use Chip': 'Swipe Transaction',
                'Merchant State': 'NV',
                'MCC': 6011
            },
            'fraud_probability': 0.65
        },
        {
            'name': 'High-Value Online',
            'transaction': {
                'User': 1, 'Card': 0,
                'Year': 2024, 'Month': 12, 'Day': 18,
                'Time': '22:45',
                'Amount': '$3500.00',
                'Use Chip': 'Online Transaction',
                'Merchant State': '',
                'MCC': 5944
            },
            'fraud_probability': 0.85
        }
    ]

    print("Rating Sample Transactions:\n")

    for test in test_cases:
        result = engine.rate_transaction(
            test['transaction'],
            test['fraud_probability']
        )

        print(f"Transaction: {test['name']}")
        print(f"  Amount: {test['transaction']['Amount']}")
        print(f"  Type: {test['transaction']['Use Chip']}")
        print(f"  Time: {test['transaction']['Time']}")
        print(f"  Fraud Probability: {test['fraud_probability']:.0%}")
        print(f"  Rating: {result['transaction_rating']} (Score: {result['rating_score']:.1f})")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Action: {result['recommendation']['action']}")
        print()


def demo_nifi_simulation():
    """Demo: Simulate NiFi transaction flow"""
    print_step(5, "NiFi Transaction Flow Simulation")

    from nifi.transaction_generator import TransactionGenerator

    generator = TransactionGenerator()

    print("Simulating NiFi transaction pipeline:\n")
    print("  GenerateFlowFile --> ExtractAttributes --> InvokeHTTP (ML) --> RouteOnAttribute\n")

    # Generate and process transactions
    transactions = []
    for i in range(5):
        if i % 3 == 0:  # Every 3rd transaction is suspicious
            trans = generator.generate_suspicious_transaction(fraud_type='random')
            trans_type = "SUSPICIOUS"
        else:
            trans = generator.generate_normal_transaction()
            trans_type = "NORMAL"

        transactions.append((trans, trans_type))

        # Simulate processing
        print(f"Transaction {i+1}: {trans_type}")
        print(f"  User: {trans['User']}, Amount: {trans['Amount']}")
        print(f"  Type: {trans['Use Chip']}")
        print(f"  ID: {trans['transaction_id']}")
        print()

        time.sleep(0.5)

    print("NiFi flow simulation complete!")
    print(f"Processed {len(transactions)} transactions")

    return transactions


def demo_full_inference(df, pipeline, model):
    """Demo: Full inference pipeline"""
    print_step(6, "Full Inference Pipeline")

    from scripts.transaction_rating import TransactionRatingEngine
    import numpy as np

    rating_engine = TransactionRatingEngine()

    # Take a few random transactions
    sample_indices = np.random.choice(len(df), size=5, replace=False)

    print("Processing live transactions:\n")

    for idx in sample_indices:
        row = df.iloc[idx]
        transaction = row.to_dict()

        # Transform transaction
        features = pipeline.transform_single(transaction, update_history=False)

        # Prepare feature vector
        feature_names = pipeline.get_feature_names()
        X = np.array([[features[f] for f in feature_names]])

        # Get prediction
        fraud_prob = float(model.predict(X)[0])

        # Get rating
        rating_result = rating_engine.rate_transaction(
            transaction,
            fraud_prob
        )

        # Display result
        actual_fraud = row.get('Is Fraud?', 'No')
        print(f"Transaction: User {row['User']}, Amount {row['Amount']}")
        print(f"  Actual: {actual_fraud}")
        print(f"  Predicted: {fraud_prob:.2%} fraud probability")
        print(f"  Rating: {rating_result['transaction_rating']} ({rating_result['risk_level']})")
        print(f"  Recommendation: {rating_result['recommendation']['action']}")
        print()


def main():
    parser = argparse.ArgumentParser(description='Fraud Detection Demo')
    parser.add_argument('--mode', choices=['full', 'quick', 'test'],
                        default='quick', help='Demo mode')
    args = parser.parse_args()

    print_header("Fraud Detection System Demo")
    print(f"Mode: {args.mode}")
    print(f"Timestamp: {datetime.now().isoformat()}")

    try:
        # Step 1: Generate data
        if args.mode == 'full':
            df = demo_data_generation(n_users=200, days=180)
        elif args.mode == 'quick':
            df = demo_data_generation(n_users=50, days=30)
        else:  # test mode - minimal data
            df = demo_data_generation(n_users=20, days=14)

        # Step 2: Feature engineering
        pipeline, features_df = demo_feature_engineering(df)

        # Step 3: Train model
        model = demo_model_training(features_df, pipeline)

        # Step 4: Transaction rating demo
        demo_transaction_rating()

        # Step 5: NiFi simulation
        demo_nifi_simulation()

        # Step 6: Full inference
        demo_full_inference(df, pipeline, model)

        print_header("Demo Complete!")

        print("Summary:")
        print(f"  - Generated {len(df):,} transactions")
        print(f"  - Trained LightGBM model with {len(pipeline.get_feature_names())} features")
        print(f"  - Demonstrated transaction rating system")
        print(f"  - Simulated NiFi transaction flow")

        print("\nNext Steps for Cloudera AI Deployment:")
        print("  1. Upload this project to Cloudera ML")
        print("  2. Run scripts/train_model_v2.py to train production model")
        print("  3. Deploy using deploy_model.py")
        print("  4. Import NiFi flow from nifi/fraud_detection_flow.json")
        print("  5. Configure NiFi to call the CML model endpoint")

    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user")
    except Exception as e:
        print(f"\nError during demo: {e}")
        raise


if __name__ == "__main__":
    main()
