#!/usr/bin/env python3
"""
Score the last year of credit card transactions using the trained model.
Generate fraud risk scores and alerts.
"""
import pandas as pd
import numpy as np
import joblib
import json
import os
from datetime import datetime

def score_transactions(df, model, feature_cols, threshold):
    """Score transactions and generate fraud predictions"""
    
    # Prepare features
    X = df[feature_cols]
    
    # Get fraud probabilities
    fraud_probabilities = model.predict(X, num_iteration=model.best_iteration)
    
    # Make predictions based on threshold
    fraud_predictions = (fraud_probabilities > threshold).astype(int)
    
    # Add scores to dataframe
    df['fraud_probability'] = fraud_probabilities
    df['fraud_prediction'] = fraud_predictions
    df['risk_level'] = pd.cut(
        fraud_probabilities,
        bins=[0, 0.3, 0.6, 0.9, 1.0],
        labels=['Low', 'Medium', 'High', 'Very High']
    )
    
    return df

def generate_alerts(scored_df):
    """Generate alerts for high-risk transactions"""
    
    # Filter high-risk transactions
    high_risk = scored_df[scored_df['fraud_probability'] > 0.7].copy()
    
    if len(high_risk) > 0:
        # Sort by fraud probability
        high_risk = high_risk.sort_values('fraud_probability', ascending=False)
        
        # Create alert summary
        alerts = []
        for _, row in high_risk.iterrows():
            alert = {
                'user_id': row['User'],
                'card_id': row['Card'],
                'transaction_date': row['datetime'].strftime('%Y-%m-%d %H:%M'),
                'amount': row['Amount'],
                'merchant': row['Merchant Name'],
                'location': f"{row['Merchant City']}, {row['Merchant State']}",
                'fraud_probability': round(row['fraud_probability'], 3),
                'risk_level': row['risk_level'],
                'transaction_type': row['Use Chip']
            }
            alerts.append(alert)
        
        return pd.DataFrame(alerts)
    else:
        return pd.DataFrame()

def main():
    print("Loading model and metadata...")
    
    # Load model
    model_path = 'models/fraud_detection_lgb.pkl'
    if not os.path.exists(model_path):
        print("Error: Model not found. Please run train_model.py first.")
        return
    
    model = joblib.load(model_path)
    
    # Load metadata
    with open('models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_cols = metadata['features']
    optimal_threshold = metadata['optimal_threshold']
    
    print(f"Model loaded with optimal threshold: {optimal_threshold:.4f}")
    
    # Load processed data
    print("\nLoading transaction data...")
    df = pd.read_csv('data/processed_transactions.csv', parse_dates=['datetime'])
    
    # Filter last year of transactions
    last_year_start = df['datetime'].max() - pd.DateOffset(years=1)
    last_year_df = df[df['datetime'] >= last_year_start].copy()
    
    print(f"Scoring {len(last_year_df)} transactions from the last year...")
    
    # Score transactions
    scored_df = score_transactions(last_year_df, model, feature_cols, optimal_threshold)
    
    # Generate summary statistics
    print("\nScoring Results Summary:")
    print(f"Total transactions scored: {len(scored_df)}")
    print(f"Predicted fraudulent transactions: {scored_df['fraud_prediction'].sum()}")
    print(f"Actual fraudulent transactions: {scored_df['is_fraud'].sum()}")
    
    print("\nRisk Level Distribution:")
    print(scored_df['risk_level'].value_counts().sort_index())
    
    # Calculate accuracy if we have true labels
    if 'is_fraud' in scored_df.columns:
        accuracy = (scored_df['fraud_prediction'] == scored_df['is_fraud']).mean()
        precision = (scored_df['fraud_prediction'] & scored_df['is_fraud']).sum() / scored_df['fraud_prediction'].sum()
        recall = (scored_df['fraud_prediction'] & scored_df['is_fraud']).sum() / scored_df['is_fraud'].sum()
        
        print(f"\nPerformance on Last Year:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
    
    # Save scored transactions
    os.makedirs('output', exist_ok=True)
    
    output_path = 'output/scored_transactions_last_year.csv'
    scored_df.to_csv(output_path, index=False)
    print(f"\nScored transactions saved to {output_path}")
    
    # Generate and save alerts
    alerts_df = generate_alerts(scored_df)
    if len(alerts_df) > 0:
        alerts_path = 'output/fraud_alerts.csv'
        alerts_df.to_csv(alerts_path, index=False)
        print(f"Generated {len(alerts_df)} fraud alerts saved to {alerts_path}")
        
        # Show top 10 alerts
        print("\nTop 10 High-Risk Transactions:")
        print(alerts_df.head(10).to_string(index=False))
    else:
        print("No high-risk transactions detected.")
    
    # Generate summary report
    summary = {
        'scoring_date': datetime.now().isoformat(),
        'period_start': last_year_start.isoformat(),
        'period_end': df['datetime'].max().isoformat(),
        'total_transactions': len(scored_df),
        'predicted_fraud_count': int(scored_df['fraud_prediction'].sum()),
        'actual_fraud_count': int(scored_df['is_fraud'].sum()),
        'high_risk_count': int((scored_df['risk_level'] == 'Very High').sum()),
        'threshold_used': optimal_threshold,
        'risk_distribution': scored_df['risk_level'].value_counts().to_dict()
    }
    
    with open('output/scoring_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("\nScoring summary saved to output/scoring_summary.json")

if __name__ == "__main__":
    main()