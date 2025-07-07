#!/usr/bin/env python3
"""
Test the deployed fraud detection model
Sends test requests to the model endpoint
"""
import os
import json
import requests
import time
from datetime import datetime

def load_deployment_info():
    """Load deployment information"""
    try:
        with open("models/deployment_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("Deployment info not found. Please run deploy_model.py first.")
        return None

def create_test_transactions():
    """Create various test transactions for testing"""
    
    # Normal transaction
    normal_transaction = {
        "Amount_clean": 45.50,
        "hour": 14,
        "day_of_week": 2,
        "day_of_month": 15,
        "is_weekend": 0,
        "is_online": 0,
        "is_chip": 1,
        "is_swipe": 0,
        "mcc_encoded": 5,
        "state_encoded": 10,
        "is_online_state": 0,
        "time_since_last": 2.5,
        "amount_mean_5": 50.0,
        "amount_std_5": 10.0,
        "amount_mean_10": 48.0,
        "amount_std_10": 12.0,
        "amount_mean_30": 52.0,
        "amount_std_30": 15.0,
        "amount_deviation": -4.5,
        "amount_zscore": -0.3,
        "trans_count_day": 3
    }
    
    # Suspicious transaction (large amount, unusual time)
    suspicious_transaction = {
        "Amount_clean": 2500.0,
        "hour": 3,
        "day_of_week": 6,
        "day_of_month": 15,
        "is_weekend": 1,
        "is_online": 1,
        "is_chip": 0,
        "is_swipe": 0,
        "mcc_encoded": 5,
        "state_encoded": 10,
        "is_online_state": 1,
        "time_since_last": 0.1,
        "amount_mean_5": 50.0,
        "amount_std_5": 10.0,
        "amount_mean_10": 45.0,
        "amount_std_10": 12.0,
        "amount_mean_30": 48.0,
        "amount_std_30": 15.0,
        "amount_deviation": 2452.0,
        "amount_zscore": 163.5,
        "trans_count_day": 15
    }
    
    # Moderately suspicious transaction
    moderate_transaction = {
        "Amount_clean": 350.0,
        "hour": 23,
        "day_of_week": 0,
        "day_of_month": 1,
        "is_weekend": 0,
        "is_online": 1,
        "is_chip": 0,
        "is_swipe": 0,
        "mcc_encoded": 8,
        "state_encoded": 15,
        "is_online_state": 1,
        "time_since_last": 0.5,
        "amount_mean_5": 80.0,
        "amount_std_5": 20.0,
        "amount_mean_10": 75.0,
        "amount_std_10": 25.0,
        "amount_mean_30": 78.0,
        "amount_std_30": 30.0,
        "amount_deviation": 272.0,
        "amount_zscore": 9.1,
        "trans_count_day": 8
    }
    
    return [
        ("Normal Transaction", normal_transaction),
        ("Suspicious Transaction", suspicious_transaction),
        ("Moderate Risk Transaction", moderate_transaction)
    ]

def test_model_locally():
    """Test the model serving script locally"""
    print("="*60)
    print("Testing Model Locally")
    print("="*60)
    
    try:
        # Import the serving script
        import sys
        sys.path.append(os.path.dirname(__file__))
        from model_serve import predict, health_check
        
        # Health check
        print("Health Check:")
        health = health_check()
        print(json.dumps(health, indent=2))
        
        if health.get("status") != "healthy":
            print("❌ Model is not healthy, cannot proceed with testing")
            return False
        
        print("\n" + "="*40)
        print("Local Prediction Tests")
        print("="*40)
        
        # Test transactions
        test_transactions = create_test_transactions()
        
        for description, transaction in test_transactions:
            print(f"\n{description}:")
            print("-" * len(description))
            print(f"Amount: ${transaction['Amount_clean']}")
            print(f"Time: {transaction['hour']:02d}:00")
            print(f"Weekend: {'Yes' if transaction['is_weekend'] else 'No'}")
            print(f"Online: {'Yes' if transaction['is_online'] else 'No'}")
            
            result = predict(transaction)
            
            print(f"\nPrediction Result:")
            print(f"  Fraud Probability: {result.get('fraud_probability', 0):.3f}")
            print(f"  Prediction: {result.get('fraud_label', 'UNKNOWN')}")
            print(f"  Risk Level: {result.get('risk_level', 'UNKNOWN')}")
            
            if 'explanation' in result:
                print(f"  Risk Factors: {', '.join(result['explanation'])}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error testing model locally: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_model_endpoint():
    """Test the deployed model endpoint via HTTP"""
    print("\n" + "="*60)
    print("Testing Deployed Model Endpoint")
    print("="*60)
    
    deployment_info = load_deployment_info()
    if not deployment_info:
        return False
    
    endpoint_url = deployment_info.get("endpoint_url")
    if not endpoint_url:
        print("❌ No endpoint URL found in deployment info")
        return False
    
    print(f"Testing endpoint: {endpoint_url}")
    
    # Note: In a real deployment, you'd need proper authentication
    # This is a placeholder for testing structure
    
    print("\n⚠️  Note: HTTP endpoint testing requires:")
    print("1. Valid authentication token/access key")
    print("2. Network access to the CML cluster")
    print("3. Deployed model to be running")
    
    print("\nExample curl commands for testing:")
    
    test_transactions = create_test_transactions()
    
    for description, transaction in test_transactions:
        print(f"\n# {description}")
        payload = {"request": transaction}
        
        curl_command = f"""curl -X POST {endpoint_url} \\
  -H 'Content-Type: application/json' \\
  -H 'Authorization: Bearer <YOUR_ACCESS_KEY>' \\
  -d '{json.dumps(payload)}'"""
        
        print(curl_command)
    
    return True

def generate_load_test():
    """Generate a load test script for the deployed model"""
    print("\n" + "="*60)
    print("Generating Load Test Script")
    print("="*60)
    
    load_test_script = '''#!/bin/bash
# Load test script for fraud detection model
# Usage: ./load_test.sh <endpoint_url> <access_key> <num_requests>

ENDPOINT_URL=$1
ACCESS_KEY=$2
NUM_REQUESTS=${3:-100}

if [ -z "$ENDPOINT_URL" ] || [ -z "$ACCESS_KEY" ]; then
    echo "Usage: $0 <endpoint_url> <access_key> [num_requests]"
    exit 1
fi

echo "Running load test with $NUM_REQUESTS requests..."
echo "Endpoint: $ENDPOINT_URL"

# Test payload
PAYLOAD='{
  "request": {
    "Amount_clean": 150.0,
    "hour": 14,
    "day_of_week": 2,
    "day_of_month": 15,
    "is_weekend": 0,
    "is_online": 0,
    "is_chip": 1,
    "is_swipe": 0,
    "mcc_encoded": 5,
    "state_encoded": 10,
    "is_online_state": 0,
    "time_since_last": 2.5,
    "amount_mean_5": 50.0,
    "amount_std_5": 10.0,
    "amount_mean_10": 48.0,
    "amount_std_10": 12.0,
    "amount_mean_30": 52.0,
    "amount_std_30": 15.0,
    "amount_deviation": -4.5,
    "amount_zscore": -0.3,
    "trans_count_day": 3
  }
}'

# Run concurrent requests
for i in $(seq 1 $NUM_REQUESTS); do
    curl -s -X POST "$ENDPOINT_URL" \\
         -H "Content-Type: application/json" \\
         -H "Authorization: Bearer $ACCESS_KEY" \\
         -d "$PAYLOAD" &
    
    # Limit concurrent requests
    if [ $((i % 10)) -eq 0 ]; then
        wait
        echo "Completed $i requests..."
    fi
done

wait
echo "Load test completed!"
'''
    
    with open("test_load.sh", "w") as f:
        f.write(load_test_script)
    
    # Make it executable
    os.chmod("test_load.sh", 0o755)
    
    print("✅ Load test script created: test_load.sh")
    print("Make it executable and run with: ./test_load.sh <endpoint> <token> <requests>")

def main():
    """Main testing function"""
    print("="*60)
    print("Fraud Detection Model Testing Suite")
    print("="*60)
    
    # Test locally first
    local_success = test_model_locally()
    
    if local_success:
        print("\n✅ Local testing completed successfully!")
        
        # Test endpoint
        endpoint_success = test_model_endpoint()
        
        if endpoint_success:
            print("\n✅ Endpoint testing information provided!")
        
        # Generate load test
        generate_load_test()
        
        print("\n" + "="*60)
        print("Testing Summary")
        print("="*60)
        print("✅ Local model testing: PASSED")
        print("ℹ️  Endpoint testing: Instructions provided")
        print("✅ Load test script: Generated")
        
    else:
        print("\n❌ Local testing failed!")

if __name__ == "__main__":
    main()