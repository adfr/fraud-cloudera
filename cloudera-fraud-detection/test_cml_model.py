#!/usr/bin/env python3
"""
Test the CML deployed fraud detection model
Provides both local testing and instructions for API testing
"""
import os
import json
import requests
from datetime import datetime

def load_deployment_info():
    """Load CML deployment information"""
    try:
        with open("models/cml_deployment_info.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("❌ CML deployment info not found. Please run deploy_model.py first.")
        return None

def test_predict_function_locally():
    """Test the predict function locally before deployment"""
    print("="*60)
    print("Testing Predict Function Locally")
    print("="*60)
    
    try:
        # Import the prediction function
        import sys
        sys.path.append(os.path.dirname(__file__))
        from predict import predict
        
        # Test cases
        test_cases = create_test_cases()
        
        success_count = 0
        for i, (description, test_data) in enumerate(test_cases, 1):
            print(f"\n{i}. {description}")
            print("-" * len(f"{i}. {description}"))
            
            # Show key input features
            print(f"Amount: ${test_data.get('Amount_clean', 0):.2f}")
            print(f"Hour: {test_data.get('hour', 'N/A')}")
            print(f"Weekend: {'Yes' if test_data.get('is_weekend') else 'No'}")
            print(f"Online: {'Yes' if test_data.get('is_online') else 'No'}")
            
            try:
                result = predict(test_data)
                
                if result.get('fraud_label') != 'ERROR':
                    print(f"✓ Prediction: {result['fraud_label']}")
                    print(f"  Risk Level: {result['risk_level']}")
                    print(f"  Probability: {result['fraud_probability']:.3f}")
                    print(f"  Confidence: {result.get('confidence', 'unknown')}")
                    
                    if 'explanation' in result:
                        print(f"  Risk Factors: {', '.join(result['explanation'])}")
                    
                    success_count += 1
                else:
                    print(f"❌ Error: {result.get('error', 'Unknown error')}")
                    
            except Exception as e:
                print(f"❌ Exception: {str(e)}")
        
        print(f"\n📊 Local Testing Summary: {success_count}/{len(test_cases)} tests passed")
        return success_count == len(test_cases)
        
    except ImportError as e:
        print(f"❌ Could not import predict function: {e}")
        return False
    except Exception as e:
        print(f"❌ Error during local testing: {e}")
        return False

def create_test_cases():
    """Create comprehensive test cases for the model"""
    return [
        ("Normal Small Purchase", {
            "Amount_clean": 25.99,
            "hour": 15,
            "day_of_week": 2,
            "day_of_month": 15,
            "is_weekend": 0,
            "is_online": 0,
            "is_chip": 1,
            "is_swipe": 0,
            "mcc_encoded": 5,
            "state_encoded": 10,
            "is_online_state": 0,
            "time_since_last": 4.5,
            "amount_mean_5": 30.0,
            "amount_std_5": 8.0,
            "amount_mean_10": 28.0,
            "amount_std_10": 10.0,
            "amount_mean_30": 32.0,
            "amount_std_30": 12.0,
            "amount_deviation": -6.01,
            "amount_zscore": -0.5,
            "trans_count_day": 2
        }),
        
        ("High-Risk Large Amount", {
            "Amount_clean": 3500.0,
            "hour": 2,
            "day_of_week": 6,
            "day_of_month": 1,
            "is_weekend": 1,
            "is_online": 1,
            "is_chip": 0,
            "is_swipe": 0,
            "mcc_encoded": 8,
            "state_encoded": 0,
            "is_online_state": 1,
            "time_since_last": 0.2,
            "amount_mean_5": 45.0,
            "amount_std_5": 12.0,
            "amount_mean_10": 48.0,
            "amount_std_10": 15.0,
            "amount_mean_30": 50.0,
            "amount_std_30": 18.0,
            "amount_deviation": 3450.0,
            "amount_zscore": 191.7,
            "trans_count_day": 12
        }),
        
        ("Moderate Risk Evening", {
            "Amount_clean": 450.0,
            "hour": 23,
            "day_of_week": 0,
            "day_of_month": 31,
            "is_weekend": 0,
            "is_online": 1,
            "is_chip": 0,
            "is_swipe": 0,
            "mcc_encoded": 12,
            "state_encoded": 5,
            "is_online_state": 1,
            "time_since_last": 1.0,
            "amount_mean_5": 85.0,
            "amount_std_5": 25.0,
            "amount_mean_10": 90.0,
            "amount_std_10": 30.0,
            "amount_mean_30": 95.0,
            "amount_std_30": 35.0,
            "amount_deviation": 355.0,
            "amount_zscore": 10.1,
            "trans_count_day": 6
        }),
        
        ("Minimal Data", {
            "Amount_clean": 75.0
        }),
        
        ("All Features Present", {
            "Amount_clean": 125.50,
            "hour": 10,
            "day_of_week": 3,
            "day_of_month": 20,
            "is_weekend": 0,
            "is_online": 0,
            "is_chip": 1,
            "is_swipe": 0,
            "mcc_encoded": 7,
            "state_encoded": 15,
            "is_online_state": 0,
            "time_since_last": 8.2,
            "amount_mean_5": 110.0,
            "amount_std_5": 20.0,
            "amount_mean_10": 115.0,
            "amount_std_10": 25.0,
            "amount_mean_30": 120.0,
            "amount_std_30": 30.0,
            "amount_deviation": 5.5,
            "amount_zscore": 0.18,
            "trans_count_day": 4
        })
    ]

def show_cml_testing_instructions():
    """Show instructions for testing the deployed CML model"""
    print("\n" + "="*60)
    print("CML Model Testing Instructions")
    print("="*60)
    
    deployment_info = load_deployment_info()
    if not deployment_info:
        return
    
    print("\n🏠 CML UI Testing:")
    print("  1. Open your CML workspace")
    print("  2. Navigate to 'Models' section")
    print(f"  3. Find 'Fraud Detection Model' (ID: {deployment_info['model_id']})")
    print("  4. Click on the model name")
    print("  5. Go to the 'Deployments' tab")
    print("  6. Click on your deployment")
    print("  7. Use the 'Test Model' button")
    
    print("\n📝 Test Payload Examples:")
    
    test_cases = create_test_cases()
    
    for i, (description, test_data) in enumerate(test_cases[:3], 1):
        print(f"\n{i}. {description}:")
        print("```json")
        print(json.dumps(test_data, indent=2))
        print("```")
    
    print("\n🔧 API Testing (if authentication is configured):")
    print("Replace <ACCESS_KEY> with your model's access key from the CML UI")
    
    print(f"""
curl -X POST "https://your-cml-host/model/{deployment_info['model_id']}/predict" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <ACCESS_KEY>" \\
  -d '{json.dumps(test_cases[0][1])}'
""")

def generate_test_script():
    """Generate a comprehensive test script for the CML model"""
    print("\n" + "="*60)
    print("Generating Test Scripts")
    print("="*60)
    
    # Python test script
    python_test_script = '''#!/usr/bin/env python3
"""
Automated testing script for CML fraud detection model
Usage: python cml_model_test.py <model_endpoint> <access_key>
"""
import requests
import json
import sys
import time

def test_cml_model(endpoint, access_key):
    """Test the CML model endpoint"""
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {access_key}"
    }
    
    test_cases = [
        {
            "name": "Normal Transaction",
            "data": {
                "Amount_clean": 45.50,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_online": 0
            }
        },
        {
            "name": "Suspicious Transaction", 
            "data": {
                "Amount_clean": 2500.0,
                "hour": 3,
                "day_of_week": 6,
                "is_weekend": 1,
                "is_online": 1,
                "amount_zscore": 150.0
            }
        }
    ]
    
    for test_case in test_cases:
        print(f"Testing: {test_case['name']}")
        
        try:
            response = requests.post(
                endpoint, 
                headers=headers, 
                json=test_case['data'],
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"  ✓ {result.get('fraud_label', 'Unknown')}")
                print(f"    Risk: {result.get('risk_level', 'Unknown')}")
                print(f"    Probability: {result.get('fraud_probability', 0):.3f}")
            else:
                print(f"  ❌ HTTP {response.status_code}: {response.text}")
                
        except Exception as e:
            print(f"  ❌ Error: {e}")
        
        time.sleep(1)  # Rate limiting

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python cml_model_test.py <endpoint> <access_key>")
        sys.exit(1)
    
    test_cml_model(sys.argv[1], sys.argv[2])
'''
    
    with open("cml_model_test.py", "w") as f:
        f.write(python_test_script)
    
    os.chmod("cml_model_test.py", 0o755)
    print("✅ Created: cml_model_test.py")
    
    # Bash test script
    bash_test_script = '''#!/bin/bash
# Simple bash test for CML fraud detection model
# Usage: ./test_cml_model.sh <endpoint> <access_key>

ENDPOINT=$1
ACCESS_KEY=$2

if [ -z "$ENDPOINT" ] || [ -z "$ACCESS_KEY" ]; then
    echo "Usage: $0 <endpoint> <access_key>"
    exit 1
fi

echo "Testing CML Fraud Detection Model"
echo "Endpoint: $ENDPOINT"

# Test 1: Normal transaction
echo "Test 1: Normal Transaction"
curl -s -X POST "$ENDPOINT" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $ACCESS_KEY" \\
  -d '{
    "Amount_clean": 45.50,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_online": 0
  }' | jq .

echo "\\n"

# Test 2: Suspicious transaction
echo "Test 2: Suspicious Transaction" 
curl -s -X POST "$ENDPOINT" \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer $ACCESS_KEY" \\
  -d '{
    "Amount_clean": 2500.0,
    "hour": 3,
    "day_of_week": 6,
    "is_weekend": 1,
    "is_online": 1,
    "amount_zscore": 150.0
  }' | jq .

echo "Testing completed!"
'''
    
    with open("test_cml_model.sh", "w") as f:
        f.write(bash_test_script)
    
    os.chmod("test_cml_model.sh", 0o755)
    print("✅ Created: test_cml_model.sh")
    
    print("\nUsage:")
    print("  python cml_model_test.py <endpoint> <access_key>")
    print("  ./test_cml_model.sh <endpoint> <access_key>")

def main():
    """Main testing function"""
    print("="*60)
    print("CML Fraud Detection Model Testing")
    print("="*60)
    
    # Test locally first
    print("\n🔬 Step 1: Local Testing")
    local_success = test_predict_function_locally()
    
    if local_success:
        print("\n✅ Local testing passed!")
    else:
        print("\n❌ Local testing failed! Fix issues before deploying.")
        return
    
    # Show CML testing instructions
    print("\n📋 Step 2: CML Testing Instructions")
    show_cml_testing_instructions()
    
    # Generate test scripts
    print("\n🛠️  Step 3: Generate Test Scripts")
    generate_test_script()
    
    print("\n" + "="*60)
    print("Testing Setup Complete!")
    print("="*60)
    print("Next steps:")
    print("1. Deploy your model: python deploy_model.py")
    print("2. Test in CML UI using the provided payloads")
    print("3. Use generated scripts for automated testing")

if __name__ == "__main__":
    main()