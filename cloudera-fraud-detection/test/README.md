# Fraud Detection Pipeline Tests

This directory contains test scripts for local testing of the fraud detection pipeline components.

## Test Scripts

1. **test_feature_engineering.py** - Tests the feature engineering pipeline
   - Creates sample credit card transaction data
   - Runs feature engineering transformations
   - Validates output features
   - Saves test data for subsequent tests

2. **test_train_model.py** - Tests the model training pipeline
   - Loads engineered features from test_feature_engineering
   - Trains a LightGBM model
   - Evaluates model performance
   - Tests prediction functionality

3. **test_score_transactions.py** - Tests the transaction scoring pipeline
   - Loads trained model and test data
   - Scores transactions for fraud probability
   - Assigns risk levels and generates alerts
   - Produces scoring reports

4. **run_all_tests.py** - Main test runner
   - Executes all tests in sequence
   - Provides summary of test results
   - Stops on first failure (since tests are dependent)

## Running Tests

### Run All Tests
```bash
python test/run_all_tests.py
```

### Run Individual Tests
```bash
# Test feature engineering only
python test/test_feature_engineering.py

# Test model training only (requires feature engineering to run first)
python test/test_train_model.py

# Test transaction scoring (requires both above to run first)
python test/test_score_transactions.py
```

## Test Data

The tests create the following artifacts:

- `data/sample_transactions.csv` - Sample transaction data
- `data/test_processed_transactions.csv` - Engineered features
- `models/test_fraud_model.pkl` - Trained test model
- `models/test_model_metadata.pkl` - Model metadata
- `output/test_scored_transactions.csv` - Scored transactions
- `output/test_fraud_alerts.csv` - Generated fraud alerts

## Requirements

The test scripts use the same dependencies as the main project. Ensure you have:
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib

## Notes

- Tests use synthetic data with 1000 sample transactions
- The fraud rate in test data is set to ~2% to match real-world scenarios
- Test models are saved separately from production models
- All test outputs are prefixed with "test_" to avoid conflicts