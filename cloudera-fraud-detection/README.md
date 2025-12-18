# Fraud Detection System for Cloudera AI

A comprehensive credit card fraud detection system using LightGBM, designed for deployment on Cloudera Machine Learning (CML) with Apache NiFi integration.

## Features

- **LightGBM-based fraud detection** with batch and real-time feature engineering
- **Transaction rating system** (A+ to F grades) with detailed risk breakdown
- **CrewAI multi-agent analysis** for deep fraud investigation
- **NiFi integration** for real-time transaction processing
- **Cloudera AI deployment** ready with REST API
- **Kaggle Credit Card Fraud Dataset** support (284K real transactions)
- **Synthetic data generation** for training and testing

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     Fraud Detection Pipeline                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│  │   NiFi       │    │   Feature    │    │   LightGBM Model     │  │
│  │  Transaction │───▶│  Engineering │───▶│   + Transaction      │  │
│  │   Input      │    │  (Batch+RT)  │    │     Rating           │  │
│  └──────────────┘    └──────────────┘    └──────────┬───────────┘  │
│                                                       │              │
│                      ┌────────────────────────────────┘              │
│                      ▼                                               │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                     Response                                   │  │
│  │  - Fraud Probability (0-1)                                    │  │
│  │  - Risk Level (Very Low → Critical)                           │  │
│  │  - Transaction Rating (A+ to F)                               │  │
│  │  - Recommendation (APPROVE, REVIEW, DECLINE)                  │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Project Structure

```
cloudera-fraud-detection/
├── agents/                          # CrewAI multi-agent system
│   ├── __init__.py
│   ├── query_agent.py              # Transaction data gathering
│   ├── pattern_agent.py            # Fraud pattern matching
│   ├── merchant_agent.py           # Online merchant research
│   ├── assessment_agent.py         # Report generation
│   ├── fraud_crew.py               # Crew orchestration
│   └── alert_pipeline.py           # Alert processing pipeline
├── scripts/
│   ├── features/                    # Feature engineering modules
│   │   ├── __init__.py
│   │   ├── batch_features.py       # Pre-computed batch features
│   │   ├── realtime_features.py    # Real-time aggregation features
│   │   └── feature_pipeline.py     # Combined feature pipeline
│   ├── generate_training_data.py   # Synthetic data generator
│   ├── train_model.py              # Original training script
│   ├── train_model_v2.py           # Enhanced training with new features
│   ├── transaction_rating.py       # Transaction rating engine
│   └── score_transactions.py       # Batch scoring script
├── nifi/
│   ├── fraud_detection_flow.json   # NiFi flow template
│   └── transaction_generator.py    # Manual transaction generator
├── config/
│   └── jobs_config.yaml            # CML job configurations
├── models/                          # Trained models (generated)
├── data/                            # Training data (generated)
├── output/                          # Scoring results (generated)
├── predict.py                       # CML prediction endpoint
├── deploy_model.py                  # CML deployment script
├── demo_fraud_detection.py          # End-to-end demo
├── requirements.txt                 # Python dependencies
└── README.md                        # This file
```

## Quick Start

### 1. Run the Demo

```bash
cd cloudera-fraud-detection
python demo_fraud_detection.py --mode quick
```

This will:
- Generate synthetic training data
- Engineer batch and real-time features
- Train a LightGBM model
- Demonstrate transaction rating
- Simulate NiFi transaction flow

### 2. Use Kaggle Credit Card Fraud Dataset (Recommended)

Download and train on the real Kaggle dataset (284,807 transactions):

```bash
# Setup Kaggle API (one-time)
# 1. Create account at kaggle.com
# 2. Go to Settings -> API -> Create New Token
# 3. Save kaggle.json to ~/.kaggle/

# Download dataset
python scripts/download_kaggle_data.py

# Train model
python scripts/train_kaggle_model.py
```

**Dataset Statistics:**
- 284,807 transactions from European cardholders
- 492 fraud cases (0.172% fraud rate)
- Features: V1-V28 (PCA), Time, Amount
- Real-world class imbalance

### 3. Generate Synthetic Training Data (Alternative)

```bash
python scripts/generate_training_data.py
```

Generates synthetic credit card transactions with realistic fraud patterns:
- High-value transactions
- Unusual time transactions
- Geographic anomalies
- Rapid succession fraud
- Online purchase bursts

### 4. Train the Model

```bash
# For Kaggle dataset
python scripts/train_kaggle_model.py

# For synthetic data
python scripts/train_model_v2.py
```

Trains a LightGBM model with:
- 40+ engineered features (batch + real-time)
- Class imbalance handling
- Early stopping
- Optimal threshold selection

### 4. Test Locally

```bash
python predict.py
```

Tests the prediction endpoint with sample transactions.

## Feature Engineering

### Batch Features (Pre-computed)

Computed offline from historical data:

| Feature | Description |
|---------|-------------|
| `user_avg_amount_30d` | User's 30-day average transaction amount |
| `user_std_amount_30d` | Standard deviation of user's amounts |
| `user_online_ratio` | Ratio of online transactions |
| `merchant_fraud_rate` | Historical fraud rate at merchant |
| `mcc_fraud_rate` | Fraud rate by merchant category |
| `state_fraud_rate` | Fraud rate by state |

### Real-Time Features (Computed at Transaction Time)

Computed dynamically for each transaction:

| Feature | Description |
|---------|-------------|
| `Amount_clean` | Transaction amount |
| `hour`, `day_of_week` | Time features |
| `is_online`, `is_chip` | Transaction type |
| `trans_count_1h` | Transactions in last hour |
| `trans_count_24h` | Transactions in last 24 hours |
| `amount_mean_5` | Rolling mean of last 5 transactions |
| `amount_zscore` | Z-score deviation from user average |
| `is_different_state` | Transaction outside home state |

## Transaction Rating System

Transactions receive a letter grade (A+ to F) based on multiple risk factors:

| Rating | Risk Score | Recommendation |
|--------|------------|----------------|
| A+ | 0-10 | Auto-approve |
| A | 10-25 | Approve |
| B | 25-45 | Approve with monitoring |
| C | 45-65 | Manual review |
| D | 65-85 | Step-up authentication |
| F | 85-100 | Decline |

### Risk Factors

- **ML Fraud Score** (40%): Model prediction
- **Amount Risk** (15%): Transaction amount vs. user profile
- **Velocity Risk** (10%): Transaction frequency
- **Time Risk** (8%): Unusual transaction times
- **Geographic Risk** (8%): Location anomalies
- **Merchant Risk** (7%): High-risk merchant categories
- **Device Risk** (7%): Transaction type (chip vs. swipe vs. online)
- **Behavioral Risk** (5%): Deviation from user patterns

## CrewAI Multi-Agent Fraud Analysis

When a high-risk alert is triggered, a CrewAI crew of specialized agents performs deep analysis:

### Agent Workflow

```
┌─────────────────────────────────────────────────────────────────────┐
│                     CrewAI Fraud Analysis                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  1. Query Agent ─────────────────────────────────────────────────   │
│     │  Gathers transaction data, user history, spending profile     │
│     ▼                                                                │
│  2. Pattern Matching Agent ─────────────────────────────────────    │
│     │  Matches against known fraud patterns                          │
│     │  (Card testing, Account takeover, Velocity abuse, etc.)       │
│     ▼                                                                │
│  3. Merchant Research Agent ────────────────────────────────────    │
│     │  Searches online for merchant compromises                      │
│     │  Checks breach databases, fraud reports                        │
│     ▼                                                                │
│  4. Assessment Writer Agent ────────────────────────────────────    │
│        Synthesizes all findings into comprehensive report            │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Using the Fraud Analysis Crew

```python
from agents import FraudAnalysisCrew

# Create crew
crew = FraudAnalysisCrew()

# Analyze an alert
alert_data = {
    "transaction_id": "TXN_001",
    "User": 123,
    "Amount": "$2,500.00",
    "Use Chip": "Online Transaction",
    "Merchant Name": "ELECTRONICS_STORE",
    "MCC": 5732,
    "fraud_probability": 0.78,
    "risk_level": "High"
}

# Full analysis (uses LLM)
result = crew.analyze_alert(alert_data)

# Quick analysis (no LLM required)
quick_result = crew.quick_analyze(alert_data)
```

### Alert Pipeline Integration

```python
from agents import AlertPipeline

# Create pipeline
pipeline = AlertPipeline(analysis_threshold=0.5)

# Process transaction result
alert = pipeline.process_transaction_result(transaction, model_result)

# Analyze pending alerts
results = pipeline.analyze_pending_alerts(max_alerts=10)
```

### Fraud Patterns Detected

| Pattern | Description | Indicators |
|---------|-------------|------------|
| Card Not Present | Online fraud with stolen cards | High-value online, electronics/gift cards |
| Card Testing | Multiple small transactions | Rapid succession, low amounts |
| Account Takeover | Compromised account | Sudden pattern change, new location |
| Velocity Abuse | High transaction frequency | >5/hour, >20/day |
| Geographic Impossibility | Impossible travel | Different cities within hours |
| High Risk Merchant | Known risky categories | Jewelry, gambling, electronics |

### Configuration

```bash
# For full analysis with LLM
export OPENAI_API_KEY=your_key_here

# For web search (merchant research)
export SERPER_API_KEY=your_key_here
```

## NiFi Integration

### Import the Flow

1. Open NiFi Canvas
2. Import `nifi/fraud_detection_flow.json`
3. Configure variables:
   - `ml.model.endpoint`: CML model URL
   - `fraud.alerts.directory`: Alert output path

### Generate Test Transactions

```bash
# Interactive mode
python nifi/transaction_generator.py

# Generate suspicious transaction
python nifi/transaction_generator.py --mode suspicious --fraud-type high_amount

# Batch generation
python nifi/transaction_generator.py --mode batch --count 100 -o transactions.json
```

### NiFi Flow Components

1. **GenerateFlowFile**: Creates test transactions
2. **ExecuteScript**: Generates random transaction data
3. **EvaluateJsonPath**: Extracts transaction attributes
4. **InvokeHTTP**: Calls CML fraud detection model
5. **RouteOnAttribute**: Routes by risk level
6. **LogAttribute**: Logs high-risk alerts
7. **PutFile**: Stores fraud alerts

## Cloudera AI Deployment

### Prerequisites

- Cloudera ML workspace
- Python 3.10+ runtime
- API key for deployment

### Deploy Model

```bash
# Set environment variables
export CML_API_HOST=https://your-cml-workspace.cloudera.site
export CML_API_KEY=your_api_key
export CML_PROJECT_ID=your_project_id

# Deploy
python deploy_model.py
```

### API Request Format

```json
{
  "User": 123,
  "Card": 0,
  "Year": 2024,
  "Month": 12,
  "Day": 18,
  "Time": "14:30",
  "Amount": "$150.00",
  "Use Chip": "Chip Transaction",
  "Merchant Name": "GROCERY_STORE",
  "Merchant State": "CA",
  "MCC": 5411
}
```

### API Response Format

```json
{
  "fraud_probability": 0.023,
  "fraud_prediction": 0,
  "fraud_label": "NORMAL",
  "risk_level": "Low",
  "transaction_rating": "A",
  "rating_score": 15.5,
  "recommendation": "APPROVE",
  "should_approve": true,
  "requires_review": false,
  "confidence": "high",
  "prediction_timestamp": "2024-12-18T14:30:00"
}
```

## Configuration

### CML Jobs (config/jobs_config.yaml)

```yaml
jobs:
  - name: feature_engineering
    script: scripts/feature_engineering.py
    cpu: 2
    memory: 4
    timeout: 1800

  - name: train_model
    script: scripts/train_model_v2.py
    cpu: 4
    memory: 8
    timeout: 3600
    depends_on: feature_engineering

  - name: deploy_model
    script: deploy_model.py
    cpu: 1
    memory: 2
    timeout: 2700
    depends_on: train_model
```

## Requirements

```
pyyaml>=6.0
python-dotenv>=0.20.0
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
joblib>=1.2.0
requests>=2.28.0

# CrewAI for multi-agent analysis
crewai>=0.28.0
crewai-tools>=0.1.0
```

## Testing

```bash
# Run all tests
python test/run_all_tests.py

# Test feature engineering
python test/test_feature_engineering.py

# Test model training
python test/test_train_model.py

# Test scoring
python test/test_score_transactions.py
```

## Environment Setup

### Environment Variables

```bash
# Template directory configuration
export TEMPLATE_DIR=template

# CML API Configuration
export CML_API_HOST=https://ml-12345.cloud.example.com
export CML_API_KEY=your_api_key_here
export CML_PROJECT_ID=project_id_here

# ML Runtime ID
export CML_RUNTIME_ID=docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-standard:2025.01.2-b15

# Default resource settings
export DEFAULT_CPU=1
export DEFAULT_MEMORY=2
export DEFAULT_TIMEOUT=3600
```

## License

Internal use only - Cloudera

## Support

For issues or questions, contact the ML Platform team.
