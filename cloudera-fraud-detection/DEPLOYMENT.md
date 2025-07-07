# CML Native Fraud Detection Model Deployment

This guide covers deploying the fraud detection model using Cloudera ML's native model deployment capabilities. This approach leverages CML's built-in model serving infrastructure for optimal performance and integration.

## Overview

The deployment uses CML's native model deployment system to create a REST API endpoint that processes individual transactions and returns fraud predictions in real-time. CML handles scaling, monitoring, and security automatically.

## Architecture

```
[Client Request] → [CML API Gateway] → [Model Runtime] → [predict.py] → [LightGBM Model] → [Response]
```

## Files

- **deploy_model.py** - CML model deployment script
- **predict.py** - Model prediction function (called by CML)
- **test_cml_model.py** - Comprehensive testing suite
- **DEPLOYMENT.md** - This documentation

## Prerequisites

### 1. Environment Variables

Create a `.env` file with CML credentials:

```bash
# Cloudera ML Configuration
CML_API_HOST=https://your-cml-workspace.cloudera.com/api/v1
CML_API_KEY=your-api-key-here
CML_PROJECT_ID=your-project-id-here
CML_RUNTIME_ID=docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-standard:2025.01.2-b15
```

### 2. Trained Model

Ensure you have a trained model in the `models/` directory:
- `models/fraud_detection_lgb.pkl` (LightGBM model)
- `models/model_metadata.json` (model metadata)

Or test model:
- `models/test_fraud_model.pkl`
- `models/test_model_metadata.pkl`

### 3. CML API Access

- Valid CML workspace access
- API key with model deployment permissions
- Project-level permissions for model creation

## Deployment Process

### 1. Test Locally

First, test the prediction function locally:

```bash
python test_cml_model.py
```

This validates:
- Model loading functionality
- Prediction logic
- Input validation
- Error handling

### 2. Deploy to CML

Deploy the model to CML:

```bash
python deploy_model.py
```

The script will:
1. Create a CML model object
2. Build the model with `predict.py`
3. Deploy with specified resources
4. Wait for deployment to be ready
5. Save deployment information

### 3. Verify Deployment

Check deployment status:
- CML UI → Models → "Fraud Detection Model"
- Verify build completed successfully
- Confirm deployment is running
- Note the model access key

## Model API

### Input Format

The model expects a JSON object with transaction features:

```json
{
  "Amount_clean": 250.0,
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
  "amount_deviation": 200.0,
  "amount_zscore": 13.3,
  "trans_count_day": 3
}
```

**Note**: Missing features are automatically set to sensible defaults (0 for most features).

### Output Format

The model returns a comprehensive prediction response:

```json
{
  "fraud_probability": 0.125,
  "fraud_prediction": 0,
  "fraud_label": "NORMAL",
  "risk_level": "Low",
  "threshold_used": 0.5,
  "prediction_timestamp": "2025-01-07T15:45:30.123456",
  "confidence": "high",
  "model_version": "2025-01-07T10:30:00",
  "model_type": "LightGBM"
}
```

For high-risk transactions (probability > 0.6):

```json
{
  "fraud_probability": 0.875,
  "fraud_prediction": 1,
  "fraud_label": "FRAUD",
  "risk_level": "Very High",
  "threshold_used": 0.5,
  "prediction_timestamp": "2025-01-07T15:45:30.123456",
  "confidence": "high",
  "explanation": [
    "Large transaction amount: $2500.00",
    "Unusual transaction time: 03:00",
    "Weekend transaction",
    "Amount deviates from user pattern (z-score: 163.5)"
  ],
  "top_features": [
    {"feature": "amount_zscore", "importance": 0.234},
    {"feature": "Amount_clean", "importance": 0.198},
    {"feature": "hour", "importance": 0.156},
    {"feature": "is_online", "importance": 0.089},
    {"feature": "is_weekend", "importance": 0.067}
  ]
}
```

### Risk Levels

- **Low**: fraud_probability < 0.3
- **Medium**: 0.3 ≤ fraud_probability < 0.6  
- **High**: 0.6 ≤ fraud_probability < 0.9
- **Very High**: fraud_probability ≥ 0.9

## Testing

### 1. CML UI Testing

1. Navigate to CML workspace
2. Go to Models → "Fraud Detection Model"
3. Click on the Deployments tab
4. Click "Test Model" button
5. Use provided test payloads

### 2. API Testing

Get the model endpoint and access key from CML UI, then:

```bash
# Using curl
curl -X POST "https://your-cml-host/model/MODEL_ID/predict" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ACCESS_KEY" \
  -d '{
    "Amount_clean": 150.0,
    "hour": 14,
    "day_of_week": 2,
    "is_weekend": 0,
    "is_online": 0
  }'

# Using generated test scripts
python cml_model_test.py <endpoint> <access_key>
./test_cml_model.sh <endpoint> <access_key>
```

### 3. Automated Testing

The test suite includes:

```bash
# Generate test scripts
python test_cml_model.py

# This creates:
# - cml_model_test.py (Python test client)
# - test_cml_model.sh (Bash test script)
```

## Configuration

### Resource Configuration

Modify `deploy_model.py` to adjust resources:

```python
deployment_request = cmlapi.CreateModelDeploymentRequest(
    model_id=model_id,
    build_id=build_id,
    cpu="2",        # CPU cores
    memory="4",     # Memory in GB
    nvidia_gpu="0", # GPU count
    replicas="2"    # Number of replicas
)
```

### Feature Defaults

Modify `predict.py` to change default values for missing features:

```python
def validate_and_prepare_input(input_data):
    for feature in features_to_use:
        if feature in input_data:
            features[feature] = input_data[feature]
        else:
            # Customize defaults here
            if 'amount' in feature.lower():
                features[feature] = 0.0
            elif feature == 'trans_count_day':
                features[feature] = 1  # Default to 1 transaction
            # ... other custom defaults
```

## Monitoring

### CML Built-in Monitoring

CML provides automatic monitoring:
- Request/response metrics
- Latency tracking
- Error rates
- Resource utilization
- Automatic health checks

### Custom Monitoring

Access monitoring in CML UI:
1. Models → Your Model → Deployments
2. Click on deployment
3. View Monitoring tab for:
   - Request volume
   - Response times
   - Error rates
   - Resource usage

### Logs

View model logs:
1. CML UI → Models → Deployment
2. Logs tab shows:
   - Model loading logs
   - Prediction requests
   - Errors and exceptions
   - Performance metrics

## Scaling

### Auto-scaling

CML can automatically scale replicas based on load:

```python
# Enable auto-scaling (future CML feature)
deployment_request.autoscaling_enabled = True
deployment_request.autoscaling_min_replicas = "1"
deployment_request.autoscaling_max_replicas = "5"
```

### Manual Scaling

Scale replicas manually in CML UI:
1. Go to model deployment
2. Click "Scale" button
3. Adjust replica count
4. Apply changes

## Security

### Authentication

- CML handles authentication automatically
- Each model gets a unique access key
- Keys can be rotated in CML UI

### Authorization

- Project-level access controls
- User-based permissions
- API key management

### Network Security

- HTTPS endpoints by default
- VPC integration
- Firewall rules through CML

## Troubleshooting

### Common Issues

1. **Model Build Fails**
   ```
   Error: Module not found
   Solution: Check predict.py imports and dependencies
   ```

2. **Deployment Fails**
   ```
   Error: Insufficient resources
   Solution: Increase CPU/memory in deployment config
   ```

3. **Prediction Errors**
   ```
   Error: Model file not found
   Solution: Verify model paths in predict.py
   ```

### Debug Steps

1. **Test Locally First**
   ```bash
   python predict.py
   ```

2. **Check Model Files**
   ```bash
   ls -la models/
   ```

3. **Review CML Logs**
   - Go to CML UI
   - Models → Deployment → Logs
   - Check for errors

4. **Validate Environment**
   ```bash
   python -c "import os; print(os.environ.get('CML_PROJECT_ID'))"
   ```

### Error Codes

- **HTTP 400**: Invalid input format
- **HTTP 401**: Authentication failed
- **HTTP 404**: Model not found
- **HTTP 500**: Internal model error
- **HTTP 503**: Model unavailable

## Updates and Versioning

### Model Updates

To deploy a new model version:

1. Train and save new model
2. Update model files
3. Run deployment script:
   ```bash
   python deploy_model.py
   ```
4. CML creates new build automatically
5. Deploy new version when ready

### Blue-Green Deployment

1. Deploy new version alongside old
2. Test new version thoroughly
3. Switch traffic gradually
4. Decommission old version

### Rollback

If issues occur:
1. Go to CML UI
2. Models → Deployment → Builds
3. Deploy previous working build
4. Investigate issues offline

## Integration Examples

### Python Client

```python
import requests
import json

class FraudDetectionClient:
    def __init__(self, endpoint, access_key):
        self.endpoint = endpoint
        self.headers = {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {access_key}'
        }
    
    def predict_fraud(self, transaction):
        response = requests.post(
            self.endpoint, 
            headers=self.headers,
            json=transaction,
            timeout=30
        )
        return response.json()

# Usage
client = FraudDetectionClient(endpoint, access_key)
result = client.predict_fraud({
    "Amount_clean": 250.0,
    "hour": 14,
    "is_online": 1
})
print(f"Risk: {result['risk_level']}")
```

### Java Client

```java
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;

public class FraudDetectionClient {
    private final String endpoint;
    private final String accessKey;
    private final HttpClient client;
    
    public FraudDetectionClient(String endpoint, String accessKey) {
        this.endpoint = endpoint;
        this.accessKey = accessKey;
        this.client = HttpClient.newHttpClient();
    }
    
    public String predictFraud(String transactionJson) throws Exception {
        HttpRequest request = HttpRequest.newBuilder()
            .uri(URI.create(endpoint))
            .header("Content-Type", "application/json")
            .header("Authorization", "Bearer " + accessKey)
            .POST(HttpRequest.BodyPublishers.ofString(transactionJson))
            .build();
            
        HttpResponse<String> response = client.send(request, 
            HttpResponse.BodyHandlers.ofString());
            
        return response.body();
    }
}
```

## Production Checklist

Before production deployment:

- [ ] Model trained and validated
- [ ] Local testing passed
- [ ] CML deployment successful
- [ ] API testing completed
- [ ] Authentication configured
- [ ] Monitoring set up
- [ ] Error handling tested
- [ ] Performance benchmarked
- [ ] Documentation updated
- [ ] Team trained on operations

## Support

For issues:
1. Check CML logs first
2. Review this documentation
3. Test locally to isolate issues
4. Contact CML administrators
5. File support tickets with logs

This deployment approach leverages CML's native capabilities for optimal performance, security, and operational simplicity.