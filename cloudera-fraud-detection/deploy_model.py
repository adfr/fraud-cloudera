#!/usr/bin/env python3
"""
Deploy fraud detection model using Cloudera ML's native model deployment capabilities
Creates a CML model that serves predictions via the platform's REST API
"""
import os
import sys
import json
import cmlapi
import time

# Load environment variables
try:
    from dotenv import load_dotenv
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-dotenv"])
    from dotenv import load_dotenv

# Load environment variables from .env file
env_path = "fraud-cloudera/cloudera-fraud-detection/.env"
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()

def create_cml_model():
    """Create a CML model using native model deployment capabilities"""
    
    # Get environment variables
    api_host = os.environ.get("CML_API_HOST")
    api_key = os.environ.get("CML_API_KEY") 
    project_id = os.environ.get("CML_PROJECT_ID")
    runtime_id = os.environ.get("CML_RUNTIME_ID", 
                                "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-jupyterlab-python3.10-standard:2025.01.2-b15")
    
    if not all([api_host, api_key, project_id]):
        raise ValueError("Missing required environment variables: CML_API_HOST, CML_API_KEY, CML_PROJECT_ID")
    
    print(f"Creating CML model in project: {project_id}")
    
    # Initialize CML API client
    client = cmlapi.default_client(api_host, api_key)
    
    try:
        # Create the model
        print("Creating CML model...")
        model_request = cmlapi.CreateModelRequest(
            name="Fraud Detection Model",
            description="Real-time fraud detection using LightGBM for credit card transactions",
            project_id=project_id,
            disable_authentication=False  # Enable authentication for security
        )
        
        model_response = client.create_model(model_request, project_id=project_id)
        model_id = model_response.id
        print(f"✓ Model created with ID: {model_id}")
        
        # Create model build
        print("\nCreating model build...")
        build_request = cmlapi.CreateModelBuildRequest(
            model_id=model_id,
            runtime_identifier=runtime_id,
            kernel="python3",
            file_path="fraud-cloudera/cloudera-fraud-detection/predict.py",
            function_name="predict"
        )
        
        build_response = client.create_model_build(
            build_request, 
            project_id=project_id, 
            model_id=model_id
        )
        build_id = build_response.id
        print(f"✓ Model build created with ID: {build_id}")
        
        # Wait for build to complete
        print("\nWaiting for model build to complete...")
        build_status = wait_for_build_completion(client, project_id, model_id, build_id)
        
        if build_status != "built":
            print(f"❌ Model build failed with status: {build_status}")
            return None, None, None
        
        print("✓ Model build completed successfully!")
        
        # Create model deployment
        print("\nCreating model deployment...")
        deployment_request = cmlapi.CreateModelDeploymentRequest(
            model_id=model_id,
            build_id=build_id,
            cpu="1",
            memory="2",
            nvidia_gpu="0",
            replicas="1"
        )
        
        deployment_response = client.create_model_deployment(
            deployment_request,
            project_id=project_id,
            model_id=model_id,
            build_id=build_id
        )
        deployment_id = deployment_response.id
        print(f"✓ Model deployment created with ID: {deployment_id}")
        
        # Wait for deployment to be ready
        print("\nWaiting for model deployment to be ready...")
        deployment_status = wait_for_deployment_ready(client, project_id, model_id, build_id, deployment_id)
        
        if deployment_status != "deployed":
            print(f"❌ Model deployment failed with status: {deployment_status}")
            return None, None, None
        
        print("✓ Model deployment is ready!")
        
        # Get model details
        model_details = client.get_model(project_id=project_id, model_id=model_id)
        
        # Save deployment information
        deployment_info = {
            "model_id": model_id,
            "build_id": build_id,
            "deployment_id": deployment_id,
            "model_name": "Fraud Detection Model",
            "access_key": getattr(model_details, 'access_key', 'Check CML UI'),
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "api_host": api_host,
            "project_id": project_id
        }
        
        os.makedirs("models", exist_ok=True)
        with open("models/cml_deployment_info.json", "w") as f:
            json.dump(deployment_info, f, indent=2)
        
        print(f"\n🎉 Model deployment completed successfully!")
        print(f"📄 Deployment info saved to: models/cml_deployment_info.json")
        
        return model_id, build_id, deployment_id
        
    except Exception as e:
        print(f"❌ Error during model deployment: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None, None

def wait_for_build_completion(client, project_id, model_id, build_id, max_wait=1800):
    """Wait for model build to complete"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            build_status_response = client.get_model_build(
                project_id=project_id, 
                model_id=model_id, 
                build_id=build_id
            )
            status = build_status_response.status
            print(f"  Build status: {status}")
            
            if status == "built":
                return status
            elif status in ["build_failed", "failed", "error"]:
                if hasattr(build_status_response, 'failure_reason'):
                    print(f"  Build failure reason: {build_status_response.failure_reason}")
                return status
            
            time.sleep(30)  # Check every 30 seconds
            
        except Exception as e:
            print(f"  Error checking build status: {e}")
            time.sleep(30)
    
    print(f"  Build timed out after {max_wait} seconds")
    return "timeout"

def wait_for_deployment_ready(client, project_id, model_id, build_id, deployment_id, max_wait=900):
    """Wait for model deployment to be ready"""
    start_time = time.time()
    while time.time() - start_time < max_wait:
        try:
            deployment_status_response = client.get_model_deployment(
                project_id=project_id,
                model_id=model_id,
                build_id=build_id,
                deployment_id=deployment_id
            )
            status = deployment_status_response.status
            print(f"  Deployment status: {status}")
            
            if status == "deployed":
                return status
            elif status in ["deploy_failed", "failed", "error", "stopped"]:
                if hasattr(deployment_status_response, 'failure_reason'):
                    print(f"  Deployment failure reason: {deployment_status_response.failure_reason}")
                return status
            
            time.sleep(15)  # Check every 15 seconds
            
        except Exception as e:
            print(f"  Error checking deployment status: {e}")
            time.sleep(15)
    
    print(f"  Deployment timed out after {max_wait} seconds")
    return "timeout"

def show_usage_instructions():
    """Show instructions for using the deployed model"""
    print("\n" + "="*60)
    print("Model Usage Instructions")
    print("="*60)
    
    try:
        with open("models/cml_deployment_info.json", "r") as f:
            info = json.load(f)
    except FileNotFoundError:
        print("❌ Deployment info not found")
        return
    
    print("\n📋 Model Information:")
    print(f"  Model ID: {info['model_id']}")
    print(f"  Build ID: {info['build_id']}")
    print(f"  Deployment ID: {info['deployment_id']}")
    print(f"  Access Key: {info['access_key']}")
    
    print("\n🔗 Access Your Model:")
    print("  1. Go to the CML UI")
    print("  2. Navigate to Models section")
    print(f"  3. Find 'Fraud Detection Model' (ID: {info['model_id']})")
    print("  4. Click on the model to see deployment details")
    print("  5. Use the 'Test Model' feature or copy the API endpoint")
    
    print("\n📝 API Usage:")
    print("  The model expects a JSON payload with transaction features:")
    print("  {")
    print('    "Amount_clean": 250.0,')
    print('    "hour": 14,')
    print('    "day_of_week": 2,')
    print('    "is_weekend": 0,')
    print('    "is_online": 1,')
    print('    "..."')
    print("  }")
    
    print("\n🧪 Test the model:")
    print("  python test_cml_model.py")

def main():
    """Main deployment function"""
    print("="*60)
    print("CML Fraud Detection Model Deployment")
    print("="*60)
    
    # Deploy the model
    model_id, build_id, deployment_id = create_cml_model()
    
    if all([model_id, build_id, deployment_id]):
        show_usage_instructions()
        print("\n✅ Deployment completed successfully!")
        return 0
    else:
        print("\n❌ Deployment failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())