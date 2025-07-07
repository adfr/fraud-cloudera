#!/usr/bin/env python3
"""
Main script to set up and run CML jobs based on configuration.
This script reads job definitions from config/jobs_config.yaml and creates them in CML.
"""
import os
import sys
import yaml
import cmlapi
import subprocess

# Install python-dotenv if not present
!pip install python-dotenv
# Load environment variables from .env file
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = "fraud-cloudera/cloudera-fraud-detection/.env"
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path)
else:
    load_dotenv()  # Fall back to default .env in current directory

project_id = os.environ["CDSW_PROJECT_ID"]

def load_config():
    """
    Load the job configuration from the YAML file
    
    Returns:
        dict: The job configuration dictionary
    """
    # Get the directory where this script is located
    script_dir = os.getcwd()
    
    # Config file is in the config directory relative to current working directory
    config_path = os.path.join(script_dir, "fraud-cloudera" , "cloudera-fraud-detection","config", "jobs_config.yaml")
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Could not find jobs_config.yaml at: {config_path}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    return config['jobs']


def setup_jobs():
    """
    Set up and create CML jobs based on configurations in config/jobs_config.yaml
    using environment variables from .env file
    
    Returns:
        dict: Mapping of job names to their created job IDs
    """
    # Get parameters from environment variables
    api_host = os.environ.get("CML_API_HOST")
    api_key = os.environ.get("CML_API_KEY")
    project_id = os.environ.get("CML_PROJECT_ID")
    default_runtime_id = os.environ.get("CML_RUNTIME_ID")
    
    # Check if required parameters are available
    if not all([api_host, api_key, project_id]):
        missing = []
        if not api_host:
            missing.append("CML_API_HOST")
        if not api_key:
            missing.append("CML_API_KEY")
        if not project_id:
            missing.append("CML_PROJECT_ID")
            
        raise ValueError(f"Missing required environment variables: {', '.join(missing)}")
    
    print(f"Setting up jobs for project: {project_id}")
    
    # Load the job configuration
    JOBS_CONFIG = load_config()
    
    # Initialize CML API client
    client = cmlapi.default_client(api_host, api_key)
    
    # Dictionary to map job names to their IDs
    job_id_map = {}
    
    # Process jobs in order (create_env first, then others)
    # Process create_env job first if it exists
    if "create_env" in JOBS_CONFIG:
        job_key = "create_env"
        job_config = JOBS_CONFIG[job_key]
        
        # Create and run the environment setup job
        print(f"Creating and running environment setup job: {job_config['name']}")
        
        # Create job request
        job_body = cmlapi.CreateJobRequest()
        
        # Set basic job parameters
        job_body.name = job_config["name"]
        job_body.script = job_config["script"]
        job_body.kernel = job_config.get("kernel", "python3")
        
        # Set runtime ID
        if "runtime_id" in job_config:
            job_body.runtime_identifier = str(job_config["runtime_id"])
        elif default_runtime_id:
            job_body.runtime_identifier = str(default_runtime_id)
        
        # Set resource requirements
        job_body.cpu = str(job_config.get("cpu", os.environ.get("DEFAULT_CPU", "1")))
        job_body.memory = str(job_config.get("memory", os.environ.get("DEFAULT_MEMORY", "2")))
        if "nvidia_gpu" in job_config:
            job_body.nvidia_gpu = job_config["nvidia_gpu"]
        
        # Set timeout
        job_body.timeout = str(job_config.get("timeout", os.environ.get("DEFAULT_TIMEOUT", "3600")))
        
        # Set arguments if provided
        if "arguments" in job_config:
            job_body.arguments = job_config["arguments"]
        
        # Set environment variables if provided
        if "environment" in job_config:
            job_body.environment = job_config["environment"]
        
        
        try:
            # Create the job
            job_response = client.create_job(job_body, project_id=project_id)
            job_id = job_response.id
            job_id_map[job_key] = job_id
            print(f"Successfully created environment job with ID: {job_id}")
            
            # Don't run immediately - we'll run all jobs at the end
        except Exception as e:
            print(f"Error creating/running environment job: {str(e)}")
    
    # Process all other jobs
    for job_key in JOBS_CONFIG:
        if job_key == "create_env":
            continue  # Skip as we've already processed it
            
        job_config = JOBS_CONFIG[job_key]
        
        print(f"Creating job: {job_config['name']}")
        
        # Create job request
        job_body = cmlapi.CreateJobRequest()
        
        # Set basic job parameters
        job_body.name = job_config["name"]
        job_body.script = job_config["script"]
        job_body.kernel = job_config.get("kernel", "python3")
        
        # Set runtime ID - required for ML Runtime projects
        if "runtime_id" in job_config:
            job_body.runtime_identifier = str(job_config["runtime_id"])
        elif default_runtime_id:
            job_body.runtime_identifier = str(default_runtime_id)
        else:
            print(f"Warning: No runtime_id specified for job '{job_config['name']}'.")
            print("ML Runtime projects require a runtime_id. Set it in config or with CML_RUNTIME_ID env var.")
        
        # Set resource requirements
        job_body.cpu = str(job_config.get("cpu", os.environ.get("DEFAULT_CPU", "1")))
        job_body.memory = str(job_config.get("memory", os.environ.get("DEFAULT_MEMORY", "2")))
        if "nvidia_gpu" in job_config:
            job_body.nvidia_gpu = job_config["nvidia_gpu"]
        
        # Set timeout
        job_body.timeout = str(job_config.get("timeout", os.environ.get("DEFAULT_TIMEOUT", "3600")))
        
        # Set arguments if provided
        if "arguments" in job_config:
            job_body.arguments = job_config["arguments"]
        
        # Set environment variables if provided
        if "environment" in job_config:
            job_body.environment = job_config["environment"]
        
        
        # Set attachments if provided
        if "attachments" in job_config:
            job_body.attachments = job_config["attachments"]
        
        # Set scheduling parameters
        if "schedule" in job_config:
            job_body.schedule = job_config["schedule"]
        elif "parent_job_id" in job_config:
            # If this is a dependent job, use the ID from our map
            parent_key = job_config["parent_job_id"]
            if parent_key in job_id_map:
                job_body.parent_job_id = job_id_map[parent_key]
            else:
                print(f"Warning: Parent job '{parent_key}' not found. Job will not be dependent.")
        
        # Create the job
        try:
            print(f"Job details for '{job_config['name']}':")
            print(f"  Script: {job_body.script}")
            print(f"  Kernel: {job_body.kernel}")
            print(f"  Runtime ID: {job_body.runtime_identifier}")
            print(f"  CPU: {job_body.cpu}")
            print(f"  Memory: {job_body.memory}")
            print(f"  GPU: {job_body.nvidia_gpu if hasattr(job_body, 'nvidia_gpu') else 'None'}")
            print(f"  Timeout: {job_body.timeout}")
            print(f"  Arguments: {job_body.arguments if hasattr(job_body, 'arguments') else 'None'}")
            print(f"  Environment: {job_body.environment if hasattr(job_body, 'environment') else 'None'}")
            print(f"  Attachments: {job_body.attachments if hasattr(job_body, 'attachments') else 'None'}")
            print(f"  Schedule: {job_body.schedule if hasattr(job_body, 'schedule') else 'None'}")
            print(f"  Parent Job ID: {job_body.parent_job_id if hasattr(job_body, 'parent_job_id') else 'None'}")
            job_response = client.create_job(job_body, project_id=project_id)
            job_id = job_response.id
            job_id_map[job_key] = job_id
            print(f"Successfully created job '{job_config['name']}' with ID: {job_id}")
        except Exception as e:
            print(f"Error creating job '{job_config['name']}': {str(e)}")
    
    return job_id_map

def run_jobs_sequentially(client, project_id, job_id_map):
    """
    Run jobs sequentially, waiting for each to complete before starting the next
    """
    import time
    
    # Define job order based on dependencies
    job_order = ["create_env", "feature_engineering", "train_model", "score_transactions"]
    
    for job_key in job_order:
        if job_key not in job_id_map:
            continue
            
        job_id = job_id_map[job_key]
        print(f"\nStarting job: {job_key} (ID: {job_id})")
        
        try:
            # Create job run
            job_run = client.create_job_run(
                cmlapi.CreateJobRunRequest(), 
                project_id=project_id,
                job_id=job_id
            )
            print(f"Job run created with ID: {job_run.id}")
            
            # Monitor job status
            while True:
                # Get latest job run status
                job_runs = client.list_job_runs(
                    project_id=project_id, 
                    job_id=job_id, 
                    sort="-created_at", 
                    page_size=1
                )
                
                if job_runs.job_runs:
                    latest_run = job_runs.job_runs[0]
                    status = latest_run.status
                    print(f"Job {job_key} status: {status}")
                    
                    if status in ["succeeded", "failed", "stopped"]:
                        if status == "succeeded":
                            print(f"Job {job_key} completed successfully")
                        else:
                            print(f"Job {job_key} failed with status: {status}")
                            return False
                        break
                
                # Wait before checking again
                time.sleep(10)
                
        except Exception as e:
            print(f"Error running job {job_key}: {str(e)}")
            return False
    
    return True

if __name__ == "__main__":
    try:
        # Initialize CML API client
        api_host = os.environ.get("CML_API_HOST")
        api_key = os.environ.get("CML_API_KEY")
        project_id = os.environ.get("CML_PROJECT_ID")
        
        client = cmlapi.default_client(api_host, api_key)
        
        # Create all jobs
        job_ids = setup_jobs()
        
        print("\nJob setup complete. Created jobs:")
        for job_name, job_id in job_ids.items():
            print(f"- {job_name}: {job_id}")
        
        # Run jobs sequentially
        print("\n" + "="*50)
        print("Starting sequential job execution...")
        print("="*50)
        
        success = run_jobs_sequentially(client, project_id, job_ids)
        
        if success:
            print("\nAll jobs completed successfully!")
        else:
            print("\nJob execution failed!")
            sys.exit(1)
            
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("\nPlease set the required environment variables in the .env file:")
        print("CML_API_HOST, CML_API_KEY, and CML_PROJECT_ID")
        print("For ML Runtime projects, also set CML_RUNTIME_ID")
        sys.exit(1)