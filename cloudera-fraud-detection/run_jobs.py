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

# Get template directory from environment or use default
template_dir = os.environ.get("TEMPLATE_DIR", "template")
print(f"Using template directory: {template_dir}")

# Load environment variables from .env file
env_path = os.path.join(template_dir, ".env")
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
    
    # Try different possible paths for the config file
    possible_paths = [
        # Direct path (when run from root of project)
        os.path.join(script_dir, "config", "jobs_config.yaml"),
        # When run from inside a cloned directory
        os.path.join(script_dir, template_dir, "config", "jobs_config.yaml"),
        # When the script is in a subdirectory
        os.path.join(os.path.dirname(script_dir), "config", "jobs_config.yaml"),
    ]
    
    config_path = None
    for path in possible_paths:
        if os.path.exists(path):
            config_path = path
            break
    
    if not config_path:
        raise FileNotFoundError(f"Could not find jobs_config.yaml in any of these locations: {possible_paths}")
    
    print(f"Loading config from: {config_path}")
    
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    
    return config['jobs']

def adjust_job_paths(job_config):
    """
    Adjust job paths to use the template directory from environment variable
    
    Args:
        job_config (dict): The job configuration dictionary
        
    Returns:
        dict: The updated job configuration
    """
    # Make a copy to avoid modifying the original
    config = dict(job_config)
    
    # Only modify paths that start with 'template/'
    if 'script' in config and config['script'].startswith('template/'):
        # Replace the hardcoded 'template/' with the actual template directory
        config['script'] = config['script'].replace('template/', f"{template_dir}/")
    
    return config

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
        job_config = adjust_job_paths(JOBS_CONFIG[job_key])
        
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
        
        # Add TEMPLATE_DIR to environment variables
        if not hasattr(job_body, 'environment'):
            job_body.environment = {}
        job_body.environment['TEMPLATE_DIR'] = template_dir
        
        try:
            # Create the job
            job_response = client.create_job(job_body, project_id=project_id)
            job_id = job_response.id
            job_id_map[job_key] = job_id
            print(f"Successfully created environment job with ID: {job_id}")
            
            # Run the job immediately
            print(f"Running environment setup job...")
            run_response = client.start_job(job_id, project_id=project_id)
            print(f"Environment setup job started with run ID: {run_response.id}")
        except Exception as e:
            print(f"Error creating/running environment job: {str(e)}")
    
    # Process all other jobs
    for job_key in JOBS_CONFIG:
        if job_key == "create_env":
            continue  # Skip as we've already processed it
            
        job_config = adjust_job_paths(JOBS_CONFIG[job_key])
        
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
        
        # Add TEMPLATE_DIR to environment variables
        if not hasattr(job_body, 'environment'):
            job_body.environment = {}
        job_body.environment['TEMPLATE_DIR'] = template_dir
        
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

if __name__ == "__main__":
    try:
        job_ids = setup_jobs()
        
        print("\nJob setup complete. Created jobs:")
        for job_name, job_id in job_ids.items():
            print(f"- {job_name}: {job_id}")
            
    except ValueError as e:
        print(f"Error: {str(e)}")
        print("\nPlease set the required environment variables in the .env file:")
        print("CML_API_HOST, CML_API_KEY, and CML_PROJECT_ID")
        print("For ML Runtime projects, also set CML_RUNTIME_ID")
        sys.exit(1)