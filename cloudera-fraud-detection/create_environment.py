#!/usr/bin/env python3
"""
Set up a Python environment with required dependencies using uv.
This is typically the first job in the workflow.
"""
import os
import subprocess
import sys
import shutil

def create_environment():
    print("Creating Python environment using uv...")
    
    # Environment name
    env_name = "project_env"
    
    # Get current directory for requirements.txt
    print("Using current directory for requirements.txt")
    
    # Check if uv is installed, install if not
    try:
        subprocess.run(["uv", "--version"], check=True, capture_output=True)
        print("uv is already installed.")
    except (subprocess.SubprocessError, FileNotFoundError):
        print("uv not found. Installing uv...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "uv"])
            print("uv installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"Error installing uv: {e}")
            return False
    
    # Get current directory
    current_dir = os.getcwd()
    
    # Path to requirements files
    requirements_file = os.path.join(current_dir, "requirements.txt")
    requirements_lock = os.path.join(current_dir, "requirements.lock")
    
    # Check if requirements.txt exists
    if not os.path.exists(requirements_file):
        print(f"Error: {requirements_file} not found.")
        return False
    
    # Check if the environment already exists
    env_path = os.path.join(current_dir, env_name)
    if os.path.exists(env_path):
        print(f"Environment {env_name} already exists. Removing it to create a fresh one.")
        try:
            shutil.rmtree(env_path)
        except Exception as e:
            print(f"Error removing existing environment: {e}")
            return False
    
    # Create a new virtual environment
    try:
        print(f"Creating new virtual environment: {env_name}")
        subprocess.check_call(["uv", "venv", env_name])
        print(f"Virtual environment '{env_name}' created successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error creating virtual environment: {e}")
        return False
    
    # Install packages from requirements.txt
    try:
        print("Installing packages from requirements.txt...")
        
        # Install directly with uv into the virtual environment
        try:
            # Use uv directly to install packages into the virtual environment
            print("Using uv to install packages directly...")
            subprocess.check_call(["uv", "pip", "install", "--python", os.path.join(env_path, "bin", "python"), "-r", requirements_file])
        except subprocess.CalledProcessError:
            print("Failed to use uv directly. Trying alternative approach...")
            # Try alternative approach with source activation
            activate_cmd = f"source {os.path.join(env_path, 'bin', 'activate')} && pip install -r {requirements_file}"
            subprocess.check_call(activate_cmd, shell=True, executable="/bin/bash")
            
        print("Packages installed successfully.")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing packages: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def is_interactive():
    """Check if we're running in an interactive environment like IPython/Jupyter"""
    try:
        # This will raise exception if not in IPython
        return bool(get_ipython())
    except NameError:
        return False

if __name__ == "__main__":
    success = create_environment()
    
    # List all installed packages
    if success:
        print("\nListing all installed packages in project_env:")
        try:
            # Try to list packages
            env_python = os.path.join(os.getcwd(), "project_env", "bin", "python")
            activate_cmd = f"source {os.path.join(os.getcwd(), 'project_env', 'bin', 'activate')} && pip list"
            subprocess.run(activate_cmd, shell=True, executable="/bin/bash")
        except Exception as e:
            print(f"Error listing packages: {e}")
        
        print("\nEnvironment setup was successful.")
        
        # Only exit if not in an interactive environment
        if not is_interactive():
            sys.exit(0)
    else:
        print("\nEnvironment setup failed.")
        
        # Only exit if not in an interactive environment
        if not is_interactive():
            sys.exit(1)