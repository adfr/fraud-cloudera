#!/usr/bin/env python3
"""
Example job script that activates a Python environment and runs another script
"""

import subprocess
import os
import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def main():
    """
    Main function to activate environment and run hello_world.py
    """
    try:
        # Get template directory name from environment variable or use default
        template_dir = os.environ.get("TEMPLATE_DIR", "template")
        logger.info(f"Using template directory: {template_dir}")
        
        # Hard-coded path to the hello_world.py script
        hello_world_script = os.path.join(template_dir, "scripts", "hello_world.py")
        
        # Check if the script exists
        if not os.path.exists(hello_world_script):
            raise FileNotFoundError(f"Could not find hello_world.py at {hello_world_script}")
        
        # Get path to the project_env Python executable
        # Use current working directory instead of __file__ which doesn't work in IPython
        project_root = os.getcwd()
        env_python = os.path.join(
            project_root, 
            "project_env",
            "bin",
            "python"
        )
        # Log the Python executable path that will be used
        logger.info(f"Using Python executable: {env_python}")
        # Check if the environment exists
        if not os.path.exists(env_python):
            logger.warning(f"Virtual environment not found at {env_python}")
            logger.info("Falling back to system Python")
            env_python = sys.executable
        
        # Command to run script with project_env Python
        cmd = f"{env_python} {hello_world_script} --name 'CML User' --repeat 3"
        
        logger.info(f"Running command: {cmd}")
        
        # Execute the command in a shell
        process = subprocess.run(
            cmd,
            shell=True,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Print output
        logger.info("Command output:")
        print(process.stdout)
        
        if process.stderr:
            logger.warning("Command stderr:")
            print(process.stderr)
            
        logger.info("Script execution completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
