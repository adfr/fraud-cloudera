#!/usr/bin/env python3
"""
Job script to run transaction scoring for fraud detection
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
    Main function to run score_transactions.py
    """
    try:
        # Get the directory where this script is located
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to the score_transactions.py script
        score_script = os.path.join(script_dir, "score_transactions.py")
        
        # Check if the script exists
        if not os.path.exists(score_script):
            raise FileNotFoundError(f"Could not find score_transactions.py at {score_script}")
        
        # Get path to the project_env Python executable
        project_root = os.path.dirname(script_dir)  # Go up one level from scripts/
        env_python = os.path.join(
            project_root, 
            "project_env",
            "bin",
            "python"
        )
        
        logger.info(f"Using Python executable: {env_python}")
        
        # Check if the environment exists
        if not os.path.exists(env_python):
            logger.warning(f"Virtual environment not found at {env_python}")
            logger.info("Falling back to system Python")
            env_python = sys.executable
        
        # Command to run script with project_env Python
        cmd = f"{env_python} {score_script}"
        
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
            
        logger.info("Transaction scoring completed successfully")
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Error executing command: {e}")
        logger.error(f"Command stderr: {e.stderr}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
