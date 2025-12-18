#!/bin/bash
# Setup script for Fraud Detection project
# Creates virtual environment and installs all dependencies

set -e

echo "============================================================"
echo "  Fraud Detection - Environment Setup"
echo "============================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    python3 -m venv venv
    echo "Virtual environment created."
else
    echo "Virtual environment already exists."
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Unset any pip user config that might interfere
unset PIP_USER

# Upgrade pip (without --user flag)
echo ""
echo "Upgrading pip..."
pip install --upgrade pip --no-user

# Install requirements (without --user flag)
echo ""
echo "Installing dependencies from requirements.txt..."
pip install -r requirements.txt --no-user

echo ""
echo "============================================================"
echo "  Setup Complete!"
echo "============================================================"
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "Next steps:"
echo "  1. Setup Kaggle credentials:"
echo "     cp config/kaggle.yaml.example config/kaggle.yaml"
echo "     # Edit config/kaggle.yaml with your credentials"
echo ""
echo "  2. Download dataset:"
echo "     python scripts/download_kaggle_data.py"
echo ""
echo "  3. Train model:"
echo "     python scripts/train_kaggle_model.py"
echo ""
