#!/bin/bash
# CML Model Build Script - installs dependencies for model deployment

echo "Installing model dependencies..."

pip install --no-cache-dir \
    joblib>=1.2.0 \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    scikit-learn>=1.0.0 \
    lightgbm>=3.3.0

echo "Dependencies installed successfully!"
