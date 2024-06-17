#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "Creating and navigating to project directory..."
mkdir -p blip3
cd blip3

echo "Setting up a virtual environment..."
python3 -m venv venv
source venv/bin/activate

echo "Installing Git LFS..."
git lfs install

echo "Cloning the model repository..."
echo TAKES A WHILE IF SLOW INTERNET...
git clone https://huggingface.co/Salesforce/xgen-mm-phi3-mini-instruct-r-v1

echo "Installing dependencies..."
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt

echo "Running tests..."
python test.py

echo "Setup complete!"
