#!/bin/bash
# Setup script for cloud GPU instance (Lambda Labs, RunPod, etc.)
set -e

echo "=========================================="
echo "Setting up GPU environment"
echo "=========================================="

# Check for GPU
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected:"
    nvidia-smi --query-gpu=name,memory.total --format=csv
else
    echo "Warning: No NVIDIA GPU detected"
fi

# Install uv if not present
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# Sync dependencies
echo "Installing dependencies..."
uv sync

# Activate venv
source .venv/bin/activate

# Setup Kaggle credentials (user needs to provide these)
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo ""
    echo "=========================================="
    echo "KAGGLE CREDENTIALS NEEDED"
    echo "=========================================="
    echo "1. Go to https://www.kaggle.com/settings"
    echo "2. Click 'Create New Token' under API"
    echo "3. Place kaggle.json in ~/.kaggle/"
    echo "4. Run: chmod 600 ~/.kaggle/kaggle.json"
    echo "=========================================="
    echo ""
fi

# Download data if not present
if [ ! -d "data" ] || [ ! -f "data/train.csv" ]; then
    echo "Downloading competition data..."
    mkdir -p data
    kaggle competitions download -c deep-past-initiative-machine-translation -p data/
    unzip -o data/*.zip -d data/
    rm -f data/*.zip
    echo "Data downloaded to data/"
fi

# Make scripts executable
chmod +x scripts/run_experiment.sh

echo ""
echo "=========================================="
echo "SETUP COMPLETE"
echo "=========================================="
echo ""
echo "Quick start:"
echo "  source .venv/bin/activate"
echo "  python experiments/status.py              # Check experiment status"
echo "  ./scripts/run_experiment.sh baseline      # Run baseline experiment"
echo ""
echo "Iteration loop:"
echo "  1. Run: ./scripts/run_experiment.sh <name> <model> <epochs> <batch> <lr>"
echo "  2. Run: python experiments/status.py      # Review results"
echo "  3. Discuss with Claude Code         # Get suggestions"
echo "  4. Repeat"
echo ""
