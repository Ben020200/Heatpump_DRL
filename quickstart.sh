#!/bin/bash
# Quick Start Script for Heatpump DRL Project

echo "==========================================="
echo "Heat Pump DRL - Quick Start"
echo "==========================================="
echo ""

# Check Python version
echo "✓ Checking Python version..."
python --version

# Install dependencies
echo ""
echo "✓ Installing dependencies..."
pip install -q -r requirements.txt

# Generate weather data
echo ""
echo "✓ Generating weather datasets..."
python utils/weather_generator.py

# Test environment
echo ""
echo "✓ Testing thermal environment..."
python test_env.py

echo ""
echo "==========================================="
echo "Setup Complete!"
echo "==========================================="
echo ""
echo "Next steps:"
echo "  1. Train a model:"
echo "     python agents/train_dqn.py --timesteps 50000 --name test_run"
echo ""
echo "  2. Or start with the Jupyter notebook:"
echo "     jupyter notebook notebooks/analysis.ipynb"
echo ""
echo "  3. Read the documentation:"
echo "     README.md and PROJECT_GUIDELINE.md"
echo ""
