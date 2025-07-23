#!/bin/bash
# AURA Development Environment Activation
conda activate aura
echo "🚀 AURA development environment activated!"
echo "📁 Project: $(pwd)"
echo "🐍 Python: $(python --version)"
echo "💻 FastAPI: $(python -c 'import fastapi; print(fastapi.__version__)')"
