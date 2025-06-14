#!/bin/bash

# AlgoForge 3.0 Installation Script
# Installs Claude Code SDK, dependencies, and sets up the environment

set -e  # Exit on any error

echo "🚀 Installing AlgoForge 3.0 - Claude Code SDK Powered Quant System"
echo "================================================================="

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
required_version="3.10"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Error: Python 3.10+ is required. Found: $python_version"
    echo "Please install Python 3.10 or newer and try again."
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Check if Node.js is installed (required for Claude Code CLI)
echo "📋 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "❌ Error: Node.js is required for Claude Code CLI"
    echo "Please install Node.js from https://nodejs.org/"
    exit 1
fi

node_version=$(node --version)
echo "✅ Node.js found: $node_version"

# Install Claude Code CLI if not already installed
echo "📋 Checking Claude Code CLI..."
if ! command -v claude-code &> /dev/null; then
    echo "🔧 Installing Claude Code CLI..."
    npm install -g @anthropic-ai/claude-code
    echo "✅ Claude Code CLI installed"
else
    echo "✅ Claude Code CLI already installed"
fi

# Create virtual environment
echo "🔧 Setting up Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "✅ Virtual environment already exists"
fi

# Activate virtual environment
source venv/bin/activate

# Upgrade pip
echo "🔧 Upgrading pip..."
pip install --upgrade pip

# Install Python dependencies
echo "🔧 Installing Python dependencies..."
pip install -r requirements.txt

echo "✅ Python dependencies installed"

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "📝 Creating .env configuration file..."
    cat > .env << EOF
# QuantConnect Configuration
QUANTCONNECT_USER_ID=357130
QUANTCONNECT_API_TOKEN=62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912

# Claude Configuration (optional overrides)
CLAUDE_TEMPERATURE=0.1
CLAUDE_MAX_TURNS=10

# Database Configuration
DATABASE_URL=sqlite:///algoforge_memory.db

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=algoforge.log
EOF
    echo "✅ Created .env file with default configuration"
    echo "📝 Please review and update the .env file with your actual credentials"
else
    echo "✅ .env file already exists"
fi

# Create logs directory
mkdir -p logs
echo "✅ Created logs directory"

# Create data directory for memory database
mkdir -p data
echo "✅ Created data directory"

# Run initial setup and tests
echo "🧪 Running initial system tests..."
python3 -c "
import asyncio
from config import config
print(f'✅ Configuration loaded successfully')
print(f'   - QuantConnect User ID: {config.quantconnect.user_id}')
print(f'   - Database URL: {config.database_url}')
print(f'   - Target CAGR: {config.targets.min_cagr*100:.1f}%')
print(f'   - Target Sharpe: {config.targets.min_sharpe}')
"

# Test imports
echo "🧪 Testing critical imports..."
python3 -c "
try:
    import claude_code_sdk
    print('✅ Claude Code SDK import successful')
except ImportError as e:
    print(f'❌ Claude Code SDK import failed: {e}')
    exit(1)

try:
    import pandas
    import numpy
    import sqlalchemy
    import aiohttp
    print('✅ All critical dependencies import successfully')
except ImportError as e:
    print(f'❌ Dependency import failed: {e}')
    exit(1)
"

echo ""
echo "🎉 AlgoForge 3.0 Installation Complete!"
echo "======================================"
echo ""
echo "📋 Next Steps:"
echo "1. Review and update the .env file with your credentials"
echo "2. Activate the virtual environment: source venv/bin/activate"
echo "3. Run the system: python algoforge_main.py"
echo ""
echo "📚 Quick Start:"
echo "   source venv/bin/activate"
echo "   python algoforge_main.py"
echo ""
echo "🔧 Development Mode:"
echo "   pip install -e .[dev]  # Install development dependencies"
echo "   pre-commit install     # Set up git hooks"
echo ""
echo "📖 For more information, see the documentation in AlgoForge 3.md"
echo ""
echo "Happy trading! 🚀📈"