#!/bin/bash

# AlgoForge 3.0 Superhuman Setup Script
# Installs MCP servers and sets up the superhuman quantitative trading system

set -e

echo "🧠 Setting up AlgoForge 3.0 SUPERHUMAN capabilities..."
echo "============================================================"

# Check if running as root (not recommended)
if [ "$EUID" -eq 0 ]; then
    echo "⚠️ Warning: Running as root. Consider running as a regular user."
fi

# Update package lists
echo "📦 Updating package lists..."
if command -v apt &> /dev/null; then
    sudo apt update
elif command -v yum &> /dev/null; then
    sudo yum check-update
elif command -v brew &> /dev/null; then
    brew update
fi

# Install Node.js if not present
echo "📋 Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "🔧 Installing Node.js..."
    if command -v apt &> /dev/null; then
        curl -fsSL https://deb.nodesource.com/setup_18.x | sudo -E bash -
        sudo apt-get install -y nodejs
    elif command -v yum &> /dev/null; then
        curl -fsSL https://rpm.nodesource.com/setup_18.x | sudo bash -
        sudo yum install -y nodejs npm
    elif command -v brew &> /dev/null; then
        brew install node
    else
        echo "❌ Please install Node.js manually from https://nodejs.org/"
        exit 1
    fi
else
    echo "✅ Node.js found: $(node --version)"
fi

# Install Python 3.11+ if not present
echo "📋 Checking Python installation..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "🔧 Installing Python 3.11..."
    if command -v apt &> /dev/null; then
        sudo apt install -y python3.11 python3.11-venv python3.11-dev
    elif command -v yum &> /dev/null; then
        sudo yum install -y python311 python311-devel
    elif command -v brew &> /dev/null; then
        brew install python@3.11
    else
        echo "❌ Please install Python 3.11+ manually"
        exit 1
    fi
else
    echo "✅ Python found: $python_version"
fi

# Install uv (modern Python package manager)
echo "🔧 Installing uv (Python package manager)..."
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.cargo/env
    echo "✅ uv installed"
else
    echo "✅ uv already installed"
fi

# Install git if not present
echo "📋 Checking Git installation..."
if ! command -v git &> /dev/null; then
    echo "🔧 Installing Git..."
    if command -v apt &> /dev/null; then
        sudo apt install -y git
    elif command -v yum &> /dev/null; then
        sudo yum install -y git
    elif command -v brew &> /dev/null; then
        brew install git
    fi
else
    echo "✅ Git found: $(git --version)"
fi

# Install PostgreSQL (for MCP database server)
echo "📋 Checking PostgreSQL installation..."
if ! command -v psql &> /dev/null; then
    echo "🔧 Installing PostgreSQL..."
    if command -v apt &> /dev/null; then
        sudo apt install -y postgresql postgresql-contrib
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    elif command -v yum &> /dev/null; then
        sudo yum install -y postgresql postgresql-server postgresql-contrib
        sudo postgresql-setup --initdb
        sudo systemctl start postgresql
        sudo systemctl enable postgresql
    elif command -v brew &> /dev/null; then
        brew install postgresql
        brew services start postgresql
    fi
    
    # Create AlgoForge database
    echo "🗄️ Setting up AlgoForge database..."
    sudo -u postgres createuser -s algoforge 2>/dev/null || true
    sudo -u postgres createdb algoforge 2>/dev/null || true
    sudo -u postgres psql -c "ALTER USER algoforge PASSWORD 'algoforge123';" 2>/dev/null || true
    
    echo "✅ PostgreSQL installed and configured"
else
    echo "✅ PostgreSQL found"
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /opt/algoforge-mcp
mkdir -p ~/.config/claude
mkdir -p ./strategies
mkdir -p ./data
mkdir -p ./results
mkdir -p ./logs

# Install QuantConnect MCP Server
echo "🔧 Installing QuantConnect MCP Server..."
if [ ! -d "/opt/quantconnect-mcp" ]; then
    cd /opt
    sudo git clone https://github.com/taylorwilsdon/quantconnect-mcp.git
    sudo chown -R $USER:$USER quantconnect-mcp
    cd quantconnect-mcp
    
    # Install with uv
    uv sync
    echo "✅ QuantConnect MCP Server installed"
else
    echo "✅ QuantConnect MCP Server already exists"
fi

# Return to AlgoForge directory
cd -

# Install Claude Code CLI
echo "🔧 Installing Claude Code CLI..."
npm install -g @anthropic-ai/claude-code
echo "✅ Claude Code CLI installed"

# Install MCP servers
echo "🔧 Installing MCP servers..."

# Install Node.js based MCP servers
npm install -g mcp-trader
npm install -g @modelcontextprotocol/server-postgres  
npm install -g @modelcontextprotocol/server-filesystem
npm install -g @modelcontextprotocol/server-github
npm install -g @modelcontextprotocol/server-brave-search
npm install -g @modelcontextprotocol/server-sequential-thinking

echo "✅ Node.js MCP servers installed"

# Install Python MCP servers
echo "🔧 Installing Python MCP servers..."
pip install finance-tools-mcp
pip install twelve-data-python
echo "✅ Python MCP servers installed"

# Install additional Python dependencies for AlgoForge
echo "🔧 Installing AlgoForge Python dependencies..."
pip install -r requirements.txt
echo "✅ AlgoForge dependencies installed"

# Create environment file template
echo "📝 Creating environment configuration..."
cat > .env.superhuman << 'EOF'
# AlgoForge 3.0 Superhuman Environment Configuration
# Fill in your actual API keys and credentials

# QuantConnect (REQUIRED)
QUANTCONNECT_USER_ID=357130
QUANTCONNECT_API_TOKEN=62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912
QUANTCONNECT_ORGANIZATION_ID=

# Financial Data APIs (OPTIONAL - enhances capabilities)
TWELVE_DATA_API_KEY=
ALPHA_VANTAGE_API_KEY=
POLYGON_API_KEY=

# Database Configuration
POSTGRES_CONNECTION_STRING=postgresql://algoforge:algoforge123@localhost:5432/algoforge

# GitHub Integration (OPTIONAL - for strategy versioning)
GITHUB_PERSONAL_ACCESS_TOKEN=

# Market Research (OPTIONAL - for enhanced research)
BRAVE_API_KEY=

# Trading Execution (OPTIONAL - for live trading)
BROKER_API_KEY=
BROKER_SECRET=

# MCP Server Configuration
MCP_SERVER_HOST=localhost
MCP_SERVER_PORT=3000

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/algoforge_superhuman.log
EOF

echo "✅ Environment template created: .env.superhuman"

# Generate Claude Desktop configuration
echo "📝 Generating Claude Desktop MCP configuration..."
python3 -c "
import json
import os
from pathlib import Path

# Get user home directory
home = Path.home()

# Determine Claude config path based on OS
import platform
system = platform.system()
if system == 'Darwin':  # macOS
    config_path = home / 'Library' / 'Application Support' / 'Claude' / 'claude_desktop_config.json'
elif system == 'Windows':
    config_path = home / 'AppData' / 'Roaming' / 'Claude' / 'claude_desktop_config.json'
else:  # Linux
    config_path = home / '.config' / 'claude' / 'claude_desktop_config.json'

# Create config directory
config_path.parent.mkdir(parents=True, exist_ok=True)

# MCP Server configuration
mcp_config = {
    'globalShortcut': 'Alt+C',
    'mcpServers': {
        'quantconnect': {
            'command': 'uv',
            'args': ['--directory', '/opt/quantconnect-mcp', 'run', 'main.py'],
            'env': {
                'QUANTCONNECT_API_KEY': '62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912',
                'QUANTCONNECT_USER_ID': '357130'
            }
        },
        'finance-data': {
            'command': 'uvx',
            'args': ['finance-tools-mcp']
        },
        'technical-analysis': {
            'command': 'npx',
            'args': ['-y', 'mcp-trader']
        },
        'database': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-postgres'],
            'env': {
                'POSTGRES_CONNECTION_STRING': 'postgresql://algoforge:algoforge123@localhost:5432/algoforge'
            }
        },
        'filesystem': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-filesystem', str(Path.cwd()), str(Path.cwd() / 'strategies'), str(Path.cwd() / 'data')]
        },
        'sequential-thinking': {
            'command': 'npx',
            'args': ['-y', '@modelcontextprotocol/server-sequential-thinking']
        }
    }
}

# Write configuration
with open(config_path, 'w') as f:
    json.dump(mcp_config, f, indent=2)

print(f'✅ Claude Desktop config written to: {config_path}')
print('🔄 Please restart Claude Desktop to load the new MCP servers')
"

# Create startup script
echo "📝 Creating startup script..."
cat > start_superhuman.sh << 'EOF'
#!/bin/bash

# AlgoForge 3.0 Superhuman Startup Script

echo "🧠 Starting AlgoForge 3.0 SUPERHUMAN mode..."

# Load environment
if [ -f ".env.superhuman" ]; then
    export $(cat .env.superhuman | grep -v '^#' | xargs)
fi

# Start PostgreSQL if not running
if ! pgrep -x "postgres" > /dev/null; then
    echo "🗄️ Starting PostgreSQL..."
    if command -v systemctl &> /dev/null; then
        sudo systemctl start postgresql
    elif command -v brew &> /dev/null; then
        brew services start postgresql
    fi
fi

# Start AlgoForge in superhuman mode
echo "🚀 Launching AlgoForge 3.0 Superhuman..."
python3 algoforge_main.py

EOF

chmod +x start_superhuman.sh

# Create MCP test script
echo "📝 Creating MCP test script..."
cat > test_mcp.py << 'EOF'
#!/usr/bin/env python3
"""
Test MCP server integration
"""
import asyncio
from mcp_integration import MCPManager

async def test_mcp():
    manager = MCPManager()
    
    print("🧪 Testing MCP server setup...")
    
    # Test server status
    status = manager.get_server_status()
    
    print("\n📊 MCP Server Status:")
    for name, info in status.items():
        icon = "✅" if info['has_required_env'] else "⚠️"
        print(f"  {icon} {name}: {info['description']}")
    
    # Test configuration generation
    config = manager.generate_claude_config()
    print(f"\n🔧 Generated config for {len(config['mcpServers'])} servers")
    
    print("\n🎉 MCP test completed!")

if __name__ == "__main__":
    asyncio.run(test_mcp())
EOF

chmod +x test_mcp.py

# Final setup verification
echo ""
echo "🧪 Running setup verification..."

# Test imports
python3 -c "
try:
    from mcp_integration import MCPManager
    from quantconnect_sync import QuantConnectSyncManager
    print('✅ All Python modules import successfully')
except ImportError as e:
    print(f'❌ Import error: {e}')
"

# Test Node.js modules
echo "📋 Verifying Node.js MCP servers..."
npm list -g --depth=0 | grep -E "(mcp-trader|@modelcontextprotocol)" && echo "✅ Node.js MCP servers verified" || echo "⚠️ Some Node.js MCP servers may not be installed"

echo ""
echo "🎉 AlgoForge 3.0 SUPERHUMAN setup completed!"
echo "=============================================="
echo ""
echo "📋 Next Steps:"
echo "1. Edit .env.superhuman with your API keys"
echo "2. Restart Claude Desktop to load MCP servers"
echo "3. Test MCP integration: python3 test_mcp.py"  
echo "4. Start superhuman mode: ./start_superhuman.sh"
echo ""
echo "🧠 Available Superhuman Capabilities:"
echo "   ├─ QuantConnect MCP: Professional trading platform integration"
echo "   ├─ Finance Data MCP: Comprehensive market data access"
echo "   ├─ Technical Analysis MCP: Advanced chart pattern detection"
echo "   ├─ Database MCP: PostgreSQL integration for data storage"
echo "   ├─ Filesystem MCP: Intelligent file management"
echo "   ├─ Sequential Thinking MCP: Enhanced reasoning capabilities"
echo "   └─ GitHub MCP: Version control integration (optional)"
echo ""
echo "🚀 Your quantitative trading system is now SUPERHUMAN!"
echo "💡 Rate limiting and sync issues are automatically handled"
echo "🔄 Local code changes sync automatically with QuantConnect"
echo ""
echo "🏆 Ready to become the best quant mind in the world!"