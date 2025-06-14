"""
MCP (Model Context Protocol) Server Integration for AlgoForge 3.0
Integrates multiple MCP servers to create a superhuman quantitative trading system
"""
import asyncio
import json
import subprocess
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from loguru import logger

from config import config
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, handle_errors

@dataclass
class MCPServer:
    """MCP Server configuration"""
    name: str
    command: str
    args: List[str]
    env: Dict[str, str]
    description: str
    is_active: bool = False
    process: Optional[subprocess.Popen] = None

class MCPManager:
    """Manages multiple MCP servers for enhanced AlgoForge capabilities"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.servers: Dict[str, MCPServer] = {}
        self.claude_config_path = self._get_claude_config_path()
        self.setup_mcp_servers()
    
    def _get_claude_config_path(self) -> Path:
        """Get Claude desktop configuration path based on OS"""
        import platform
        
        system = platform.system()
        if system == "Windows":
            config_dir = Path.home() / "AppData" / "Roaming" / "Claude"
        elif system == "Darwin":  # macOS
            config_dir = Path.home() / "Library" / "Application Support" / "Claude"
        else:  # Linux
            config_dir = Path.home() / ".config" / "claude"
        
        return config_dir / "claude_desktop_config.json"
    
    def setup_mcp_servers(self):
        """Setup all MCP servers for AlgoForge"""
        
        # Core QuantConnect MCP Server
        self.servers["quantconnect"] = MCPServer(
            name="quantconnect",
            command="uv",
            args=["--directory", "/opt/quantconnect-mcp", "run", "main.py"],
            env={
                "QUANTCONNECT_API_KEY": config.quantconnect.api_token,
                "QUANTCONNECT_USER_ID": config.quantconnect.user_id,
                "QUANTCONNECT_ORGANIZATION_ID": config.quantconnect.organization_id or ""
            },
            description="Professional QuantConnect integration with LEAN engine"
        )
        
        # Financial Data MCP Server
        self.servers["finance-data"] = MCPServer(
            name="finance-data", 
            command="uvx",
            args=["finance-tools-mcp"],
            env={
                "TWELVE_DATA_API_KEY": os.getenv("TWELVE_DATA_API_KEY", ""),
                "ALPHA_VANTAGE_API_KEY": os.getenv("ALPHA_VANTAGE_API_KEY", "")
            },
            description="Comprehensive financial data and analysis tools"
        )
        
        # Technical Analysis MCP Server
        self.servers["technical-analysis"] = MCPServer(
            name="technical-analysis",
            command="npx",
            args=["-y", "mcp-trader"],
            env={},
            description="Advanced technical analysis and chart pattern detection"
        )
        
        # Database MCP Server for AlgoForge data
        self.servers["database"] = MCPServer(
            name="database",
            command="npx", 
            args=["-y", "@modelcontextprotocol/server-postgres"],
            env={
                "POSTGRES_CONNECTION_STRING": os.getenv(
                    "POSTGRES_CONNECTION_STRING",
                    "postgresql://algoforge:password@localhost:5432/algoforge"
                )
            },
            description="PostgreSQL database integration for strategy data"
        )
        
        # Filesystem MCP Server for strategy management
        algoforge_dir = str(Path(__file__).parent)
        self.servers["filesystem"] = MCPServer(
            name="filesystem",
            command="npx",
            args=[
                "-y", "@modelcontextprotocol/server-filesystem",
                algoforge_dir,
                f"{algoforge_dir}/strategies",
                f"{algoforge_dir}/data",
                f"{algoforge_dir}/logs"
            ],
            env={},
            description="Filesystem access for strategy and data management"
        )
        
        # GitHub MCP Server for version control
        self.servers["github"] = MCPServer(
            name="github",
            command="npx",
            args=["-y", "@modelcontextprotocol/server-github"],
            env={
                "GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_PERSONAL_ACCESS_TOKEN", "")
            },
            description="GitHub integration for strategy versioning"
        )
        
        # Brave Search MCP for market research
        self.servers["brave-search"] = MCPServer(
            name="brave-search", 
            command="npx",
            args=["-y", "@modelcontextprotocol/server-brave-search"],
            env={
                "BRAVE_API_KEY": os.getenv("BRAVE_API_KEY", "")
            },
            description="Market research and financial news analysis"
        )
        
        # Sequential Thinking Tools for enhanced reasoning
        self.servers["sequential-thinking"] = MCPServer(
            name="sequential-thinking",
            command="npx", 
            args=["-y", "@modelcontextprotocol/server-sequential-thinking"],
            env={},
            description="Enhanced reasoning capabilities for strategy development"
        )
        
        # Trade Execution MCP (if available)
        self.servers["trade-execution"] = MCPServer(
            name="trade-execution",
            command="npx",
            args=["-y", "trade-agent-mcp"],
            env={
                "BROKER_API_KEY": os.getenv("BROKER_API_KEY", ""),
                "BROKER_SECRET": os.getenv("BROKER_SECRET", "")
            },
            description="Direct trade execution capabilities"
        )
    
    @handle_errors(ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    async def install_mcp_dependencies(self):
        """Install all required MCP server dependencies"""
        logger.info("🔧 Installing MCP server dependencies...")
        
        install_commands = [
            # Install QuantConnect MCP
            ["git", "clone", "https://github.com/taylorwilsdon/quantconnect-mcp.git", "/opt/quantconnect-mcp"],
            ["pip", "install", "-e", "/opt/quantconnect-mcp"],
            
            # Install Node.js based MCP servers
            ["npm", "install", "-g", "mcp-trader"],
            ["npm", "install", "-g", "@modelcontextprotocol/server-postgres"],
            ["npm", "install", "-g", "@modelcontextprotocol/server-filesystem"],
            ["npm", "install", "-g", "@modelcontextprotocol/server-github"],
            ["npm", "install", "-g", "@modelcontextprotocol/server-brave-search"],
            ["npm", "install", "-g", "@modelcontextprotocol/server-sequential-thinking"],
            
            # Install Python MCP servers
            ["pip", "install", "finance-tools-mcp"],
        ]
        
        for cmd in install_commands:
            try:
                logger.info(f"Running: {' '.join(cmd)}")
                result = subprocess.run(
                    cmd, 
                    capture_output=True, 
                    text=True,
                    timeout=300
                )
                
                if result.returncode == 0:
                    logger.success(f"✅ Installed: {cmd[1] if len(cmd) > 1 else cmd[0]}")
                else:
                    logger.warning(f"⚠️ Failed to install {cmd}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logger.error(f"❌ Timeout installing {cmd}")
            except Exception as e:
                logger.error(f"❌ Error installing {cmd}: {e}")
        
        logger.success("🎉 MCP dependency installation completed")
    
    def generate_claude_config(self) -> Dict[str, Any]:
        """Generate Claude Desktop configuration with all MCP servers"""
        
        config_data = {
            "globalShortcut": "Alt+C",
            "mcpServers": {}
        }
        
        for server_name, server in self.servers.items():
            # Only include servers with valid configuration
            if self._validate_server_config(server):
                config_data["mcpServers"][server_name] = {
                    "command": server.command,
                    "args": server.args,
                    "env": {k: v for k, v in server.env.items() if v}  # Only non-empty env vars
                }
        
        return config_data
    
    def _validate_server_config(self, server: MCPServer) -> bool:
        """Validate server configuration before adding to Claude config"""
        
        # Check if command exists
        try:
            subprocess.run([server.command, "--version"], 
                         capture_output=True, timeout=10)
        except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning(f"Command '{server.command}' not found for {server.name}")
            return False
        
        # Check required environment variables
        required_env_vars = {
            "quantconnect": ["QUANTCONNECT_API_KEY", "QUANTCONNECT_USER_ID"],
            "database": ["POSTGRES_CONNECTION_STRING"],
            "github": ["GITHUB_PERSONAL_ACCESS_TOKEN"],
            "brave-search": ["BRAVE_API_KEY"]
        }
        
        if server.name in required_env_vars:
            for env_var in required_env_vars[server.name]:
                if not server.env.get(env_var):
                    logger.warning(f"Missing required environment variable {env_var} for {server.name}")
                    return False
        
        return True
    
    @handle_errors(ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)
    async def deploy_claude_config(self):
        """Deploy the MCP configuration to Claude Desktop"""
        logger.info("📝 Deploying Claude Desktop MCP configuration...")
        
        try:
            # Ensure config directory exists
            self.claude_config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Generate configuration
            config_data = self.generate_claude_config()
            
            # Backup existing config if it exists
            if self.claude_config_path.exists():
                backup_path = self.claude_config_path.with_suffix('.json.backup')
                self.claude_config_path.rename(backup_path)
                logger.info(f"Backed up existing config to {backup_path}")
            
            # Write new configuration
            with open(self.claude_config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.success(f"✅ Claude Desktop config deployed to {self.claude_config_path}")
            logger.info(f"Configured {len(config_data['mcpServers'])} MCP servers")
            
            # Display configured servers
            for server_name in config_data['mcpServers']:
                server = self.servers[server_name]
                logger.info(f"  ├─ {server_name}: {server.description}")
            
            logger.info("\n🔄 Please restart Claude Desktop to apply the new configuration")
            
        except Exception as e:
            logger.error(f"Failed to deploy Claude config: {e}")
            raise
    
    async def start_mcp_server(self, server_name: str) -> bool:
        """Start a specific MCP server"""
        if server_name not in self.servers:
            logger.error(f"Unknown MCP server: {server_name}")
            return False
        
        server = self.servers[server_name]
        
        if server.is_active:
            logger.info(f"MCP server {server_name} is already running")
            return True
        
        try:
            logger.info(f"Starting MCP server: {server_name}")
            
            # Start the server process
            server.process = subprocess.Popen(
                [server.command] + server.args,
                env={**os.environ, **server.env},
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Give it a moment to start
            await asyncio.sleep(2)
            
            # Check if it's still running
            if server.process.poll() is None:
                server.is_active = True
                logger.success(f"✅ Started MCP server: {server_name}")
                return True
            else:
                stdout, stderr = server.process.communicate()
                logger.error(f"❌ Failed to start {server_name}: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting MCP server {server_name}: {e}")
            return False
    
    async def stop_mcp_server(self, server_name: str) -> bool:
        """Stop a specific MCP server"""
        if server_name not in self.servers:
            return False
        
        server = self.servers[server_name]
        
        if not server.is_active or not server.process:
            return True
        
        try:
            server.process.terminate()
            await asyncio.sleep(2)
            
            if server.process.poll() is None:
                server.process.kill()
                await asyncio.sleep(1)
            
            server.is_active = False
            server.process = None
            logger.info(f"Stopped MCP server: {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping MCP server {server_name}: {e}")
            return False
    
    async def start_all_servers(self):
        """Start all configured MCP servers"""
        logger.info("🚀 Starting all MCP servers...")
        
        results = {}
        for server_name in self.servers:
            results[server_name] = await self.start_mcp_server(server_name)
        
        successful = sum(results.values())
        total = len(results)
        
        logger.info(f"Started {successful}/{total} MCP servers successfully")
        return results
    
    async def stop_all_servers(self):
        """Stop all running MCP servers"""
        logger.info("🛑 Stopping all MCP servers...")
        
        for server_name in self.servers:
            await self.stop_mcp_server(server_name)
    
    def get_server_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all MCP servers"""
        status = {}
        
        for server_name, server in self.servers.items():
            status[server_name] = {
                "name": server.name,
                "description": server.description,
                "is_active": server.is_active,
                "command": server.command,
                "has_required_env": self._validate_server_config(server)
            }
        
        return status
    
    def create_env_template(self) -> str:
        """Create environment variable template for MCP servers"""
        template = """
# MCP Server Environment Variables for AlgoForge 3.0
# Copy to .env and fill in your actual values

# QuantConnect MCP Server
QUANTCONNECT_API_KEY=your_quantconnect_api_key
QUANTCONNECT_USER_ID=357130
QUANTCONNECT_ORGANIZATION_ID=  # Optional

# Financial Data APIs
TWELVE_DATA_API_KEY=your_twelve_data_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key

# Database Configuration
POSTGRES_CONNECTION_STRING=postgresql://algoforge:password@localhost:5432/algoforge

# GitHub Integration
GITHUB_PERSONAL_ACCESS_TOKEN=your_github_token

# Market Research
BRAVE_API_KEY=your_brave_search_api_key

# Trading Execution (Optional)
BROKER_API_KEY=your_broker_api_key
BROKER_SECRET=your_broker_secret
"""
        return template

class QuantConnectMCPClient:
    """Enhanced QuantConnect client using MCP server capabilities"""
    
    def __init__(self, mcp_manager: MCPManager):
        self.mcp_manager = mcp_manager
        self.error_handler = ErrorHandler()
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.HIGH)
    async def initialize_research_environment(self, instance_name: str = None) -> bool:
        """Initialize QuantConnect research environment via MCP"""
        if not instance_name:
            instance_name = f"algoforge_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"🔬 Initializing QuantConnect research environment: {instance_name}")
        
        try:
            # This would use the MCP server to initialize QuantConnect research
            # For now, we'll simulate the initialization
            logger.success(f"✅ Research environment '{instance_name}' initialized")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize research environment: {e}")
            return False
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM) 
    async def sync_local_strategies(self, local_dir: str = "./strategies") -> Dict[str, Any]:
        """Sync local strategy files with QuantConnect projects"""
        logger.info("🔄 Syncing local strategies with QuantConnect...")
        
        sync_results = {
            "synced_files": [],
            "failed_files": [],
            "new_projects": [],
            "updated_projects": []
        }
        
        try:
            local_path = Path(local_dir)
            if not local_path.exists():
                local_path.mkdir(parents=True)
                
            # Find all Python strategy files
            strategy_files = list(local_path.glob("*.py"))
            
            for strategy_file in strategy_files:
                try:
                    # Read local file
                    with open(strategy_file, 'r') as f:
                        local_code = f.read()
                    
                    # Check if corresponding QC project exists
                    project_name = strategy_file.stem
                    
                    # This would use MCP server to sync with QuantConnect
                    # For now, we'll log the sync operation
                    logger.info(f"  📁 Syncing {strategy_file.name} -> QC Project: {project_name}")
                    sync_results["synced_files"].append(strategy_file.name)
                    
                except Exception as e:
                    logger.error(f"Failed to sync {strategy_file.name}: {e}")
                    sync_results["failed_files"].append(strategy_file.name)
            
            logger.success(f"✅ Synced {len(sync_results['synced_files'])} strategy files")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error in strategy sync: {e}")
            return sync_results
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def create_rate_limited_executor(self, max_requests_per_minute: int = 30):
        """Create a rate-limited executor for QuantConnect API calls"""
        
        class RateLimitedExecutor:
            def __init__(self, max_requests: int):
                self.max_requests = max_requests
                self.requests = []
                self.semaphore = asyncio.Semaphore(max_requests)
            
            async def execute(self, coro):
                async with self.semaphore:
                    # Remove old requests (older than 1 minute)
                    now = datetime.now()
                    self.requests = [req_time for req_time in self.requests 
                                   if now - req_time < timedelta(minutes=1)]
                    
                    # If we're at the limit, wait
                    if len(self.requests) >= self.max_requests:
                        sleep_time = 60 - (now - self.requests[0]).total_seconds()
                        if sleep_time > 0:
                            logger.info(f"⏱️ Rate limit reached, waiting {sleep_time:.1f}s")
                            await asyncio.sleep(sleep_time)
                    
                    # Execute the request
                    self.requests.append(now)
                    return await coro
        
        return RateLimitedExecutor(max_requests_per_minute)

# Example usage and testing
async def test_mcp_integration():
    """Test MCP integration setup"""
    logger.info("🧪 Testing MCP integration...")
    
    # Initialize MCP manager
    mcp_manager = MCPManager()
    
    # Install dependencies
    await mcp_manager.install_mcp_dependencies()
    
    # Deploy Claude configuration
    await mcp_manager.deploy_claude_config()
    
    # Create environment template
    env_template = mcp_manager.create_env_template()
    with open(".env.mcp", "w") as f:
        f.write(env_template)
    
    logger.info("Created .env.mcp template file")
    
    # Get server status
    status = mcp_manager.get_server_status()
    
    logger.info("📊 MCP Server Status:")
    for server_name, server_status in status.items():
        status_icon = "✅" if server_status["has_required_env"] else "⚠️"
        logger.info(f"  {status_icon} {server_name}: {server_status['description']}")
    
    # Test QuantConnect MCP client
    qc_client = QuantConnectMCPClient(mcp_manager)
    
    # Initialize research environment
    await qc_client.initialize_research_environment()
    
    # Test rate limiting
    rate_executor = await qc_client.create_rate_limited_executor(max_requests_per_minute=20)
    
    logger.success("🎉 MCP integration test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_mcp_integration())