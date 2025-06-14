"""
QuantConnect API Client for AlgoForge 3.0
Handles all interactions with QuantConnect platform including backtesting, live trading, and result retrieval
"""
import asyncio
import aiohttp
import json
import base64
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timezone
from dataclasses import dataclass
from loguru import logger
from config import config
from resilience_framework import (
    retry_with_backoff, RetryConfig, FailureType, 
    safe_api_request, resilience_manager, CircuitBreakerConfig
)

@dataclass
class BacktestResult:
    """QuantConnect backtest result wrapper"""
    project_id: int
    backtest_id: str
    name: str
    created: datetime
    completed: datetime
    progress: float
    result: Optional[Dict[str, Any]] = None
    statistics: Optional[Dict[str, float]] = None
    charts: Optional[Dict[str, Any]] = None
    
    @property
    def is_complete(self) -> bool:
        return self.progress >= 1.0
    
    @property
    def cagr(self) -> Optional[float]:
        if self.statistics:
            return self.statistics.get('Compounding Annual Return')
    
    @property
    def sharpe(self) -> Optional[float]:
        if self.statistics:
            return self.statistics.get('Sharpe Ratio')
    
    @property
    def max_drawdown(self) -> Optional[float]:
        if self.statistics:
            return abs(self.statistics.get('Drawdown', 0))
    
    @property
    def total_trades(self) -> Optional[int]:
        if self.statistics:
            return int(self.statistics.get('Total Trades', 0))
    
    def meets_targets(self) -> bool:
        """Check if backtest meets AlgoForge targets"""
        if not self.statistics:
            return False
            
        cagr_ok = (self.cagr or 0) >= config.targets.min_cagr
        sharpe_ok = (self.sharpe or 0) >= config.targets.min_sharpe
        dd_ok = (self.max_drawdown or 1) <= config.targets.max_drawdown
        
        return cagr_ok and sharpe_ok and dd_ok

class QuantConnectClient:
    """Async QuantConnect API client with comprehensive backtesting and project management"""
    
    def __init__(self):
        self.user_id = config.quantconnect.user_id
        self.api_token = config.quantconnect.api_token
        self.base_url = config.quantconnect.base_url
        self.auth_header = self._create_auth_header()
        self.session: Optional[aiohttp.ClientSession] = None
    
    def _create_auth_header(self) -> str:
        """Create basic auth header for QuantConnect API"""
        credentials = f"{self.user_id}:{self.api_token}"
        encoded = base64.b64encode(credentials.encode()).decode()
        return f"Basic {encoded}"
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession(
            headers={"Authorization": self.auth_header},
            timeout=aiohttp.ClientTimeout(total=300)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    @retry_with_backoff(
        retry_config=RetryConfig(max_attempts=5, base_delay=2.0, max_delay=120.0),
        failure_types=[FailureType.NETWORK, FailureType.API]
    )
    async def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make authenticated request to QuantConnect API with resilience"""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = await safe_api_request(
                self.session, method, url, 
                rate_limiter_name="quantconnect",
                **kwargs
            )
            return await response.json()
        except aiohttp.ClientError as e:
            if "429" in str(e) or "rate limit" in str(e).lower():
                # This will trigger the retry mechanism
                raise Exception(f"QuantConnect rate limited: {e}")
            else:
                logger.error(f"QuantConnect API error: {e}")
                raise
    
    async def create_project(self, name: str, language: str = "Py") -> int:
        """Create new QuantConnect project"""
        logger.info(f"Creating project: {name}")
        
        data = {
            "projectName": name,
            "language": language
        }
        
        result = await self._request("POST", "projects/create", json=data)
        project_id = result["projects"][0]["projectId"]
        
        logger.success(f"Created project {name} with ID: {project_id}")
        return project_id
    
    async def upload_file(self, project_id: int, filename: str, content: str) -> bool:
        """Upload file to QuantConnect project"""
        logger.info(f"Uploading {filename} to project {project_id}")
        
        data = {
            "projectId": project_id,
            "name": filename,
            "content": content
        }
        
        result = await self._request("POST", "files/create", json=data)
        success = result.get("success", False)
        
        if success:
            logger.success(f"Uploaded {filename} successfully")
        else:
            logger.error(f"Failed to upload {filename}")
        
        return success
    
    async def compile_project(self, project_id: int) -> bool:
        """Compile QuantConnect project"""
        logger.info(f"Compiling project {project_id}")
        
        data = {"projectId": project_id}
        result = await self._request("POST", "compile/create", json=data)
        
        compile_id = result["compileId"]
        
        # Wait for compilation to complete
        while True:
            await asyncio.sleep(2)
            compile_result = await self._request("GET", f"compile/read?compileId={compile_id}")
            
            if compile_result["state"] == "BuildSuccess":
                logger.success(f"Project {project_id} compiled successfully")
                return True
            elif compile_result["state"] == "BuildError":
                logger.error(f"Compilation failed: {compile_result.get('logs', 'Unknown error')}")
                return False
    
    async def create_backtest(self, project_id: int, name: str) -> str:
        """Create and run backtest"""
        logger.info(f"Creating backtest '{name}' for project {project_id}")
        
        data = {
            "projectId": project_id,
            "name": name
        }
        
        result = await self._request("POST", "backtests/create", json=data)
        backtest_id = result["backtestId"]
        
        logger.success(f"Created backtest {name} with ID: {backtest_id}")
        return backtest_id
    
    async def get_backtest_result(self, project_id: int, backtest_id: str) -> BacktestResult:
        """Get backtest result with full statistics"""
        result = await self._request("GET", f"backtests/read?projectId={project_id}&backtestId={backtest_id}")
        
        backtest_data = result["backtests"][0]
        
        # Parse dates
        created = datetime.fromisoformat(backtest_data["created"].replace('Z', '+00:00'))
        completed = None
        if backtest_data.get("completed"):
            completed = datetime.fromisoformat(backtest_data["completed"].replace('Z', '+00:00'))
        
        # Extract statistics if available
        statistics = None
        charts = None
        if backtest_data.get("result"):
            statistics = backtest_data["result"].get("Statistics", {})
            charts = backtest_data["result"].get("Charts", {})
        
        return BacktestResult(
            project_id=project_id,
            backtest_id=backtest_id,
            name=backtest_data["name"],
            created=created,
            completed=completed,
            progress=backtest_data["progress"],
            result=backtest_data.get("result"),
            statistics=statistics,
            charts=charts
        )
    
    async def wait_for_backtest_completion(self, project_id: int, backtest_id: str, timeout_minutes: int = 30) -> BacktestResult:
        """Wait for backtest to complete and return results"""
        logger.info(f"Waiting for backtest {backtest_id} to complete...")
        
        start_time = datetime.now()
        timeout_seconds = timeout_minutes * 60
        
        while True:
            result = await self.get_backtest_result(project_id, backtest_id)
            
            if result.is_complete:
                logger.success(f"Backtest {backtest_id} completed successfully")
                return result
            
            # Check timeout
            if (datetime.now() - start_time).total_seconds() > timeout_seconds:
                logger.warning(f"Backtest {backtest_id} timed out after {timeout_minutes} minutes")
                return result
            
            logger.info(f"Backtest progress: {result.progress*100:.1f}%")
            await asyncio.sleep(10)
    
    async def list_projects(self) -> List[Dict[str, Any]]:
        """List all projects"""
        result = await self._request("GET", "projects/read")
        return result["projects"]
    
    async def delete_project(self, project_id: int) -> bool:
        """Delete project"""
        logger.info(f"Deleting project {project_id}")
        
        data = {"projectId": project_id}
        result = await self._request("POST", "projects/delete", json=data)
        
        success = result.get("success", False)
        if success:
            logger.success(f"Deleted project {project_id}")
        
        return success
    
    async def get_node_usage(self) -> Dict[str, Any]:
        """Get current node usage statistics"""
        try:
            result = await self._request("GET", "usage/read")
            return result
        except Exception as e:
            logger.warning(f"Could not fetch node usage: {e}")
            return {"available_nodes": config.quantconnect.node_count}

# Example usage function
async def test_quantconnect_integration():
    """Test the QuantConnect integration"""
    async with QuantConnectClient() as client:
        # List projects
        projects = await client.list_projects()
        logger.info(f"Found {len(projects)} existing projects")
        
        # Get node usage
        usage = await client.get_node_usage()
        logger.info(f"Node usage: {usage}")
        
        return True

if __name__ == "__main__":
    asyncio.run(test_quantconnect_integration())