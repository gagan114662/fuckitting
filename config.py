"""
Configuration settings for AlgoForge 3.0 - Claude Code SDK powered quant system
"""
import os
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

load_dotenv()

class QuantConnectConfig(BaseModel):
    """QuantConnect API configuration"""
    user_id: str = Field(default="357130", description="QuantConnect User ID")
    api_token: str = Field(default="62d0a329b3c854f1f61d29114eb02a7c724b361878a85d7953d0ba0e2b053912", description="QuantConnect API Token")
    base_url: str = Field(default="https://www.quantconnect.com/api/v2", description="QuantConnect API Base URL")
    organization_id: Optional[str] = Field(default=None, description="Organization ID for team accounts")
    node_count: int = Field(default=2, description="Number of compute nodes available")

class TradingTargets(BaseModel):
    """Trading performance targets"""
    min_cagr: float = Field(default=0.25, description="Minimum CAGR > 25%")
    min_sharpe: float = Field(default=1.0, description="Minimum Sharpe ratio > 1")
    max_drawdown: float = Field(default=0.20, description="Maximum drawdown < 20%")
    min_avg_profit_per_trade: float = Field(default=0.0075, description="Minimum average profit per trade > 0.75%")

class ClaudeConfig(BaseModel):
    """Claude Code SDK configuration"""
    system_prompt: str = Field(
        default="You are an expert quantitative analyst and algorithmic trading strategist. Generate efficient, vectorized, and well-documented Python code for QuantConnect. Always follow best practices for performance optimization and risk management.",
        description="System prompt for Claude Code SDK"
    )
    max_turns: int = Field(default=10, description="Maximum conversation turns")
    allowed_tools: List[str] = Field(
        default=["Read", "Write", "Bash", "Edit", "MultiEdit", "Glob", "Grep"], 
        description="Allowed tools for Claude"
    )
    temperature: float = Field(default=0.1, description="Lower temperature for more focused code generation")

class AlgoForgeConfig(BaseModel):
    """Main AlgoForge configuration"""
    quantconnect: QuantConnectConfig = Field(default_factory=QuantConnectConfig)
    targets: TradingTargets = Field(default_factory=TradingTargets)
    claude: ClaudeConfig = Field(default_factory=ClaudeConfig)
    
    # Database settings
    database_url: str = Field(default="sqlite:///algoforge_memory.db", description="Database URL for memory storage")
    
    # Research settings
    research_sources: List[str] = Field(
        default=["arxiv", "ssrn", "quantitative-finance", "risk-management"],
        description="Research paper sources"
    )
    
    # Validation settings
    validation_periods: Dict[str, Any] = Field(
        default={
            "backtest_years": 15,
            "oos_ratio": 0.2,
            "walk_forward_steps": 12,
            "monte_carlo_runs": 1000
        },
        description="Validation test parameters"
    )
    
    # Memory and learning settings
    learning_config: Dict[str, Any] = Field(
        default={
            "memory_retention_days": 365,
            "strategy_success_threshold": 0.6,
            "ensemble_max_strategies": 10,
            "rebalance_frequency": "monthly"
        },
        description="Learning and memory configuration"
    )

# Global config instance
config = AlgoForgeConfig()