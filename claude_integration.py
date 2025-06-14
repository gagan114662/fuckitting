"""
Claude Code SDK Integration for AlgoForge 3.0
Intelligent hypothesis generation and strategy code creation using Claude
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, AsyncGenerator
from dataclasses import dataclass
from datetime import datetime
from loguru import logger
from claude_code_sdk import query, ClaudeCodeOptions
from config import config

@dataclass
class TradingHypothesis:
    """Trading hypothesis with metadata"""
    id: str
    description: str
    rationale: str
    expected_return: float
    risk_level: str  # low, medium, high
    asset_classes: List[str]
    timeframe: str
    indicators: List[str]
    created_at: datetime
    source_research: Optional[str] = None
    confidence_score: float = 0.5

@dataclass
class GeneratedStrategy:
    """Generated trading strategy with code and metadata"""
    hypothesis_id: str
    name: str
    code: str
    description: str
    parameters: Dict[str, Any]
    expected_metrics: Dict[str, float]
    risk_controls: List[str]
    created_at: datetime
    version: int = 1

class ClaudeQuantAnalyst:
    """Claude-powered quantitative analyst for hypothesis generation and strategy creation"""
    
    def __init__(self):
        self.options = ClaudeCodeOptions(
            system_prompt=config.claude.system_prompt,
            max_turns=config.claude.max_turns,
            allowed_tools=config.claude.allowed_tools,
            permission_mode='acceptEdits'
        )
    
    async def generate_hypotheses_from_research(self, research_text: str, market_context: Dict[str, Any]) -> List[TradingHypothesis]:
        """Generate trading hypotheses from research papers"""
        prompt = f"""
        Analyze the following research and generate 3-5 actionable trading hypotheses for QuantConnect.
        
        Research Content:
        {research_text[:5000]}  # Limit to avoid token overflow
        
        Market Context:
        {json.dumps(market_context, indent=2)}
        
        Generate hypotheses that:
        1. Have clear theoretical foundation from the research
        2. Are implementable in QuantConnect with Python
        3. Have specific expected returns and risk profiles
        4. Use quantifiable indicators and signals
        5. Target the performance goals: CAGR > 25%, Sharpe > 1, Max DD < 20%
        
        Return a JSON list of hypotheses with this structure:
        {{
            "id": "unique_identifier",
            "description": "Brief hypothesis description",
            "rationale": "Why this should work based on research",
            "expected_return": 0.3,  # Expected annual return
            "risk_level": "medium",
            "asset_classes": ["equities", "etf"],
            "timeframe": "daily",
            "indicators": ["rsi", "macd", "volume"],
            "confidence_score": 0.75
        }}
        """
        
        hypotheses = []
        
        try:
            async for message in query(prompt=prompt, options=self.options):
                if message.type == "text":
                    # Extract JSON from response
                    content = message.content
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                    else:
                        json_content = content
                    
                    try:
                        hypothesis_data = json.loads(json_content)
                        if isinstance(hypothesis_data, list):
                            for h in hypothesis_data:
                                hypotheses.append(TradingHypothesis(
                                    id=h["id"],
                                    description=h["description"],
                                    rationale=h["rationale"],
                                    expected_return=h["expected_return"],
                                    risk_level=h["risk_level"],
                                    asset_classes=h["asset_classes"],
                                    timeframe=h["timeframe"],
                                    indicators=h["indicators"],
                                    created_at=datetime.now(),
                                    source_research=research_text[:200],
                                    confidence_score=h.get("confidence_score", 0.5)
                                ))
                    except json.JSONDecodeError as e:
                        logger.error(f"Failed to parse hypothesis JSON: {e}")
        
        except Exception as e:
            logger.error(f"Error generating hypotheses: {e}")
        
        logger.info(f"Generated {len(hypotheses)} hypotheses from research")
        return hypotheses
    
    async def generate_strategy_code(self, hypothesis: TradingHypothesis, historical_performance: Optional[Dict] = None) -> GeneratedStrategy:
        """Generate QuantConnect strategy code from hypothesis"""
        
        performance_context = ""
        if historical_performance:
            performance_context = f"""
            Historical Performance Context:
            - Similar strategies achieved: {historical_performance.get('avg_return', 'N/A')} return
            - Common failure modes: {historical_performance.get('failure_modes', [])}
            - Successful patterns: {historical_performance.get('success_patterns', [])}
            """
        
        prompt = f"""
        Generate a complete QuantConnect algorithm implementation for this trading hypothesis:
        
        Hypothesis: {hypothesis.description}
        Rationale: {hypothesis.rationale}
        Target Return: {hypothesis.expected_return*100:.1f}%
        Risk Level: {hypothesis.risk_level}
        Asset Classes: {hypothesis.asset_classes}
        Timeframe: {hypothesis.timeframe}
        Indicators: {hypothesis.indicators}
        
        {performance_context}
        
        Requirements:
        1. Complete QCAlgorithm class with proper imports
        2. Vectorized operations using numpy/pandas where possible
        3. Proper risk management (position sizing, stop losses)
        4. Parameter optimization ready (use self.GetParameter())
        5. Comprehensive logging for debugging
        6. Target metrics: CAGR > 25%, Sharpe > 1, Max DD < 20%
        7. Include warm-up period and universe selection
        8. Efficient data handling for large backtests
        
        Generate clean, production-ready Python code with:
        - Clear variable names and functions
        - Docstrings for all methods
        - Error handling
        - Performance optimizations
        - No hardcoded values (use parameters)
        
        Also provide a JSON metadata block with:
        {{
            "parameters": {{"param_name": "default_value"}},
            "expected_metrics": {{"cagr": 0.3, "sharpe": 1.2, "max_dd": 0.15}},
            "risk_controls": ["position_sizing", "stop_loss", "drawdown_limit"]
        }}
        """
        
        strategy_code = ""
        metadata = {}
        
        try:
            async for message in query(prompt=prompt, options=self.options):
                if message.type == "text":
                    content = message.content
                    
                    # Extract Python code
                    if "```python" in content:
                        code_start = content.find("```python") + 9
                        code_end = content.find("```", code_start)
                        strategy_code = content[code_start:code_end].strip()
                    
                    # Extract metadata JSON
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        try:
                            metadata = json.loads(json_content)
                        except json.JSONDecodeError:
                            logger.warning("Could not parse strategy metadata JSON")
        
        except Exception as e:
            logger.error(f"Error generating strategy code: {e}")
            return None
        
        if not strategy_code:
            logger.error("No strategy code generated")
            return None
        
        strategy = GeneratedStrategy(
            hypothesis_id=hypothesis.id,
            name=f"Strategy_{hypothesis.id}",
            code=strategy_code,
            description=hypothesis.description,
            parameters=metadata.get("parameters", {}),
            expected_metrics=metadata.get("expected_metrics", {}),
            risk_controls=metadata.get("risk_controls", []),
            created_at=datetime.now()
        )
        
        logger.success(f"Generated strategy code for hypothesis {hypothesis.id}")
        return strategy
    
    async def optimize_strategy_code(self, strategy: GeneratedStrategy, performance_issues: List[str]) -> GeneratedStrategy:
        """Optimize strategy code based on performance issues"""
        
        prompt = f"""
        Optimize this QuantConnect strategy to address performance issues:
        
        Current Strategy:
        ```python
        {strategy.code}
        ```
        
        Performance Issues to Address:
        {json.dumps(performance_issues, indent=2)}
        
        Optimization Goals:
        1. Improve vectorization and reduce loops
        2. Optimize memory usage
        3. Reduce computational complexity
        4. Fix any runtime errors
        5. Enhance risk management
        6. Improve signal quality
        
        Provide the optimized code with improvements clearly documented.
        """
        
        optimized_code = ""
        
        try:
            async for message in query(prompt=prompt, options=self.options):
                if message.type == "text":
                    content = message.content
                    
                    if "```python" in content:
                        code_start = content.find("```python") + 9
                        code_end = content.find("```", code_start)
                        optimized_code = content[code_start:code_end].strip()
        
        except Exception as e:
            logger.error(f"Error optimizing strategy: {e}")
            return strategy
        
        if optimized_code:
            strategy.code = optimized_code
            strategy.version += 1
            logger.success(f"Optimized strategy {strategy.name} to version {strategy.version}")
        
        return strategy
    
    async def analyze_backtest_results(self, strategy: GeneratedStrategy, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze backtest results and provide insights"""
        
        prompt = f"""
        Analyze these backtest results for the trading strategy and provide insights:
        
        Strategy: {strategy.name}
        Description: {strategy.description}
        
        Backtest Results:
        {json.dumps(backtest_results, indent=2)}
        
        Provide analysis in JSON format:
        {{
            "performance_grade": "A/B/C/D/F",
            "strengths": ["list of what worked well"],
            "weaknesses": ["list of issues identified"],
            "improvement_suggestions": ["specific actionable improvements"],
            "risk_assessment": "detailed risk analysis",
            "market_regime_analysis": "how strategy performs in different market conditions",
            "parameter_sensitivity": "analysis of parameter robustness"
        }}
        """
        
        analysis = {}
        
        try:
            async for message in query(prompt=prompt, options=self.options):
                if message.type == "text":
                    content = message.content
                    
                    # Extract JSON analysis
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        try:
                            analysis = json.loads(json_content)
                        except json.JSONDecodeError:
                            logger.warning("Could not parse analysis JSON")
        
        except Exception as e:
            logger.error(f"Error analyzing backtest results: {e}")
        
        return analysis

# Example usage
async def test_claude_integration():
    """Test Claude integration"""
    analyst = ClaudeQuantAnalyst()
    
    # Test hypothesis generation
    sample_research = """
    Recent research shows that momentum strategies combined with mean reversion
    can provide superior risk-adjusted returns. The key is identifying the optimal
    lookback periods and position sizing methods.
    """
    
    market_context = {
        "current_vix": 20.5,
        "market_regime": "neutral",
        "sector_rotation": "technology_leading"
    }
    
    hypotheses = await analyst.generate_hypotheses_from_research(sample_research, market_context)
    logger.info(f"Generated {len(hypotheses)} hypotheses")
    
    if hypotheses:
        # Test strategy generation
        strategy = await analyst.generate_strategy_code(hypotheses[0])
        if strategy:
            logger.success(f"Generated strategy: {strategy.name}")
            return True
    
    return False

if __name__ == "__main__":
    asyncio.run(test_claude_integration())