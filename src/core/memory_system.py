"""
Memory and Learning System for AlgoForge 3.0
Stores strategy performance, learns from successes/failures, and optimizes future generations
"""
import asyncio
import json
import sqlite3
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.sqlite import JSON
from loguru import logger
from config import config
from claude_integration import TradingHypothesis, GeneratedStrategy

Base = declarative_base()

class StrategyRecord(Base):
    """Database model for strategy performance records"""
    __tablename__ = 'strategies'
    
    id = Column(Integer, primary_key=True)
    hypothesis_id = Column(String, nullable=False)
    strategy_name = Column(String, nullable=False)
    description = Column(Text)
    code_hash = Column(String)  # Hash of the strategy code for versioning
    parameters = Column(JSON)
    
    # Performance metrics
    cagr = Column(Float)
    sharpe_ratio = Column(Float)
    max_drawdown = Column(Float)
    total_trades = Column(Integer)
    win_rate = Column(Float)
    profit_factor = Column(Float)
    avg_profit_per_trade = Column(Float)
    
    # Validation results
    oos_performance = Column(JSON)  # Out-of-sample performance
    walk_forward_results = Column(JSON)
    monte_carlo_results = Column(JSON)
    crisis_test_results = Column(JSON)
    
    # Learning metadata
    success_score = Column(Float)  # Combined success metric
    market_regime_performance = Column(JSON)  # Performance by market conditions
    failure_modes = Column(JSON)  # Documented failure patterns
    success_patterns = Column(JSON)  # Documented success patterns
    
    # Timestamps
    created_at = Column(DateTime, default=datetime.utcnow)
    last_updated = Column(DateTime, default=datetime.utcnow)
    
    # Status
    is_active = Column(Boolean, default=True)
    is_live_ready = Column(Boolean, default=False)

class MarketRegime(Base):
    """Market regime tracking for context-aware learning"""
    __tablename__ = 'market_regimes'
    
    id = Column(Integer, primary_key=True)
    date = Column(DateTime, nullable=False)
    regime_type = Column(String)  # bull, bear, sideways, high_vol, low_vol
    vix_level = Column(Float)
    market_breadth = Column(Float)
    sector_rotation = Column(String)
    economic_indicators = Column(JSON)

class LearningInsight(Base):
    """Learned insights from strategy analysis"""
    __tablename__ = 'learning_insights'
    
    id = Column(Integer, primary_key=True)
    insight_type = Column(String)  # parameter, indicator, risk_control, market_timing
    description = Column(Text)
    evidence_strategies = Column(JSON)  # List of strategy IDs supporting this insight
    confidence_score = Column(Float)
    market_conditions = Column(JSON)  # When this insight applies
    created_at = Column(DateTime, default=datetime.utcnow)

class AlgoForgeMemory:
    """Memory and learning system for AlgoForge"""
    
    def __init__(self, db_url: str = None):
        self.db_url = db_url or config.database_url
        self.engine = create_engine(self.db_url)
        Base.metadata.create_all(self.engine)
        self.SessionLocal = sessionmaker(bind=self.engine)
        
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    async def store_strategy_performance(self, strategy: GeneratedStrategy, backtest_results: Dict[str, Any]) -> int:
        """Store strategy performance in memory database"""
        session = self.get_session()
        
        try:
            # Extract key metrics from backtest results
            statistics = backtest_results.get('Statistics', {})
            
            record = StrategyRecord(
                hypothesis_id=strategy.hypothesis_id,
                strategy_name=strategy.name,
                description=strategy.description,
                code_hash=hash(strategy.code),
                parameters=strategy.parameters,
                cagr=statistics.get('Compounding Annual Return'),
                sharpe_ratio=statistics.get('Sharpe Ratio'),
                max_drawdown=abs(statistics.get('Drawdown', 0)),
                total_trades=int(statistics.get('Total Trades', 0)),
                win_rate=statistics.get('Win Rate', 0),
                profit_factor=statistics.get('Profit-Loss Ratio', 0),
                avg_profit_per_trade=statistics.get('Average Trade', 0),
                success_score=self._calculate_success_score(statistics),
                market_regime_performance={},
                failure_modes=[],
                success_patterns=[]
            )
            
            session.add(record)
            session.commit()
            
            strategy_id = record.id
            logger.success(f"Stored strategy performance for {strategy.name} (ID: {strategy_id})")
            
            return strategy_id
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error storing strategy performance: {e}")
            raise
        finally:
            session.close()
    
    def _calculate_success_score(self, statistics: Dict[str, Any]) -> float:
        """Calculate overall success score based on multiple metrics"""
        cagr = statistics.get('Compounding Annual Return', 0)
        sharpe = statistics.get('Sharpe Ratio', 0)
        max_dd = abs(statistics.get('Drawdown', 1))
        win_rate = statistics.get('Win Rate', 0)
        
        # Weight different metrics
        cagr_score = min(cagr / config.targets.min_cagr, 2.0)  # Cap at 2x target
        sharpe_score = min(sharpe / config.targets.min_sharpe, 2.0)
        dd_score = max(0, 1 - (max_dd / config.targets.max_drawdown))
        win_rate_score = win_rate
        
        # Combined score (0-1 scale)
        success_score = (cagr_score * 0.3 + sharpe_score * 0.3 + dd_score * 0.25 + win_rate_score * 0.15)
        return min(success_score, 1.0)
    
    async def get_best_performing_strategies(self, limit: int = 10) -> List[StrategyRecord]:
        """Get top performing strategies from memory"""
        session = self.get_session()
        
        try:
            strategies = session.query(StrategyRecord)\
                .filter(StrategyRecord.is_active == True)\
                .order_by(StrategyRecord.success_score.desc())\
                .limit(limit)\
                .all()
            
            return strategies
            
        finally:
            session.close()
    
    async def get_similar_strategies(self, hypothesis: TradingHypothesis, limit: int = 5) -> List[StrategyRecord]:
        """Find similar strategies based on hypothesis characteristics"""
        session = self.get_session()
        
        try:
            # Simple similarity search - in practice, could use vector embeddings
            strategies = session.query(StrategyRecord)\
                .filter(StrategyRecord.is_active == True)\
                .order_by(StrategyRecord.success_score.desc())\
                .limit(limit)\
                .all()
            
            return strategies
            
        finally:
            session.close()
    
    async def learn_from_failures(self, strategy_id: int, failure_analysis: Dict[str, Any]):
        """Record failure patterns for future learning"""
        session = self.get_session()
        
        try:
            strategy = session.query(StrategyRecord).filter(StrategyRecord.id == strategy_id).first()
            if strategy:
                failure_modes = strategy.failure_modes or []
                failure_modes.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'analysis': failure_analysis,
                    'lessons': failure_analysis.get('improvement_suggestions', [])
                })
                
                strategy.failure_modes = failure_modes
                strategy.last_updated = datetime.utcnow()
                session.commit()
                
                logger.info(f"Recorded failure analysis for strategy {strategy_id}")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording failure analysis: {e}")
        finally:
            session.close()
    
    async def learn_from_successes(self, strategy_id: int, success_analysis: Dict[str, Any]):
        """Record success patterns for future learning"""
        session = self.get_session()
        
        try:
            strategy = session.query(StrategyRecord).filter(StrategyRecord.id == strategy_id).first()
            if strategy:
                success_patterns = strategy.success_patterns or []
                success_patterns.append({
                    'timestamp': datetime.utcnow().isoformat(),
                    'analysis': success_analysis,
                    'key_factors': success_analysis.get('strengths', [])
                })
                
                strategy.success_patterns = success_patterns
                strategy.last_updated = datetime.utcnow()
                session.commit()
                
                logger.info(f"Recorded success analysis for strategy {strategy_id}")
                
        except Exception as e:
            session.rollback()
            logger.error(f"Error recording success analysis: {e}")
        finally:
            session.close()
    
    async def generate_learning_insights(self) -> List[Dict[str, Any]]:
        """Generate insights from accumulated strategy data"""
        session = self.get_session()
        
        try:
            # Get all successful strategies
            successful_strategies = session.query(StrategyRecord)\
                .filter(StrategyRecord.success_score >= 0.7)\
                .all()
            
            # Get all failed strategies
            failed_strategies = session.query(StrategyRecord)\
                .filter(StrategyRecord.success_score < 0.3)\
                .all()
            
            insights = []
            
            # Parameter analysis
            if successful_strategies:
                insights.extend(self._analyze_parameter_patterns(successful_strategies))
            
            # Failure mode analysis
            if failed_strategies:
                insights.extend(self._analyze_failure_patterns(failed_strategies))
            
            # Market regime analysis
            insights.extend(self._analyze_market_regime_performance(successful_strategies))
            
            return insights
            
        finally:
            session.close()
    
    def _analyze_parameter_patterns(self, strategies: List[StrategyRecord]) -> List[Dict[str, Any]]:
        """Analyze successful parameter patterns"""
        insights = []
        
        # Collect all parameters
        all_params = {}
        for strategy in strategies:
            if strategy.parameters:
                for param, value in strategy.parameters.items():
                    if param not in all_params:
                        all_params[param] = []
                    all_params[param].append(value)
        
        # Find patterns
        for param, values in all_params.items():
            if len(values) >= 3:  # Need at least 3 data points
                insights.append({
                    'type': 'parameter_pattern',
                    'parameter': param,
                    'successful_range': f"{min(values):.2f} - {max(values):.2f}",
                    'avg_value': sum(values) / len(values),
                    'evidence_count': len(values),
                    'confidence': min(len(values) / 10, 1.0)
                })
        
        return insights
    
    def _analyze_failure_patterns(self, strategies: List[StrategyRecord]) -> List[Dict[str, Any]]:
        """Analyze common failure patterns"""
        insights = []
        
        common_failures = {}
        for strategy in strategies:
            if strategy.failure_modes:
                for failure in strategy.failure_modes:
                    for lesson in failure.get('lessons', []):
                        if lesson not in common_failures:
                            common_failures[lesson] = 0
                        common_failures[lesson] += 1
        
        # Find most common failure patterns
        for failure, count in common_failures.items():
            if count >= 2:  # Appears in at least 2 strategies
                insights.append({
                    'type': 'failure_pattern',
                    'pattern': failure,
                    'frequency': count,
                    'confidence': min(count / 5, 1.0)
                })
        
        return insights
    
    def _analyze_market_regime_performance(self, strategies: List[StrategyRecord]) -> List[Dict[str, Any]]:
        """Analyze performance across different market regimes"""
        insights = []
        
        # This would integrate with market regime data
        # For now, return placeholder insights
        insights.append({
            'type': 'market_regime',
            'insight': 'More data needed for market regime analysis',
            'confidence': 0.1
        })
        
        return insights
    
    async def get_historical_performance_context(self, hypothesis: TradingHypothesis) -> Dict[str, Any]:
        """Get historical performance context for similar strategies"""
        session = self.get_session()
        
        try:
            # Find similar strategies (simplified similarity)
            similar_strategies = await self.get_similar_strategies(hypothesis)
            
            if not similar_strategies:
                return {'avg_return': None, 'success_rate': 0, 'common_issues': []}
            
            returns = [s.cagr for s in similar_strategies if s.cagr]
            success_rate = len([s for s in similar_strategies if s.success_score >= 0.6]) / len(similar_strategies)
            
            # Extract common failure modes
            common_issues = []
            for strategy in similar_strategies:
                if strategy.failure_modes:
                    for failure in strategy.failure_modes:
                        common_issues.extend(failure.get('lessons', []))
            
            return {
                'avg_return': sum(returns) / len(returns) if returns else None,
                'success_rate': success_rate,
                'common_issues': list(set(common_issues))[:5],  # Top 5 unique issues
                'sample_size': len(similar_strategies)
            }
            
        finally:
            session.close()

# Example usage
async def test_memory_system():
    """Test the memory system"""
    memory = AlgoForgeMemory()
    
    # Create sample strategy for testing
    from claude_integration import GeneratedStrategy
    sample_strategy = GeneratedStrategy(
        hypothesis_id="test_001",
        name="Test Strategy",
        code="# Sample code",
        description="Test strategy for memory system",
        parameters={"lookback": 20, "threshold": 0.05},
        expected_metrics={"cagr": 0.3, "sharpe": 1.2},
        risk_controls=["stop_loss"],
        created_at=datetime.now()
    )
    
    # Sample backtest results
    backtest_results = {
        'Statistics': {
            'Compounding Annual Return': 0.28,
            'Sharpe Ratio': 1.15,
            'Drawdown': -0.18,
            'Total Trades': 150,
            'Win Rate': 0.58
        }
    }
    
    # Store performance
    strategy_id = await memory.store_strategy_performance(sample_strategy, backtest_results)
    logger.info(f"Stored test strategy with ID: {strategy_id}")
    
    # Generate insights
    insights = await memory.generate_learning_insights()
    logger.info(f"Generated {len(insights)} learning insights")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_memory_system())