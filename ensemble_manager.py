"""
Ensemble Portfolio Management and Rebalancing for AlgoForge 3.0
Advanced portfolio construction and management using multiple validated strategies
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from scipy.optimize import minimize
from loguru import logger
import json

from memory_system import AlgoForgeMemory, StrategyRecord
from live_trading import LiveTradingManager, LiveTradingMetrics, DeploymentConfig, TradingMode
from claude_integration import ClaudeQuantAnalyst
from config import config

@dataclass
class StrategyAllocation:
    """Strategy allocation in ensemble portfolio"""
    strategy_name: str
    target_weight: float
    current_weight: float
    performance_score: float
    correlation_score: float
    risk_score: float
    last_rebalance: datetime

@dataclass
class EnsemblePortfolio:
    """Complete ensemble portfolio definition"""
    portfolio_id: str
    name: str
    created_date: datetime
    strategy_allocations: List[StrategyAllocation]
    rebalance_frequency: str  # daily, weekly, monthly
    risk_budget: float
    target_volatility: float
    max_strategy_weight: float
    min_strategy_weight: float
    
    # Performance tracking
    inception_value: float
    current_value: float
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    max_drawdown: float
    
    # Risk metrics
    var_95: float
    expected_shortfall: float
    beta: float
    
    last_rebalance: datetime
    next_rebalance: datetime
    status: str  # active, paused, stopped

class PortfolioOptimizer:
    """Advanced portfolio optimization using multiple objectives"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% risk-free rate
    
    def optimize_weights(self, strategy_data: List[Dict[str, Any]], constraints: Dict[str, Any]) -> Dict[str, float]:
        """Optimize portfolio weights using multiple objectives"""
        logger.info("🎯 Optimizing ensemble portfolio weights...")
        
        n_strategies = len(strategy_data)
        if n_strategies < 2:
            logger.warning("Need at least 2 strategies for optimization")
            return {}
        
        # Extract returns and risk metrics
        returns = np.array([s['expected_return'] for s in strategy_data])
        volatilities = np.array([s['volatility'] for s in strategy_data])
        sharpe_ratios = np.array([s['sharpe_ratio'] for s in strategy_data])
        
        # Estimate correlation matrix (simplified)
        correlation_matrix = self._estimate_correlation_matrix(strategy_data)
        
        # Calculate covariance matrix
        cov_matrix = np.outer(volatilities, volatilities) * correlation_matrix
        
        # Define optimization objectives
        def objective(weights):
            portfolio_return = np.dot(weights, returns)
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
            
            # Multi-objective: maximize Sharpe ratio and minimize concentration
            concentration_penalty = np.sum(weights ** 2)  # Herfindahl index
            
            # Minimize negative Sharpe + concentration penalty
            return -portfolio_sharpe + 0.1 * concentration_penalty
        
        # Constraints
        constraints_list = [
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},  # Weights sum to 1
        ]
        
        # Individual weight bounds
        min_weight = constraints.get('min_strategy_weight', 0.05)
        max_weight = constraints.get('max_strategy_weight', 0.4)
        bounds = [(min_weight, max_weight) for _ in range(n_strategies)]
        
        # Initial guess (equal weights)
        x0 = np.ones(n_strategies) / n_strategies
        
        # Optimize
        try:
            result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints_list)
            
            if result.success:
                weights = result.x
                
                # Create weight dictionary
                weight_dict = {}
                for i, strategy in enumerate(strategy_data):
                    weight_dict[strategy['name']] = weights[i]
                
                # Log optimization results
                portfolio_return = np.dot(weights, returns)
                portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
                portfolio_sharpe = (portfolio_return - self.risk_free_rate) / np.sqrt(portfolio_variance)
                
                logger.success(f"✅ Optimization completed:")
                logger.info(f"   Expected Return: {portfolio_return:.2%}")
                logger.info(f"   Expected Volatility: {np.sqrt(portfolio_variance):.2%}")
                logger.info(f"   Expected Sharpe: {portfolio_sharpe:.2f}")
                
                return weight_dict
            else:
                logger.error(f"Optimization failed: {result.message}")
                return self._equal_weight_fallback(strategy_data)
                
        except Exception as e:
            logger.error(f"Error in optimization: {e}")
            return self._equal_weight_fallback(strategy_data)
    
    def _estimate_correlation_matrix(self, strategy_data: List[Dict]) -> np.ndarray:
        """Estimate correlation matrix between strategies"""
        n = len(strategy_data)
        
        # Simplified correlation estimation based on strategy characteristics
        correlation_matrix = np.eye(n)
        
        for i in range(n):
            for j in range(i+1, n):
                # Estimate correlation based on strategy similarity
                similarity = self._calculate_strategy_similarity(strategy_data[i], strategy_data[j])
                correlation = 0.1 + 0.8 * similarity  # Base correlation + similarity component
                correlation_matrix[i, j] = correlation
                correlation_matrix[j, i] = correlation
        
        return correlation_matrix
    
    def _calculate_strategy_similarity(self, strategy1: Dict, strategy2: Dict) -> float:
        """Calculate similarity between two strategies"""
        # Simple similarity based on asset classes and timeframes
        asset_overlap = len(set(strategy1.get('asset_classes', [])) & set(strategy2.get('asset_classes', [])))
        total_assets = len(set(strategy1.get('asset_classes', [])) | set(strategy2.get('asset_classes', [])))
        
        timeframe_similarity = 1.0 if strategy1.get('timeframe') == strategy2.get('timeframe') else 0.3
        
        if total_assets == 0:
            asset_similarity = 0.5
        else:
            asset_similarity = asset_overlap / total_assets
        
        return (asset_similarity * 0.7 + timeframe_similarity * 0.3)
    
    def _equal_weight_fallback(self, strategy_data: List[Dict]) -> Dict[str, float]:
        """Fallback to equal weights if optimization fails"""
        logger.warning("Using equal weight fallback")
        weight = 1.0 / len(strategy_data)
        return {strategy['name']: weight for strategy in strategy_data}

class EnsembleManager:
    """Manages ensemble portfolios and rebalancing"""
    
    def __init__(self):
        self.memory = AlgoForgeMemory()
        self.live_trading_manager = LiveTradingManager()
        self.optimizer = PortfolioOptimizer()
        self.claude_analyst = ClaudeQuantAnalyst()
        self.active_portfolios: Dict[str, EnsemblePortfolio] = {}
    
    async def create_ensemble_portfolio(self, portfolio_name: str, strategy_criteria: Dict[str, Any]) -> Optional[EnsemblePortfolio]:
        """Create new ensemble portfolio from top strategies"""
        logger.info(f"🏗️ Creating ensemble portfolio: {portfolio_name}")
        
        try:
            # Get candidate strategies
            candidates = await self._select_candidate_strategies(strategy_criteria)
            
            if len(candidates) < 2:
                logger.error("Need at least 2 strategies for ensemble portfolio")
                return None
            
            # Prepare strategy data for optimization
            strategy_data = []
            for candidate in candidates:
                strategy_data.append({
                    'name': candidate.strategy_name,
                    'expected_return': candidate.cagr or 0,
                    'volatility': self._estimate_strategy_volatility(candidate),
                    'sharpe_ratio': candidate.sharpe_ratio or 0,
                    'asset_classes': self._extract_asset_classes(candidate),
                    'timeframe': self._extract_timeframe(candidate)
                })
            
            # Optimize weights
            optimization_constraints = {
                'min_strategy_weight': config.learning_config.get('min_strategy_weight', 0.1),
                'max_strategy_weight': config.learning_config.get('max_strategy_weight', 0.4),
                'target_volatility': 0.15
            }
            
            optimal_weights = self.optimizer.optimize_weights(strategy_data, optimization_constraints)
            
            if not optimal_weights:
                logger.error("Portfolio optimization failed")
                return None
            
            # Create strategy allocations
            allocations = []
            for candidate in candidates:
                if candidate.strategy_name in optimal_weights:
                    allocation = StrategyAllocation(
                        strategy_name=candidate.strategy_name,
                        target_weight=optimal_weights[candidate.strategy_name],
                        current_weight=0.0,
                        performance_score=candidate.success_score,
                        correlation_score=0.5,  # Will be calculated
                        risk_score=min(candidate.max_drawdown or 0.2, 1.0),
                        last_rebalance=datetime.now()
                    )
                    allocations.append(allocation)
            
            # Calculate portfolio metrics
            portfolio_metrics = self._calculate_portfolio_metrics(allocations, strategy_data)
            
            # Create ensemble portfolio
            portfolio = EnsemblePortfolio(
                portfolio_id=f"ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                name=portfolio_name,
                created_date=datetime.now(),
                strategy_allocations=allocations,
                rebalance_frequency=config.learning_config.get('rebalance_frequency', 'monthly'),
                risk_budget=0.15,
                target_volatility=0.15,
                max_strategy_weight=0.4,
                min_strategy_weight=0.1,
                inception_value=1000000,  # $1M default
                current_value=1000000,
                total_return=0.0,
                annualized_return=portfolio_metrics['expected_return'],
                volatility=portfolio_metrics['expected_volatility'],
                sharpe_ratio=portfolio_metrics['expected_sharpe'],
                max_drawdown=0.0,
                var_95=0.0,
                expected_shortfall=0.0,
                beta=1.0,
                last_rebalance=datetime.now(),
                next_rebalance=self._calculate_next_rebalance_date('monthly'),
                status='active'
            )
            
            # Store portfolio
            self.active_portfolios[portfolio.portfolio_id] = portfolio
            
            logger.success(f"✅ Created ensemble portfolio '{portfolio_name}' with {len(allocations)} strategies")
            
            return portfolio
            
        except Exception as e:
            logger.error(f"Error creating ensemble portfolio: {e}")
            return None
    
    async def _select_candidate_strategies(self, criteria: Dict[str, Any]) -> List[StrategyRecord]:
        """Select candidate strategies for ensemble"""
        min_success_score = criteria.get('min_success_score', 0.7)
        max_strategies = criteria.get('max_strategies', 10)
        min_validation_period = criteria.get('min_validation_days', 30)
        
        session = self.memory.get_session()
        try:
            # Query top strategies
            candidates = session.query(StrategyRecord)\
                .filter(StrategyRecord.is_active == True)\
                .filter(StrategyRecord.success_score >= min_success_score)\
                .filter(StrategyRecord.created_at <= datetime.utcnow() - timedelta(days=min_validation_period))\
                .order_by(StrategyRecord.success_score.desc())\
                .limit(max_strategies)\
                .all()
            
            logger.info(f"Selected {len(candidates)} candidate strategies")
            return candidates
            
        finally:
            session.close()
    
    def _estimate_strategy_volatility(self, strategy: StrategyRecord) -> float:
        """Estimate strategy volatility from available data"""
        # Simple volatility estimation
        if strategy.sharpe_ratio and strategy.cagr:
            estimated_vol = abs(strategy.cagr) / max(abs(strategy.sharpe_ratio), 0.1)
            return min(estimated_vol, 0.5)  # Cap at 50%
        return 0.15  # Default 15% volatility
    
    def _extract_asset_classes(self, strategy: StrategyRecord) -> List[str]:
        """Extract asset classes from strategy parameters"""
        # This would analyze the strategy code/parameters to determine asset classes
        return ['equities']  # Default
    
    def _extract_timeframe(self, strategy: StrategyRecord) -> str:
        """Extract trading timeframe from strategy"""
        # This would analyze the strategy code to determine timeframe
        return 'daily'  # Default
    
    def _calculate_portfolio_metrics(self, allocations: List[StrategyAllocation], strategy_data: List[Dict]) -> Dict[str, float]:
        """Calculate expected portfolio metrics"""
        weights = np.array([a.target_weight for a in allocations])
        returns = np.array([s['expected_return'] for s in strategy_data])
        volatilities = np.array([s['volatility'] for s in strategy_data])
        
        # Portfolio expected return
        portfolio_return = np.dot(weights, returns)
        
        # Portfolio volatility (simplified - assumes low correlation)
        portfolio_variance = np.dot(weights ** 2, volatilities ** 2)
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Portfolio Sharpe ratio
        portfolio_sharpe = (portfolio_return - 0.02) / portfolio_volatility
        
        return {
            'expected_return': portfolio_return,
            'expected_volatility': portfolio_volatility,
            'expected_sharpe': portfolio_sharpe
        }
    
    def _calculate_next_rebalance_date(self, frequency: str) -> datetime:
        """Calculate next rebalance date"""
        now = datetime.now()
        
        if frequency == 'daily':
            return now + timedelta(days=1)
        elif frequency == 'weekly':
            return now + timedelta(weeks=1)
        elif frequency == 'monthly':
            return now + timedelta(days=30)
        else:
            return now + timedelta(days=30)  # Default monthly
    
    async def rebalance_portfolio(self, portfolio_id: str) -> bool:
        """Rebalance ensemble portfolio"""
        if portfolio_id not in self.active_portfolios:
            logger.error(f"Portfolio {portfolio_id} not found")
            return False
        
        portfolio = self.active_portfolios[portfolio_id]
        logger.info(f"⚖️ Rebalancing portfolio: {portfolio.name}")
        
        try:
            # Get current performance of all strategies
            current_performance = await self._get_current_strategy_performance(portfolio)
            
            # Analyze if rebalancing is needed
            rebalance_analysis = await self._analyze_rebalancing_needs(portfolio, current_performance)
            
            if not rebalance_analysis['needs_rebalancing']:
                logger.info("Portfolio is well-balanced, no rebalancing needed")
                return True
            
            # Calculate new weights
            new_weights = await self._calculate_new_weights(portfolio, current_performance, rebalance_analysis)
            
            if new_weights:
                # Execute rebalancing
                success = await self._execute_rebalancing(portfolio, new_weights)
                
                if success:
                    # Update portfolio
                    portfolio.last_rebalance = datetime.now()
                    portfolio.next_rebalance = self._calculate_next_rebalance_date(portfolio.rebalance_frequency)
                    
                    logger.success(f"✅ Portfolio {portfolio.name} rebalanced successfully")
                    return True
            
        except Exception as e:
            logger.error(f"Error rebalancing portfolio {portfolio_id}: {e}")
        
        return False
    
    async def _get_current_strategy_performance(self, portfolio: EnsemblePortfolio) -> Dict[str, Dict]:
        """Get current performance metrics for all strategies in portfolio"""
        performance = {}
        
        for allocation in portfolio.strategy_allocations:
            # Get live trading metrics if available
            live_metrics = await self.live_trading_manager.get_live_trading_metrics(allocation.strategy_name)
            
            if live_metrics:
                performance[allocation.strategy_name] = {
                    'current_return': live_metrics.total_return,
                    'sharpe_ratio': live_metrics.sharpe_ratio,
                    'max_drawdown': live_metrics.max_drawdown,
                    'volatility': self._calculate_strategy_volatility(live_metrics),
                    'recent_performance': self._calculate_recent_performance(live_metrics)
                }
            else:
                # Fall back to historical performance
                performance[allocation.strategy_name] = {
                    'current_return': 0,
                    'sharpe_ratio': allocation.performance_score,
                    'max_drawdown': allocation.risk_score,
                    'volatility': 0.15,
                    'recent_performance': 0
                }
        
        return performance
    
    def _calculate_strategy_volatility(self, metrics: LiveTradingMetrics) -> float:
        """Calculate strategy volatility from live metrics"""
        # This would use historical returns to calculate volatility
        # For now, return estimated volatility
        return 0.15
    
    def _calculate_recent_performance(self, metrics: LiveTradingMetrics) -> float:
        """Calculate recent performance for rebalancing decisions"""
        # This would analyze recent performance trends
        return metrics.total_return
    
    async def _analyze_rebalancing_needs(self, portfolio: EnsemblePortfolio, performance: Dict) -> Dict[str, Any]:
        """Analyze if portfolio needs rebalancing"""
        
        # Check weight drift
        max_drift = 0
        for allocation in portfolio.strategy_allocations:
            drift = abs(allocation.current_weight - allocation.target_weight)
            max_drift = max(max_drift, drift)
        
        # Check performance divergence
        performance_scores = [performance[alloc.strategy_name]['sharpe_ratio'] for alloc in portfolio.strategy_allocations]
        performance_std = np.std(performance_scores)
        
        # Check risk levels
        risk_scores = [performance[alloc.strategy_name]['max_drawdown'] for alloc in portfolio.strategy_allocations]
        max_risk = max(risk_scores)
        
        needs_rebalancing = (
            max_drift > 0.05 or  # 5% weight drift threshold
            performance_std > 1.0 or  # High performance divergence
            max_risk > 0.25  # Risk threshold exceeded
        )
        
        return {
            'needs_rebalancing': needs_rebalancing,
            'max_weight_drift': max_drift,
            'performance_divergence': performance_std,
            'max_risk_level': max_risk,
            'reasons': self._get_rebalancing_reasons(max_drift, performance_std, max_risk)
        }
    
    def _get_rebalancing_reasons(self, drift: float, divergence: float, risk: float) -> List[str]:
        """Get reasons for rebalancing"""
        reasons = []
        
        if drift > 0.05:
            reasons.append(f"Weight drift exceeded threshold: {drift:.2%}")
        if divergence > 1.0:
            reasons.append(f"Performance divergence: {divergence:.2f}")
        if risk > 0.25:
            reasons.append(f"Risk level exceeded: {risk:.2%}")
        
        return reasons
    
    async def _calculate_new_weights(self, portfolio: EnsemblePortfolio, performance: Dict, analysis: Dict) -> Optional[Dict[str, float]]:
        """Calculate new portfolio weights"""
        
        # Use Claude to analyze and suggest new weights
        prompt = f"""
        Analyze this ensemble portfolio and suggest optimal rebalancing:
        
        Current Portfolio: {portfolio.name}
        Current Allocations:
        {json.dumps([asdict(alloc) for alloc in portfolio.strategy_allocations], indent=2, default=str)}
        
        Current Performance:
        {json.dumps(performance, indent=2)}
        
        Rebalancing Analysis:
        {json.dumps(analysis, indent=2)}
        
        Suggest new weights in JSON format:
        {{
            "strategy_name_1": 0.35,
            "strategy_name_2": 0.25,
            "strategy_name_3": 0.40
        }}
        
        Consider:
        1. Recent performance trends
        2. Risk-adjusted returns
        3. Diversification benefits
        4. Maximum 40% per strategy, minimum 10%
        """
        
        try:
            async for message in self.claude_analyst.query(prompt=prompt, options=self.claude_analyst.options):
                if message.type == "text":
                    content = message.content
                    
                    if "```json" in content:
                        json_start = content.find("```json") + 7
                        json_end = content.find("```", json_start)
                        json_content = content[json_start:json_end].strip()
                        
                        try:
                            new_weights = json.loads(json_content)
                            
                            # Validate weights
                            if self._validate_weights(new_weights, portfolio):
                                return new_weights
                        except json.JSONDecodeError:
                            logger.warning("Could not parse weight recommendations")
        
        except Exception as e:
            logger.error(f"Error calculating new weights: {e}")
        
        return None
    
    def _validate_weights(self, weights: Dict[str, float], portfolio: EnsemblePortfolio) -> bool:
        """Validate proposed weights"""
        # Check sum to 1
        total_weight = sum(weights.values())
        if abs(total_weight - 1.0) > 0.01:
            logger.warning(f"Weights don't sum to 1: {total_weight}")
            return False
        
        # Check individual weight limits
        for strategy_name, weight in weights.items():
            if weight < portfolio.min_strategy_weight or weight > portfolio.max_strategy_weight:
                logger.warning(f"Weight for {strategy_name} outside limits: {weight}")
                return False
        
        # Check all strategies are included
        expected_strategies = {alloc.strategy_name for alloc in portfolio.strategy_allocations}
        provided_strategies = set(weights.keys())
        
        if expected_strategies != provided_strategies:
            logger.warning("Mismatch in strategy names")
            return False
        
        return True
    
    async def _execute_rebalancing(self, portfolio: EnsemblePortfolio, new_weights: Dict[str, float]) -> bool:
        """Execute portfolio rebalancing"""
        logger.info("🔄 Executing portfolio rebalancing...")
        
        try:
            # Update allocations
            for allocation in portfolio.strategy_allocations:
                if allocation.strategy_name in new_weights:
                    allocation.target_weight = new_weights[allocation.strategy_name]
                    allocation.last_rebalance = datetime.now()
            
            # In a real system, this would:
            # 1. Calculate position changes needed
            # 2. Submit rebalancing orders to each strategy
            # 3. Monitor execution
            # 4. Update current weights after execution
            
            logger.success("✅ Rebalancing executed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error executing rebalancing: {e}")
            return False
    
    async def monitor_all_portfolios(self) -> Dict[str, Dict[str, Any]]:
        """Monitor all active ensemble portfolios"""
        logger.info("📊 Monitoring all ensemble portfolios...")
        
        monitoring_results = {}
        
        for portfolio_id, portfolio in self.active_portfolios.items():
            try:
                # Check if rebalancing is due
                if datetime.now() >= portfolio.next_rebalance:
                    logger.info(f"Portfolio {portfolio.name} due for rebalancing")
                    await self.rebalance_portfolio(portfolio_id)
                
                # Get current metrics
                current_performance = await self._get_current_strategy_performance(portfolio)
                portfolio_metrics = self._calculate_current_portfolio_metrics(portfolio, current_performance)
                
                monitoring_results[portfolio_id] = {
                    'portfolio': asdict(portfolio),
                    'current_metrics': portfolio_metrics,
                    'strategy_performance': current_performance,
                    'last_monitored': datetime.now().isoformat()
                }
                
            except Exception as e:
                logger.error(f"Error monitoring portfolio {portfolio_id}: {e}")
                monitoring_results[portfolio_id] = {'error': str(e)}
        
        return monitoring_results
    
    def _calculate_current_portfolio_metrics(self, portfolio: EnsemblePortfolio, performance: Dict) -> Dict[str, float]:
        """Calculate current portfolio-level metrics"""
        weights = np.array([alloc.current_weight for alloc in portfolio.strategy_allocations])
        
        if sum(weights) == 0:  # No current weights set
            weights = np.array([alloc.target_weight for alloc in portfolio.strategy_allocations])
        
        returns = np.array([performance[alloc.strategy_name]['current_return'] for alloc in portfolio.strategy_allocations])
        sharpe_ratios = np.array([performance[alloc.strategy_name]['sharpe_ratio'] for alloc in portfolio.strategy_allocations])
        
        # Portfolio metrics
        portfolio_return = np.dot(weights, returns)
        portfolio_sharpe = np.dot(weights, sharpe_ratios)
        
        return {
            'current_return': portfolio_return,
            'weighted_sharpe': portfolio_sharpe,
            'number_of_strategies': len(portfolio.strategy_allocations),
            'diversification_ratio': 1 / np.sum(weights ** 2)  # Inverse Herfindahl index
        }

# Example usage and testing
async def test_ensemble_manager():
    """Test the ensemble manager"""
    manager = EnsembleManager()
    
    # Create test ensemble portfolio
    criteria = {
        'min_success_score': 0.6,
        'max_strategies': 5,
        'min_validation_days': 7  # Reduced for testing
    }
    
    portfolio = await manager.create_ensemble_portfolio("Test_Ensemble_Portfolio", criteria)
    
    if portfolio:
        logger.success(f"Created portfolio: {portfolio.name}")
        logger.info(f"Strategies: {len(portfolio.strategy_allocations)}")
        logger.info(f"Expected Return: {portfolio.annualized_return:.2%}")
        logger.info(f"Expected Sharpe: {portfolio.sharpe_ratio:.2f}")
        
        # Test monitoring
        monitoring_results = await manager.monitor_all_portfolios()
        logger.info(f"Monitoring {len(monitoring_results)} portfolios")
        
        return True
    else:
        logger.warning("Failed to create ensemble portfolio")
        return False

if __name__ == "__main__":
    asyncio.run(test_ensemble_manager())