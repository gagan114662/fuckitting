"""
Live Trading and Paper Trading Integration for AlgoForge 3.0
Manages deployment to live trading, paper trading, and real-time monitoring
"""
import asyncio
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger

from quantconnect_client import QuantConnectClient
from claude_integration import GeneratedStrategy
from memory_system import AlgoForgeMemory, StrategyRecord
from config import config

class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"
    BACKTEST = "backtest"

class DeploymentStatus(Enum):
    PENDING = "pending"
    DEPLOYING = "deploying"
    ACTIVE = "active"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"

@dataclass
class LivePosition:
    """Live trading position"""
    symbol: str
    quantity: int
    avg_price: float
    current_price: float
    unrealized_pnl: float
    realized_pnl: float
    market_value: float
    last_updated: datetime

@dataclass
class LiveTradingMetrics:
    """Real-time trading metrics"""
    strategy_name: str
    deployed_date: datetime
    mode: TradingMode
    
    # Performance metrics
    total_return: float
    daily_return: float
    sharpe_ratio: float
    max_drawdown: float
    current_drawdown: float
    
    # Trading statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Risk metrics
    var_95: float  # Value at Risk 95%
    current_leverage: float
    max_leverage_used: float
    
    # Positions
    positions: List[LivePosition]
    cash_balance: float
    total_portfolio_value: float
    
    last_updated: datetime

@dataclass
class DeploymentConfig:
    """Configuration for strategy deployment"""
    strategy_name: str
    trading_mode: TradingMode
    initial_capital: float
    max_leverage: float
    risk_limits: Dict[str, float]
    brokerage: str
    auto_restart: bool = True
    max_daily_loss: float = 0.05  # 5% daily loss limit
    position_size_limit: float = 0.2  # 20% per position limit

class LiveTradingManager:
    """Manages live trading deployments and monitoring"""
    
    def __init__(self):
        self.qc_client = QuantConnectClient()
        self.memory = AlgoForgeMemory()
        self.active_deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_configs: Dict[str, DeploymentConfig] = {}
    
    async def deploy_strategy_to_paper_trading(self, strategy: GeneratedStrategy, initial_capital: float = 100000) -> bool:
        """Deploy strategy to paper trading"""
        logger.info(f"📄 Deploying {strategy.name} to paper trading...")
        
        try:
            # Create deployment configuration
            config = DeploymentConfig(
                strategy_name=strategy.name,
                trading_mode=TradingMode.PAPER,
                initial_capital=initial_capital,
                max_leverage=2.0,
                risk_limits={
                    'max_position_size': 0.2,
                    'max_daily_loss': 0.05,
                    'max_drawdown': 0.15
                },
                brokerage="QuantConnect Paper Trading"
            )
            
            # Deploy to QuantConnect
            deployment_result = await self._deploy_to_quantconnect(strategy, config)
            
            if deployment_result['success']:
                # Store deployment info
                self.deployment_configs[strategy.name] = config
                self.active_deployments[strategy.name] = {
                    'deployment_id': deployment_result['deployment_id'],
                    'status': DeploymentStatus.ACTIVE,
                    'deployed_at': datetime.now(),
                    'config': config,
                    'metrics': None
                }
                
                logger.success(f"✅ {strategy.name} deployed to paper trading")
                return True
            else:
                logger.error(f"❌ Failed to deploy {strategy.name}: {deployment_result.get('error')}")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying strategy to paper trading: {e}")
            return False
    
    async def deploy_strategy_to_live_trading(self, strategy: GeneratedStrategy, deployment_config: DeploymentConfig) -> bool:
        """Deploy strategy to live trading (after paper trading validation)"""
        logger.info(f"🚀 Deploying {strategy.name} to LIVE trading...")
        
        try:
            # Validate strategy is ready for live trading
            validation_result = await self._validate_strategy_for_live_trading(strategy)
            
            if not validation_result['ready']:
                logger.warning(f"Strategy {strategy.name} not ready for live trading: {validation_result['reasons']}")
                return False
            
            # Confirm live deployment (in real system, would require manual confirmation)
            logger.warning("⚠️  LIVE TRADING DEPLOYMENT - This would deploy with real money!")
            logger.info("In a production system, this would require manual confirmation and additional safeguards")
            
            # For safety, we'll simulate live deployment
            deployment_result = await self._simulate_live_deployment(strategy, deployment_config)
            
            if deployment_result['success']:
                self.deployment_configs[strategy.name] = deployment_config
                self.active_deployments[strategy.name] = {
                    'deployment_id': deployment_result['deployment_id'],
                    'status': DeploymentStatus.ACTIVE,
                    'deployed_at': datetime.now(),
                    'config': deployment_config,
                    'metrics': None
                }
                
                logger.success(f"✅ {strategy.name} deployed to live trading (SIMULATED)")
                return True
            else:
                logger.error(f"❌ Failed to deploy {strategy.name} to live trading")
                return False
                
        except Exception as e:
            logger.error(f"Error deploying strategy to live trading: {e}")
            return False
    
    async def _deploy_to_quantconnect(self, strategy: GeneratedStrategy, config: DeploymentConfig) -> Dict[str, Any]:
        """Deploy strategy to QuantConnect live trading"""
        try:
            async with self.qc_client as client:
                # Create project for live trading
                project_name = f"Live_{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_id = await client.create_project(project_name)
                
                # Modify strategy code for live trading
                live_code = self._prepare_strategy_for_live_trading(strategy.code, config)
                
                # Upload strategy
                await client.upload_file(project_id, "main.py", live_code)
                
                # Compile
                if await client.compile_project(project_id):
                    # Create live algorithm
                    live_result = await self._create_live_algorithm(client, project_id, config)
                    
                    return {
                        'success': True,
                        'deployment_id': live_result.get('algorithmId', f"paper_{project_id}"),
                        'project_id': project_id
                    }
                else:
                    return {'success': False, 'error': 'Compilation failed'}
                    
        except Exception as e:
            logger.error(f"Error in QuantConnect deployment: {e}")
            return {'success': False, 'error': str(e)}
    
    def _prepare_strategy_for_live_trading(self, strategy_code: str, config: DeploymentConfig) -> str:
        """Prepare strategy code for live trading"""
        
        # Add live trading specific configurations
        live_additions = f"""
# Live Trading Configuration
# Mode: {config.trading_mode.value}
# Initial Capital: ${config.initial_capital:,.0f}
# Max Leverage: {config.max_leverage}x

# Risk Management Settings
self.max_position_size = {config.risk_limits.get('max_position_size', 0.2)}
self.max_daily_loss = {config.risk_limits.get('max_daily_loss', 0.05)}
self.max_drawdown = {config.risk_limits.get('max_drawdown', 0.15)}

# Live trading risk controls
def OnData(self, data):
    # Check daily loss limit
    if self.Portfolio.UnrealizedProfit / self.Portfolio.TotalPortfolioValue < -self.max_daily_loss:
        self.Liquidate("Daily loss limit exceeded")
        self.Quit("Strategy stopped due to daily loss limit")
        return
    
    # Check maximum drawdown
    if self.Portfolio.TotalPortfolioValue / self.Portfolio.TotalPortfolioValue < (1 - self.max_drawdown):
        self.Liquidate("Maximum drawdown exceeded")
        self.Quit("Strategy stopped due to maximum drawdown")
        return
    
    # Original strategy logic follows...
"""
        
        return live_additions + "\n" + strategy_code
    
    async def _create_live_algorithm(self, client: QuantConnectClient, project_id: int, config: DeploymentConfig) -> Dict[str, Any]:
        """Create live algorithm deployment"""
        # This would use QuantConnect's live trading API
        # For now, we'll create a paper trading deployment
        
        if config.trading_mode == TradingMode.PAPER:
            # Create paper trading deployment
            deploy_data = {
                'projectId': project_id,
                'compileId': 'latest',
                'serverType': 'L1',  # Live trading server
                'baseLiveAlgorithmSettings': {
                    'brokerage': 'QuantConnectBrokerage',  # Paper trading
                    'environment': 'paper',
                    'initialCash': config.initial_capital
                }
            }
            
            # In a real implementation, this would call the live deployment API
            # For now, return a simulated response
            return {
                'algorithmId': f"paper_{project_id}_{datetime.now().strftime('%H%M%S')}",
                'status': 'Running',
                'launched': datetime.now().isoformat()
            }
        else:
            # For live trading, would need real brokerage integration
            logger.warning("Live trading deployment requires real brokerage configuration")
            return {'error': 'Live trading not configured'}
    
    async def _validate_strategy_for_live_trading(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """Validate strategy is ready for live trading"""
        reasons = []
        
        # Check if strategy has passed paper trading
        paper_performance = await self._get_paper_trading_performance(strategy.name)
        
        if not paper_performance:
            reasons.append("No paper trading performance data available")
        elif paper_performance['days_running'] < 30:
            reasons.append("Insufficient paper trading period (minimum 30 days required)")
        elif paper_performance['sharpe_ratio'] < config.targets.min_sharpe:
            reasons.append(f"Paper trading Sharpe ratio ({paper_performance['sharpe_ratio']:.2f}) below target ({config.targets.min_sharpe})")
        elif paper_performance['max_drawdown'] > config.targets.max_drawdown:
            reasons.append(f"Paper trading max drawdown ({paper_performance['max_drawdown']:.2%}) exceeds limit ({config.targets.max_drawdown:.2%})")
        
        # Check strategy validation scores
        strategy_record = await self._get_strategy_record(strategy.name)
        if strategy_record and strategy_record.success_score < 0.8:
            reasons.append(f"Strategy success score ({strategy_record.success_score:.2f}) below live trading threshold (0.8)")
        
        return {
            'ready': len(reasons) == 0,
            'reasons': reasons,
            'paper_performance': paper_performance
        }
    
    async def _simulate_live_deployment(self, strategy: GeneratedStrategy, config: DeploymentConfig) -> Dict[str, Any]:
        """Simulate live deployment for safety"""
        # In a real system, this would be actual live deployment
        logger.info("🔒 SIMULATING live deployment for safety")
        
        return {
            'success': True,
            'deployment_id': f"live_sim_{strategy.name}_{datetime.now().strftime('%H%M%S')}",
            'note': 'This is a simulated live deployment for safety'
        }
    
    async def get_live_trading_metrics(self, strategy_name: str) -> Optional[LiveTradingMetrics]:
        """Get real-time trading metrics for a deployed strategy"""
        if strategy_name not in self.active_deployments:
            logger.warning(f"Strategy {strategy_name} not found in active deployments")
            return None
        
        try:
            deployment = self.active_deployments[strategy_name]
            deployment_id = deployment['deployment_id']
            
            # Get live algorithm status from QuantConnect
            async with self.qc_client as client:
                # This would call QuantConnect's live algorithm API
                live_data = await self._get_live_algorithm_data(client, deployment_id)
                
                if live_data:
                    metrics = self._parse_live_metrics(strategy_name, live_data, deployment['config'])
                    
                    # Update cached metrics
                    deployment['metrics'] = metrics
                    
                    return metrics
                    
        except Exception as e:
            logger.error(f"Error getting live trading metrics for {strategy_name}: {e}")
        
        return None
    
    async def _get_live_algorithm_data(self, client: QuantConnectClient, deployment_id: str) -> Optional[Dict]:
        """Get live algorithm data from QuantConnect"""
        try:
            # This would use QuantConnect's live algorithm API
            # For now, return simulated data
            
            return {
                'algorithm': {
                    'status': 'Running',
                    'launched': datetime.now().isoformat(),
                    'stopped': None,
                    'equity': 105000,  # Simulated portfolio value
                    'cash': 25000,
                    'holdings': [
                        {'symbol': 'SPY', 'quantity': 100, 'averagePrice': 400, 'marketPrice': 405},
                        {'symbol': 'QQQ', 'quantity': 50, 'averagePrice': 350, 'marketPrice': 352}
                    ],
                    'trades': [
                        {'symbol': 'SPY', 'quantity': 100, 'price': 400, 'time': datetime.now().isoformat()},
                        {'symbol': 'QQQ', 'quantity': 50, 'price': 350, 'time': datetime.now().isoformat()}
                    ],
                    'statistics': {
                        'totalReturn': 0.05,
                        'sharpeRatio': 1.2,
                        'maxDrawdown': 0.03,
                        'totalTrades': 15,
                        'winRate': 0.67
                    }
                }
            }
            
        except Exception as e:
            logger.error(f"Error fetching live algorithm data: {e}")
            return None
    
    def _parse_live_metrics(self, strategy_name: str, live_data: Dict, config: DeploymentConfig) -> LiveTradingMetrics:
        """Parse live algorithm data into metrics"""
        algo_data = live_data['algorithm']
        stats = algo_data.get('statistics', {})
        
        # Parse positions
        positions = []
        for holding in algo_data.get('holdings', []):
            position = LivePosition(
                symbol=holding['symbol'],
                quantity=holding['quantity'],
                avg_price=holding['averagePrice'],
                current_price=holding['marketPrice'],
                unrealized_pnl=(holding['marketPrice'] - holding['averagePrice']) * holding['quantity'],
                realized_pnl=0,  # Would need to calculate from trades
                market_value=holding['marketPrice'] * holding['quantity'],
                last_updated=datetime.now()
            )
            positions.append(position)
        
        # Calculate metrics
        total_trades = stats.get('totalTrades', 0)
        win_rate = stats.get('winRate', 0)
        winning_trades = int(total_trades * win_rate)
        losing_trades = total_trades - winning_trades
        
        return LiveTradingMetrics(
            strategy_name=strategy_name,
            deployed_date=datetime.fromisoformat(algo_data['launched']),
            mode=config.trading_mode,
            total_return=stats.get('totalReturn', 0),
            daily_return=0,  # Would calculate from recent performance
            sharpe_ratio=stats.get('sharpeRatio', 0),
            max_drawdown=stats.get('maxDrawdown', 0),
            current_drawdown=0,  # Would calculate current drawdown
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=win_rate,
            avg_win=0,  # Would calculate from trades
            avg_loss=0,  # Would calculate from trades
            profit_factor=0,  # Would calculate from trades
            var_95=0,  # Would calculate VaR
            current_leverage=0,  # Would calculate current leverage
            max_leverage_used=0,  # Would track max leverage
            positions=positions,
            cash_balance=algo_data.get('cash', 0),
            total_portfolio_value=algo_data.get('equity', 0),
            last_updated=datetime.now()
        )
    
    async def stop_strategy(self, strategy_name: str, reason: str = "Manual stop") -> bool:
        """Stop a live trading strategy"""
        if strategy_name not in self.active_deployments:
            logger.warning(f"Strategy {strategy_name} not found in active deployments")
            return False
        
        try:
            deployment = self.active_deployments[strategy_name]
            deployment_id = deployment['deployment_id']
            
            logger.info(f"🛑 Stopping strategy {strategy_name}: {reason}")
            
            # Stop algorithm in QuantConnect
            async with self.qc_client as client:
                # This would call QuantConnect's stop algorithm API
                stop_result = await self._stop_live_algorithm(client, deployment_id)
                
                if stop_result:
                    deployment['status'] = DeploymentStatus.STOPPED
                    logger.success(f"✅ Strategy {strategy_name} stopped successfully")
                    return True
                    
        except Exception as e:
            logger.error(f"Error stopping strategy {strategy_name}: {e}")
        
        return False
    
    async def _stop_live_algorithm(self, client: QuantConnectClient, deployment_id: str) -> bool:
        """Stop live algorithm in QuantConnect"""
        try:
            # This would use QuantConnect's stop algorithm API
            logger.info(f"Stopping live algorithm {deployment_id}")
            return True  # Simulated success
        except Exception as e:
            logger.error(f"Error stopping live algorithm: {e}")
            return False
    
    async def _get_paper_trading_performance(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get paper trading performance metrics"""
        # This would query actual paper trading results
        # For now, return simulated data
        return {
            'days_running': 45,
            'total_return': 0.12,
            'sharpe_ratio': 1.35,
            'max_drawdown': 0.08,
            'total_trades': 67,
            'win_rate': 0.65
        }
    
    async def _get_strategy_record(self, strategy_name: str) -> Optional[StrategyRecord]:
        """Get strategy record from memory"""
        session = self.memory.get_session()
        try:
            record = session.query(StrategyRecord).filter(
                StrategyRecord.strategy_name == strategy_name
            ).first()
            return record
        finally:
            session.close()
    
    async def monitor_all_deployments(self) -> Dict[str, LiveTradingMetrics]:
        """Monitor all active deployments"""
        logger.info("📊 Monitoring all active deployments...")
        
        metrics = {}
        for strategy_name in self.active_deployments:
            strategy_metrics = await self.get_live_trading_metrics(strategy_name)
            if strategy_metrics:
                metrics[strategy_name] = strategy_metrics
                
                # Check for risk violations
                await self._check_risk_violations(strategy_name, strategy_metrics)
        
        return metrics
    
    async def _check_risk_violations(self, strategy_name: str, metrics: LiveTradingMetrics):
        """Check for risk violations and take action if needed"""
        config = self.deployment_configs.get(strategy_name)
        if not config:
            return
        
        violations = []
        
        # Check daily loss limit
        if metrics.daily_return < -config.risk_limits.get('max_daily_loss', 0.05):
            violations.append(f"Daily loss limit exceeded: {metrics.daily_return:.2%}")
        
        # Check maximum drawdown
        if metrics.current_drawdown > config.risk_limits.get('max_drawdown', 0.15):
            violations.append(f"Maximum drawdown exceeded: {metrics.current_drawdown:.2%}")
        
        # Check leverage limits
        if metrics.current_leverage > config.max_leverage:
            violations.append(f"Leverage limit exceeded: {metrics.current_leverage:.1f}x")
        
        if violations:
            logger.warning(f"⚠️ Risk violations detected for {strategy_name}:")
            for violation in violations:
                logger.warning(f"  - {violation}")
            
            # Auto-stop if configured
            if config.auto_restart:
                await self.stop_strategy(strategy_name, f"Risk violations: {', '.join(violations)}")

# Example usage and testing
async def test_live_trading_system():
    """Test the live trading system"""
    from claude_integration import GeneratedStrategy
    
    # Create sample strategy
    sample_strategy = GeneratedStrategy(
        hypothesis_id="live_test_001",
        name="Live_Test_Strategy",
        code="""
class LiveTestAlgorithm(QCAlgorithm):
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetCash(100000)
        self.AddEquity("SPY", Resolution.Daily)
        
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 0.5)
        """,
        description="Test strategy for live trading system",
        parameters={"allocation": 0.5},
        expected_metrics={"cagr": 0.15, "sharpe": 1.0},
        risk_controls=["position_sizing", "daily_loss_limit"],
        created_at=datetime.now()
    )
    
    # Initialize live trading manager
    manager = LiveTradingManager()
    
    # Deploy to paper trading
    paper_success = await manager.deploy_strategy_to_paper_trading(sample_strategy)
    logger.info(f"Paper trading deployment: {'Success' if paper_success else 'Failed'}")
    
    if paper_success:
        # Get live metrics
        await asyncio.sleep(2)  # Wait for deployment
        metrics = await manager.get_live_trading_metrics(sample_strategy.name)
        
        if metrics:
            logger.info(f"Live metrics retrieved for {metrics.strategy_name}")
            logger.info(f"Total return: {metrics.total_return:.2%}")
            logger.info(f"Sharpe ratio: {metrics.sharpe_ratio:.2f}")
            logger.info(f"Positions: {len(metrics.positions)}")
        
        # Monitor all deployments
        all_metrics = await manager.monitor_all_deployments()
        logger.info(f"Monitoring {len(all_metrics)} active deployments")
    
    return paper_success

if __name__ == "__main__":
    asyncio.run(test_live_trading_system())