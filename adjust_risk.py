#!/usr/bin/env python3
"""
Dynamic Risk Adjustment System
Automatically adjusts risk parameters based on market conditions and strategy performance
"""
import asyncio
import numpy as np
import json
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
from dataclasses import dataclass, asdict
from enum import Enum
import sqlite3

class RiskLevel(Enum):
    CONSERVATIVE = "conservative"
    MODERATE = "moderate"
    AGGRESSIVE = "aggressive"
    ULTRA_CONSERVATIVE = "ultra_conservative"

class MarketRegime(Enum):
    BULL = "bull"
    BEAR = "bear"
    SIDEWAYS = "sideways"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"

@dataclass
class RiskParameters:
    max_position_size: float  # Maximum position size as % of portfolio
    stop_loss_percent: float  # Stop loss as % of position
    take_profit_percent: float  # Take profit as % of position
    max_daily_risk: float  # Maximum daily risk as % of portfolio
    max_correlation: float  # Maximum correlation between positions
    volatility_multiplier: float  # Multiplier for volatility-based sizing
    sharpe_threshold: float  # Minimum Sharpe ratio threshold
    max_drawdown_limit: float  # Maximum acceptable drawdown
    leverage_limit: float  # Maximum leverage allowed
    diversification_requirement: int  # Minimum number of uncorrelated positions

@dataclass
class RiskAdjustment:
    timestamp: str
    strategy_name: str
    old_parameters: RiskParameters
    new_parameters: RiskParameters
    trigger_reason: str
    market_regime: str
    volatility_level: float
    recent_performance: float
    adjustment_magnitude: float

class RiskManager:
    """Dynamic risk management system with autonomous adjustments"""
    
    def __init__(self, config_dir: str = "config", data_dir: str = "data"):
        self.config_dir = Path(config_dir)
        self.data_dir = Path(data_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize risk profiles
        self.risk_profiles = self._initialize_risk_profiles()
        self.current_risk_parameters = {}
        self.adjustment_history = []
        
        # Initialize database
        self.db_path = self.config_dir / "risk_management.db"
        self._init_risk_database()
        
        # Load existing parameters
        self._load_current_parameters()
    
    def _initialize_risk_profiles(self) -> Dict[RiskLevel, RiskParameters]:
        """Initialize predefined risk profiles"""
        return {
            RiskLevel.ULTRA_CONSERVATIVE: RiskParameters(
                max_position_size=0.02,  # 2%
                stop_loss_percent=0.01,  # 1%
                take_profit_percent=0.02,  # 2%
                max_daily_risk=0.005,  # 0.5%
                max_correlation=0.3,
                volatility_multiplier=0.5,
                sharpe_threshold=1.5,
                max_drawdown_limit=0.03,  # 3%
                leverage_limit=1.0,
                diversification_requirement=10
            ),
            RiskLevel.CONSERVATIVE: RiskParameters(
                max_position_size=0.05,  # 5%
                stop_loss_percent=0.02,  # 2%
                take_profit_percent=0.04,  # 4%
                max_daily_risk=0.01,  # 1%
                max_correlation=0.5,
                volatility_multiplier=0.75,
                sharpe_threshold=1.2,
                max_drawdown_limit=0.05,  # 5%
                leverage_limit=1.5,
                diversification_requirement=8
            ),
            RiskLevel.MODERATE: RiskParameters(
                max_position_size=0.10,  # 10%
                stop_loss_percent=0.03,  # 3%
                take_profit_percent=0.06,  # 6%
                max_daily_risk=0.02,  # 2%
                max_correlation=0.7,
                volatility_multiplier=1.0,
                sharpe_threshold=1.0,
                max_drawdown_limit=0.10,  # 10%
                leverage_limit=2.0,
                diversification_requirement=5
            ),
            RiskLevel.AGGRESSIVE: RiskParameters(
                max_position_size=0.20,  # 20%
                stop_loss_percent=0.05,  # 5%
                take_profit_percent=0.10,  # 10%
                max_daily_risk=0.05,  # 5%
                max_correlation=0.8,
                volatility_multiplier=1.5,
                sharpe_threshold=0.8,
                max_drawdown_limit=0.20,  # 20%
                leverage_limit=3.0,
                diversification_requirement=3
            )
        }
    
    def _init_risk_database(self):
        """Initialize SQLite database for risk management"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS risk_parameters (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        risk_level TEXT NOT NULL,
                        parameters TEXT NOT NULL,
                        is_current BOOLEAN DEFAULT TRUE
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS risk_adjustments (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        strategy_name TEXT NOT NULL,
                        old_parameters TEXT,
                        new_parameters TEXT,
                        trigger_reason TEXT,
                        market_regime TEXT,
                        volatility_level REAL,
                        recent_performance REAL,
                        adjustment_magnitude REAL
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS market_conditions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        regime TEXT NOT NULL,
                        volatility REAL,
                        trend_strength REAL,
                        correlation_level REAL,
                        risk_on_off REAL
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_strategy_name ON risk_parameters(strategy_name)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON risk_adjustments(timestamp)
                ''')
        
        except Exception as e:
            logger.error(f"Failed to initialize risk database: {e}")
    
    def _load_current_parameters(self):
        """Load current risk parameters from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT DISTINCT strategy_name FROM risk_parameters 
                    WHERE is_current = TRUE
                ''')
                
                for row in cursor.fetchall():
                    strategy_name = row['strategy_name']
                    params = self._get_current_parameters(strategy_name)
                    if params:
                        self.current_risk_parameters[strategy_name] = params
                        
        except Exception as e:
            logger.error(f"Error loading current parameters: {e}")
    
    async def adjust_risk_conservative(self) -> Dict[str, Any]:
        """Adjust risk to conservative levels"""
        logger.info("🛡️ Adjusting risk to conservative levels...")
        
        adjustment_results = {
            'timestamp': datetime.now().isoformat(),
            'adjustment_type': 'conservative',
            'strategies_adjusted': 0,
            'adjustments': [],
            'success': True
        }
        
        try:
            # Get current market conditions
            market_conditions = await self._assess_market_conditions()
            
            # Adjust all strategies to conservative settings
            for strategy_name in self._get_active_strategies():
                old_params = self.current_risk_parameters.get(
                    strategy_name, 
                    self.risk_profiles[RiskLevel.MODERATE]
                )
                
                # Use ultra-conservative profile for high-risk situations
                if market_conditions['volatility'] > 0.3 or market_conditions['regime'] == MarketRegime.BEAR:
                    new_params = self.risk_profiles[RiskLevel.ULTRA_CONSERVATIVE]
                    risk_level = RiskLevel.ULTRA_CONSERVATIVE
                else:
                    new_params = self.risk_profiles[RiskLevel.CONSERVATIVE]
                    risk_level = RiskLevel.CONSERVATIVE
                
                # Apply adjustment
                success = await self._apply_risk_adjustment(
                    strategy_name=strategy_name,
                    new_parameters=new_params,
                    trigger_reason="manual_conservative_adjustment",
                    market_regime=market_conditions['regime'].value,
                    volatility_level=market_conditions['volatility']
                )
                
                if success:
                    adjustment_results['strategies_adjusted'] += 1
                    adjustment_results['adjustments'].append({
                        'strategy': strategy_name,
                        'new_risk_level': risk_level.value,
                        'position_size_reduction': (old_params.max_position_size - new_params.max_position_size) / old_params.max_position_size
                    })
                    
                    logger.success(f"✅ Adjusted {strategy_name} to {risk_level.value} risk")
                else:
                    adjustment_results['success'] = False
            
            if adjustment_results['strategies_adjusted'] > 0:
                logger.success(f"✅ Conservative risk adjustment completed: {adjustment_results['strategies_adjusted']} strategies adjusted")
            else:
                logger.warning("⚠️ No strategies were adjusted")
            
            return adjustment_results
            
        except Exception as e:
            logger.error(f"❌ Error in conservative risk adjustment: {e}")
            adjustment_results['success'] = False
            adjustment_results['error'] = str(e)
            return adjustment_results
    
    async def _assess_market_conditions(self) -> Dict[str, Any]:
        """Assess current market conditions"""
        try:
            # In a real implementation, this would fetch live market data
            # For now, we'll simulate market condition assessment
            
            # Load recent market data if available
            market_data = await self._load_recent_market_data()
            
            if market_data is not None and len(market_data) > 0:
                # Calculate volatility
                returns = market_data.get('returns', [0.01] * 100)
                volatility = np.std(returns) * np.sqrt(252)  # Annualized volatility
                
                # Determine regime
                recent_returns = returns[-20:] if len(returns) >= 20 else returns
                avg_return = np.mean(recent_returns)
                
                if avg_return > 0.002 and volatility < 0.2:
                    regime = MarketRegime.BULL
                elif avg_return < -0.002:
                    regime = MarketRegime.BEAR
                elif volatility > 0.3:
                    regime = MarketRegime.HIGH_VOLATILITY
                elif volatility < 0.1:
                    regime = MarketRegime.LOW_VOLATILITY
                else:
                    regime = MarketRegime.SIDEWAYS
                    
                # Calculate trend strength
                trend_strength = abs(avg_return) / volatility if volatility > 0 else 0
                
                # Risk on/off indicator
                risk_on_off = 0.5 + (avg_return / 0.01) * 0.3  # Scale between 0 and 1
                risk_on_off = max(0, min(1, risk_on_off))
                
            else:
                # Default values when no data available
                volatility = 0.2
                regime = MarketRegime.SIDEWAYS
                trend_strength = 0.5
                risk_on_off = 0.5
            
            conditions = {
                'regime': regime,
                'volatility': volatility,
                'trend_strength': trend_strength,
                'correlation_level': 0.6,  # Simplified
                'risk_on_off': risk_on_off,
                'timestamp': datetime.now().isoformat()
            }
            
            # Save to database
            await self._save_market_conditions(conditions)
            
            return conditions
            
        except Exception as e:
            logger.error(f"Error assessing market conditions: {e}")
            # Return safe defaults
            return {
                'regime': MarketRegime.SIDEWAYS,
                'volatility': 0.25,
                'trend_strength': 0.5,
                'correlation_level': 0.6,
                'risk_on_off': 0.4
            }
    
    async def _load_recent_market_data(self) -> Optional[Dict[str, List[float]]]:
        """Load recent market data for analysis"""
        try:
            # Look for market data files
            data_files = [
                self.data_dir / "market_data.csv",
                self.data_dir / "trading_data.csv",
                self.data_dir / "price_data.csv"
            ]
            
            for data_file in data_files:
                if data_file.exists():
                    import pandas as pd
                    df = pd.read_csv(data_file)
                    
                    # Extract relevant columns
                    if 'returns' in df.columns:
                        returns = df['returns'].dropna().tolist()
                    elif 'close' in df.columns:
                        prices = df['close'].dropna()
                        returns = prices.pct_change().dropna().tolist()
                    else:
                        continue
                    
                    return {
                        'returns': returns[-1000:],  # Last 1000 observations
                        'timestamps': df.get('timestamp', df.get('date', [])).tolist()[-1000:]
                    }
            
            # Generate synthetic data if no real data available
            np.random.seed(int(datetime.now().timestamp()) % 10000)
            n_points = 500
            returns = np.random.normal(0.0005, 0.015, n_points).tolist()
            
            return {
                'returns': returns,
                'timestamps': [datetime.now().isoformat()] * n_points
            }
            
        except Exception as e:
            logger.debug(f"Error loading market data: {e}")
            return None
    
    async def _save_market_conditions(self, conditions: Dict[str, Any]):
        """Save market conditions to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO market_conditions 
                    (timestamp, regime, volatility, trend_strength, correlation_level, risk_on_off)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    conditions['timestamp'],
                    conditions['regime'].value if hasattr(conditions['regime'], 'value') else str(conditions['regime']),
                    conditions['volatility'],
                    conditions['trend_strength'],
                    conditions['correlation_level'],
                    conditions['risk_on_off']
                ))
                
        except Exception as e:
            logger.debug(f"Error saving market conditions: {e}")
    
    def _get_active_strategies(self) -> List[str]:
        """Get list of active strategies"""
        try:
            # Check for strategy files
            strategies_dir = Path("strategies")
            if strategies_dir.exists():
                strategy_files = [f.stem for f in strategies_dir.glob("*.py") if f.stem != "__init__"]
                if strategy_files:
                    return strategy_files
            
            # Fallback: return strategies from database
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT DISTINCT strategy_name FROM risk_parameters')
                db_strategies = [row[0] for row in cursor.fetchall()]
                if db_strategies:
                    return db_strategies
            
            # Default strategies if none found
            return ['momentum_strategy', 'mean_reversion_strategy', 'volatility_strategy']
            
        except Exception as e:
            logger.debug(f"Error getting active strategies: {e}")
            return ['default_strategy']
    
    async def _apply_risk_adjustment(self, strategy_name: str, new_parameters: RiskParameters,
                                   trigger_reason: str, market_regime: str, volatility_level: float) -> bool:
        """Apply risk parameter adjustment"""
        try:
            # Get old parameters
            old_parameters = self.current_risk_parameters.get(
                strategy_name, 
                self.risk_profiles[RiskLevel.MODERATE]
            )
            
            # Calculate adjustment magnitude
            adjustment_magnitude = self._calculate_adjustment_magnitude(old_parameters, new_parameters)
            
            # Update current parameters
            self.current_risk_parameters[strategy_name] = new_parameters
            
            # Save to database
            await self._save_risk_parameters(strategy_name, new_parameters)
            
            # Create adjustment record
            adjustment = RiskAdjustment(
                timestamp=datetime.now().isoformat(),
                strategy_name=strategy_name,
                old_parameters=old_parameters,
                new_parameters=new_parameters,
                trigger_reason=trigger_reason,
                market_regime=market_regime,
                volatility_level=volatility_level,
                recent_performance=0.0,  # Would be calculated from recent results
                adjustment_magnitude=adjustment_magnitude
            )
            
            # Save adjustment record
            await self._save_risk_adjustment(adjustment)
            
            # Apply parameters to strategy files if they exist
            await self._update_strategy_risk_parameters(strategy_name, new_parameters)
            
            logger.debug(f"Applied risk adjustment for {strategy_name} (magnitude: {adjustment_magnitude:.3f})")
            return True
            
        except Exception as e:
            logger.error(f"Error applying risk adjustment for {strategy_name}: {e}")
            return False
    
    def _calculate_adjustment_magnitude(self, old_params: RiskParameters, new_params: RiskParameters) -> float:
        """Calculate the magnitude of risk adjustment"""
        try:
            # Calculate relative changes in key parameters
            position_size_change = abs(new_params.max_position_size - old_params.max_position_size) / old_params.max_position_size
            stop_loss_change = abs(new_params.stop_loss_percent - old_params.stop_loss_percent) / old_params.stop_loss_percent
            daily_risk_change = abs(new_params.max_daily_risk - old_params.max_daily_risk) / old_params.max_daily_risk
            
            # Average the changes
            magnitude = (position_size_change + stop_loss_change + daily_risk_change) / 3
            
            return magnitude
            
        except Exception as e:
            logger.debug(f"Error calculating adjustment magnitude: {e}")
            return 0.0
    
    async def _save_risk_parameters(self, strategy_name: str, parameters: RiskParameters):
        """Save risk parameters to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                # Mark previous parameters as not current
                conn.execute('''
                    UPDATE risk_parameters 
                    SET is_current = FALSE 
                    WHERE strategy_name = ? AND is_current = TRUE
                ''', (strategy_name,))
                
                # Insert new parameters
                conn.execute('''
                    INSERT INTO risk_parameters 
                    (strategy_name, timestamp, risk_level, parameters, is_current)
                    VALUES (?, ?, ?, ?, TRUE)
                ''', (
                    strategy_name,
                    datetime.now().isoformat(),
                    'custom',  # Could be mapped to actual risk level
                    json.dumps(asdict(parameters))
                ))
                
        except Exception as e:
            logger.error(f"Error saving risk parameters: {e}")
    
    async def _save_risk_adjustment(self, adjustment: RiskAdjustment):
        """Save risk adjustment record"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO risk_adjustments 
                    (timestamp, strategy_name, old_parameters, new_parameters, trigger_reason,
                     market_regime, volatility_level, recent_performance, adjustment_magnitude)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    adjustment.timestamp,
                    adjustment.strategy_name,
                    json.dumps(asdict(adjustment.old_parameters)),
                    json.dumps(asdict(adjustment.new_parameters)),
                    adjustment.trigger_reason,
                    adjustment.market_regime,
                    adjustment.volatility_level,
                    adjustment.recent_performance,
                    adjustment.adjustment_magnitude
                ))
                
        except Exception as e:
            logger.error(f"Error saving risk adjustment: {e}")
    
    async def _update_strategy_risk_parameters(self, strategy_name: str, parameters: RiskParameters):
        """Update strategy files with new risk parameters"""
        try:
            strategy_file = Path("strategies") / f"{strategy_name}.py"
            if not strategy_file.exists():
                return
            
            # Read strategy file
            with open(strategy_file, 'r') as f:
                content = f.read()
            
            # Create risk parameter section
            risk_params_section = f'''
# Auto-generated risk parameters - Updated {datetime.now().isoformat()}
RISK_PARAMETERS = {{
    'max_position_size': {parameters.max_position_size},
    'stop_loss_percent': {parameters.stop_loss_percent},
    'take_profit_percent': {parameters.take_profit_percent},
    'max_daily_risk': {parameters.max_daily_risk},
    'max_correlation': {parameters.max_correlation},
    'volatility_multiplier': {parameters.volatility_multiplier},
    'sharpe_threshold': {parameters.sharpe_threshold},
    'max_drawdown_limit': {parameters.max_drawdown_limit},
    'leverage_limit': {parameters.leverage_limit},
    'diversification_requirement': {parameters.diversification_requirement}
}}
'''
            
            # Replace or add risk parameters section
            if 'RISK_PARAMETERS = {' in content:
                # Replace existing parameters
                import re
                pattern = r'RISK_PARAMETERS = \{[^}]*\}'
                content = re.sub(pattern, risk_params_section.strip(), content, flags=re.DOTALL)
            else:
                # Add parameters at the top after imports
                lines = content.split('\n')
                insert_index = 0
                for i, line in enumerate(lines):
                    if (line.strip().startswith('import ') or 
                        line.strip().startswith('from ') or 
                        line.strip().startswith('#')):
                        insert_index = i + 1
                    else:
                        break
                
                lines.insert(insert_index, risk_params_section)
                content = '\n'.join(lines)
            
            # Write updated file
            with open(strategy_file, 'w') as f:
                f.write(content)
            
            logger.debug(f"Updated risk parameters in {strategy_file}")
            
        except Exception as e:
            logger.debug(f"Error updating strategy file {strategy_name}: {e}")
    
    def _get_current_parameters(self, strategy_name: str) -> Optional[RiskParameters]:
        """Get current risk parameters for a strategy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT parameters FROM risk_parameters 
                    WHERE strategy_name = ? AND is_current = TRUE
                    ORDER BY timestamp DESC LIMIT 1
                ''', (strategy_name,))
                
                result = cursor.fetchone()
                if result:
                    params_dict = json.loads(result['parameters'])
                    return RiskParameters(**params_dict)
                
                return None
                
        except Exception as e:
            logger.debug(f"Error getting current parameters for {strategy_name}: {e}")
            return None
    
    async def autonomous_risk_adjustment(self) -> Dict[str, Any]:
        """Autonomous risk adjustment based on market conditions and performance"""
        logger.info("🤖 Starting autonomous risk adjustment...")
        
        adjustment_results = {
            'timestamp': datetime.now().isoformat(),
            'strategies_analyzed': 0,
            'strategies_adjusted': 0,
            'adjustments': [],
            'market_conditions': {},
            'success': True
        }
        
        try:
            # Assess market conditions
            market_conditions = await self._assess_market_conditions()
            adjustment_results['market_conditions'] = {
                'regime': market_conditions['regime'].value,
                'volatility': market_conditions['volatility'],
                'risk_on_off': market_conditions['risk_on_off']
            }
            
            # Analyze each strategy
            for strategy_name in self._get_active_strategies():
                adjustment_results['strategies_analyzed'] += 1
                
                # Get current parameters
                current_params = self.current_risk_parameters.get(
                    strategy_name,
                    self.risk_profiles[RiskLevel.MODERATE]
                )
                
                # Determine optimal risk level based on conditions
                optimal_risk_level = self._determine_optimal_risk_level(market_conditions, strategy_name)
                new_params = self._calculate_dynamic_parameters(
                    base_params=self.risk_profiles[optimal_risk_level],
                    market_conditions=market_conditions,
                    strategy_name=strategy_name
                )
                
                # Check if adjustment is needed
                adjustment_magnitude = self._calculate_adjustment_magnitude(current_params, new_params)
                
                if adjustment_magnitude > 0.05:  # 5% threshold for adjustment
                    success = await self._apply_risk_adjustment(
                        strategy_name=strategy_name,
                        new_parameters=new_params,
                        trigger_reason="autonomous_market_adjustment",
                        market_regime=market_conditions['regime'].value,
                        volatility_level=market_conditions['volatility']
                    )
                    
                    if success:
                        adjustment_results['strategies_adjusted'] += 1
                        adjustment_results['adjustments'].append({
                            'strategy': strategy_name,
                            'risk_level': optimal_risk_level.value,
                            'adjustment_magnitude': adjustment_magnitude,
                            'reason': f"Market regime: {market_conditions['regime'].value}, Volatility: {market_conditions['volatility']:.3f}"
                        })
                        
                        logger.success(f"✅ Autonomously adjusted {strategy_name} to {optimal_risk_level.value}")
                    else:
                        adjustment_results['success'] = False
                else:
                    logger.debug(f"No significant adjustment needed for {strategy_name} (magnitude: {adjustment_magnitude:.3f})")
            
            if adjustment_results['strategies_adjusted'] > 0:
                logger.success(f"✅ Autonomous risk adjustment completed: {adjustment_results['strategies_adjusted']} strategies adjusted")
            else:
                logger.info("ℹ️ No strategies required risk adjustment")
            
            return adjustment_results
            
        except Exception as e:
            logger.error(f"❌ Error in autonomous risk adjustment: {e}")
            adjustment_results['success'] = False
            adjustment_results['error'] = str(e)
            return adjustment_results
    
    def _determine_optimal_risk_level(self, market_conditions: Dict[str, Any], strategy_name: str) -> RiskLevel:
        """Determine optimal risk level based on market conditions"""
        try:
            regime = market_conditions['regime']
            volatility = market_conditions['volatility']
            risk_on_off = market_conditions['risk_on_off']
            
            # Risk-off conditions
            if (regime == MarketRegime.BEAR or 
                volatility > 0.4 or 
                risk_on_off < 0.3):
                return RiskLevel.ULTRA_CONSERVATIVE
            
            # High volatility but not bearish
            elif volatility > 0.3 or risk_on_off < 0.4:
                return RiskLevel.CONSERVATIVE
            
            # Favorable conditions
            elif (regime == MarketRegime.BULL and 
                  volatility < 0.2 and 
                  risk_on_off > 0.7):
                return RiskLevel.AGGRESSIVE
            
            # Default moderate risk
            else:
                return RiskLevel.MODERATE
                
        except Exception as e:
            logger.debug(f"Error determining optimal risk level: {e}")
            return RiskLevel.CONSERVATIVE  # Safe default
    
    def _calculate_dynamic_parameters(self, base_params: RiskParameters, 
                                    market_conditions: Dict[str, Any], 
                                    strategy_name: str) -> RiskParameters:
        """Calculate dynamic risk parameters based on market conditions"""
        try:
            volatility = market_conditions['volatility']
            risk_on_off = market_conditions['risk_on_off']
            
            # Adjust parameters based on volatility
            volatility_adjustment = 1.0 - (volatility - 0.2) * 2  # Reduce size in high vol
            volatility_adjustment = max(0.3, min(1.5, volatility_adjustment))
            
            # Adjust parameters based on risk sentiment
            sentiment_adjustment = 0.5 + risk_on_off * 0.5  # 0.5 to 1.0 range
            
            # Combined adjustment factor
            adjustment_factor = (volatility_adjustment + sentiment_adjustment) / 2
            
            # Create adjusted parameters
            adjusted_params = RiskParameters(
                max_position_size=base_params.max_position_size * adjustment_factor,
                stop_loss_percent=base_params.stop_loss_percent / adjustment_factor,  # Tighter stops in risky conditions
                take_profit_percent=base_params.take_profit_percent * adjustment_factor,
                max_daily_risk=base_params.max_daily_risk * adjustment_factor,
                max_correlation=base_params.max_correlation,
                volatility_multiplier=base_params.volatility_multiplier * (2 - adjustment_factor),
                sharpe_threshold=base_params.sharpe_threshold / adjustment_factor,
                max_drawdown_limit=base_params.max_drawdown_limit * adjustment_factor,
                leverage_limit=base_params.leverage_limit * adjustment_factor,
                diversification_requirement=max(base_params.diversification_requirement, 
                                              int(base_params.diversification_requirement / adjustment_factor))
            )
            
            # Ensure parameters stay within reasonable bounds
            adjusted_params.max_position_size = max(0.01, min(0.5, adjusted_params.max_position_size))
            adjusted_params.stop_loss_percent = max(0.005, min(0.1, adjusted_params.stop_loss_percent))
            adjusted_params.max_daily_risk = max(0.001, min(0.1, adjusted_params.max_daily_risk))
            adjusted_params.leverage_limit = max(1.0, min(5.0, adjusted_params.leverage_limit))
            
            return adjusted_params
            
        except Exception as e:
            logger.error(f"Error calculating dynamic parameters: {e}")
            return base_params  # Return base parameters if calculation fails
    
    def get_risk_status_report(self) -> Dict[str, Any]:
        """Generate comprehensive risk status report"""
        try:
            report = {
                'timestamp': datetime.now().isoformat(),
                'strategies': {},
                'market_conditions': {},
                'recent_adjustments': [],
                'overall_risk_level': 'unknown'
            }
            
            # Get market conditions
            market_conditions = asyncio.run(self._assess_market_conditions())
            report['market_conditions'] = {
                'regime': market_conditions['regime'].value,
                'volatility': market_conditions['volatility'],
                'risk_on_off': market_conditions['risk_on_off'],
                'trend_strength': market_conditions['trend_strength']
            }
            
            # Get strategy risk parameters
            total_risk_score = 0
            strategy_count = 0
            
            for strategy_name in self._get_active_strategies():
                current_params = self.current_risk_parameters.get(strategy_name)
                if current_params:
                    strategy_count += 1
                    
                    # Calculate risk score (0-1, higher = more risk)
                    risk_score = (
                        current_params.max_position_size * 2 +  # Weight by 2
                        current_params.max_daily_risk * 10 +    # Weight by 10
                        current_params.leverage_limit / 5       # Weight by 1/5
                    ) / 3
                    
                    total_risk_score += risk_score
                    
                    report['strategies'][strategy_name] = {
                        'max_position_size': current_params.max_position_size,
                        'stop_loss_percent': current_params.stop_loss_percent,
                        'max_daily_risk': current_params.max_daily_risk,
                        'leverage_limit': current_params.leverage_limit,
                        'risk_score': risk_score
                    }
            
            # Calculate overall risk level
            if strategy_count > 0:
                avg_risk_score = total_risk_score / strategy_count
                if avg_risk_score < 0.3:
                    report['overall_risk_level'] = 'conservative'
                elif avg_risk_score < 0.6:
                    report['overall_risk_level'] = 'moderate'
                else:
                    report['overall_risk_level'] = 'aggressive'
            
            # Get recent adjustments
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM risk_adjustments 
                    ORDER BY timestamp DESC LIMIT 10
                ''')
                
                for row in cursor.fetchall():
                    report['recent_adjustments'].append({
                        'timestamp': row['timestamp'],
                        'strategy': row['strategy_name'],
                        'reason': row['trigger_reason'],
                        'adjustment_magnitude': row['adjustment_magnitude']
                    })
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating risk status report: {e}")
            return {'error': str(e)}

async def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Dynamic risk adjustment system")
    parser.add_argument("--conservative", action="store_true", help="Adjust all strategies to conservative risk")
    parser.add_argument("--autonomous", action="store_true", help="Run autonomous risk adjustment")
    parser.add_argument("--status", action="store_true", help="Show risk status report")
    parser.add_argument("--strategy", help="Adjust risk for specific strategy")
    
    args = parser.parse_args()
    
    risk_manager = RiskManager()
    
    if args.conservative:
        result = await risk_manager.adjust_risk_conservative()
        if result['success']:
            print(f"✅ Conservative risk adjustment completed:")
            print(f"  Strategies adjusted: {result['strategies_adjusted']}")
            for adj in result['adjustments']:
                print(f"  - {adj['strategy']}: {adj['new_risk_level']} (position size reduced by {adj['position_size_reduction']:.1%})")
            return True
        else:
            print("❌ Conservative risk adjustment failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
            return False
    
    elif args.autonomous:
        result = await risk_manager.autonomous_risk_adjustment()
        if result['success']:
            print(f"✅ Autonomous risk adjustment completed:")
            print(f"  Strategies analyzed: {result['strategies_analyzed']}")
            print(f"  Strategies adjusted: {result['strategies_adjusted']}")
            print(f"  Market regime: {result['market_conditions']['regime']}")
            print(f"  Volatility: {result['market_conditions']['volatility']:.3f}")
            
            for adj in result['adjustments']:
                print(f"  - {adj['strategy']}: {adj['risk_level']} (magnitude: {adj['adjustment_magnitude']:.3f})")
            
            return True
        else:
            print("❌ Autonomous risk adjustment failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
            return False
    
    elif args.status:
        report = risk_manager.get_risk_status_report()
        if 'error' not in report:
            print("Risk Status Report:")
            print(f"  Overall risk level: {report['overall_risk_level']}")
            print(f"  Market regime: {report['market_conditions']['regime']}")
            print(f"  Volatility: {report['market_conditions']['volatility']:.3f}")
            print(f"  Risk sentiment: {report['market_conditions']['risk_on_off']:.3f}")
            
            print("\nStrategy Risk Parameters:")
            for name, params in report['strategies'].items():
                print(f"  {name}:")
                print(f"    Max position: {params['max_position_size']:.1%}")
                print(f"    Stop loss: {params['stop_loss_percent']:.1%}")
                print(f"    Max daily risk: {params['max_daily_risk']:.1%}")
                print(f"    Risk score: {params['risk_score']:.3f}")
            
            if report['recent_adjustments']:
                print(f"\nRecent Adjustments ({len(report['recent_adjustments'])}):")
                for adj in report['recent_adjustments'][:5]:
                    print(f"  {adj['timestamp'][:19]}: {adj['strategy']} - {adj['reason']}")
            
            return True
        else:
            print(f"❌ Error getting risk status: {report['error']}")
            return False
    
    else:
        # Default: run autonomous adjustment
        result = await risk_manager.autonomous_risk_adjustment()
        return result['success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)