"""
Real-time Market Regime Detection for AlgoForge 3.0
Detects and analyzes market regimes to optimize strategy selection and risk management
"""
import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import norm
from loguru import logger
import json

from config import config

class MarketRegime(Enum):
    BULL_LOW_VOL = "bull_low_vol"
    BULL_HIGH_VOL = "bull_high_vol"
    BEAR_LOW_VOL = "bear_low_vol"
    BEAR_HIGH_VOL = "bear_high_vol"
    SIDEWAYS_LOW_VOL = "sideways_low_vol"
    SIDEWAYS_HIGH_VOL = "sideways_high_vol"
    CRISIS = "crisis"
    RECOVERY = "recovery"

@dataclass
class RegimeSignals:
    """Market regime signals and indicators"""
    vix_level: float
    vix_percentile: float
    market_return_1m: float
    market_return_3m: float
    market_return_6m: float
    volatility_1m: float
    volatility_3m: float
    trend_strength: float
    momentum_score: float
    breadth_indicator: float
    term_structure: float
    credit_spreads: float
    fear_greed_index: float

@dataclass
class RegimeAnalysis:
    """Complete market regime analysis"""
    current_regime: MarketRegime
    regime_probability: float
    regime_duration: int  # days in current regime
    regime_stability: float  # 0-1, how stable the regime is
    signals: RegimeSignals
    
    # Regime transition probabilities
    transition_probabilities: Dict[MarketRegime, float]
    
    # Strategy recommendations
    favorable_strategies: List[str]
    unfavorable_strategies: List[str]
    
    # Risk adjustments
    recommended_leverage: float
    recommended_position_sizing: float
    
    timestamp: datetime

class MarketDataFetcher:
    """Fetches market data for regime detection"""
    
    def __init__(self):
        self.tickers = {
            'market': '^GSPC',  # S&P 500
            'vix': '^VIX',      # VIX
            'bonds': '^TNX',    # 10-year Treasury
            'dollar': 'DX-Y.NYB', # Dollar Index
            'gold': 'GC=F',     # Gold
            'oil': 'CL=F'       # Oil
        }
    
    async def fetch_current_market_data(self, lookback_days: int = 252) -> Dict[str, pd.DataFrame]:
        """Fetch current market data for regime analysis"""
        logger.info("📊 Fetching market data for regime detection...")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        market_data = {}
        
        try:
            for name, ticker in self.tickers.items():
                try:
                    data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                    if not data.empty:
                        market_data[name] = data
                        logger.debug(f"Fetched {len(data)} days of data for {name}")
                    else:
                        logger.warning(f"No data available for {name} ({ticker})")
                except Exception as e:
                    logger.warning(f"Error fetching data for {name}: {e}")
                    continue
            
            # Fetch additional market breadth data
            breadth_data = await self._fetch_market_breadth_data(start_date, end_date)
            if breadth_data:
                market_data['breadth'] = breadth_data
            
            logger.success(f"✅ Fetched data for {len(market_data)} market indicators")
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return {}
    
    async def _fetch_market_breadth_data(self, start_date: datetime, end_date: datetime) -> Optional[pd.DataFrame]:
        """Fetch market breadth indicators"""
        try:
            # Fetch data for major ETFs to calculate breadth
            breadth_tickers = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
            breadth_data = {}
            
            for ticker in breadth_tickers:
                data = yf.download(ticker, start=start_date, end=end_date, progress=False)
                if not data.empty:
                    breadth_data[ticker] = data['Close']
            
            if breadth_data:
                breadth_df = pd.DataFrame(breadth_data)
                return breadth_df
            
        except Exception as e:
            logger.warning(f"Error fetching breadth data: {e}")
        
        return None

class RegimeDetector:
    """Advanced market regime detection using multiple methodologies"""
    
    def __init__(self):
        self.data_fetcher = MarketDataFetcher()
        self.lookback_window = 252  # 1 year of data
        self.regime_history: List[RegimeAnalysis] = []
    
    async def detect_current_regime(self) -> RegimeAnalysis:
        """Detect current market regime"""
        logger.info("🔍 Detecting current market regime...")
        
        try:
            # Fetch market data
            market_data = await self.data_fetcher.fetch_current_market_data(self.lookback_window)
            
            if not market_data:
                logger.error("No market data available for regime detection")
                return self._create_default_regime_analysis()
            
            # Calculate regime signals
            signals = self._calculate_regime_signals(market_data)
            
            # Detect regime using multiple methods
            regime_scores = self._calculate_regime_scores(signals, market_data)
            
            # Determine most likely regime
            current_regime = max(regime_scores.items(), key=lambda x: x[1])[0]
            regime_probability = regime_scores[current_regime]
            
            # Calculate regime characteristics
            regime_duration = self._calculate_regime_duration(current_regime)
            regime_stability = self._calculate_regime_stability(regime_scores)
            
            # Calculate transition probabilities
            transition_probs = self._calculate_transition_probabilities(current_regime, signals)
            
            # Generate strategy recommendations
            favorable_strategies, unfavorable_strategies = self._get_strategy_recommendations(current_regime, signals)
            
            # Calculate risk adjustments
            recommended_leverage, recommended_position_sizing = self._calculate_risk_adjustments(current_regime, signals)
            
            # Create regime analysis
            analysis = RegimeAnalysis(
                current_regime=current_regime,
                regime_probability=regime_probability,
                regime_duration=regime_duration,
                regime_stability=regime_stability,
                signals=signals,
                transition_probabilities=transition_probs,
                favorable_strategies=favorable_strategies,
                unfavorable_strategies=unfavorable_strategies,
                recommended_leverage=recommended_leverage,
                recommended_position_sizing=recommended_position_sizing,
                timestamp=datetime.now()
            )
            
            # Store in history
            self.regime_history.append(analysis)
            if len(self.regime_history) > 100:  # Keep last 100 analyses
                self.regime_history.pop(0)
            
            logger.success(f"✅ Detected regime: {current_regime.value} (confidence: {regime_probability:.2%})")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in regime detection: {e}")
            return self._create_default_regime_analysis()
    
    def _calculate_regime_signals(self, market_data: Dict[str, pd.DataFrame]) -> RegimeSignals:
        """Calculate all regime detection signals"""
        
        # Get latest market data
        market_prices = market_data.get('market', pd.DataFrame())
        vix_data = market_data.get('vix', pd.DataFrame())
        
        if market_prices.empty or vix_data.empty:
            return self._create_default_signals()
        
        # Calculate returns
        returns = market_prices['Close'].pct_change().dropna()
        
        # VIX metrics
        vix_current = vix_data['Close'].iloc[-1] if not vix_data.empty else 20.0
        vix_historical = vix_data['Close'].dropna()
        vix_percentile = (vix_historical <= vix_current).mean() if len(vix_historical) > 0 else 0.5
        
        # Return metrics
        return_1m = returns.tail(21).mean() * 21 if len(returns) >= 21 else 0
        return_3m = returns.tail(63).mean() * 252 if len(returns) >= 63 else 0
        return_6m = returns.tail(126).mean() * 252 if len(returns) >= 126 else 0
        
        # Volatility metrics
        vol_1m = returns.tail(21).std() * np.sqrt(252) if len(returns) >= 21 else 0.15
        vol_3m = returns.tail(63).std() * np.sqrt(252) if len(returns) >= 63 else 0.15
        
        # Trend strength (using moving averages)
        prices = market_prices['Close']
        if len(prices) >= 50:
            ma_20 = prices.rolling(20).mean().iloc[-1]
            ma_50 = prices.rolling(50).mean().iloc[-1]
            current_price = prices.iloc[-1]
            
            trend_strength = (current_price - ma_50) / ma_50 if ma_50 > 0 else 0
        else:
            trend_strength = 0
        
        # Momentum score
        if len(prices) >= 20:
            momentum_score = (prices.iloc[-1] / prices.iloc[-20] - 1) if prices.iloc[-20] > 0 else 0
        else:
            momentum_score = 0
        
        # Market breadth
        breadth_indicator = self._calculate_breadth_indicator(market_data.get('breadth'))
        
        # Term structure (simplified)
        bonds_data = market_data.get('bonds', pd.DataFrame())
        if not bonds_data.empty and len(bonds_data) >= 20:
            term_structure = bonds_data['Close'].iloc[-1] - bonds_data['Close'].rolling(20).mean().iloc[-1]
        else:
            term_structure = 0
        
        # Credit spreads (approximated using VIX as proxy)
        credit_spreads = (vix_current - 15) / 15  # Normalized credit risk proxy
        
        # Fear & Greed index (approximated)
        fear_greed_index = self._calculate_fear_greed_proxy(vix_current, return_1m, vol_1m)
        
        return RegimeSignals(
            vix_level=vix_current,
            vix_percentile=vix_percentile,
            market_return_1m=return_1m,
            market_return_3m=return_3m,
            market_return_6m=return_6m,
            volatility_1m=vol_1m,
            volatility_3m=vol_3m,
            trend_strength=trend_strength,
            momentum_score=momentum_score,
            breadth_indicator=breadth_indicator,
            term_structure=term_structure,
            credit_spreads=credit_spreads,
            fear_greed_index=fear_greed_index
        )
    
    def _calculate_breadth_indicator(self, breadth_data: Optional[pd.DataFrame]) -> float:
        """Calculate market breadth indicator"""
        if breadth_data is None or breadth_data.empty:
            return 0.5  # Neutral
        
        try:
            # Calculate percentage of securities above their 20-day moving average
            above_ma = 0
            total = 0
            
            for column in breadth_data.columns:
                prices = breadth_data[column].dropna()
                if len(prices) >= 20:
                    ma_20 = prices.rolling(20).mean().iloc[-1]
                    current = prices.iloc[-1]
                    if current > ma_20:
                        above_ma += 1
                    total += 1
            
            return above_ma / total if total > 0 else 0.5
            
        except Exception as e:
            logger.warning(f"Error calculating breadth indicator: {e}")
            return 0.5
    
    def _calculate_fear_greed_proxy(self, vix: float, returns: float, volatility: float) -> float:
        """Calculate fear & greed index proxy"""
        # Normalize components (0 = extreme fear, 1 = extreme greed)
        vix_component = max(0, min(1, (50 - vix) / 40))  # Inverted VIX
        return_component = max(0, min(1, (returns + 0.2) / 0.4))  # Returns component
        vol_component = max(0, min(1, (0.4 - volatility) / 0.3))  # Volatility component
        
        # Weighted average
        fear_greed = (vix_component * 0.4 + return_component * 0.3 + vol_component * 0.3)
        return fear_greed
    
    def _calculate_regime_scores(self, signals: RegimeSignals, market_data: Dict) -> Dict[MarketRegime, float]:
        """Calculate probability scores for each regime"""
        scores = {}
        
        # Bull markets (positive returns, various volatility)
        bull_score = max(0, signals.market_return_3m) * (1 + signals.momentum_score) * signals.breadth_indicator
        
        scores[MarketRegime.BULL_LOW_VOL] = bull_score * max(0, 1 - signals.vix_level / 20)
        scores[MarketRegime.BULL_HIGH_VOL] = bull_score * min(1, signals.vix_level / 25)
        
        # Bear markets (negative returns)
        bear_score = max(0, -signals.market_return_3m) * max(0, -signals.momentum_score) * (1 - signals.breadth_indicator)
        
        scores[MarketRegime.BEAR_LOW_VOL] = bear_score * max(0, 1 - signals.vix_level / 25)
        scores[MarketRegime.BEAR_HIGH_VOL] = bear_score * min(1, signals.vix_level / 20)
        
        # Sideways markets (low trend strength)
        sideways_score = max(0, 1 - abs(signals.trend_strength) * 2) * max(0, 1 - abs(signals.momentum_score) * 3)
        
        scores[MarketRegime.SIDEWAYS_LOW_VOL] = sideways_score * max(0, 1 - signals.vix_level / 20)
        scores[MarketRegime.SIDEWAYS_HIGH_VOL] = sideways_score * min(1, signals.vix_level / 25)
        
        # Crisis regime (extreme fear conditions)
        crisis_score = (
            min(1, signals.vix_level / 40) * 0.4 +  # High VIX
            max(0, -signals.market_return_1m * 5) * 0.3 +  # Sharp decline
            max(0, signals.volatility_1m / 0.5) * 0.3  # High volatility
        )
        scores[MarketRegime.CRISIS] = crisis_score
        
        # Recovery regime (improving from crisis)
        if len(self.regime_history) > 0 and self.regime_history[-1].current_regime == MarketRegime.CRISIS:
            recovery_score = max(0, signals.market_return_1m) * (1 - signals.vix_level / 30)
            scores[MarketRegime.RECOVERY] = recovery_score
        else:
            scores[MarketRegime.RECOVERY] = 0
        
        # Normalize scores
        total_score = sum(scores.values())
        if total_score > 0:
            scores = {regime: score / total_score for regime, score in scores.items()}
        else:
            # Default to sideways low vol if no clear signals
            scores = {regime: 0.0 for regime in MarketRegime}
            scores[MarketRegime.SIDEWAYS_LOW_VOL] = 1.0
        
        return scores
    
    def _calculate_regime_duration(self, current_regime: MarketRegime) -> int:
        """Calculate how long we've been in the current regime"""
        if not self.regime_history:
            return 1
        
        duration = 1
        for analysis in reversed(self.regime_history):
            if analysis.current_regime == current_regime:
                duration += 1
            else:
                break
        
        return duration
    
    def _calculate_regime_stability(self, regime_scores: Dict[MarketRegime, float]) -> float:
        """Calculate how stable the current regime is"""
        # Higher stability when one regime has much higher probability than others
        sorted_scores = sorted(regime_scores.values(), reverse=True)
        if len(sorted_scores) >= 2:
            stability = (sorted_scores[0] - sorted_scores[1]) / max(sorted_scores[0], 0.01)
        else:
            stability = sorted_scores[0] if sorted_scores else 0
        
        return min(1.0, stability)
    
    def _calculate_transition_probabilities(self, current_regime: MarketRegime, signals: RegimeSignals) -> Dict[MarketRegime, float]:
        """Calculate probabilities of transitioning to other regimes"""
        transitions = {}
        
        # Simplified transition model based on current signals
        for regime in MarketRegime:
            if regime == current_regime:
                transitions[regime] = 0.0  # Can't transition to same regime
            else:
                # Calculate transition probability based on signal distances
                transition_prob = self._calculate_regime_distance(regime, signals)
                transitions[regime] = max(0, min(0.5, transition_prob))  # Cap at 50%
        
        return transitions
    
    def _calculate_regime_distance(self, target_regime: MarketRegime, signals: RegimeSignals) -> float:
        """Calculate how close current signals are to target regime"""
        # Simplified distance calculation
        if target_regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]:
            return max(0, signals.market_return_3m) * signals.breadth_indicator
        elif target_regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
            return max(0, -signals.market_return_3m) * (1 - signals.breadth_indicator)
        elif target_regime == MarketRegime.CRISIS:
            return min(1, signals.vix_level / 40)
        else:
            return 0.1  # Default low probability
    
    def _get_strategy_recommendations(self, regime: MarketRegime, signals: RegimeSignals) -> Tuple[List[str], List[str]]:
        """Get strategy recommendations based on regime"""
        
        favorable = []
        unfavorable = []
        
        if regime in [MarketRegime.BULL_LOW_VOL, MarketRegime.BULL_HIGH_VOL]:
            favorable = ["momentum", "trend_following", "growth", "risk_parity"]
            unfavorable = ["mean_reversion", "volatility_targeting", "defensive"]
            
        elif regime in [MarketRegime.BEAR_LOW_VOL, MarketRegime.BEAR_HIGH_VOL]:
            favorable = ["mean_reversion", "defensive", "volatility_targeting", "short_strategies"]
            unfavorable = ["momentum", "trend_following", "growth"]
            
        elif regime in [MarketRegime.SIDEWAYS_LOW_VOL, MarketRegime.SIDEWAYS_HIGH_VOL]:
            favorable = ["mean_reversion", "pairs_trading", "market_neutral", "range_bound"]
            unfavorable = ["momentum", "trend_following"]
            
        elif regime == MarketRegime.CRISIS:
            favorable = ["defensive", "volatility_trading", "crisis_alpha", "safe_haven"]
            unfavorable = ["momentum", "trend_following", "growth", "leverage"]
            
        elif regime == MarketRegime.RECOVERY:
            favorable = ["momentum", "value", "small_cap", "recovery_plays"]
            unfavorable = ["defensive", "safe_haven"]
        
        return favorable, unfavorable
    
    def _calculate_risk_adjustments(self, regime: MarketRegime, signals: RegimeSignals) -> Tuple[float, float]:
        """Calculate recommended leverage and position sizing adjustments"""
        
        base_leverage = 1.0
        base_position_size = 0.2
        
        # Adjust based on regime
        if regime in [MarketRegime.BULL_LOW_VOL]:
            leverage_multiplier = 1.2
            position_multiplier = 1.1
            
        elif regime in [MarketRegime.BULL_HIGH_VOL, MarketRegime.BEAR_LOW_VOL]:
            leverage_multiplier = 0.8
            position_multiplier = 0.9
            
        elif regime in [MarketRegime.BEAR_HIGH_VOL, MarketRegime.SIDEWAYS_HIGH_VOL]:
            leverage_multiplier = 0.6
            position_multiplier = 0.7
            
        elif regime == MarketRegime.CRISIS:
            leverage_multiplier = 0.3
            position_multiplier = 0.5
            
        elif regime == MarketRegime.RECOVERY:
            leverage_multiplier = 1.1
            position_multiplier = 1.0
            
        else:  # Sideways low vol
            leverage_multiplier = 0.9
            position_multiplier = 0.8
        
        # Further adjust based on VIX level
        vix_adjustment = max(0.5, min(1.5, 20 / max(signals.vix_level, 10)))
        
        recommended_leverage = base_leverage * leverage_multiplier * vix_adjustment
        recommended_position_sizing = base_position_size * position_multiplier * vix_adjustment
        
        return recommended_leverage, recommended_position_sizing
    
    def _create_default_regime_analysis(self) -> RegimeAnalysis:
        """Create default regime analysis when detection fails"""
        return RegimeAnalysis(
            current_regime=MarketRegime.SIDEWAYS_LOW_VOL,
            regime_probability=0.5,
            regime_duration=1,
            regime_stability=0.5,
            signals=self._create_default_signals(),
            transition_probabilities={regime: 0.1 for regime in MarketRegime},
            favorable_strategies=["market_neutral", "diversified"],
            unfavorable_strategies=["high_risk"],
            recommended_leverage=0.8,
            recommended_position_sizing=0.15,
            timestamp=datetime.now()
        )
    
    def _create_default_signals(self) -> RegimeSignals:
        """Create default signals when data is unavailable"""
        return RegimeSignals(
            vix_level=20.0,
            vix_percentile=0.5,
            market_return_1m=0.0,
            market_return_3m=0.0,
            market_return_6m=0.0,
            volatility_1m=0.15,
            volatility_3m=0.15,
            trend_strength=0.0,
            momentum_score=0.0,
            breadth_indicator=0.5,
            term_structure=0.0,
            credit_spreads=0.0,
            fear_greed_index=0.5
        )

class RegimeBasedRiskManager:
    """Risk management that adapts to market regimes"""
    
    def __init__(self, regime_detector: RegimeDetector):
        self.regime_detector = regime_detector
    
    async def get_regime_adjusted_parameters(self, strategy_name: str) -> Dict[str, float]:
        """Get regime-adjusted parameters for a strategy"""
        
        # Get current regime
        regime_analysis = await self.regime_detector.detect_current_regime()
        
        # Base parameters
        base_params = {
            'position_size': 0.2,
            'stop_loss': 0.05,
            'take_profit': 0.1,
            'max_leverage': 2.0
        }
        
        # Adjust based on regime
        if regime_analysis.current_regime == MarketRegime.CRISIS:
            adjustments = {
                'position_size': 0.5,  # Reduce position size
                'stop_loss': 0.7,     # Tighter stops
                'take_profit': 0.8,   # Take profits earlier
                'max_leverage': 0.3   # Reduce leverage significantly
            }
        elif regime_analysis.current_regime in [MarketRegime.BEAR_HIGH_VOL, MarketRegime.SIDEWAYS_HIGH_VOL]:
            adjustments = {
                'position_size': 0.7,
                'stop_loss': 0.8,
                'take_profit': 0.9,
                'max_leverage': 0.6
            }
        elif regime_analysis.current_regime == MarketRegime.BULL_LOW_VOL:
            adjustments = {
                'position_size': 1.2,
                'stop_loss': 1.1,
                'take_profit': 1.2,
                'max_leverage': 1.3
            }
        else:
            adjustments = {
                'position_size': 1.0,
                'stop_loss': 1.0,
                'take_profit': 1.0,
                'max_leverage': 1.0
            }
        
        # Apply adjustments
        adjusted_params = {}
        for param, base_value in base_params.items():
            adjustment = adjustments.get(param, 1.0)
            adjusted_params[param] = base_value * adjustment
        
        return adjusted_params

# Example usage and testing
async def test_regime_detector():
    """Test the market regime detector"""
    detector = RegimeDetector()
    
    # Detect current regime
    analysis = await detector.detect_current_regime()
    
    logger.info("Market Regime Analysis:")
    logger.info(f"Current Regime: {analysis.current_regime.value}")
    logger.info(f"Confidence: {analysis.regime_probability:.2%}")
    logger.info(f"Duration: {analysis.regime_duration} days")
    logger.info(f"Stability: {analysis.regime_stability:.2f}")
    
    logger.info("Key Signals:")
    logger.info(f"VIX Level: {analysis.signals.vix_level:.1f}")
    logger.info(f"3M Return: {analysis.signals.market_return_3m:.2%}")
    logger.info(f"Volatility: {analysis.signals.volatility_1m:.2%}")
    logger.info(f"Trend Strength: {analysis.signals.trend_strength:.2f}")
    
    logger.info("Strategy Recommendations:")
    logger.info(f"Favorable: {', '.join(analysis.favorable_strategies)}")
    logger.info(f"Unfavorable: {', '.join(analysis.unfavorable_strategies)}")
    
    logger.info("Risk Adjustments:")
    logger.info(f"Recommended Leverage: {analysis.recommended_leverage:.1f}x")
    logger.info(f"Position Sizing: {analysis.recommended_position_sizing:.1%}")
    
    # Test risk manager
    risk_manager = RegimeBasedRiskManager(detector)
    adjusted_params = await risk_manager.get_regime_adjusted_parameters("test_strategy")
    
    logger.info("Regime-Adjusted Parameters:")
    for param, value in adjusted_params.items():
        logger.info(f"{param}: {value:.3f}")
    
    return analysis

if __name__ == "__main__":
    asyncio.run(test_regime_detector())