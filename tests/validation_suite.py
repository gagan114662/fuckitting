"""
Advanced Validation Suite for AlgoForge 3.0
Implements comprehensive strategy validation including Monte Carlo, Walk-Forward, and Crisis testing
"""
import asyncio
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from loguru import logger
from scipy import stats
import json

from quantconnect_client import QuantConnectClient, BacktestResult
from claude_integration import GeneratedStrategy
from config import config

@dataclass
class ValidationResult:
    """Comprehensive validation result for a strategy"""
    strategy_name: str
    
    # Basic backtest results
    base_backtest: Dict[str, Any]
    
    # Out-of-sample results
    oos_performance: Dict[str, float]
    oos_passed: bool
    
    # Walk-forward results
    walk_forward_results: List[Dict[str, Any]]
    walk_forward_stability: float
    
    # Monte Carlo results
    monte_carlo_results: Dict[str, Any]
    monte_carlo_confidence: float
    
    # Parameter sensitivity
    parameter_sensitivity: Dict[str, Any]
    
    # Crisis testing
    crisis_test_results: Dict[str, Any]
    
    # Overall validation score
    overall_score: float
    validation_passed: bool
    
    # Recommendations
    recommendations: List[str]

class OutOfSampleValidator:
    """Out-of-sample validation using holdout data"""
    
    def __init__(self, qc_client: QuantConnectClient):
        self.qc_client = qc_client
    
    async def validate_strategy(self, strategy: GeneratedStrategy, oos_ratio: float = 0.2) -> Dict[str, Any]:
        """Run out-of-sample validation on a strategy"""
        logger.info(f"🔍 Running out-of-sample validation for {strategy.name}")
        
        try:
            # Create modified strategy code for OOS testing
            oos_strategy_code = self._modify_strategy_for_oos(strategy.code, oos_ratio)
            
            # Create temporary strategy for OOS testing
            oos_strategy = GeneratedStrategy(
                hypothesis_id=strategy.hypothesis_id,
                name=f"{strategy.name}_OOS",
                code=oos_strategy_code,
                description=f"OOS validation of {strategy.description}",
                parameters=strategy.parameters,
                expected_metrics=strategy.expected_metrics,
                risk_controls=strategy.risk_controls,
                created_at=datetime.now()
            )
            
            # Run OOS backtest
            async with self.qc_client as client:
                # Create project
                project_name = f"OOS_{strategy.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_id = await client.create_project(project_name)
                
                # Upload strategy
                await client.upload_file(project_id, "main.py", oos_strategy_code)
                
                # Compile and run
                if await client.compile_project(project_id):
                    backtest_id = await client.create_backtest(project_id, f"OOS_Test_{strategy.name}")
                    result = await client.wait_for_backtest_completion(project_id, backtest_id)
                    
                    if result and result.statistics:
                        oos_metrics = {
                            'cagr': result.cagr or 0,
                            'sharpe': result.sharpe or 0,
                            'max_drawdown': result.max_drawdown or 1,
                            'total_trades': result.total_trades or 0
                        }
                        
                        # Compare with expected metrics
                        performance_ratio = self._calculate_performance_ratio(
                            oos_metrics, strategy.expected_metrics
                        )
                        
                        return {
                            'oos_metrics': oos_metrics,
                            'performance_ratio': performance_ratio,
                            'passed': performance_ratio >= 0.7,  # OOS should retain 70% of IS performance
                            'degradation': 1 - performance_ratio
                        }
                    
                # Clean up
                await client.delete_project(project_id)
                
        except Exception as e:
            logger.error(f"Error in OOS validation: {e}")
        
        return {'passed': False, 'error': 'Validation failed'}
    
    def _modify_strategy_for_oos(self, strategy_code: str, oos_ratio: float) -> str:
        """Modify strategy code to use only out-of-sample data"""
        # Simple modification - in practice, would need more sophisticated date handling
        modified_code = strategy_code.replace(
            "self.SetStartDate(", 
            f"# OOS Testing - Using last {oos_ratio*100:.0f}% of data\n        self.SetStartDate("
        )
        
        # Add OOS date calculation
        oos_code_addition = f"""
        # Out-of-sample validation setup
        total_years = {config.validation_periods['backtest_years']}
        oos_years = int(total_years * {oos_ratio})
        start_year = {datetime.now().year} - oos_years
        self.SetStartDate(start_year, 1, 1)
        """
        
        return oos_code_addition + "\n" + modified_code
    
    def _calculate_performance_ratio(self, oos_metrics: Dict, expected_metrics: Dict) -> float:
        """Calculate performance ratio between OOS and expected metrics"""
        if not expected_metrics:
            return 0.0
        
        # Compare key metrics
        cagr_ratio = (oos_metrics.get('cagr', 0) / max(expected_metrics.get('cagr', 0.01), 0.01))
        sharpe_ratio = (oos_metrics.get('sharpe', 0) / max(expected_metrics.get('sharpe', 0.01), 0.01))
        
        # Weight the ratios
        performance_ratio = (cagr_ratio * 0.6 + sharpe_ratio * 0.4)
        return min(performance_ratio, 2.0)  # Cap at 200%

class WalkForwardValidator:
    """Walk-forward validation for strategy robustness"""
    
    def __init__(self, qc_client: QuantConnectClient):
        self.qc_client = qc_client
    
    async def validate_strategy(self, strategy: GeneratedStrategy, window_months: int = 12) -> Dict[str, Any]:
        """Run walk-forward validation"""
        logger.info(f"📊 Running walk-forward validation for {strategy.name}")
        
        results = []
        total_years = config.validation_periods['backtest_years']
        num_windows = max(1, (total_years * 12) // window_months - 1)
        
        try:
            for i in range(num_windows):
                # Calculate window dates
                end_year = datetime.now().year - (i * window_months // 12)
                start_year = end_year - (window_months // 12) - 1
                
                # Create window-specific strategy
                window_code = self._create_window_strategy(strategy.code, start_year, end_year)
                
                # Run backtest for this window
                window_result = await self._run_window_backtest(
                    strategy, window_code, f"WF_{i+1}"
                )
                
                if window_result:
                    results.append({
                        'window': i + 1,
                        'start_year': start_year,
                        'end_year': end_year,
                        'cagr': window_result.get('cagr', 0),
                        'sharpe': window_result.get('sharpe', 0),
                        'max_drawdown': window_result.get('max_drawdown', 1),
                        'total_trades': window_result.get('total_trades', 0)
                    })
                
                # Limit concurrent backtests
                if i > 0 and i % 3 == 0:
                    await asyncio.sleep(10)  # Rate limiting
            
            # Analyze walk-forward stability
            stability_metrics = self._analyze_walk_forward_stability(results)
            
            return {
                'results': results,
                'stability_score': stability_metrics['stability_score'],
                'cagr_consistency': stability_metrics['cagr_consistency'],
                'sharpe_consistency': stability_metrics['sharpe_consistency'],
                'passed': stability_metrics['stability_score'] >= 0.6
            }
            
        except Exception as e:
            logger.error(f"Error in walk-forward validation: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _create_window_strategy(self, strategy_code: str, start_year: int, end_year: int) -> str:
        """Create strategy code for specific time window"""
        window_setup = f"""
        # Walk-forward window: {start_year} to {end_year}
        self.SetStartDate({start_year}, 1, 1)
        self.SetEndDate({end_year}, 12, 31)
        """
        
        return window_setup + "\n" + strategy_code
    
    async def _run_window_backtest(self, strategy: GeneratedStrategy, window_code: str, window_name: str) -> Optional[Dict]:
        """Run backtest for a specific window"""
        try:
            async with self.qc_client as client:
                project_name = f"WF_{strategy.name}_{window_name}_{datetime.now().strftime('%H%M%S')}"
                project_id = await client.create_project(project_name)
                
                await client.upload_file(project_id, "main.py", window_code)
                
                if await client.compile_project(project_id):
                    backtest_id = await client.create_backtest(project_id, f"WF_{window_name}")
                    result = await client.wait_for_backtest_completion(project_id, backtest_id, timeout_minutes=15)
                    
                    if result and result.statistics:
                        return {
                            'cagr': result.cagr,
                            'sharpe': result.sharpe,
                            'max_drawdown': result.max_drawdown,
                            'total_trades': result.total_trades
                        }
                
                await client.delete_project(project_id)
                
        except Exception as e:
            logger.error(f"Error in window backtest {window_name}: {e}")
        
        return None
    
    def _analyze_walk_forward_stability(self, results: List[Dict]) -> Dict[str, float]:
        """Analyze stability across walk-forward windows"""
        if not results:
            return {'stability_score': 0.0, 'cagr_consistency': 0.0, 'sharpe_consistency': 0.0}
        
        # Extract metrics
        cagrs = [r['cagr'] for r in results if r['cagr'] is not None]
        sharpes = [r['sharpe'] for r in results if r['sharpe'] is not None]
        
        if not cagrs or not sharpes:
            return {'stability_score': 0.0, 'cagr_consistency': 0.0, 'sharpe_consistency': 0.0}
        
        # Calculate consistency (inverse of coefficient of variation)
        cagr_consistency = 1 / (1 + (np.std(cagrs) / max(abs(np.mean(cagrs)), 0.01)))
        sharpe_consistency = 1 / (1 + (np.std(sharpes) / max(abs(np.mean(sharpes)), 0.01)))
        
        # Overall stability score
        stability_score = (cagr_consistency * 0.6 + sharpe_consistency * 0.4)
        
        return {
            'stability_score': stability_score,
            'cagr_consistency': cagr_consistency,
            'sharpe_consistency': sharpe_consistency
        }

class MonteCarloValidator:
    """Monte Carlo validation for strategy robustness"""
    
    def __init__(self):
        self.num_simulations = config.validation_periods['monte_carlo_runs']
    
    def validate_strategy_returns(self, backtest_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run Monte Carlo validation on strategy returns"""
        logger.info("🎲 Running Monte Carlo validation...")
        
        try:
            # Extract returns from backtest results
            returns = self._extract_returns_from_backtest(backtest_results)
            
            if not returns:
                return {'passed': False, 'error': 'No returns data available'}
            
            # Run Monte Carlo simulations
            simulations = []
            for i in range(self.num_simulations):
                # Bootstrap resample returns
                resampled_returns = np.random.choice(returns, size=len(returns), replace=True)
                
                # Calculate cumulative performance
                cumulative_return = np.prod(1 + np.array(resampled_returns)) - 1
                annual_return = (1 + cumulative_return) ** (252 / len(returns)) - 1
                
                # Calculate other metrics
                sharpe = np.mean(resampled_returns) / max(np.std(resampled_returns), 0.001) * np.sqrt(252)
                max_dd = self._calculate_max_drawdown(resampled_returns)
                
                simulations.append({
                    'annual_return': annual_return,
                    'sharpe': sharpe,
                    'max_drawdown': max_dd
                })
            
            # Analyze simulation results
            analysis = self._analyze_monte_carlo_results(simulations)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in Monte Carlo validation: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _extract_returns_from_backtest(self, backtest_results: Dict) -> List[float]:
        """Extract daily returns from backtest results"""
        # This would extract actual daily returns from QuantConnect results
        # For now, generate synthetic returns based on statistics
        if not backtest_results.get('Statistics'):
            return []
        
        stats = backtest_results['Statistics']
        annual_return = stats.get('Compounding Annual Return', 0)
        volatility = stats.get('Annual Standard Deviation', 0.15)
        
        # Generate synthetic daily returns (placeholder)
        num_days = 252 * config.validation_periods['backtest_years']
        daily_return = annual_return / 252
        daily_vol = volatility / np.sqrt(252)
        
        returns = np.random.normal(daily_return, daily_vol, num_days)
        return returns.tolist()
    
    def _calculate_max_drawdown(self, returns: List[float]) -> float:
        """Calculate maximum drawdown from returns"""
        cumulative = np.cumprod(1 + np.array(returns))
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        return abs(np.min(drawdown))
    
    def _analyze_monte_carlo_results(self, simulations: List[Dict]) -> Dict[str, Any]:
        """Analyze Monte Carlo simulation results"""
        # Extract metrics
        returns = [s['annual_return'] for s in simulations]
        sharpes = [s['sharpe'] for s in simulations]
        drawdowns = [s['max_drawdown'] for s in simulations]
        
        # Calculate confidence intervals
        return_ci = np.percentile(returns, [5, 25, 50, 75, 95])
        sharpe_ci = np.percentile(sharpes, [5, 25, 50, 75, 95])
        dd_ci = np.percentile(drawdowns, [5, 25, 50, 75, 95])
        
        # Probability of meeting targets
        prob_cagr = np.mean(np.array(returns) >= config.targets.min_cagr)
        prob_sharpe = np.mean(np.array(sharpes) >= config.targets.min_sharpe)
        prob_dd = np.mean(np.array(drawdowns) <= config.targets.max_drawdown)
        
        # Overall confidence
        confidence_score = (prob_cagr + prob_sharpe + prob_dd) / 3
        
        return {
            'return_confidence_intervals': return_ci.tolist(),
            'sharpe_confidence_intervals': sharpe_ci.tolist(),
            'drawdown_confidence_intervals': dd_ci.tolist(),
            'probability_meet_cagr_target': prob_cagr,
            'probability_meet_sharpe_target': prob_sharpe,
            'probability_meet_dd_target': prob_dd,
            'overall_confidence': confidence_score,
            'passed': confidence_score >= 0.6
        }

class CrisisTestValidator:
    """Crisis testing validator for extreme market conditions"""
    
    CRISIS_PERIODS = [
        {'name': '2008 Financial Crisis', 'start': '2008-01-01', 'end': '2009-03-31'},
        {'name': 'COVID-19 Crash', 'start': '2020-02-01', 'end': '2020-05-01'},
        {'name': 'Dot-com Crash', 'start': '2000-03-01', 'end': '2002-12-31'},
        {'name': 'Flash Crash 2010', 'start': '2010-05-01', 'end': '2010-06-30'},
    ]
    
    def __init__(self, qc_client: QuantConnectClient):
        self.qc_client = qc_client
    
    async def validate_strategy(self, strategy: GeneratedStrategy) -> Dict[str, Any]:
        """Test strategy performance during historical crisis periods"""
        logger.info(f"⚠️ Running crisis testing for {strategy.name}")
        
        crisis_results = []
        
        try:
            for crisis in self.CRISIS_PERIODS:
                # Create crisis-specific strategy
                crisis_code = self._create_crisis_strategy(
                    strategy.code, crisis['start'], crisis['end']
                )
                
                # Run crisis backtest
                result = await self._run_crisis_backtest(
                    strategy, crisis_code, crisis['name']
                )
                
                if result:
                    crisis_results.append({
                        'crisis_name': crisis['name'],
                        'period': f"{crisis['start']} to {crisis['end']}",
                        'cagr': result.get('cagr', 0),
                        'max_drawdown': result.get('max_drawdown', 1),
                        'survived': result.get('max_drawdown', 1) <= 0.5  # Survived if DD < 50%
                    })
                
                # Rate limiting
                await asyncio.sleep(5)
            
            # Analyze crisis performance
            analysis = self._analyze_crisis_performance(crisis_results)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in crisis testing: {e}")
            return {'passed': False, 'error': str(e)}
    
    def _create_crisis_strategy(self, strategy_code: str, start_date: str, end_date: str) -> str:
        """Create strategy code for crisis period"""
        start_parts = start_date.split('-')
        end_parts = end_date.split('-')
        
        crisis_setup = f"""
        # Crisis testing period: {start_date} to {end_date}
        self.SetStartDate({start_parts[0]}, {start_parts[1]}, {start_parts[2]})
        self.SetEndDate({end_parts[0]}, {end_parts[1]}, {end_parts[2]})
        """
        
        return crisis_setup + "\n" + strategy_code
    
    async def _run_crisis_backtest(self, strategy: GeneratedStrategy, crisis_code: str, crisis_name: str) -> Optional[Dict]:
        """Run backtest for crisis period"""
        try:
            async with self.qc_client as client:
                project_name = f"Crisis_{strategy.name}_{crisis_name.replace(' ', '_')}"
                project_id = await client.create_project(project_name)
                
                await client.upload_file(project_id, "main.py", crisis_code)
                
                if await client.compile_project(project_id):
                    backtest_id = await client.create_backtest(project_id, f"Crisis_{crisis_name}")
                    result = await client.wait_for_backtest_completion(project_id, backtest_id, timeout_minutes=10)
                    
                    if result and result.statistics:
                        return {
                            'cagr': result.cagr,
                            'max_drawdown': result.max_drawdown,
                            'total_trades': result.total_trades
                        }
                
                await client.delete_project(project_id)
                
        except Exception as e:
            logger.error(f"Error in crisis backtest {crisis_name}: {e}")
        
        return None
    
    def _analyze_crisis_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance across crisis periods"""
        if not results:
            return {'passed': False, 'survival_rate': 0.0}
        
        # Calculate survival rate
        survived_count = sum(1 for r in results if r['survived'])
        survival_rate = survived_count / len(results)
        
        # Average drawdown during crises
        avg_crisis_dd = np.mean([r['max_drawdown'] for r in results])
        
        # Strategy resilience score
        resilience_score = (survival_rate * 0.7 + (1 - min(avg_crisis_dd, 1.0)) * 0.3)
        
        return {
            'crisis_results': results,
            'survival_rate': survival_rate,
            'average_crisis_drawdown': avg_crisis_dd,
            'resilience_score': resilience_score,
            'passed': survival_rate >= 0.5 and avg_crisis_dd <= 0.4
        }

class ComprehensiveValidator:
    """Comprehensive validation orchestrator"""
    
    def __init__(self, qc_client: QuantConnectClient):
        self.qc_client = qc_client
        self.oos_validator = OutOfSampleValidator(qc_client)
        self.wf_validator = WalkForwardValidator(qc_client)
        self.mc_validator = MonteCarloValidator()
        self.crisis_validator = CrisisTestValidator(qc_client)
    
    async def comprehensive_validation(self, strategy: GeneratedStrategy, base_backtest: Dict[str, Any]) -> ValidationResult:
        """Run comprehensive validation suite"""
        logger.info(f"🔬 Running comprehensive validation for {strategy.name}")
        
        try:
            # Run all validation tests
            logger.info("Running out-of-sample validation...")
            oos_result = await self.oos_validator.validate_strategy(strategy)
            
            logger.info("Running walk-forward validation...")
            wf_result = await self.wf_validator.validate_strategy(strategy)
            
            logger.info("Running Monte Carlo validation...")
            mc_result = self.mc_validator.validate_strategy_returns(base_backtest)
            
            logger.info("Running crisis testing...")
            crisis_result = await self.crisis_validator.validate_strategy(strategy)
            
            # Calculate overall validation score
            scores = []
            if oos_result.get('passed'): scores.append(0.8)
            if wf_result.get('passed'): scores.append(0.9)
            if mc_result.get('passed'): scores.append(0.7)
            if crisis_result.get('passed'): scores.append(0.6)
            
            overall_score = np.mean(scores) if scores else 0.0
            validation_passed = overall_score >= 0.7
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                oos_result, wf_result, mc_result, crisis_result
            )
            
            # Create comprehensive result
            result = ValidationResult(
                strategy_name=strategy.name,
                base_backtest=base_backtest,
                oos_performance=oos_result,
                oos_passed=oos_result.get('passed', False),
                walk_forward_results=wf_result.get('results', []),
                walk_forward_stability=wf_result.get('stability_score', 0.0),
                monte_carlo_results=mc_result,
                monte_carlo_confidence=mc_result.get('overall_confidence', 0.0),
                parameter_sensitivity={},  # TODO: Implement parameter sensitivity
                crisis_test_results=crisis_result,
                overall_score=overall_score,
                validation_passed=validation_passed,
                recommendations=recommendations
            )
            
            logger.success(f"✅ Comprehensive validation complete for {strategy.name}")
            logger.info(f"Overall Score: {overall_score:.2f} | Passed: {validation_passed}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error in comprehensive validation: {e}")
            raise
    
    def _generate_recommendations(self, oos_result: Dict, wf_result: Dict, mc_result: Dict, crisis_result: Dict) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        if not oos_result.get('passed'):
            recommendations.append("Strategy shows overfitting - consider simplifying model or reducing parameters")
        
        if wf_result.get('stability_score', 0) < 0.6:
            recommendations.append("Strategy lacks robustness across time periods - review parameter stability")
        
        if mc_result.get('overall_confidence', 0) < 0.6:
            recommendations.append("Low statistical confidence - consider longer backtest period or different approach")
        
        if not crisis_result.get('passed'):
            recommendations.append("Poor crisis performance - implement stronger risk controls and position sizing")
        
        if not recommendations:
            recommendations.append("Strategy passes all validation tests - ready for paper trading")
        
        return recommendations

# Testing function
async def test_validation_suite():
    """Test the validation suite"""
    from claude_integration import GeneratedStrategy
    
    # Create sample strategy
    sample_strategy = GeneratedStrategy(
        hypothesis_id="test_validation",
        name="Test_Validation_Strategy",
        code="# Sample strategy code",
        description="Test strategy for validation suite",
        parameters={"param1": 20},
        expected_metrics={"cagr": 0.25, "sharpe": 1.2},
        risk_controls=["stop_loss"],
        created_at=datetime.now()
    )
    
    # Sample backtest results
    sample_backtest = {
        'Statistics': {
            'Compounding Annual Return': 0.28,
            'Sharpe Ratio': 1.15,
            'Drawdown': -0.18,
            'Annual Standard Deviation': 0.15
        }
    }
    
    # Run validation
    qc_client = QuantConnectClient()
    validator = ComprehensiveValidator(qc_client)
    
    result = await validator.comprehensive_validation(sample_strategy, sample_backtest)
    
    logger.info(f"Validation Results for {result.strategy_name}:")
    logger.info(f"Overall Score: {result.overall_score:.2f}")
    logger.info(f"Validation Passed: {result.validation_passed}")
    logger.info("Recommendations:")
    for rec in result.recommendations:
        logger.info(f"  - {rec}")
    
    return result

if __name__ == "__main__":
    asyncio.run(test_validation_suite())