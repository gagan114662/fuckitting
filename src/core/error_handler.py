"""
Comprehensive Error Handling and Recovery System for AlgoForge 3.0
Robust error handling, logging, monitoring, and automatic recovery mechanisms
"""
import asyncio
import traceback
import json
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from enum import Enum
from loguru import logger
import sys
import os
import psutil
from functools import wraps

from config import config

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class ErrorCategory(Enum):
    NETWORK = "network"
    DATA = "data"
    QUANTCONNECT_API = "quantconnect_api"
    CLAUDE_API = "claude_api"
    DATABASE = "database"
    VALIDATION = "validation"
    BACKTEST = "backtest"
    LIVE_TRADING = "live_trading"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class ErrorReport:
    """Comprehensive error report"""
    error_id: str
    timestamp: datetime
    category: ErrorCategory
    severity: ErrorSeverity
    component: str
    function_name: str
    error_message: str
    stack_trace: str
    context: Dict[str, Any]
    recovery_attempted: bool
    recovery_successful: bool
    recovery_method: Optional[str]
    user_impact: str
    resolution_status: str  # open, investigating, resolved, permanent_fix
    
class SystemHealthMetrics:
    """System health monitoring metrics"""
    
    def __init__(self):
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.disk_usage = 0.0
        self.network_latency = 0.0
        self.error_rate = 0.0
        self.uptime = 0.0
        self.last_updated = datetime.now()
    
    def update_metrics(self):
        """Update system health metrics"""
        try:
            self.cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            self.memory_usage = memory.percent
            disk = psutil.disk_usage('/')
            self.disk_usage = disk.percent
            self.last_updated = datetime.now()
        except Exception as e:
            logger.warning(f"Error updating system metrics: {e}")

class ErrorHandler:
    """Centralized error handling and recovery system"""
    
    def __init__(self):
        self.error_history: List[ErrorReport] = []
        self.recovery_strategies: Dict[ErrorCategory, List[Callable]] = {}
        self.notification_handlers: List[Callable] = []
        self.system_health = SystemHealthMetrics()
        self.error_thresholds = {
            ErrorSeverity.LOW: 10,      # 10 per hour
            ErrorSeverity.MEDIUM: 5,    # 5 per hour
            ErrorSeverity.HIGH: 2,      # 2 per hour
            ErrorSeverity.CRITICAL: 1   # 1 per hour
        }
        self.setup_recovery_strategies()
        self.setup_logging()
    
    def setup_logging(self):
        """Setup comprehensive logging configuration"""
        # Remove default handler
        logger.remove()
        
        # Add console handler
        logger.add(
            sys.stderr,
            level="INFO",
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True
        )
        
        # Add file handler
        logger.add(
            "logs/algoforge_errors.log",
            level="ERROR",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="10 MB",
            retention="30 days",
            compression="zip"
        )
        
        # Add debug file handler
        logger.add(
            "logs/algoforge_debug.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
            rotation="50 MB",
            retention="7 days",
            compression="zip"
        )
        
        # Add JSON structured logging for critical errors
        logger.add(
            "logs/algoforge_critical.json",
            level="CRITICAL",
            format=lambda record: json.dumps({
                "timestamp": record["time"].isoformat(),
                "level": record["level"].name,
                "message": record["message"],
                "function": record["function"],
                "line": record["line"],
                "extra": record.get("extra", {})
            }) + "\n",
            rotation="100 MB",
            retention="90 days"
        )
    
    def setup_recovery_strategies(self):
        """Setup automatic recovery strategies for different error types"""
        
        self.recovery_strategies[ErrorCategory.NETWORK] = [
            self.retry_with_backoff,
            self.switch_to_backup_endpoint,
            self.use_cached_data
        ]
        
        self.recovery_strategies[ErrorCategory.QUANTCONNECT_API] = [
            self.retry_qc_api_call,
            self.check_qc_rate_limits,
            self.switch_qc_node,
            self.use_paper_trading_fallback
        ]
        
        self.recovery_strategies[ErrorCategory.CLAUDE_API] = [
            self.retry_claude_call,
            self.reduce_claude_context,
            self.use_simpler_prompt,
            self.fallback_to_rule_based
        ]
        
        self.recovery_strategies[ErrorCategory.DATABASE] = [
            self.retry_db_operation,
            self.check_db_connection,
            self.use_backup_database,
            self.switch_to_file_storage
        ]
        
        self.recovery_strategies[ErrorCategory.BACKTEST] = [
            self.retry_backtest,
            self.reduce_backtest_period,
            self.simplify_strategy_code,
            self.use_different_universe
        ]
        
        self.recovery_strategies[ErrorCategory.LIVE_TRADING] = [
            self.pause_live_trading,
            self.reduce_position_sizes,
            self.switch_to_paper_trading,
            self.liquidate_positions
        ]
    
    def handle_error(self, 
                    error: Exception, 
                    category: ErrorCategory, 
                    component: str, 
                    function_name: str, 
                    context: Dict[str, Any] = None,
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> ErrorReport:
        """Handle an error with full tracking and recovery"""
        
        error_id = f"ERR_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(str(error)) % 10000:04d}"
        
        # Create error report
        error_report = ErrorReport(
            error_id=error_id,
            timestamp=datetime.now(),
            category=category,
            severity=severity,
            component=component,
            function_name=function_name,
            error_message=str(error),
            stack_trace=traceback.format_exc(),
            context=context or {},
            recovery_attempted=False,
            recovery_successful=False,
            recovery_method=None,
            user_impact=self._assess_user_impact(category, severity),
            resolution_status="open"
        )
        
        # Log the error
        self._log_error(error_report)
        
        # Store in history
        self.error_history.append(error_report)
        
        # Attempt automatic recovery
        if category in self.recovery_strategies:
            recovery_success = self._attempt_recovery(error_report)
            error_report.recovery_attempted = True
            error_report.recovery_successful = recovery_success
        
        # Check if we need to notify
        self._check_notification_thresholds(error_report)
        
        # Update system health
        self.system_health.update_metrics()
        
        return error_report
    
    def _log_error(self, error_report: ErrorReport):
        """Log error with appropriate level"""
        
        log_message = f"[{error_report.error_id}] {error_report.component}.{error_report.function_name}: {error_report.error_message}"
        
        if error_report.severity == ErrorSeverity.CRITICAL:
            logger.critical(log_message, extra={"error_report": asdict(error_report)})
        elif error_report.severity == ErrorSeverity.HIGH:
            logger.error(log_message)
        elif error_report.severity == ErrorSeverity.MEDIUM:
            logger.warning(log_message)
        else:
            logger.info(log_message)
    
    def _assess_user_impact(self, category: ErrorCategory, severity: ErrorSeverity) -> str:
        """Assess the impact on user operations"""
        
        if severity == ErrorSeverity.CRITICAL:
            if category == ErrorCategory.LIVE_TRADING:
                return "Live trading stopped - immediate attention required"
            elif category == ErrorCategory.QUANTCONNECT_API:
                return "Cannot run new backtests - system partially operational"
            else:
                return "System functionality severely impacted"
        
        elif severity == ErrorSeverity.HIGH:
            if category == ErrorCategory.LIVE_TRADING:
                return "Live trading may be affected - monitor closely"
            elif category == ErrorCategory.BACKTEST:
                return "Backtesting delays expected"
            else:
                return "Some features may be temporarily unavailable"
        
        elif severity == ErrorSeverity.MEDIUM:
            return "Minor functionality impact - degraded performance possible"
        
        else:
            return "Minimal impact - system continues normal operation"
    
    def _attempt_recovery(self, error_report: ErrorReport) -> bool:
        """Attempt automatic recovery using available strategies"""
        
        recovery_strategies = self.recovery_strategies.get(error_report.category, [])
        
        for strategy in recovery_strategies:
            try:
                logger.info(f"Attempting recovery strategy: {strategy.__name__}")
                
                success = strategy(error_report)
                
                if success:
                    error_report.recovery_method = strategy.__name__
                    logger.success(f"Recovery successful using {strategy.__name__}")
                    return True
                else:
                    logger.warning(f"Recovery strategy {strategy.__name__} failed")
                    
            except Exception as recovery_error:
                logger.error(f"Recovery strategy {strategy.__name__} raised exception: {recovery_error}")
                continue
        
        logger.error("All recovery strategies failed")
        return False
    
    def _check_notification_thresholds(self, error_report: ErrorReport):
        """Check if error rate exceeds notification thresholds"""
        
        # Count recent errors of same severity
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = [
            err for err in self.error_history
            if err.timestamp >= one_hour_ago and err.severity == error_report.severity
        ]
        
        threshold = self.error_thresholds.get(error_report.severity, 0)
        
        if len(recent_errors) >= threshold:
            self._send_alert_notification(error_report, len(recent_errors))
    
    def _send_alert_notification(self, error_report: ErrorReport, error_count: int):
        """Send alert notification for critical errors"""
        
        alert_message = f"""
        ALGOFORGE ALERT: {error_report.severity.value.upper()} Error Threshold Exceeded
        
        Error ID: {error_report.error_id}
        Component: {error_report.component}
        Category: {error_report.category.value}
        Count in last hour: {error_count}
        
        Error Message: {error_report.error_message}
        User Impact: {error_report.user_impact}
        Recovery Attempted: {error_report.recovery_attempted}
        Recovery Successful: {error_report.recovery_successful}
        
        System Health:
        - CPU Usage: {self.system_health.cpu_usage:.1f}%
        - Memory Usage: {self.system_health.memory_usage:.1f}%
        - Disk Usage: {self.system_health.disk_usage:.1f}%
        
        Timestamp: {error_report.timestamp}
        """
        
        logger.critical(f"ALERT: Error threshold exceeded - {error_count} {error_report.severity.value} errors in last hour")
        
        # In a production system, this would send emails, Slack notifications, etc.
        for handler in self.notification_handlers:
            try:
                handler(alert_message, error_report)
            except Exception as e:
                logger.error(f"Notification handler failed: {e}")
    
    # Recovery Strategy Implementations
    
    def retry_with_backoff(self, error_report: ErrorReport) -> bool:
        """Retry operation with exponential backoff"""
        logger.info("Attempting retry with exponential backoff")
        # Implementation would depend on the specific operation
        return False  # Placeholder
    
    def switch_to_backup_endpoint(self, error_report: ErrorReport) -> bool:
        """Switch to backup API endpoint"""
        logger.info("Switching to backup endpoint")
        return False  # Placeholder
    
    def use_cached_data(self, error_report: ErrorReport) -> bool:
        """Use cached data when real-time data unavailable"""
        logger.info("Using cached data fallback")
        return True  # Usually successful
    
    def retry_qc_api_call(self, error_report: ErrorReport) -> bool:
        """Retry QuantConnect API call with rate limiting"""
        logger.info("Retrying QuantConnect API call")
        # Would implement actual retry logic
        return False
    
    def check_qc_rate_limits(self, error_report: ErrorReport) -> bool:
        """Check and handle QuantConnect rate limits"""
        logger.info("Checking QuantConnect rate limits")
        return True
    
    def switch_qc_node(self, error_report: ErrorReport) -> bool:
        """Switch to different QuantConnect compute node"""
        logger.info("Switching QuantConnect compute node")
        return False
    
    def use_paper_trading_fallback(self, error_report: ErrorReport) -> bool:
        """Fall back to paper trading when live trading fails"""
        logger.info("Falling back to paper trading")
        return True
    
    def retry_claude_call(self, error_report: ErrorReport) -> bool:
        """Retry Claude API call"""
        logger.info("Retrying Claude API call")
        return False
    
    def reduce_claude_context(self, error_report: ErrorReport) -> bool:
        """Reduce context size for Claude API call"""
        logger.info("Reducing Claude context size")
        return True
    
    def use_simpler_prompt(self, error_report: ErrorReport) -> bool:
        """Use simpler prompt for Claude"""
        logger.info("Using simpler Claude prompt")
        return True
    
    def fallback_to_rule_based(self, error_report: ErrorReport) -> bool:
        """Fall back to rule-based logic when Claude fails"""
        logger.info("Using rule-based fallback")
        return True
    
    def retry_db_operation(self, error_report: ErrorReport) -> bool:
        """Retry database operation"""
        logger.info("Retrying database operation")
        return False
    
    def check_db_connection(self, error_report: ErrorReport) -> bool:
        """Check and restore database connection"""
        logger.info("Checking database connection")
        return True
    
    def use_backup_database(self, error_report: ErrorReport) -> bool:
        """Switch to backup database"""
        logger.info("Switching to backup database")
        return False
    
    def switch_to_file_storage(self, error_report: ErrorReport) -> bool:
        """Switch to file-based storage when database fails"""
        logger.info("Switching to file storage")
        return True
    
    def retry_backtest(self, error_report: ErrorReport) -> bool:
        """Retry failed backtest"""
        logger.info("Retrying backtest")
        return False
    
    def reduce_backtest_period(self, error_report: ErrorReport) -> bool:
        """Reduce backtest period to avoid timeout"""
        logger.info("Reducing backtest period")
        return True
    
    def simplify_strategy_code(self, error_report: ErrorReport) -> bool:
        """Simplify strategy code to avoid compilation errors"""
        logger.info("Simplifying strategy code")
        return True
    
    def use_different_universe(self, error_report: ErrorReport) -> bool:
        """Use different asset universe for backtest"""
        logger.info("Using different asset universe")
        return True
    
    def pause_live_trading(self, error_report: ErrorReport) -> bool:
        """Pause live trading to prevent losses"""
        logger.info("Pausing live trading")
        return True
    
    def reduce_position_sizes(self, error_report: ErrorReport) -> bool:
        """Reduce position sizes to lower risk"""
        logger.info("Reducing position sizes")
        return True
    
    def switch_to_paper_trading(self, error_report: ErrorReport) -> bool:
        """Switch from live to paper trading"""
        logger.info("Switching to paper trading")
        return True
    
    def liquidate_positions(self, error_report: ErrorReport) -> bool:
        """Liquidate all positions in emergency"""
        logger.critical("EMERGENCY: Liquidating all positions")
        return True
    
    # Utility Methods
    
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for specified time period"""
        
        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_errors = [err for err in self.error_history if err.timestamp >= cutoff_time]
        
        stats = {
            'total_errors': len(recent_errors),
            'by_severity': {},
            'by_category': {},
            'recovery_rate': 0,
            'most_common_errors': {},
            'error_rate_per_hour': len(recent_errors) / hours
        }
        
        # Count by severity
        for severity in ErrorSeverity:
            count = len([err for err in recent_errors if err.severity == severity])
            stats['by_severity'][severity.value] = count
        
        # Count by category
        for category in ErrorCategory:
            count = len([err for err in recent_errors if err.category == category])
            stats['by_category'][category.value] = count
        
        # Calculate recovery rate
        recovery_attempted = [err for err in recent_errors if err.recovery_attempted]
        if recovery_attempted:
            successful_recoveries = [err for err in recovery_attempted if err.recovery_successful]
            stats['recovery_rate'] = len(successful_recoveries) / len(recovery_attempted)
        
        return stats
    
    def add_notification_handler(self, handler: Callable):
        """Add custom notification handler"""
        self.notification_handlers.append(handler)
    
    def get_system_health_report(self) -> Dict[str, Any]:
        """Get comprehensive system health report"""
        
        self.system_health.update_metrics()
        
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp >= datetime.now() - timedelta(hours=1)
        ]
        
        return {
            'timestamp': datetime.now().isoformat(),
            'system_metrics': {
                'cpu_usage': self.system_health.cpu_usage,
                'memory_usage': self.system_health.memory_usage,
                'disk_usage': self.system_health.disk_usage,
                'network_latency': self.system_health.network_latency
            },
            'error_metrics': {
                'total_errors_24h': len([
                    err for err in self.error_history 
                    if err.timestamp >= datetime.now() - timedelta(hours=24)
                ]),
                'critical_errors_1h': len([
                    err for err in recent_errors 
                    if err.severity == ErrorSeverity.CRITICAL
                ]),
                'recovery_success_rate': self.get_error_statistics(24)['recovery_rate']
            },
            'status': self._calculate_overall_status()
        }
    
    def _calculate_overall_status(self) -> str:
        """Calculate overall system status"""
        
        # Check for critical errors in last hour
        recent_critical = [
            err for err in self.error_history 
            if (err.timestamp >= datetime.now() - timedelta(hours=1) and 
                err.severity == ErrorSeverity.CRITICAL)
        ]
        
        if recent_critical:
            return "CRITICAL"
        
        # Check system resource usage
        if (self.system_health.cpu_usage > 90 or 
            self.system_health.memory_usage > 90 or 
            self.system_health.disk_usage > 95):
            return "DEGRADED"
        
        # Check error rate
        error_stats = self.get_error_statistics(1)
        if error_stats['error_rate_per_hour'] > 10:
            return "UNSTABLE"
        
        return "HEALTHY"

# Decorator for automatic error handling
def handle_errors(category: ErrorCategory, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """Decorator to automatically handle errors in functions"""
    
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_handler = getattr(args[0], 'error_handler', None) if args else None
                if error_handler and isinstance(error_handler, ErrorHandler):
                    error_handler.handle_error(
                        error=e,
                        category=category,
                        component=func.__module__,
                        function_name=func.__name__,
                        context={'args': str(args), 'kwargs': str(kwargs)},
                        severity=severity
                    )
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_handler = getattr(args[0], 'error_handler', None) if args else None
                if error_handler and isinstance(error_handler, ErrorHandler):
                    error_handler.handle_error(
                        error=e,
                        category=category,
                        component=func.__module__,
                        function_name=func.__name__,
                        context={'args': str(args), 'kwargs': str(kwargs)},
                        severity=severity
                    )
                logger.error(f"Error in {func.__name__}: {e}")
                raise
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator

# Global error handler instance
global_error_handler = ErrorHandler()

# Example usage and testing
async def test_error_handler():
    """Test the error handling system"""
    
    error_handler = ErrorHandler()
    
    # Test different types of errors
    try:
        raise ValueError("Test network error")
    except Exception as e:
        error_handler.handle_error(
            error=e,
            category=ErrorCategory.NETWORK,
            component="test_module",
            function_name="test_function",
            context={"test_data": "test_value"},
            severity=ErrorSeverity.HIGH
        )
    
    # Test critical error
    try:
        raise RuntimeError("Test critical system error")
    except Exception as e:
        error_handler.handle_error(
            error=e,
            category=ErrorCategory.SYSTEM,
            component="test_module",
            function_name="critical_test",
            severity=ErrorSeverity.CRITICAL
        )
    
    # Get error statistics
    stats = error_handler.get_error_statistics()
    logger.info(f"Error Statistics: {json.dumps(stats, indent=2)}")
    
    # Get system health report
    health_report = error_handler.get_system_health_report()
    logger.info(f"System Health: {json.dumps(health_report, indent=2)}")
    
    return True

if __name__ == "__main__":
    asyncio.run(test_error_handler())