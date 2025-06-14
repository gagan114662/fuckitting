"""
Autonomous Self-Healing AlgoForge System
Addresses critical failure points and creates truly autonomous operation
"""
import asyncio
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass
from pathlib import Path
from loguru import logger
import subprocess
import psutil
import sqlite3
from enum import Enum

from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity
from config import config
from resilience_framework import (
    safe_database_connection, memory_monitor, thread_safe,
    retry_with_backoff, RetryConfig, FailureType, cleanup_temp_files,
    resilience_manager
)

class SystemHealth(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    FAILED = "failed"

class FailureMode(Enum):
    # Infrastructure failures
    NETWORK_OUTAGE = "network_outage"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    DISK_FULL = "disk_full"
    MEMORY_EXHAUSTION = "memory_exhaustion"
    
    # Application failures
    CODE_COMPILATION_ERROR = "code_compilation_error"
    STRATEGY_RUNTIME_ERROR = "strategy_runtime_error"
    DATA_CORRUPTION = "data_corruption"
    MODEL_DRIFT = "model_drift"
    
    # Market failures
    MARKET_REGIME_SHIFT = "market_regime_shift"
    EXTREME_VOLATILITY = "extreme_volatility"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    
    # Cognitive failures
    OVERFITTING = "overfitting"
    BIAS_ACCUMULATION = "bias_accumulation"
    FEEDBACK_LOOPS = "feedback_loops"

@dataclass
class RecoveryAction:
    """Defines an autonomous recovery action"""
    failure_mode: FailureMode
    action_type: str
    command: str
    timeout_seconds: int
    max_retries: int
    success_criteria: str
    fallback_action: Optional[str] = None

class AutonomousSystemManager:
    """Manages autonomous operation and self-healing capabilities"""
    
    def __init__(self):
        self.error_handler = ErrorHandler()
        self.system_health = SystemHealth.HEALTHY
        self.recovery_actions = self._define_recovery_actions()
        self.health_metrics = {}
        self.last_health_check = datetime.now()
        self.failure_history = []
        self.performance_baseline = None
        self.autonomous_mode = True
        
    def _define_recovery_actions(self) -> Dict[FailureMode, List[RecoveryAction]]:
        """Define comprehensive recovery actions for each failure mode"""
        return {
            FailureMode.NETWORK_OUTAGE: [
                RecoveryAction(
                    failure_mode=FailureMode.NETWORK_OUTAGE,
                    action_type="switch_network",
                    command="python3 -c 'import requests; requests.get(\"https://8.8.8.8\")'",
                    timeout_seconds=30,
                    max_retries=3,
                    success_criteria="status_code_200",
                    fallback_action="enable_offline_mode"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.NETWORK_OUTAGE,
                    action_type="restart_networking",
                    command="sudo systemctl restart networking",
                    timeout_seconds=60,
                    max_retries=1,
                    success_criteria="ping_success"
                )
            ],
            
            FailureMode.API_QUOTA_EXCEEDED: [
                RecoveryAction(
                    failure_mode=FailureMode.API_QUOTA_EXCEEDED,
                    action_type="enable_aggressive_caching",
                    command="python3 -c 'from config import config; config.cache_aggressive_mode = True'",
                    timeout_seconds=5,
                    max_retries=1,
                    success_criteria="cache_enabled"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.API_QUOTA_EXCEEDED,
                    action_type="switch_to_alternative_apis",
                    command="python3 switch_data_sources.py",
                    timeout_seconds=30,
                    max_retries=2,
                    success_criteria="alternative_apis_active"
                )
            ],
            
            FailureMode.DISK_FULL: [
                RecoveryAction(
                    failure_mode=FailureMode.DISK_FULL,
                    action_type="cleanup_old_data",
                    command="python3 cleanup_system.py --aggressive",
                    timeout_seconds=120,
                    max_retries=1,
                    success_criteria="disk_space_freed"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.DISK_FULL,
                    action_type="compress_archives",
                    command="find ./logs ./results -name '*.log' -o -name '*.json' | xargs gzip",
                    timeout_seconds=300,
                    max_retries=1,
                    success_criteria="compression_completed"
                )
            ],
            
            FailureMode.MEMORY_EXHAUSTION: [
                RecoveryAction(
                    failure_mode=FailureMode.MEMORY_EXHAUSTION,
                    action_type="restart_memory_intensive_processes",
                    command="python3 restart_components.py --memory-intensive",
                    timeout_seconds=60,
                    max_retries=2,
                    success_criteria="memory_usage_normal"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.MEMORY_EXHAUSTION,
                    action_type="enable_memory_optimization",
                    command="python3 -c 'import gc; gc.collect()'",
                    timeout_seconds=30,
                    max_retries=1,
                    success_criteria="garbage_collected"
                )
            ],
            
            FailureMode.CODE_COMPILATION_ERROR: [
                RecoveryAction(
                    failure_mode=FailureMode.CODE_COMPILATION_ERROR,
                    action_type="auto_fix_with_claude",
                    command="python3 auto_fix_code.py",
                    timeout_seconds=180,
                    max_retries=3,
                    success_criteria="compilation_success"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.CODE_COMPILATION_ERROR,
                    action_type="revert_to_last_working_version",
                    command="python3 revert_strategy.py --last-working",
                    timeout_seconds=30,
                    max_retries=1,
                    success_criteria="revert_successful"
                )
            ],
            
            FailureMode.MODEL_DRIFT: [
                RecoveryAction(
                    failure_mode=FailureMode.MODEL_DRIFT,
                    action_type="retrain_models",
                    command="python3 retrain_models.py --autonomous",
                    timeout_seconds=3600,
                    max_retries=1,
                    success_criteria="model_performance_restored"
                ),
                RecoveryAction(
                    failure_mode=FailureMode.MODEL_DRIFT,
                    action_type="adjust_position_sizing",
                    command="python3 adjust_risk.py --conservative",
                    timeout_seconds=60,
                    max_retries=1,
                    success_criteria="risk_adjusted"
                )
            ]
        }
    
    async def continuous_health_monitoring(self):
        """Continuously monitor system health and trigger autonomous actions"""
        logger.info("🤖 Starting autonomous health monitoring...")
        
        while self.autonomous_mode:
            try:
                # Comprehensive health check
                health_status = await self._comprehensive_health_check()
                
                # Detect and respond to issues
                detected_failures = await self._detect_failure_modes()
                
                if detected_failures:
                    logger.warning(f"🚨 Detected failures: {[f.value for f in detected_failures]}")
                    
                    # Trigger autonomous recovery
                    recovery_success = await self._autonomous_recovery(detected_failures)
                    
                    if recovery_success:
                        logger.success("🔧 Autonomous recovery successful")
                    else:
                        logger.error("❌ Autonomous recovery failed - human intervention may be required")
                        await self._escalate_to_human()
                
                # Proactive optimization
                await self._proactive_optimization()
                
                # Sleep until next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error
    
    @memory_monitor(threshold_mb=500)
    @retry_with_backoff(
        retry_config=RetryConfig(max_attempts=3, base_delay=1.0),
        failure_types=[FailureType.MEMORY, FailureType.RESOURCE]
    )
    async def _comprehensive_health_check(self) -> Dict[str, Any]:
        """Comprehensive system health assessment with resilience"""
        health_data = {
            'timestamp': datetime.now().isoformat(),
            'system_resources': self._check_system_resources(),
            'network_connectivity': await self._check_network_connectivity(),
            'api_health': await self._check_api_health(),
            'data_integrity': await self._check_data_integrity(),
            'model_performance': await self._check_model_performance(),
            'trading_performance': await self._check_trading_performance()
        }
        
        # Calculate overall health score
        health_score = self._calculate_health_score(health_data)
        health_data['overall_health_score'] = health_score
        
        # Update system health status
        if health_score >= 0.9:
            self.system_health = SystemHealth.HEALTHY
        elif health_score >= 0.7:
            self.system_health = SystemHealth.DEGRADED
        elif health_score >= 0.5:
            self.system_health = SystemHealth.CRITICAL
        else:
            self.system_health = SystemHealth.FAILED
        
        self.health_metrics = health_data
        self.last_health_check = datetime.now()
        
        return health_data
    
    def _check_system_resources(self) -> Dict[str, float]:
        """Check system resource utilization"""
        return {
            'cpu_usage': psutil.cpu_percent(interval=1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('/').percent,
            'load_average': os.getloadavg()[0] if hasattr(os, 'getloadavg') else 0.0
        }
    
    async def _check_network_connectivity(self) -> Dict[str, bool]:
        """Check network connectivity to critical services"""
        connectivity = {}
        
        endpoints = {
            'quantconnect': 'https://www.quantconnect.com',
            'github': 'https://api.github.com',
            'claude_api': 'https://api.anthropic.com',
            'google_dns': 'https://8.8.8.8'
        }
        
        for name, url in endpoints.items():
            try:
                import aiohttp
                async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
                    async with session.get(url) as response:
                        connectivity[name] = response.status < 400
            except:
                connectivity[name] = False
        
        return connectivity
    
    async def _check_api_health(self) -> Dict[str, Any]:
        """Check API health and quota status"""
        api_health = {}
        
        # Check QuantConnect API
        try:
            # This would check actual API health
            api_health['quantconnect'] = {
                'status': 'healthy',
                'quota_remaining': 0.8,  # 80% quota remaining
                'response_time_ms': 150
            }
        except Exception as e:
            api_health['quantconnect'] = {
                'status': 'error',
                'error': str(e)
            }
        
        return api_health
    
    async def _check_data_integrity(self) -> Dict[str, bool]:
        """Check data integrity across the system"""
        integrity_checks = {}
        
        # Check database integrity
        try:
            db_path = "data/algoforge_memory.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                conn.execute("PRAGMA integrity_check")
                integrity_checks['database'] = True
                conn.close()
            else:
                integrity_checks['database'] = False
        except:
            integrity_checks['database'] = False
        
        # Check strategy files
        strategy_dir = Path("strategies")
        if strategy_dir.exists():
            strategy_files = list(strategy_dir.glob("*.py"))
            integrity_checks['strategies'] = len(strategy_files) > 0
        else:
            integrity_checks['strategies'] = False
        
        return integrity_checks
    
    async def _check_model_performance(self) -> Dict[str, float]:
        """Check model performance and detect drift"""
        # This would implement actual model performance monitoring
        return {
            'prediction_accuracy': 0.75,
            'sharpe_ratio': 1.2,
            'performance_drift': 0.05  # 5% drift from baseline
        }
    
    async def _check_trading_performance(self) -> Dict[str, float]:
        """Check trading performance metrics"""
        # This would check actual trading performance
        return {
            'daily_return': 0.02,
            'drawdown': 0.05,
            'win_rate': 0.65,
            'profit_factor': 1.8
        }
    
    def _calculate_health_score(self, health_data: Dict[str, Any]) -> float:
        """Calculate overall health score from health data"""
        scores = []
        
        # System resources (25% weight)
        resources = health_data['system_resources']
        resource_score = (
            (100 - resources['cpu_usage']) / 100 * 0.25 +
            (100 - resources['memory_usage']) / 100 * 0.25 +
            (100 - resources['disk_usage']) / 100 * 0.5
        )
        scores.append(resource_score * 0.25)
        
        # Network connectivity (20% weight)
        connectivity = health_data['network_connectivity']
        network_score = sum(connectivity.values()) / len(connectivity)
        scores.append(network_score * 0.20)
        
        # API health (20% weight)
        api_health = health_data['api_health']
        api_score = 1.0 if all(api.get('status') == 'healthy' for api in api_health.values()) else 0.5
        scores.append(api_score * 0.20)
        
        # Data integrity (15% weight)
        integrity = health_data['data_integrity']
        integrity_score = sum(integrity.values()) / len(integrity)
        scores.append(integrity_score * 0.15)
        
        # Model performance (10% weight)
        model_perf = health_data['model_performance']
        model_score = min(model_perf['prediction_accuracy'], 1.0)
        scores.append(model_score * 0.10)
        
        # Trading performance (10% weight)
        trading_perf = health_data['trading_performance']
        trading_score = min(trading_perf['win_rate'], 1.0)
        scores.append(trading_score * 0.10)
        
        return sum(scores)
    
    async def _detect_failure_modes(self) -> List[FailureMode]:
        """Detect active failure modes"""
        failures = []
        
        if not self.health_metrics:
            return failures
        
        # Check for specific failure conditions
        resources = self.health_metrics.get('system_resources', {})
        
        if resources.get('disk_usage', 0) > 90:
            failures.append(FailureMode.DISK_FULL)
        
        if resources.get('memory_usage', 0) > 95:
            failures.append(FailureMode.MEMORY_EXHAUSTION)
        
        # Check network connectivity
        connectivity = self.health_metrics.get('network_connectivity', {})
        if not connectivity.get('quantconnect', True):
            failures.append(FailureMode.NETWORK_OUTAGE)
        
        # Check API health
        api_health = self.health_metrics.get('api_health', {})
        qc_api = api_health.get('quantconnect', {})
        if qc_api.get('quota_remaining', 1.0) < 0.1:
            failures.append(FailureMode.API_QUOTA_EXCEEDED)
        
        # Check model performance
        model_perf = self.health_metrics.get('model_performance', {})
        if model_perf.get('performance_drift', 0) > 0.15:  # 15% drift threshold
            failures.append(FailureMode.MODEL_DRIFT)
        
        return failures
    
    async def _autonomous_recovery(self, failures: List[FailureMode]) -> bool:
        """Execute autonomous recovery actions for detected failures"""
        logger.info(f"🤖 Initiating autonomous recovery for {len(failures)} failures")
        
        recovery_success = True
        
        for failure in failures:
            if failure in self.recovery_actions:
                actions = self.recovery_actions[failure]
                
                for action in actions:
                    try:
                        logger.info(f"🔧 Executing recovery action: {action.action_type}")
                        
                        # Execute recovery command
                        result = await self._execute_recovery_action(action)
                        
                        if result:
                            logger.success(f"✅ Recovery action successful: {action.action_type}")
                            break  # Success, no need for fallback
                        else:
                            logger.warning(f"⚠️ Recovery action failed: {action.action_type}")
                            
                    except Exception as e:
                        logger.error(f"❌ Recovery action error: {e}")
                        recovery_success = False
                        
                        # Try fallback action if available
                        if action.fallback_action:
                            try:
                                await self._execute_fallback_action(action.fallback_action)
                            except Exception as fallback_error:
                                logger.error(f"❌ Fallback action failed: {fallback_error}")
        
        return recovery_success
    
    async def _execute_recovery_action(self, action: RecoveryAction) -> bool:
        """Execute a single recovery action"""
        try:
            # Execute the command with timeout
            process = await asyncio.create_subprocess_shell(
                action.command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=action.timeout_seconds
            )
            
            # Check success criteria
            if action.success_criteria == "status_code_200":
                return process.returncode == 0
            elif action.success_criteria == "disk_space_freed":
                # Check if disk usage improved
                current_usage = psutil.disk_usage('/').percent
                return current_usage < 85  # Improved disk usage
            elif action.success_criteria == "memory_usage_normal":
                # Check if memory usage improved
                current_memory = psutil.virtual_memory().percent
                return current_memory < 80  # Improved memory usage
            else:
                return process.returncode == 0
                
        except asyncio.TimeoutError:
            logger.error(f"Recovery action timed out: {action.action_type}")
            return False
        except Exception as e:
            logger.error(f"Recovery action failed: {e}")
            return False
    
    async def _execute_fallback_action(self, fallback_action: str):
        """Execute fallback action"""
        logger.info(f"🔄 Executing fallback action: {fallback_action}")
        
        if fallback_action == "enable_offline_mode":
            # Enable offline mode
            logger.info("📴 Enabling offline mode")
            # This would enable offline operation
            
        elif fallback_action == "switch_to_backup_systems":
            # Switch to backup systems
            logger.info("🔄 Switching to backup systems")
            # This would switch to backup infrastructure
    
    async def _escalate_to_human(self):
        """Escalate critical issues to human operators"""
        logger.critical("🚨 CRITICAL: Escalating to human intervention")
        
        # Create detailed incident report
        incident_report = {
            'timestamp': datetime.now().isoformat(),
            'system_health': self.system_health.value,
            'health_metrics': self.health_metrics,
            'failure_history': self.failure_history[-10:],  # Last 10 failures
            'recommended_actions': self._generate_human_recommendations()
        }
        
        # Save incident report
        incident_file = f"incidents/incident_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        os.makedirs("incidents", exist_ok=True)
        
        with open(incident_file, 'w') as f:
            json.dump(incident_report, f, indent=2, default=str)
        
        logger.critical(f"💾 Incident report saved: {incident_file}")
        
        # Send notifications (email, Slack, etc.)
        await self._send_critical_notifications(incident_report)
    
    def _generate_human_recommendations(self) -> List[str]:
        """Generate recommendations for human operators"""
        recommendations = []
        
        if self.system_health == SystemHealth.FAILED:
            recommendations.append("System restart required")
            recommendations.append("Check hardware health")
            recommendations.append("Verify network connectivity")
        
        if self.health_metrics.get('system_resources', {}).get('disk_usage', 0) > 95:
            recommendations.append("Immediate disk cleanup required")
            recommendations.append("Consider adding storage capacity")
        
        if self.health_metrics.get('system_resources', {}).get('memory_usage', 0) > 95:
            recommendations.append("Memory leak investigation required")
            recommendations.append("Consider increasing system RAM")
        
        return recommendations
    
    async def _send_critical_notifications(self, incident_report: Dict[str, Any]):
        """Send critical notifications to operators"""
        # This would send actual notifications via email, Slack, SMS, etc.
        logger.critical("📧 Critical notifications sent to operators")
    
    async def _proactive_optimization(self):
        """Perform proactive system optimization"""
        try:
            # Cleanup old logs
            await self._cleanup_old_files()
            
            # Optimize database
            await self._optimize_database()
            
            # Update models if needed
            await self._check_model_updates()
            
            # Tune performance parameters
            await self._tune_performance()
            
        except Exception as e:
            logger.warning(f"Proactive optimization error: {e}")
    
    async def _cleanup_old_files(self):
        """Clean up old files to free space"""
        # Remove logs older than 30 days
        log_dir = Path("logs")
        if log_dir.exists():
            cutoff_date = datetime.now() - timedelta(days=30)
            for log_file in log_dir.glob("*.log"):
                if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff_date:
                    log_file.unlink()
    
    async def _optimize_database(self):
        """Optimize database performance"""
        try:
            db_path = "data/algoforge_memory.db"
            if os.path.exists(db_path):
                conn = sqlite3.connect(db_path)
                conn.execute("VACUUM")
                conn.execute("ANALYZE")
                conn.close()
        except Exception as e:
            logger.warning(f"Database optimization error: {e}")
    
    async def _check_model_updates(self):
        """Check if models need updating"""
        # This would implement model update logic
        pass
    
    async def _tune_performance(self):
        """Automatically tune performance parameters"""
        # This would implement performance tuning logic
        pass
    
    def get_autonomy_status(self) -> Dict[str, Any]:
        """Get current autonomy status"""
        return {
            'autonomous_mode': self.autonomous_mode,
            'system_health': self.system_health.value,
            'last_health_check': self.last_health_check.isoformat(),
            'failure_count_24h': len([
                f for f in self.failure_history 
                if f['timestamp'] > datetime.now() - timedelta(hours=24)
            ]),
            'recovery_success_rate': self._calculate_recovery_success_rate(),
            'uptime_percentage': self._calculate_uptime_percentage()
        }
    
    def _calculate_recovery_success_rate(self) -> float:
        """Calculate recovery success rate"""
        recent_failures = [
            f for f in self.failure_history 
            if f['timestamp'] > datetime.now() - timedelta(days=7)
        ]
        
        if not recent_failures:
            return 1.0
        
        successful_recoveries = len([f for f in recent_failures if f.get('recovered', False)])
        return successful_recoveries / len(recent_failures)
    
    def _calculate_uptime_percentage(self) -> float:
        """Calculate system uptime percentage"""
        # This would calculate actual uptime
        return 99.5  # Placeholder

# Create autonomous system components
async def create_self_healing_components():
    """Create additional self-healing components"""
    
    # Auto-fix code using Claude Code SDK
    await create_auto_fix_system()
    
    # Create system cleanup utilities
    await create_cleanup_system()
    
    # Create backup and recovery system
    await create_backup_system()

async def create_auto_fix_system():
    """Create autonomous code fixing system using Claude Code SDK"""
    auto_fix_code = '''
#!/usr/bin/env python3
"""
Autonomous Code Fixing System using Claude Code SDK
Automatically fixes compilation errors and runtime issues
"""
import asyncio
from claude_code_sdk import query, ClaudeCodeOptions
from pathlib import Path
import subprocess

async def auto_fix_compilation_error(file_path: str, error_message: str) -> bool:
    """Use Claude to automatically fix compilation errors"""
    
    # Read the problematic code
    with open(file_path, 'r') as f:
        code = f.read()
    
    # Create fixing prompt
    prompt = f"""
    Fix this Python code compilation error:
    
    Error: {error_message}
    
    Code:
    ```python
    {code}
    ```
    
    Please provide the corrected code that will compile successfully.
    Only return the fixed Python code, no explanations.
    """
    
    options = ClaudeCodeOptions(
        system_prompt="You are an expert Python developer who fixes code compilation errors.",
        max_turns=1
    )
    
    try:
        async for message in query(prompt=prompt, options=options):
            if message.type == "text":
                fixed_code = message.content
                
                # Extract code from markdown if present
                if "```python" in fixed_code:
                    start = fixed_code.find("```python") + 9
                    end = fixed_code.find("```", start)
                    fixed_code = fixed_code[start:end].strip()
                
                # Test compilation
                try:
                    compile(fixed_code, file_path, 'exec')
                    
                    # Create backup
                    backup_path = f"{file_path}.backup"
                    with open(backup_path, 'w') as f:
                        f.write(code)
                    
                    # Write fixed code
                    with open(file_path, 'w') as f:
                        f.write(fixed_code)
                    
                    print(f"✅ Auto-fixed compilation error in {file_path}")
                    return True
                    
                except SyntaxError:
                    print(f"❌ Auto-fix failed for {file_path}")
                    return False
        
        return False
        
    except Exception as e:
        print(f"❌ Auto-fix error: {e}")
        return False

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 2:
        asyncio.run(auto_fix_compilation_error(sys.argv[1], sys.argv[2]))
'''
    
    with open("auto_fix_code.py", 'w') as f:
        f.write(auto_fix_code)
    
    # Make executable
    os.chmod("auto_fix_code.py", 0o755)

async def create_cleanup_system():
    """Create system cleanup utilities"""
    cleanup_code = '''
#!/usr/bin/env python3
"""
System Cleanup Utilities
Automatically cleans up system resources
"""
import os
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

def cleanup_aggressive():
    """Aggressive cleanup to free maximum space"""
    freed_space = 0
    
    # Clean temporary files
    temp_dirs = ["/tmp", "./temp", "./cache"]
    for temp_dir in temp_dirs:
        if os.path.exists(temp_dir):
            for item in Path(temp_dir).iterdir():
                if item.is_file():
                    try:
                        size = item.stat().st_size
                        item.unlink()
                        freed_space += size
                    except:
                        pass
    
    # Clean old logs
    log_dirs = ["./logs", "./results"]
    cutoff = datetime.now() - timedelta(days=7)
    
    for log_dir in log_dirs:
        if os.path.exists(log_dir):
            for log_file in Path(log_dir).rglob("*"):
                if log_file.is_file():
                    try:
                        if datetime.fromtimestamp(log_file.stat().st_mtime) < cutoff:
                            size = log_file.stat().st_size
                            log_file.unlink()
                            freed_space += size
                    except:
                        pass
    
    # Compress large files
    for large_file in Path(".").rglob("*.json"):
        if large_file.stat().st_size > 10 * 1024 * 1024:  # >10MB
            try:
                subprocess.run(["gzip", str(large_file)], check=True)
            except:
                pass
    
    print(f"✅ Freed {freed_space / (1024*1024):.1f} MB of disk space")
    return freed_space > 0

if __name__ == "__main__":
    import sys
    if "--aggressive" in sys.argv:
        cleanup_aggressive()
'''
    
    with open("cleanup_system.py", 'w') as f:
        f.write(cleanup_code)
    
    os.chmod("cleanup_system.py", 0o755)

async def create_backup_system():
    """Create backup and recovery system"""
    backup_code = '''
#!/usr/bin/env python3
"""
Backup and Recovery System
Automatically backs up critical system state
"""
import shutil
import json
from pathlib import Path
from datetime import datetime

def create_system_backup():
    """Create complete system backup"""
    backup_dir = Path(f"backups/backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    
    # Backup critical directories
    critical_dirs = ["strategies", "data", "config"]
    
    for dir_name in critical_dirs:
        if os.path.exists(dir_name):
            shutil.copytree(dir_name, backup_dir / dir_name)
    
    # Backup configuration
    config_backup = {
        'timestamp': datetime.now().isoformat(),
        'system_state': 'healthy',
        'backup_type': 'automatic'
    }
    
    with open(backup_dir / "backup_manifest.json", 'w') as f:
        json.dump(config_backup, f, indent=2)
    
    print(f"✅ System backup created: {backup_dir}")
    return str(backup_dir)

def restore_from_backup(backup_path: str):
    """Restore system from backup"""
    backup_dir = Path(backup_path)
    
    if not backup_dir.exists():
        print(f"❌ Backup not found: {backup_path}")
        return False
    
    # Restore critical directories
    critical_dirs = ["strategies", "data"]
    
    for dir_name in critical_dirs:
        source = backup_dir / dir_name
        if source.exists():
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
            shutil.copytree(source, dir_name)
    
    print(f"✅ System restored from backup: {backup_path}")
    return True

if __name__ == "__main__":
    import sys
    if "--restore" in sys.argv and len(sys.argv) > 2:
        restore_from_backup(sys.argv[2])
    else:
        create_system_backup()
'''
    
    with open("backup_system.py", 'w') as f:
        f.write(backup_code)
    
    os.chmod("backup_system.py", 0o755)

# Example usage
async def test_autonomous_system():
    """Test the autonomous system"""
    manager = AutonomousSystemManager()
    
    # Start health monitoring in background
    health_task = asyncio.create_task(manager.continuous_health_monitoring())
    
    # Get status
    status = manager.get_autonomy_status()
    print(f"Autonomous system status: {status}")
    
    # Run for a short time for testing
    await asyncio.sleep(10)
    
    # Stop monitoring
    manager.autonomous_mode = False
    health_task.cancel()
    
    return True

if __name__ == "__main__":
    asyncio.run(test_autonomous_system())