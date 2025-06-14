#!/usr/bin/env python3
"""
Component Restart Manager
Manages restarting of memory-intensive processes and system components
"""
import os
import sys
import subprocess
import psutil
import signal
import time
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
from loguru import logger
import json

class ComponentManager:
    """Manages system components and process lifecycle"""
    
    def __init__(self):
        self.component_configs = self._load_component_configs()
        self.restart_history = []
        
    def _load_component_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load component configuration"""
        return {
            'algoforge_main': {
                'command': ['python3', 'algoforge_main.py'],
                'cwd': '.',
                'memory_threshold_mb': 2048,  # 2GB
                'cpu_threshold_percent': 80,
                'restart_delay': 5,
                'max_restarts': 3,
                'priority': 'high'
            },
            'autonomous_system': {
                'command': ['python3', 'autonomous_system.py'],
                'cwd': '.',
                'memory_threshold_mb': 1024,  # 1GB
                'cpu_threshold_percent': 70,
                'restart_delay': 3,
                'max_restarts': 5,
                'priority': 'high'
            },
            'mcp_servers': {
                'command': ['npx', '-y', '@modelcontextprotocol/server-filesystem'],
                'cwd': '.',
                'memory_threshold_mb': 512,  # 512MB
                'cpu_threshold_percent': 60,
                'restart_delay': 2,
                'max_restarts': 10,
                'priority': 'medium'
            },
            'postgres': {
                'command': ['sudo', 'systemctl', 'restart', 'postgresql'],
                'cwd': '.',
                'memory_threshold_mb': 1024,
                'cpu_threshold_percent': 75,
                'restart_delay': 10,
                'max_restarts': 2,
                'priority': 'critical'
            },
            'quantconnect_sync': {
                'command': ['python3', '-c', 'from quantconnect_sync import QuantConnectSyncManager; import asyncio; asyncio.run(QuantConnectSyncManager().start_continuous_sync())'],
                'cwd': '.',
                'memory_threshold_mb': 256,
                'cpu_threshold_percent': 50,
                'restart_delay': 5,
                'max_restarts': 10,
                'priority': 'medium'
            }
        }
    
    def restart_memory_intensive_processes(self) -> Dict[str, Any]:
        """Restart processes that are using excessive memory"""
        logger.info("🔄 Restarting memory-intensive processes...")
        
        restart_results = {
            'timestamp': datetime.now().isoformat(),
            'processes_checked': 0,
            'processes_restarted': 0,
            'restart_details': [],
            'memory_freed_mb': 0,
            'success': True
        }
        
        try:
            # Get all running processes
            for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent', 'cmdline']):
                try:
                    proc_info = proc.info
                    memory_mb = proc_info['memory_info'].rss / (1024 * 1024)
                    cpu_percent = proc_info['cpu_percent']
                    
                    restart_results['processes_checked'] += 1
                    
                    # Check if process matches our components and exceeds thresholds
                    for component_name, config in self.component_configs.items():
                        if self._is_component_process(proc_info, component_name, config):
                            if (memory_mb > config['memory_threshold_mb'] or 
                                cpu_percent > config['cpu_threshold_percent']):
                                
                                logger.info(f"🎯 Restarting {component_name} (Memory: {memory_mb:.1f}MB, CPU: {cpu_percent:.1f}%)")
                                
                                restart_result = self._restart_component(component_name, config, proc)
                                restart_results['restart_details'].append(restart_result)
                                
                                if restart_result['success']:
                                    restart_results['processes_restarted'] += 1
                                    restart_results['memory_freed_mb'] += memory_mb
                                else:
                                    restart_results['success'] = False
                                
                                break
                    
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    continue
            
            # Restart system services if needed
            restart_results.update(self._restart_system_services())
            
            # Record restart history
            self.restart_history.append(restart_results)
            self._save_restart_history()
            
            if restart_results['processes_restarted'] > 0:
                logger.success(f"✅ Restarted {restart_results['processes_restarted']} memory-intensive processes")
                logger.info(f"💾 Freed approximately {restart_results['memory_freed_mb']:.1f} MB of memory")
            else:
                logger.info("ℹ️ No memory-intensive processes needed restarting")
            
            return restart_results
            
        except Exception as e:
            logger.error(f"❌ Error restarting memory-intensive processes: {e}")
            restart_results['success'] = False
            restart_results['error'] = str(e)
            return restart_results
    
    def _is_component_process(self, proc_info: Dict[str, Any], component_name: str, config: Dict[str, Any]) -> bool:
        """Check if process matches a component"""
        try:
            cmdline = proc_info.get('cmdline', [])
            if not cmdline:
                return False
            
            cmdline_str = ' '.join(cmdline).lower()
            
            # Match based on component name and command patterns
            if component_name == 'algoforge_main':
                return 'algoforge_main.py' in cmdline_str
            elif component_name == 'autonomous_system':
                return 'autonomous_system.py' in cmdline_str
            elif component_name == 'mcp_servers':
                return 'mcp' in cmdline_str or 'modelcontextprotocol' in cmdline_str
            elif component_name == 'quantconnect_sync':
                return 'quantconnect_sync' in cmdline_str
            elif component_name == 'postgres':
                return proc_info.get('name', '').lower() in ['postgres', 'postgresql']
            
            return False
            
        except Exception:
            return False
    
    def _restart_component(self, component_name: str, config: Dict[str, Any], proc: psutil.Process) -> Dict[str, Any]:
        """Restart a specific component"""
        restart_result = {
            'component': component_name,
            'timestamp': datetime.now().isoformat(),
            'old_pid': proc.pid,
            'new_pid': None,
            'success': False,
            'error': None
        }
        
        try:
            # Gracefully terminate the process
            logger.debug(f"Terminating {component_name} (PID: {proc.pid})")
            
            try:
                proc.terminate()
                # Wait for graceful shutdown
                proc.wait(timeout=10)
            except psutil.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning(f"Force killing {component_name} (PID: {proc.pid})")
                proc.kill()
                proc.wait(timeout=5)
            
            # Wait before restarting
            time.sleep(config.get('restart_delay', 5))
            
            # Restart the component
            if component_name == 'postgres':
                # Special handling for system services
                result = subprocess.run(
                    config['command'],
                    cwd=config.get('cwd', '.'),
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                restart_result['success'] = result.returncode == 0
                if not restart_result['success']:
                    restart_result['error'] = result.stderr
            else:
                # Regular process restart
                new_proc = subprocess.Popen(
                    config['command'],
                    cwd=config.get('cwd', '.'),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                restart_result['new_pid'] = new_proc.pid
                restart_result['success'] = True
                
                logger.debug(f"Restarted {component_name} with new PID: {new_proc.pid}")
            
            return restart_result
            
        except Exception as e:
            logger.error(f"Failed to restart {component_name}: {e}")
            restart_result['error'] = str(e)
            return restart_result
    
    def _restart_system_services(self) -> Dict[str, Any]:
        """Restart critical system services if needed"""
        service_results = {
            'services_checked': 0,
            'services_restarted': 0,
            'service_details': []
        }
        
        critical_services = ['postgresql']
        
        for service in critical_services:
            try:
                service_results['services_checked'] += 1
                
                # Check service status
                status_result = subprocess.run(
                    ['systemctl', 'is-active', service],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                
                if status_result.returncode != 0:  # Service is not active
                    logger.info(f"🔄 Restarting inactive service: {service}")
                    
                    restart_result = subprocess.run(
                        ['sudo', 'systemctl', 'restart', service],
                        capture_output=True,
                        text=True,
                        timeout=30
                    )
                    
                    service_detail = {
                        'service': service,
                        'action': 'restart',
                        'success': restart_result.returncode == 0,
                        'error': restart_result.stderr if restart_result.returncode != 0 else None
                    }
                    
                    service_results['service_details'].append(service_detail)
                    
                    if service_detail['success']:
                        service_results['services_restarted'] += 1
                        logger.success(f"✅ Restarted service: {service}")
                    else:
                        logger.error(f"❌ Failed to restart service {service}: {restart_result.stderr}")
                
            except Exception as e:
                logger.debug(f"Error checking/restarting service {service}: {e}")
        
        return service_results
    
    def restart_specific_component(self, component_name: str) -> bool:
        """Restart a specific component by name"""
        logger.info(f"🔄 Restarting specific component: {component_name}")
        
        if component_name not in self.component_configs:
            logger.error(f"❌ Unknown component: {component_name}")
            return False
        
        config = self.component_configs[component_name]
        
        try:
            # Find running processes for this component
            target_processes = []
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    if self._is_component_process(proc.info, component_name, config):
                        target_processes.append(proc)
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if not target_processes:
                logger.warning(f"⚠️ No running processes found for {component_name}")
                # Try to start the component anyway
                return self._start_component(component_name, config)
            
            # Restart each found process
            success_count = 0
            for proc in target_processes:
                restart_result = self._restart_component(component_name, config, proc)
                if restart_result['success']:
                    success_count += 1
            
            success = success_count > 0
            if success:
                logger.success(f"✅ Successfully restarted {component_name}")
            else:
                logger.error(f"❌ Failed to restart {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error restarting component {component_name}: {e}")
            return False
    
    def _start_component(self, component_name: str, config: Dict[str, Any]) -> bool:
        """Start a component that's not currently running"""
        try:
            logger.info(f"🚀 Starting component: {component_name}")
            
            if component_name == 'postgres':
                result = subprocess.run(
                    ['sudo', 'systemctl', 'start', 'postgresql'],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                success = result.returncode == 0
            else:
                proc = subprocess.Popen(
                    config['command'],
                    cwd=config.get('cwd', '.'),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                success = True
                logger.debug(f"Started {component_name} with PID: {proc.pid}")
            
            if success:
                logger.success(f"✅ Started component: {component_name}")
            else:
                logger.error(f"❌ Failed to start component: {component_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Error starting component {component_name}: {e}")
            return False
    
    def check_component_health(self) -> Dict[str, Any]:
        """Check health status of all components"""
        health_report = {
            'timestamp': datetime.now().isoformat(),
            'components': {},
            'overall_health': 'healthy',
            'unhealthy_components': []
        }
        
        for component_name, config in self.component_configs.items():
            component_health = {
                'running': False,
                'memory_usage_mb': 0,
                'cpu_usage_percent': 0,
                'process_count': 0,
                'exceeds_thresholds': False,
                'status': 'unknown'
            }
            
            try:
                # Find processes for this component
                component_processes = []
                for proc in psutil.process_iter(['pid', 'name', 'memory_info', 'cpu_percent', 'cmdline']):
                    try:
                        if self._is_component_process(proc.info, component_name, config):
                            component_processes.append(proc)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
                
                if component_processes:
                    component_health['running'] = True
                    component_health['process_count'] = len(component_processes)
                    
                    # Calculate total resource usage
                    total_memory = sum(proc.memory_info().rss for proc in component_processes)
                    component_health['memory_usage_mb'] = total_memory / (1024 * 1024)
                    
                    # Average CPU usage
                    cpu_values = [proc.cpu_percent() for proc in component_processes]
                    component_health['cpu_usage_percent'] = sum(cpu_values) / len(cpu_values) if cpu_values else 0
                    
                    # Check thresholds
                    memory_exceeded = component_health['memory_usage_mb'] > config['memory_threshold_mb']
                    cpu_exceeded = component_health['cpu_usage_percent'] > config['cpu_threshold_percent']
                    component_health['exceeds_thresholds'] = memory_exceeded or cpu_exceeded
                    
                    if component_health['exceeds_thresholds']:
                        component_health['status'] = 'unhealthy'
                        health_report['unhealthy_components'].append(component_name)
                    else:
                        component_health['status'] = 'healthy'
                else:
                    component_health['status'] = 'not_running'
                    if config['priority'] in ['high', 'critical']:
                        health_report['unhealthy_components'].append(component_name)
                
            except Exception as e:
                component_health['status'] = 'error'
                component_health['error'] = str(e)
                health_report['unhealthy_components'].append(component_name)
            
            health_report['components'][component_name] = component_health
        
        # Determine overall health
        if health_report['unhealthy_components']:
            health_report['overall_health'] = 'unhealthy'
        
        return health_report
    
    def _save_restart_history(self):
        """Save restart history to file"""
        try:
            history_file = Path("logs/restart_history.json")
            history_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(history_file, 'w') as f:
                json.dump(self.restart_history, f, indent=2)
                
        except Exception as e:
            logger.debug(f"Could not save restart history: {e}")
    
    def get_restart_statistics(self) -> Dict[str, Any]:
        """Get restart statistics"""
        if not self.restart_history:
            return {'total_restarts': 0}
        
        total_restarts = sum(r.get('processes_restarted', 0) for r in self.restart_history)
        total_memory_freed = sum(r.get('memory_freed_mb', 0) for r in self.restart_history)
        
        return {
            'total_restart_sessions': len(self.restart_history),
            'total_processes_restarted': total_restarts,
            'total_memory_freed_mb': total_memory_freed,
            'average_memory_freed_per_session': total_memory_freed / len(self.restart_history) if self.restart_history else 0,
            'last_restart': self.restart_history[-1]['timestamp'] if self.restart_history else None
        }

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Component restart manager")
    parser.add_argument("--memory-intensive", action="store_true", help="Restart memory-intensive processes")
    parser.add_argument("--component", help="Restart specific component")
    parser.add_argument("--health", action="store_true", help="Check component health")
    parser.add_argument("--stats", action="store_true", help="Show restart statistics")
    
    args = parser.parse_args()
    
    manager = ComponentManager()
    
    if args.memory_intensive:
        result = manager.restart_memory_intensive_processes()
        if result['success']:
            print(f"✅ Restarted {result['processes_restarted']} processes")
            print(f"💾 Freed {result['memory_freed_mb']:.1f} MB of memory")
            return True
        else:
            print("❌ Failed to restart memory-intensive processes")
            return False
    
    elif args.component:
        success = manager.restart_specific_component(args.component)
        if success:
            print(f"✅ Successfully restarted {args.component}")
            return True
        else:
            print(f"❌ Failed to restart {args.component}")
            return False
    
    elif args.health:
        health = manager.check_component_health()
        print(f"Overall health: {health['overall_health']}")
        print(f"Components checked: {len(health['components'])}")
        if health['unhealthy_components']:
            print(f"Unhealthy components: {', '.join(health['unhealthy_components'])}")
        
        for name, component in health['components'].items():
            status_icon = "✅" if component['status'] == 'healthy' else "❌"
            print(f"  {status_icon} {name}: {component['status']}")
            if component['running']:
                print(f"    Memory: {component['memory_usage_mb']:.1f} MB")
                print(f"    CPU: {component['cpu_usage_percent']:.1f}%")
        
        return health['overall_health'] == 'healthy'
    
    elif args.stats:
        stats = manager.get_restart_statistics()
        print("Restart Statistics:")
        print(f"  Total restart sessions: {stats['total_restart_sessions']}")
        print(f"  Total processes restarted: {stats['total_processes_restarted']}")
        print(f"  Total memory freed: {stats['total_memory_freed_mb']:.1f} MB")
        print(f"  Average memory freed per session: {stats['average_memory_freed_per_session']:.1f} MB")
        if stats['last_restart']:
            print(f"  Last restart: {stats['last_restart']}")
        return True
    
    else:
        # Default: restart memory-intensive processes
        result = manager.restart_memory_intensive_processes()
        return result['success']

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)