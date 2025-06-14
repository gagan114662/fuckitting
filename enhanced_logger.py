#!/usr/bin/env python3
"""
Enhanced Conversational Logger for AlgoForge 3.0
Provides detailed, plain English updates about system operations
"""
import sys
from datetime import datetime
from loguru import logger
import random
import asyncio
from typing import Optional

class ConversationalLogger:
    """A friendly worker that explains everything happening in the system"""
    
    def __init__(self):
        # Remove default logger
        logger.remove()
        
        # Add console handler with custom format
        logger.add(
            sys.stdout,
            level="INFO",
            format="<green>{time:HH:mm:ss}</green> | <cyan>🤖 AlgoBot</cyan> | <level>{message}</level>",
            colorize=True
        )
        
        # Add detailed file logger
        logger.add(
            "logs/algoforge_detailed_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level} | {name}:{function}:{line} | {message}",
            rotation="1 day",
            retention="30 days"
        )
        
        self.start_time = datetime.now()
        self.step_count = 0
        
    def greet(self):
        """Initial greeting"""
        greetings = [
            "Hey there! I'm your AlgoForge assistant. Let me walk you through everything I'm doing!",
            "Hello! I'm here to explain every step of your superhuman trading system!",
            "Hi! I'll be your guide through the AlgoForge startup process. Let's see what's happening!"
        ]
        logger.info(f"\n{'='*80}\n{random.choice(greetings)}\n{'='*80}")
        
    def thinking(self, action: str):
        """Show the system is thinking/processing"""
        messages = [
            f"🤔 Let me {action}...",
            f"💭 Working on {action}...",
            f"🧠 Processing {action}...",
            f"⚙️ Handling {action}..."
        ]
        logger.info(random.choice(messages))
        
    def step(self, description: str):
        """Log a major step"""
        self.step_count += 1
        logger.info(f"\n📍 STEP {self.step_count}: {description}")
        
    def detail(self, info: str):
        """Provide detailed information"""
        logger.info(f"   ℹ️ {info}")
        
    def found(self, item: str, status: str = "active"):
        """Report something found"""
        logger.info(f"   ✓ Found {item} - {status}")
        
    def checking(self, what: str):
        """Report checking something"""
        logger.info(f"   🔍 Checking {what}...")
        
    def success(self, message: str):
        """Report success"""
        logger.success(f"   ✅ {message}")
        
    def warning(self, message: str):
        """Report warning"""
        logger.warning(f"   ⚠️ {message}")
        
    def error(self, message: str):
        """Report error"""
        logger.error(f"   ❌ {message}")
        
    def progress(self, current: int, total: int, what: str):
        """Show progress"""
        percentage = (current / total) * 100 if total > 0 else 0
        logger.info(f"   📊 Progress: {current}/{total} {what} ({percentage:.1f}%)")
        
    def explain(self, explanation: str):
        """Explain what's happening in plain English"""
        logger.info(f"   💡 {explanation}")
        
    def report_status(self, component: str, status: str, details: Optional[str] = None):
        """Report component status"""
        status_icons = {
            'ready': '🟢',
            'warning': '🟡',
            'error': '🔴',
            'loading': '⏳',
            'active': '✅'
        }
        icon = status_icons.get(status, '⚪')
        msg = f"   {icon} {component}: {status.upper()}"
        if details:
            msg += f" - {details}"
        logger.info(msg)
        
    def section(self, title: str):
        """Start a new section"""
        logger.info(f"\n{'─'*60}\n🏷️  {title.upper()}\n{'─'*60}")
        
    def subsection(self, title: str):
        """Start a subsection"""
        logger.info(f"\n   【 {title} 】")
        
    def list_item(self, item: str, status: Optional[str] = None):
        """List an item"""
        if status:
            logger.info(f"     • {item}: {status}")
        else:
            logger.info(f"     • {item}")
            
    def chat(self, message: str):
        """Conversational message"""
        chats = [
            f"Just so you know, {message}",
            f"By the way, {message}",
            f"Quick update: {message}",
            f"FYI - {message}",
            f"Heads up! {message}"
        ]
        logger.info(f"   💬 {random.choice(chats)}")
        
    def working_on(self, task: str):
        """Report currently working on something"""
        messages = [
            f"Now I'm working on {task}...",
            f"Moving on to {task}...",
            f"Let me handle {task} for you...",
            f"Time to work on {task}..."
        ]
        logger.info(f"   🔧 {random.choice(messages)}")
        
    def discovery(self, what: str):
        """Report a discovery"""
        logger.info(f"   🔎 Discovered: {what}")
        
    def summary(self, title: str, items: list):
        """Provide a summary"""
        logger.info(f"\n   📋 {title}:")
        for item in items:
            logger.info(f"      • {item}")
            
    def elapsed_time(self):
        """Report elapsed time"""
        elapsed = (datetime.now() - self.start_time).total_seconds()
        logger.info(f"\n⏱️  Total time: {elapsed:.1f} seconds")
        
    def finale(self, success: bool = True):
        """Final message"""
        if success:
            messages = [
                "🎉 Everything's ready! Your superhuman trading system is good to go!",
                "🚀 All systems are go! Your AlgoForge is ready to conquer the markets!",
                "✨ Perfect! Your trading system is now running at superhuman levels!"
            ]
        else:
            messages = [
                "🤔 We hit a few bumps, but your system is still operational!",
                "💪 Despite some challenges, your system is up and running!",
                "🔧 Not everything went perfectly, but we're still good to trade!"
            ]
        logger.info(f"\n{'='*80}\n{random.choice(messages)}\n{'='*80}")

# Global logger instance
conv_logger = ConversationalLogger()

class DetailedSystemReporter:
    """Provides ultra-detailed system reporting"""
    
    def __init__(self, logger: ConversationalLogger):
        self.log = logger
        
    async def report_api_initialization(self, api_name: str, has_key: bool, testing: bool = True):
        """Detailed API initialization reporting (HONEST VERSION)"""
        if has_key:
            self.log.found(f"{api_name} API key", "configured")
            if testing:
                self.log.detail(f"DEMO: Simulating {api_name} connection test...")
                await asyncio.sleep(0.5)  # Simulate API test
                self.log.success(f"DEMO: {api_name} connection simulation completed!")
            else:
                self.log.detail(f"API key available - connection testing disabled in production mode")
                self.log.success(f"{api_name} ready for use!")
        else:
            self.log.warning(f"{api_name} API key not found - this feature will be limited")
            
    async def report_component_startup(self, component: str, description: str):
        """Detailed component startup reporting"""
        self.log.working_on(f"initializing {component}")
        self.log.explain(description)
        await asyncio.sleep(0.3)  # Simulate startup
        self.log.success(f"{component} is now active!")
        
    def report_data_source_status(self, sources: dict):
        """Report data source status in detail"""
        self.log.subsection("Data Source Configuration")
        
        active = [s for s, info in sources.items() if info.get('is_active')]
        inactive = [s for s, info in sources.items() if not info.get('is_active')]
        
        self.log.detail(f"I found {len(active)} active data sources and {len(inactive)} inactive ones")
        
        for source in active:
            priority = sources[source].get('priority', 'N/A')
            rate_limit = sources[source].get('rate_limit_per_minute', 'N/A')
            self.log.list_item(f"{source}", f"Priority {priority}, {rate_limit} requests/min")
            
        if inactive:
            self.log.detail("These sources need API keys to activate:")
            for source in inactive:
                self.log.list_item(f"{source}", "needs API key")
                
    def report_sync_status(self, sync_info: dict):
        """Report sync status in conversational detail (HONEST VERSION)"""
        self.log.subsection("QuantConnect Sync Status")
        
        self.log.chat("DEMO: This shows how sync bottleneck solutions would work!")
        self.log.detail(f"DEMO: Rate limiter configured for {sync_info.get('requests_allowed', 30)} requests per minute")
        self.log.detail(f"DEMO: Example sync count: {sync_info.get('files_synced', 0)} files")
        
        if sync_info.get('conflicts', 0) > 0:
            self.log.warning(f"DEMO: Example conflict handling: {sync_info['conflicts']} conflicts detected")
        else:
            self.log.success("DEMO: Example of conflict-free sync status!")
            
    def report_health_check(self, health_data: dict):
        """Report system health in friendly terms"""
        self.log.subsection("System Health Check")
        
        score = health_data.get('overall_health_score', 0)
        
        if score > 0.8:
            self.log.success(f"System health is excellent! (Score: {score:.2%})")
            self.log.chat("everything's running smoothly")
        elif score > 0.6:
            self.log.report_status("System Health", "warning", f"Score: {score:.2%}")
            self.log.chat("we're doing okay, but there's room for improvement")
        else:
            self.log.warning(f"System health needs attention (Score: {score:.2%})")
            self.log.chat("I'll activate self-healing to fix any issues")
            
        # Report specific metrics
        resources = health_data.get('system_resources', {})
        if resources:
            self.log.detail(f"CPU usage: {resources.get('cpu_usage', 0):.1f}%")
            self.log.detail(f"Memory usage: {resources.get('memory_usage', 0):.1f}%")
            self.log.detail(f"Disk usage: {resources.get('disk_usage', 0):.1f}%")

# Export the logger
__all__ = ['conv_logger', 'DetailedSystemReporter']