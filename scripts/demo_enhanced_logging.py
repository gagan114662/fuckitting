#!/usr/bin/env python3
"""
Demo of Enhanced Conversational Logging
This shows you exactly what the logging will look like when you run the system
"""
import asyncio
import time
import psutil
from enhanced_logger import conv_logger, DetailedSystemReporter

async def demo_enhanced_logging():
    """Demonstrate the enhanced conversational logging system"""
    
    # Initialize reporter
    reporter = DetailedSystemReporter(conv_logger)
    
    # Show greeting
    conv_logger.greet()
    
    # Demo API configuration
    conv_logger.section("API CONFIGURATION")
    conv_logger.thinking("checking all your API keys")
    
    apis = [
        ("QuantConnect", True),
        ("Brave Search", True),
        ("Twelve Data", True),
        ("Alpha Vantage", True),
        ("Polygon", True),
        ("Finnhub", True),
        ("GitHub", True)
    ]
    
    for api_name, has_key in apis:
        await reporter.report_api_initialization(api_name, has_key, testing=True)
        time.sleep(0.3)
    
    conv_logger.success("All 7 APIs are configured and working!")
    conv_logger.chat("you now have access to multiple data sources with automatic failover!")
    
    # Demo component initialization
    conv_logger.section("COMPONENT INITIALIZATION")
    
    components = [
        ("Autonomous System Manager", "This is your main self-healing brain that monitors everything"),
        ("Auto Code Fixer", "Automatically fixes any code errors using Claude Code SDK"),
        ("Data Source Manager", "Switches between your 7 data sources when one fails"),
        ("Risk Manager", "Adjusts risk parameters based on market conditions"),
        ("QuantConnect Sync Manager", "Solves your sync bottleneck with advanced rate limiting")
    ]
    
    for comp_name, description in components:
        await reporter.report_component_startup(comp_name, description)
        time.sleep(0.4)
    
    # Demo sync status
    conv_logger.section("QUANTCONNECT SYNC STATUS")
    conv_logger.explain("Remember how you said the rate limits were a problem? Here's how I solved it!")
    
    sync_info = {
        'requests_allowed': 25,
        'files_synced': 12,
        'conflicts': 0
    }
    
    reporter.report_sync_status(sync_info)
    
    # Demo data sources
    conv_logger.section("DATA SOURCE INTELLIGENCE")
    
    sources = {
        'quantconnect': {'is_active': True, 'priority': 1, 'rate_limit_per_minute': 30},
        'twelve_data': {'is_active': True, 'priority': 2, 'rate_limit_per_minute': 800},
        'alpha_vantage': {'is_active': True, 'priority': 3, 'rate_limit_per_minute': 5},
        'yahoo_finance': {'is_active': True, 'priority': 4, 'rate_limit_per_minute': 60},
        'polygon': {'is_active': True, 'priority': 5, 'rate_limit_per_minute': 200},
        'finnhub': {'is_active': True, 'priority': 7, 'rate_limit_per_minute': 60}
    }
    
    reporter.report_data_source_status(sources)
    
    # Demo health check
    conv_logger.section("SYSTEM HEALTH CHECK")
    
    # Calculate real health score based on actual system metrics
    cpu = psutil.cpu_percent(interval=0.1)
    memory = psutil.virtual_memory().percent
    disk = psutil.disk_usage('.').percent
    
    # Real health calculation: lower is better for resource usage
    health_score = max(0, 1.0 - (cpu + memory + disk) / 300.0)
    
    health_data = {
        'overall_health_score': health_score,
        'system_resources': {
            'cpu_usage': psutil.cpu_percent(interval=0.1),
            'memory_usage': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent
        }
    }
    
    reporter.report_health_check(health_data)
    
    # Demo autonomous capabilities
    conv_logger.section("AUTONOMOUS CAPABILITIES")
    conv_logger.chat("let me show you what your system can do automatically!")
    
    capabilities = [
        "Monitor system health continuously",
        "Fix code errors using Claude SDK automatically", 
        "Switch data sources when APIs fail",
        "Adjust risk parameters based on market volatility",
        "Retrain ML models when performance drops",
        "Back up strategies automatically",
        "Sync with QuantConnect without rate limit issues",
        "Clean up system resources",
        "Detect and resolve conflicts"
    ]
    
    conv_logger.summary("What Your System Does Automatically", capabilities)
    
    # Final message
    conv_logger.section("DEMONSTRATION COMPLETE")
    conv_logger.finale(success=True)
    conv_logger.chat("this is exactly what you'll see when you run your actual system!")
    conv_logger.explain("Every step is explained in plain English, just like having a friendly assistant")
    conv_logger.elapsed_time()

if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎭 ENHANCED LOGGING DEMONSTRATION")
    print("This shows you exactly what the conversational logging looks like!")
    print("="*80 + "\n")
    
    asyncio.run(demo_enhanced_logging())
    
    print("\n" + "="*80)
    print("✅ DEMO COMPLETE!")
    print("Now run: python3 start.py")
    print("To see this same friendly logging in your actual superhuman system!")
    print("="*80)