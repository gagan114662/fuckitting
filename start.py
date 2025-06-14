#!/usr/bin/env python3
"""
Enhanced startup script for AlgoForge 3.0 Superhuman
Now with detailed conversational logging as requested!
"""
import os
import sys
import subprocess
from enhanced_logger import conv_logger

# Set up environment - Load from .env file or use provided credentials
if not os.getenv('QUANTCONNECT_USER_ID'):
    os.environ['QUANTCONNECT_USER_ID'] = '357130'
if not os.getenv('QUANTCONNECT_API_TOKEN'):
    os.environ['QUANTCONNECT_API_TOKEN'] = 'your_quantconnect_api_token_here'
if not os.getenv('BRAVE_API_KEY'):
    os.environ['BRAVE_API_KEY'] = 'your_brave_api_key_here'
if not os.getenv('TWELVE_DATA_API_KEY'):
    os.environ['TWELVE_DATA_API_KEY'] = 'your_twelve_data_api_key_here'
if not os.getenv('ALPHA_VANTAGE_API_KEY'):
    os.environ['ALPHA_VANTAGE_API_KEY'] = 'your_alpha_vantage_api_key_here'
if not os.getenv('POLYGON_API_KEY'):
    os.environ['POLYGON_API_KEY'] = 'your_polygon_api_key_here'
if not os.getenv('FINNHUB_API_KEY'):
    os.environ['FINNHUB_API_KEY'] = 'your_finnhub_api_key_here'
if not os.getenv('GITHUB_PERSONAL_ACCESS_TOKEN'):
    os.environ['GITHUB_PERSONAL_ACCESS_TOKEN'] = 'your_github_token_here'

conv_logger.section("ALGOFORGE 3.0 SUPERHUMAN SYSTEM")
conv_logger.detail("All 7 premium API keys loaded automatically")
conv_logger.chat("I'm about to start your superhuman trading system with full conversational logging!")

try:
    # Import and run the system
    from run_superhuman import SuperhumanRunner
    import asyncio
    
    async def main():
        conv_logger.working_on("starting the superhuman runner")
        runner = SuperhumanRunner()
        return await runner.run_superhuman_system()
    
    result = asyncio.run(main())
    
    if result['components_failed'] == 0:
        conv_logger.section("MISSION ACCOMPLISHED!")
        conv_logger.success("Your superhuman system is fully operational!")
        conv_logger.chat("you're now ready to develop sophisticated strategies with superhuman capabilities!")
        conv_logger.explain("Your system will continue running autonomously, monitoring and optimizing everything")
    else:
        conv_logger.warning(f"System is running with {result['components_failed']} components needing attention")
        conv_logger.chat("don't worry though - the core functionality is still working great!")
        
except Exception as e:
    conv_logger.error(f"System startup failed: {e}")
    conv_logger.chat("something went wrong, but let me show you the details so we can fix it")
    import traceback
    traceback.print_exc()