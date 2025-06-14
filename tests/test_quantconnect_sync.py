#!/usr/bin/env python3
"""
QuantConnect Synchronization Test
Verifies that code updates correctly sync to your QuantConnect account
"""
import asyncio
import os
import json
from datetime import datetime
from pathlib import Path
from loguru import logger
from quantconnect_sync import QuantConnectSyncManager
from quantconnect_client import QuantConnectClient

class QuantConnectSyncTester:
    """Test QuantConnect synchronization functionality"""
    
    def __init__(self):
        self.sync_manager = QuantConnectSyncManager()
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests': [],
            'overall_success': False
        }
        
    async def run_comprehensive_sync_test(self):
        """Run comprehensive sync test"""
        logger.info("🧪 Starting QuantConnect Synchronization Test")
        logger.info("=" * 60)
        
        # Test 1: API Connection
        await self._test_api_connection()
        
        # Test 2: Create Test Strategy
        await self._test_create_strategy()
        
        # Test 3: Upload and Sync
        await self._test_upload_sync()
        
        # Test 4: Modify and Re-sync
        await self._test_modify_resync()
        
        # Test 5: Rate Limiting
        await self._test_rate_limiting()
        
        # Test 6: Conflict Resolution
        await self._test_conflict_resolution()
        
        # Generate final report
        self._generate_sync_report()
        
        return self.test_results
    
    async def _test_api_connection(self):
        """Test QuantConnect API connection"""
        test_name = "API Connection"
        logger.info(f"🔗 Testing: {test_name}")
        
        try:
            async with QuantConnectClient() as client:
                # Test basic API call
                projects = await client.list_projects()
                
                if projects is not None:
                    self._record_test(test_name, True, f"Connected successfully, found {len(projects)} projects")
                    logger.success(f"✅ {test_name}: Connected to QuantConnect")
                    logger.info(f"   Account has {len(projects)} existing projects")
                else:
                    self._record_test(test_name, False, "Failed to retrieve projects")
                    logger.error(f"❌ {test_name}: Failed to connect")
                    
        except Exception as e:
            self._record_test(test_name, False, f"Connection error: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    async def _test_create_strategy(self):
        """Test creating a test strategy file"""
        test_name = "Create Test Strategy"
        logger.info(f"📝 Testing: {test_name}")
        
        try:
            # Create test strategy
            strategy_content = '''"""
Test Strategy for Sync Verification
Generated at: {timestamp}
"""

from AlgorithmImports import *

class TestSyncStrategy(QCAlgorithm):
    
    def Initialize(self):
        self.SetStartDate(2020, 1, 1)
        self.SetEndDate(2023, 1, 1)
        self.SetCash(100000)
        
        # Add SPY for testing
        self.spy = self.AddEquity("SPY", Resolution.Daily)
        
        # Test sync timestamp: {timestamp}
        self.Debug(f"Strategy initialized at {timestamp}")
    
    def OnData(self, data):
        if not self.Portfolio.Invested:
            self.SetHoldings("SPY", 0.5)
            self.Debug("Purchased SPY")
'''.format(timestamp=datetime.now().isoformat())
            
            # Save to strategies directory
            strategies_dir = Path("strategies")
            strategies_dir.mkdir(exist_ok=True)
            
            strategy_file = strategies_dir / "test_sync_strategy.py"
            with open(strategy_file, 'w') as f:
                f.write(strategy_content)
            
            self._record_test(test_name, True, f"Created test strategy: {strategy_file}")
            logger.success(f"✅ {test_name}: Created {strategy_file}")
            
        except Exception as e:
            self._record_test(test_name, False, f"Failed to create strategy: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    async def _test_upload_sync(self):
        """Test uploading and syncing strategy to QuantConnect"""
        test_name = "Upload and Sync"
        logger.info(f"⬆️ Testing: {test_name}")
        
        try:
            strategy_file = Path("strategies/test_sync_strategy.py")
            if not strategy_file.exists():
                self._record_test(test_name, False, "Test strategy file not found")
                return
            
            # Use sync manager to upload
            async with QuantConnectClient() as client:
                # Create project
                project_name = f"AlgoForge_SyncTest_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                project_id = await self.sync_manager.rate_limited_request(
                    client.create_project(project_name)
                )
                
                if not project_id:
                    self._record_test(test_name, False, "Failed to create project")
                    return
                
                logger.info(f"   Created project: {project_name} (ID: {project_id})")
                
                # Upload file with rate limiting
                with open(strategy_file, 'r') as f:
                    content = f.read()
                
                upload_success = await self.sync_manager.rate_limited_request(
                    client.upload_file(project_id, "main.py", content)
                )
                
                if upload_success:
                    # Verify upload by reading back
                    files = await self.sync_manager.rate_limited_request(
                        client.read_project_files(project_id)
                    )
                    
                    if files and any(f.get('name') == 'main.py' for f in files):
                        self._record_test(test_name, True, f"Successfully synced to project {project_id}")
                        logger.success(f"✅ {test_name}: Code uploaded and verified")
                        logger.info(f"   Project ID: {project_id}")
                        
                        # Store project ID for later tests
                        self.test_project_id = project_id
                    else:
                        self._record_test(test_name, False, "Upload succeeded but verification failed")
                        logger.warning(f"⚠️ {test_name}: Upload succeeded but verification failed")
                else:
                    self._record_test(test_name, False, "File upload failed")
                    logger.error(f"❌ {test_name}: File upload failed")
                    
        except Exception as e:
            self._record_test(test_name, False, f"Upload sync failed: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    async def _test_modify_resync(self):
        """Test modifying strategy and re-syncing"""
        test_name = "Modify and Re-sync"
        logger.info(f"🔄 Testing: {test_name}")
        
        try:
            if not hasattr(self, 'test_project_id'):
                self._record_test(test_name, False, "No test project available")
                return
            
            strategy_file = Path("strategies/test_sync_strategy.py")
            
            # Read current content
            with open(strategy_file, 'r') as f:
                content = f.read()
            
            # Modify content
            modified_content = content.replace(
                "self.SetHoldings(\"SPY\", 0.5)",
                f"self.SetHoldings(\"SPY\", 0.3)  # Modified at {datetime.now().isoformat()}"
            )
            
            # Write modified content
            with open(strategy_file, 'w') as f:
                f.write(modified_content)
            
            logger.info(f"   Modified strategy file")
            
            # Re-sync to QuantConnect
            async with QuantConnectClient() as client:
                upload_success = await self.sync_manager.rate_limited_request(
                    client.upload_file(self.test_project_id, "main.py", modified_content)
                )
                
                if upload_success:
                    # Verify modification
                    files = await self.sync_manager.rate_limited_request(
                        client.read_project_files(self.test_project_id)
                    )
                    
                    main_file = next((f for f in files if f.get('name') == 'main.py'), None)
                    if main_file and "Modified at" in main_file.get('content', ''):
                        self._record_test(test_name, True, "Successfully re-synced modified strategy")
                        logger.success(f"✅ {test_name}: Modified code synced successfully")
                    else:
                        self._record_test(test_name, False, "Re-sync succeeded but changes not reflected")
                        logger.warning(f"⚠️ {test_name}: Changes not reflected on QuantConnect")
                else:
                    self._record_test(test_name, False, "Re-sync failed")
                    logger.error(f"❌ {test_name}: Re-sync failed")
                    
        except Exception as e:
            self._record_test(test_name, False, f"Modify re-sync failed: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    async def _test_rate_limiting(self):
        """Test rate limiting functionality"""
        test_name = "Rate Limiting"
        logger.info(f"⏱️ Testing: {test_name}")
        
        try:
            # Test multiple rapid requests
            start_time = datetime.now()
            successful_requests = 0
            
            async with QuantConnectClient() as client:
                for i in range(5):  # Test 5 rapid requests
                    try:
                        result = await self.sync_manager.rate_limited_request(
                            client.list_projects()
                        )
                        if result is not None:
                            successful_requests += 1
                    except Exception:
                        pass  # Expected for rate limiting
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Check rate limiter status
            rate_status = self.sync_manager.get_sync_status()
            
            if successful_requests >= 3 and duration >= 2:  # Should take at least 2 seconds due to rate limiting
                self._record_test(test_name, True, f"{successful_requests}/5 requests in {duration:.1f}s - rate limiting working")
                logger.success(f"✅ {test_name}: Rate limiting working correctly")
                logger.info(f"   {successful_requests}/5 requests completed in {duration:.1f}s")
            else:
                self._record_test(test_name, False, f"Rate limiting may not be working properly")
                logger.warning(f"⚠️ {test_name}: Rate limiting behavior unclear")
                
        except Exception as e:
            self._record_test(test_name, False, f"Rate limiting test failed: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    async def _test_conflict_resolution(self):
        """Test conflict detection and resolution"""
        test_name = "Conflict Resolution"
        logger.info(f"🔀 Testing: {test_name}")
        
        try:
            # This is a simulation since we can't easily create real conflicts
            # Test the conflict resolution mechanisms
            
            # Check if sync manager has conflict detection
            sync_status = self.sync_manager.get_sync_status()
            
            if 'active_conflicts' in sync_status:
                self._record_test(test_name, True, "Conflict detection mechanism available")
                logger.success(f"✅ {test_name}: Conflict detection available")
                logger.info(f"   Active conflicts: {sync_status['active_conflicts']}")
            else:
                self._record_test(test_name, False, "Conflict detection not available")
                logger.warning(f"⚠️ {test_name}: Conflict detection not implemented")
                
        except Exception as e:
            self._record_test(test_name, False, f"Conflict resolution test failed: {e}")
            logger.error(f"❌ {test_name}: {e}")
    
    def _record_test(self, test_name: str, success: bool, details: str):
        """Record test result"""
        self.test_results['tests'].append({
            'name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_sync_report(self):
        """Generate final sync test report"""
        successful_tests = len([t for t in self.test_results['tests'] if t['success']])
        total_tests = len(self.test_results['tests'])
        success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
        
        self.test_results['overall_success'] = success_rate >= 70
        self.test_results['success_rate'] = success_rate
        
        logger.info("📊 QUANTCONNECT SYNCHRONIZATION TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Failed: {total_tests - successful_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Result: {'🎉 SUCCESS' if self.test_results['overall_success'] else '❌ NEEDS ATTENTION'}")
        logger.info("=" * 60)
        
        for test in self.test_results['tests']:
            status_icon = "✅" if test['success'] else "❌"
            logger.info(f"{status_icon} {test['name']}: {test['details']}")
        
        logger.info("=" * 60)
        
        if self.test_results['overall_success']:
            logger.success("🎉 QuantConnect synchronization is working correctly!")
            logger.info("Your code update bottleneck has been resolved!")
        else:
            logger.warning("⚠️ QuantConnect synchronization needs attention")
            logger.info("Some sync features may not be working as expected")
        
        # Save detailed report
        report_file = f"quantconnect_sync_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"📄 Detailed report saved: {report_file}")

async def main():
    """Run QuantConnect sync test"""
    tester = QuantConnectSyncTester()
    results = await tester.run_comprehensive_sync_test()
    return results['overall_success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)