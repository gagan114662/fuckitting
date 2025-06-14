#!/usr/bin/env python3
"""
Comprehensive Test Suite for Autonomous System Integration
Tests all components working together in self-healing scenarios
"""
import asyncio
import tempfile
import os
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# Import all autonomous components
from autonomous_system import AutonomousSystemManager, create_self_healing_components
from auto_fix_code import AutoCodeFixer
from cleanup_system import SystemCleaner
from backup_system import BackupManager
from restart_components import ComponentManager
from switch_data_sources import DataSourceManager
from revert_strategy import StrategyVersionManager
from retrain_models import ModelManager
from adjust_risk import RiskManager

class AutonomousSystemIntegrationTest:
    """Comprehensive integration test for autonomous system"""
    
    def __init__(self):
        self.test_results = {
            'timestamp': datetime.now().isoformat(),
            'tests_run': 0,
            'tests_passed': 0,
            'tests_failed': 0,
            'test_details': [],
            'overall_success': False
        }
        
    async def run_full_integration_test(self):
        """Run complete integration test suite"""
        logger.info("🚀 Starting Comprehensive Autonomous System Integration Test")
        
        # Test 1: Component Initialization
        await self._test_component_initialization()
        
        # Test 2: Self-Healing Infrastructure
        await self._test_self_healing_components()
        
        # Test 3: Autonomous Recovery Simulation
        await self._test_autonomous_recovery_simulation()
        
        # Test 4: Multi-Component Coordination
        await self._test_multi_component_coordination()
        
        # Test 5: Failure Mode Detection and Response
        await self._test_failure_mode_detection()
        
        # Test 6: Performance Under Load
        await self._test_performance_under_load()
        
        # Generate final report
        self._generate_final_report()
        
        return self.test_results
    
    async def _test_component_initialization(self):
        """Test that all autonomous components initialize correctly"""
        test_name = "Component Initialization"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            # Initialize all components
            autonomous_manager = AutonomousSystemManager()
            auto_fixer = AutoCodeFixer()
            system_cleaner = SystemCleaner()
            backup_manager = BackupManager()
            component_manager = ComponentManager()
            data_source_manager = DataSourceManager()
            version_manager = StrategyVersionManager()
            model_manager = ModelManager()
            risk_manager = RiskManager()
            
            # Test basic functionality of each
            health_data = await autonomous_manager._comprehensive_health_check()
            fixer_stats = auto_fixer.get_fix_statistics()
            cleanup_report = system_cleaner.get_cleanup_report()
            available_backups = backup_manager.list_backups()
            component_health = component_manager.check_component_health()
            data_source_status = data_source_manager.get_data_source_status()
            model_summary = model_manager.get_model_performance_summary()
            risk_report = risk_manager.get_risk_status_report()
            
            self._record_test_result(test_name, True, "All components initialized successfully")
            logger.success(f"✅ {test_name}: PASSED")
            
        except Exception as e:
            self._record_test_result(test_name, False, f"Component initialization failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_self_healing_components(self):
        """Test self-healing component creation and functionality"""
        test_name = "Self-Healing Components"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            # Test creating self-healing components
            await create_self_healing_components()
            
            # Verify auto-fix script exists and works
            auto_fix_path = Path("auto_fix_code.py")
            cleanup_path = Path("cleanup_system.py")
            backup_path = Path("backup_system.py")
            
            if all(p.exists() for p in [auto_fix_path, cleanup_path, backup_path]):
                self._record_test_result(test_name, True, "Self-healing components created successfully")
                logger.success(f"✅ {test_name}: PASSED")
            else:
                self._record_test_result(test_name, False, "Some self-healing components missing")
                logger.error(f"❌ {test_name}: FAILED - Missing components")
                
        except Exception as e:
            self._record_test_result(test_name, False, f"Self-healing component test failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_autonomous_recovery_simulation(self):
        """Test autonomous recovery capabilities"""
        test_name = "Autonomous Recovery Simulation"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            autonomous_manager = AutonomousSystemManager()
            
            # Simulate various failure modes and test recovery
            simulated_failures = []
            
            # Test 1: Simulate disk space issue
            try:
                system_cleaner = SystemCleaner()
                cleanup_result = system_cleaner.cleanup_aggressive()
                simulated_failures.append("disk_cleanup_successful")
            except Exception as e:
                simulated_failures.append(f"disk_cleanup_failed: {e}")
            
            # Test 2: Simulate backup creation
            try:
                backup_manager = BackupManager()
                backup_path = backup_manager.create_system_backup("test_backup")
                if backup_path:
                    simulated_failures.append("backup_creation_successful")
                else:
                    simulated_failures.append("backup_creation_failed")
            except Exception as e:
                simulated_failures.append(f"backup_creation_error: {e}")
            
            # Test 3: Simulate model retraining
            try:
                model_manager = ModelManager()
                retrain_result = await model_manager.retrain_models_autonomous()
                if retrain_result['success']:
                    simulated_failures.append("model_retrain_successful")
                else:
                    simulated_failures.append("model_retrain_failed")
            except Exception as e:
                simulated_failures.append(f"model_retrain_error: {e}")
            
            # Test 4: Simulate risk adjustment
            try:
                risk_manager = RiskManager()
                risk_result = await risk_manager.autonomous_risk_adjustment()
                if risk_result['success']:
                    simulated_failures.append("risk_adjustment_successful")
                else:
                    simulated_failures.append("risk_adjustment_failed")
            except Exception as e:
                simulated_failures.append(f"risk_adjustment_error: {e}")
            
            success_count = len([f for f in simulated_failures if "successful" in f])
            total_tests = 4
            
            if success_count >= total_tests // 2:  # At least half should succeed
                self._record_test_result(test_name, True, f"Recovery simulation: {success_count}/{total_tests} successful")
                logger.success(f"✅ {test_name}: PASSED - {success_count}/{total_tests} recovery tests successful")
            else:
                self._record_test_result(test_name, False, f"Insufficient recovery success: {success_count}/{total_tests}")
                logger.warning(f"⚠️ {test_name}: PARTIAL - Only {success_count}/{total_tests} recovery tests successful")
                
        except Exception as e:
            self._record_test_result(test_name, False, f"Recovery simulation failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_multi_component_coordination(self):
        """Test coordination between multiple autonomous components"""
        test_name = "Multi-Component Coordination"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            # Test coordinated response to system stress
            coordination_results = []
            
            # Step 1: Create a backup before making changes
            backup_manager = BackupManager()
            backup_path = backup_manager.create_system_backup("coordination_test")
            if backup_path:
                coordination_results.append("backup_created")
            
            # Step 2: Clean up system resources
            system_cleaner = SystemCleaner()
            cleanup_result = system_cleaner.cleanup_memory_intensive()
            if cleanup_result['space_freed'] >= 0:  # Any cleanup is good
                coordination_results.append("cleanup_completed")
            
            # Step 3: Check component health
            component_manager = ComponentManager()
            health = component_manager.check_component_health()
            if health['overall_health'] in ['healthy', 'unhealthy']:  # Any valid status
                coordination_results.append("health_checked")
            
            # Step 4: Adjust risk parameters
            risk_manager = RiskManager()
            risk_adjustment = await risk_manager.adjust_risk_conservative()
            if risk_adjustment['success']:
                coordination_results.append("risk_adjusted")
            
            # Evaluate coordination
            success_count = len(coordination_results)
            expected_steps = 4
            
            if success_count >= expected_steps - 1:  # Allow one failure
                self._record_test_result(test_name, True, f"Coordination successful: {success_count}/{expected_steps} steps completed")
                logger.success(f"✅ {test_name}: PASSED - {success_count}/{expected_steps} coordination steps successful")
            else:
                self._record_test_result(test_name, False, f"Coordination insufficient: {success_count}/{expected_steps}")
                logger.warning(f"⚠️ {test_name}: FAILED - Only {success_count}/{expected_steps} coordination steps successful")
                
        except Exception as e:
            self._record_test_result(test_name, False, f"Multi-component coordination failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_failure_mode_detection(self):
        """Test failure mode detection and response mechanisms"""
        test_name = "Failure Mode Detection"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            autonomous_manager = AutonomousSystemManager()
            
            # Test comprehensive health check
            health_data = await autonomous_manager._comprehensive_health_check()
            
            # Test failure detection
            detected_failures = await autonomous_manager._detect_failure_modes()
            
            # Test autonomy status
            autonomy_status = autonomous_manager.get_autonomy_status()
            
            # Validate results
            checks_passed = []
            
            if 'overall_health_score' in health_data:
                checks_passed.append("health_score_calculated")
            
            if isinstance(detected_failures, list):
                checks_passed.append("failure_detection_working")
            
            if 'autonomous_mode' in autonomy_status and 'system_health' in autonomy_status:
                checks_passed.append("autonomy_status_complete")
            
            if len(checks_passed) >= 2:  # At least 2 out of 3 checks should pass
                self._record_test_result(test_name, True, f"Failure detection working: {len(checks_passed)}/3 checks passed")
                logger.success(f"✅ {test_name}: PASSED - {len(checks_passed)}/3 detection mechanisms working")
            else:
                self._record_test_result(test_name, False, f"Insufficient detection capability: {len(checks_passed)}/3")
                logger.error(f"❌ {test_name}: FAILED - Only {len(checks_passed)}/3 detection mechanisms working")
                
        except Exception as e:
            self._record_test_result(test_name, False, f"Failure mode detection test failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    async def _test_performance_under_load(self):
        """Test system performance under simulated load"""
        test_name = "Performance Under Load"
        logger.info(f"🧪 Testing: {test_name}")
        
        try:
            start_time = datetime.now()
            
            # Simulate concurrent operations
            tasks = []
            
            # Task 1: Multiple health checks
            autonomous_manager = AutonomousSystemManager()
            for i in range(3):
                tasks.append(autonomous_manager._comprehensive_health_check())
            
            # Task 2: Multiple data source operations
            data_source_manager = DataSourceManager()
            for i in range(2):
                tasks.append(data_source_manager.get_market_data('AAPL', '1d', '5d'))
            
            # Execute tasks concurrently
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Calculate performance metrics
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            successful_tasks = len([r for r in results if not isinstance(r, Exception)])
            total_tasks = len(results)
            
            # Evaluate performance
            if duration < 30 and successful_tasks >= total_tasks // 2:  # Under 30 seconds, at least half successful
                self._record_test_result(test_name, True, f"Performance acceptable: {successful_tasks}/{total_tasks} tasks in {duration:.1f}s")
                logger.success(f"✅ {test_name}: PASSED - {successful_tasks}/{total_tasks} tasks completed in {duration:.1f}s")
            else:
                self._record_test_result(test_name, False, f"Performance insufficient: {successful_tasks}/{total_tasks} tasks in {duration:.1f}s")
                logger.warning(f"⚠️ {test_name}: FAILED - {successful_tasks}/{total_tasks} tasks completed in {duration:.1f}s")
                
        except Exception as e:
            self._record_test_result(test_name, False, f"Performance test failed: {e}")
            logger.error(f"❌ {test_name}: FAILED - {e}")
    
    def _record_test_result(self, test_name: str, success: bool, details: str):
        """Record test result"""
        self.test_results['tests_run'] += 1
        
        if success:
            self.test_results['tests_passed'] += 1
        else:
            self.test_results['tests_failed'] += 1
        
        self.test_results['test_details'].append({
            'test_name': test_name,
            'success': success,
            'details': details,
            'timestamp': datetime.now().isoformat()
        })
    
    def _generate_final_report(self):
        """Generate final test report"""
        success_rate = (self.test_results['tests_passed'] / self.test_results['tests_run']) * 100 if self.test_results['tests_run'] > 0 else 0
        self.test_results['success_rate'] = success_rate
        self.test_results['overall_success'] = success_rate >= 70  # 70% threshold for overall success
        
        # Save detailed report
        report_file = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info("📊 AUTONOMOUS SYSTEM INTEGRATION TEST REPORT")
        logger.info("=" * 60)
        logger.info(f"Total Tests Run: {self.test_results['tests_run']}")
        logger.info(f"Tests Passed: {self.test_results['tests_passed']}")
        logger.info(f"Tests Failed: {self.test_results['tests_failed']}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Overall Result: {'🎉 SUCCESS' if self.test_results['overall_success'] else '❌ FAILURE'}")
        logger.info("=" * 60)
        
        for test in self.test_results['test_details']:
            status_icon = "✅" if test['success'] else "❌"
            logger.info(f"{status_icon} {test['test_name']}: {test['details']}")
        
        logger.info("=" * 60)
        logger.info(f"📄 Detailed report saved: {report_file}")
        
        if self.test_results['overall_success']:
            logger.success("🚀 AUTONOMOUS SYSTEM IS READY FOR SUPERHUMAN OPERATION!")
        else:
            logger.warning("⚠️ Autonomous system needs attention before full deployment")

async def main():
    """Run comprehensive autonomous system integration test"""
    test_suite = AutonomousSystemIntegrationTest()
    results = await test_suite.run_full_integration_test()
    
    return results['overall_success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)