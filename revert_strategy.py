#!/usr/bin/env python3
"""
Strategy Reversion System
Automatically reverts to last working strategy versions when compilation/runtime errors occur
"""
import os
import json
import shutil
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import hashlib
import sqlite3

class StrategyVersionManager:
    """Manages strategy versions and provides reversion capabilities"""
    
    def __init__(self, strategies_dir: str = "strategies", versions_dir: str = "strategy_versions"):
        self.strategies_dir = Path(strategies_dir)
        self.versions_dir = Path(versions_dir)
        self.versions_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize version database
        self.db_path = self.versions_dir / "strategy_versions.db"
        self._init_version_database()
        
        self.reversion_history = []
        
    def _init_version_database(self):
        """Initialize SQLite database for version tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_versions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        version_number INTEGER NOT NULL,
                        file_path TEXT NOT NULL,
                        version_path TEXT NOT NULL,
                        checksum TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        compilation_status TEXT DEFAULT 'unknown',
                        backtest_results TEXT,
                        performance_metrics TEXT,
                        is_working BOOLEAN DEFAULT TRUE,
                        created_by TEXT DEFAULT 'system',
                        notes TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS strategy_reversions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        strategy_name TEXT NOT NULL,
                        from_version INTEGER,
                        to_version INTEGER NOT NULL,
                        reversion_reason TEXT,
                        timestamp TEXT NOT NULL,
                        success BOOLEAN,
                        error_message TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_strategy_name ON strategy_versions(strategy_name)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON strategy_versions(timestamp)
                ''')
        
        except Exception as e:
            logger.error(f"Failed to initialize version database: {e}")
    
    def save_strategy_version(self, strategy_file: str, notes: str = "", performance_metrics: Dict[str, Any] = None) -> bool:
        """Save a new version of a strategy"""
        try:
            strategy_path = Path(strategy_file)
            if not strategy_path.exists():
                logger.error(f"Strategy file not found: {strategy_file}")
                return False
            
            strategy_name = strategy_path.stem
            
            # Read strategy content
            with open(strategy_path, 'r') as f:
                content = f.read()
            
            # Calculate checksum
            checksum = hashlib.md5(content.encode()).hexdigest()
            
            # Get next version number
            version_number = self._get_next_version_number(strategy_name)
            
            # Create version directory
            version_dir = self.versions_dir / strategy_name
            version_dir.mkdir(parents=True, exist_ok=True)
            
            # Save version file
            version_filename = f"{strategy_name}_v{version_number}.py"
            version_path = version_dir / version_filename
            
            shutil.copy2(strategy_path, version_path)
            
            # Test compilation
            compilation_status = self._test_compilation(version_path)
            
            # Save to database
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO strategy_versions 
                    (strategy_name, version_number, file_path, version_path, checksum, 
                     timestamp, compilation_status, performance_metrics, notes)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_name,
                    version_number,
                    str(strategy_path),
                    str(version_path),
                    checksum,
                    datetime.now().isoformat(),
                    compilation_status,
                    json.dumps(performance_metrics) if performance_metrics else None,
                    notes
                ))
            
            logger.success(f"✅ Saved strategy version: {strategy_name} v{version_number}")
            logger.info(f"   ├─ Compilation: {compilation_status}")
            logger.info(f"   ├─ File: {version_path}")
            logger.info(f"   └─ Checksum: {checksum[:8]}...")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to save strategy version: {e}")
            return False
    
    def _get_next_version_number(self, strategy_name: str) -> int:
        """Get the next version number for a strategy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute(
                    'SELECT MAX(version_number) FROM strategy_versions WHERE strategy_name = ?',
                    (strategy_name,)
                )
                result = cursor.fetchone()
                return (result[0] if result[0] is not None else 0) + 1
        except Exception:
            return 1
    
    def _test_compilation(self, strategy_path: Path) -> str:
        """Test if strategy compiles successfully"""
        try:
            with open(strategy_path, 'r') as f:
                code = f.read()
            
            compile(code, str(strategy_path), 'exec')
            return 'success'
            
        except SyntaxError as e:
            return f'syntax_error: {str(e)}'
        except Exception as e:
            return f'error: {str(e)}'
    
    def revert_to_last_working_version(self, strategy_name: str) -> bool:
        """Revert strategy to last known working version"""
        logger.info(f"🔄 Reverting {strategy_name} to last working version...")
        
        try:
            # Find last working version
            last_working = self._find_last_working_version(strategy_name)
            
            if not last_working:
                logger.error(f"❌ No working version found for {strategy_name}")
                return False
            
            # Get current version for comparison
            current_version = self._get_current_version(strategy_name)
            
            # Perform reversion
            success = self._perform_reversion(strategy_name, last_working, current_version)
            
            # Record reversion
            self._record_reversion(
                strategy_name=strategy_name,
                from_version=current_version.get('version_number') if current_version else None,
                to_version=last_working['version_number'],
                reason='compilation_or_runtime_error',
                success=success
            )
            
            if success:
                logger.success(f"✅ Successfully reverted {strategy_name} to version {last_working['version_number']}")
                logger.info(f"   ├─ Timestamp: {last_working['timestamp']}")
                logger.info(f"   ├─ Compilation: {last_working['compilation_status']}")
                logger.info(f"   └─ Notes: {last_working.get('notes', 'None')}")
            else:
                logger.error(f"❌ Failed to revert {strategy_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reverting strategy {strategy_name}: {e}")
            return False
    
    def _find_last_working_version(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Find the last known working version of a strategy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM strategy_versions 
                    WHERE strategy_name = ? 
                    AND compilation_status = 'success'
                    AND is_working = TRUE
                    ORDER BY version_number DESC
                    LIMIT 1
                ''', (strategy_name,))
                
                result = cursor.fetchone()
                if result:
                    return dict(result)
                
                # If no explicitly working version, find last successful compilation
                cursor = conn.execute('''
                    SELECT * FROM strategy_versions 
                    WHERE strategy_name = ? 
                    AND compilation_status = 'success'
                    ORDER BY version_number DESC
                    LIMIT 1
                ''', (strategy_name,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Error finding last working version: {e}")
            return None
    
    def _get_current_version(self, strategy_name: str) -> Optional[Dict[str, Any]]:
        """Get current version information"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM strategy_versions 
                    WHERE strategy_name = ? 
                    ORDER BY version_number DESC
                    LIMIT 1
                ''', (strategy_name,))
                
                result = cursor.fetchone()
                return dict(result) if result else None
                
        except Exception as e:
            logger.error(f"Error getting current version: {e}")
            return None
    
    def _perform_reversion(self, strategy_name: str, target_version: Dict[str, Any], current_version: Optional[Dict[str, Any]]) -> bool:
        """Perform the actual file reversion"""
        try:
            target_version_path = Path(target_version['version_path'])
            current_file_path = Path(target_version['file_path'])
            
            if not target_version_path.exists():
                logger.error(f"Target version file not found: {target_version_path}")
                return False
            
            # Create backup of current file if it exists
            if current_file_path.exists():
                backup_path = current_file_path.with_suffix('.backup')
                shutil.copy2(current_file_path, backup_path)
                logger.debug(f"Created backup: {backup_path}")
            
            # Copy target version to current location
            current_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target_version_path, current_file_path)
            
            # Verify the reversion
            with open(current_file_path, 'r') as f:
                content = f.read()
            
            actual_checksum = hashlib.md5(content.encode()).hexdigest()
            expected_checksum = target_version['checksum']
            
            if actual_checksum != expected_checksum:
                logger.warning(f"Checksum mismatch after reversion (expected: {expected_checksum}, actual: {actual_checksum})")
                return False
            
            # Test compilation of reverted file
            compilation_result = self._test_compilation(current_file_path)
            if compilation_result != 'success':
                logger.warning(f"Reverted file has compilation issues: {compilation_result}")
                # Don't fail the reversion, but log the issue
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing reversion: {e}")
            return False
    
    def _record_reversion(self, strategy_name: str, from_version: Optional[int], to_version: int, 
                         reason: str, success: bool, error_message: str = None):
        """Record reversion in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO strategy_reversions 
                    (strategy_name, from_version, to_version, reversion_reason, timestamp, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    strategy_name,
                    from_version,
                    to_version,
                    reason,
                    datetime.now().isoformat(),
                    success,
                    error_message
                ))
                
        except Exception as e:
            logger.error(f"Failed to record reversion: {e}")
    
    def list_strategy_versions(self, strategy_name: str) -> List[Dict[str, Any]]:
        """List all versions of a strategy"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM strategy_versions 
                    WHERE strategy_name = ? 
                    ORDER BY version_number DESC
                ''', (strategy_name,))
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error listing strategy versions: {e}")
            return []
    
    def revert_to_specific_version(self, strategy_name: str, version_number: int) -> bool:
        """Revert to a specific version number"""
        logger.info(f"🔄 Reverting {strategy_name} to version {version_number}...")
        
        try:
            # Find target version
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM strategy_versions 
                    WHERE strategy_name = ? AND version_number = ?
                ''', (strategy_name, version_number))
                
                target_version = cursor.fetchone()
                if not target_version:
                    logger.error(f"❌ Version {version_number} not found for {strategy_name}")
                    return False
                
                target_version = dict(target_version)
            
            # Get current version
            current_version = self._get_current_version(strategy_name)
            
            # Perform reversion
            success = self._perform_reversion(strategy_name, target_version, current_version)
            
            # Record reversion
            self._record_reversion(
                strategy_name=strategy_name,
                from_version=current_version.get('version_number') if current_version else None,
                to_version=version_number,
                reason='manual_reversion',
                success=success
            )
            
            if success:
                logger.success(f"✅ Successfully reverted {strategy_name} to version {version_number}")
            else:
                logger.error(f"❌ Failed to revert {strategy_name} to version {version_number}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error reverting to specific version: {e}")
            return False
    
    def mark_version_as_working(self, strategy_name: str, version_number: int, 
                               performance_metrics: Dict[str, Any] = None) -> bool:
        """Mark a version as working with optional performance metrics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE strategy_versions 
                    SET is_working = TRUE, performance_metrics = ?
                    WHERE strategy_name = ? AND version_number = ?
                ''', (
                    json.dumps(performance_metrics) if performance_metrics else None,
                    strategy_name,
                    version_number
                ))
                
                if conn.total_changes > 0:
                    logger.success(f"✅ Marked {strategy_name} v{version_number} as working")
                    return True
                else:
                    logger.warning(f"⚠️ Version {version_number} not found for {strategy_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error marking version as working: {e}")
            return False
    
    def mark_version_as_broken(self, strategy_name: str, version_number: int, reason: str = "") -> bool:
        """Mark a version as broken"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    UPDATE strategy_versions 
                    SET is_working = FALSE, notes = ?
                    WHERE strategy_name = ? AND version_number = ?
                ''', (reason, strategy_name, version_number))
                
                if conn.total_changes > 0:
                    logger.info(f"Marked {strategy_name} v{version_number} as broken: {reason}")
                    return True
                else:
                    logger.warning(f"Version {version_number} not found for {strategy_name}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error marking version as broken: {e}")
            return False
    
    def get_reversion_history(self, strategy_name: str = None) -> List[Dict[str, Any]]:
        """Get reversion history"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                
                if strategy_name:
                    cursor = conn.execute('''
                        SELECT * FROM strategy_reversions 
                        WHERE strategy_name = ? 
                        ORDER BY timestamp DESC
                    ''', (strategy_name,))
                else:
                    cursor = conn.execute('''
                        SELECT * FROM strategy_reversions 
                        ORDER BY timestamp DESC
                    ''')
                
                return [dict(row) for row in cursor.fetchall()]
                
        except Exception as e:
            logger.error(f"Error getting reversion history: {e}")
            return []
    
    def cleanup_old_versions(self, strategy_name: str = None, keep_versions: int = 10):
        """Cleanup old versions, keeping only the specified number"""
        try:
            strategies_to_clean = [strategy_name] if strategy_name else self._get_all_strategy_names()
            
            for strat_name in strategies_to_clean:
                versions = self.list_strategy_versions(strat_name)
                
                if len(versions) > keep_versions:
                    # Keep working versions and recent versions
                    versions_to_delete = []
                    working_versions = [v for v in versions if v.get('is_working')]
                    recent_versions = versions[:keep_versions]
                    
                    for version in versions[keep_versions:]:
                        if version not in working_versions and version not in recent_versions:
                            versions_to_delete.append(version)
                    
                    for version in versions_to_delete:
                        try:
                            # Delete version file
                            version_path = Path(version['version_path'])
                            if version_path.exists():
                                version_path.unlink()
                            
                            # Remove from database
                            with sqlite3.connect(self.db_path) as conn:
                                conn.execute(
                                    'DELETE FROM strategy_versions WHERE id = ?',
                                    (version['id'],)
                                )
                            
                            logger.debug(f"Deleted old version: {strat_name} v{version['version_number']}")
                            
                        except Exception as e:
                            logger.warning(f"Failed to delete version {version['id']}: {e}")
                    
                    if versions_to_delete:
                        logger.info(f"🗑️ Cleaned up {len(versions_to_delete)} old versions of {strat_name}")
                        
        except Exception as e:
            logger.error(f"Error cleaning up old versions: {e}")
    
    def _get_all_strategy_names(self) -> List[str]:
        """Get all strategy names from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute('SELECT DISTINCT strategy_name FROM strategy_versions')
                return [row[0] for row in cursor.fetchall()]
        except Exception:
            return []

def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Strategy version management and reversion")
    parser.add_argument("--last-working", help="Revert strategy to last working version")
    parser.add_argument("--version", type=int, help="Revert to specific version number")
    parser.add_argument("--strategy", required=True, help="Strategy name")
    parser.add_argument("--save", action="store_true", help="Save current version")
    parser.add_argument("--list", action="store_true", help="List all versions")
    parser.add_argument("--mark-working", type=int, help="Mark version as working")
    parser.add_argument("--mark-broken", type=int, help="Mark version as broken")
    parser.add_argument("--cleanup", action="store_true", help="Cleanup old versions")
    parser.add_argument("--notes", help="Notes for saving version")
    
    args = parser.parse_args()
    
    manager = StrategyVersionManager()
    
    if args.save:
        strategy_file = f"strategies/{args.strategy}.py"
        success = manager.save_strategy_version(strategy_file, args.notes or "Manual save")
        if success:
            print(f"✅ Saved version of {args.strategy}")
            return True
        else:
            print(f"❌ Failed to save version of {args.strategy}")
            return False
    
    elif args.last_working:
        success = manager.revert_to_last_working_version(args.strategy)
        if success:
            print(f"✅ Reverted {args.strategy} to last working version")
            return True
        else:
            print(f"❌ Failed to revert {args.strategy}")
            return False
    
    elif args.version:
        success = manager.revert_to_specific_version(args.strategy, args.version)
        if success:
            print(f"✅ Reverted {args.strategy} to version {args.version}")
            return True
        else:
            print(f"❌ Failed to revert {args.strategy} to version {args.version}")
            return False
    
    elif args.list:
        versions = manager.list_strategy_versions(args.strategy)
        print(f"Versions of {args.strategy}:")
        for version in versions:
            working_icon = "✅" if version.get('is_working') else "❌"
            compile_icon = "✅" if version['compilation_status'] == 'success' else "❌"
            print(f"  v{version['version_number']} {working_icon} {compile_icon} - {version['timestamp']}")
            if version.get('notes'):
                print(f"    Notes: {version['notes']}")
        return True
    
    elif args.mark_working:
        success = manager.mark_version_as_working(args.strategy, args.mark_working)
        if success:
            print(f"✅ Marked {args.strategy} v{args.mark_working} as working")
            return True
        else:
            print(f"❌ Failed to mark version as working")
            return False
    
    elif args.mark_broken:
        reason = args.notes or "Marked as broken"
        success = manager.mark_version_as_broken(args.strategy, args.mark_broken, reason)
        if success:
            print(f"✅ Marked {args.strategy} v{args.mark_broken} as broken")
            return True
        else:
            print(f"❌ Failed to mark version as broken")
            return False
    
    elif args.cleanup:
        manager.cleanup_old_versions(args.strategy)
        print(f"✅ Cleaned up old versions of {args.strategy}")
        return True
    
    else:
        print("No action specified. Use --help for options.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)