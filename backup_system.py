
#!/usr/bin/env python3
"""
Backup and Recovery System
Automatically backs up critical system state
"""
import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from loguru import logger
from resilience_framework import (
    safe_file_operation, thread_safe, retry_with_backoff,
    RetryConfig, FailureType
)

class BackupManager:
    """System backup and recovery management"""
    
    def __init__(self):
        self.backup_history = []
        self.backup_base_dir = Path("backups")
        self.backup_base_dir.mkdir(parents=True, exist_ok=True)
        
    def create_system_backup(self, backup_type: str = "manual") -> str:
        """Create complete system backup"""
        return create_system_backup()
    
    def restore_from_backup(self, backup_path: str) -> bool:
        """Restore system from backup"""
        return restore_from_backup(backup_path)
    
    def list_backups(self) -> list:
        """List all available backups"""
        try:
            backups = []
            for backup_dir in self.backup_base_dir.glob("backup_*"):
                if backup_dir.is_dir():
                    manifest_file = backup_dir / "backup_manifest.json"
                    if manifest_file.exists():
                        with open(manifest_file, 'r') as f:
                            manifest = json.load(f)
                        backups.append({
                            'path': str(backup_dir),
                            'timestamp': manifest.get('timestamp'),
                            'type': manifest.get('backup_type', 'unknown'),
                            'size_mb': sum(f.stat().st_size for f in backup_dir.rglob('*') if f.is_file()) / (1024*1024)
                        })
            return sorted(backups, key=lambda x: x['timestamp'], reverse=True)
        except Exception as e:
            logger.error(f"Error listing backups: {e}")
            return []
    
    def cleanup_old_backups(self, keep_count: int = 10):
        """Clean up old backups, keeping only the specified number"""
        try:
            backups = self.list_backups()
            if len(backups) > keep_count:
                for backup in backups[keep_count:]:
                    backup_path = Path(backup['path'])
                    if backup_path.exists():
                        shutil.rmtree(backup_path)
                        logger.info(f"Removed old backup: {backup_path}")
            return True
        except Exception as e:
            logger.error(f"Error cleaning up backups: {e}")
            return False

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
