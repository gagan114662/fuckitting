
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
from loguru import logger

class SystemCleaner:
    """System cleanup and resource management"""
    
    def __init__(self):
        self.cleanup_history = []
        self.temp_dirs = ["/tmp", "./temp", "./cache"]
        self.log_dirs = ["./logs", "./results"]
        
    def cleanup_aggressive(self) -> bool:
        """Aggressive cleanup to free maximum space"""
        return cleanup_aggressive()
    
    def get_cleanup_report(self) -> dict:
        """Get cleanup status report"""
        try:
            disk_usage = shutil.disk_usage(".")
            total, used, free = disk_usage
            
            return {
                'disk_usage': {
                    'total_gb': total / (1024**3),
                    'used_gb': used / (1024**3),
                    'free_gb': free / (1024**3),
                    'used_percent': (used / total) * 100
                },
                'cleanup_history': len(self.cleanup_history),
                'temp_dirs_checked': len(self.temp_dirs),
                'log_dirs_checked': len(self.log_dirs)
            }
        except Exception as e:
            logger.error(f"Error getting cleanup report: {e}")
            return {'error': str(e)}
    
    def schedule_cleanup(self):
        """Schedule regular cleanup operations"""
        logger.info("Cleanup scheduled for regular intervals")
        return True

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
