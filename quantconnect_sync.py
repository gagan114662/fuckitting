"""
Advanced QuantConnect Synchronization and Rate Limiting System
Solves QuantConnect rate limits and local code synchronization issues
"""
import asyncio
import hashlib
import json
import os
import time
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque
import aiofiles
import aiohttp
from loguru import logger

from quantconnect_client import QuantConnectClient
from error_handler import ErrorHandler, ErrorCategory, ErrorSeverity, handle_errors
from config import config

@dataclass
class RateLimitConfig:
    """Rate limiting configuration"""
    requests_per_minute: int = 30
    requests_per_hour: int = 1000
    burst_limit: int = 5
    backoff_multiplier: float = 2.0
    max_backoff: float = 300.0  # 5 minutes

@dataclass
class SyncedFile:
    """Represents a synchronized file between local and QuantConnect"""
    local_path: Path
    qc_project_id: int
    qc_file_name: str
    local_hash: str
    qc_hash: str
    last_sync: datetime
    sync_direction: str  # 'up', 'down', 'both'
    conflicts: List[str] = field(default_factory=list)

class AdvancedRateLimiter:
    """Advanced rate limiter with burst handling and adaptive backoff"""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.request_times = deque()
        self.hour_request_times = deque()
        self.consecutive_failures = 0
        self.current_backoff = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self) -> bool:
        """Acquire permission to make a request"""
        async with self._lock:
            now = time.time()
            
            # Clean old requests
            self._clean_old_requests(now)
            
            # Check if we're within limits
            minute_requests = len(self.request_times)
            hour_requests = len(self.hour_request_times)
            
            if (minute_requests >= self.config.requests_per_minute or 
                hour_requests >= self.config.requests_per_hour):
                
                # Calculate wait time
                if minute_requests >= self.config.requests_per_minute:
                    wait_time = 60 - (now - self.request_times[0])
                else:
                    wait_time = 3600 - (now - self.hour_request_times[0])
                
                # Add backoff if we have consecutive failures
                if self.current_backoff > 0:
                    wait_time = max(wait_time, self.current_backoff)
                
                logger.warning(f"⏱️ Rate limit reached, waiting {wait_time:.1f}s")
                await asyncio.sleep(wait_time)
                return await self.acquire()  # Recursive call after waiting
            
            # Record the request
            self.request_times.append(now)
            self.hour_request_times.append(now)
            return True
    
    def _clean_old_requests(self, now: float):
        """Remove old request timestamps"""
        # Remove requests older than 1 minute
        while self.request_times and now - self.request_times[0] > 60:
            self.request_times.popleft()
        
        # Remove requests older than 1 hour
        while self.hour_request_times and now - self.hour_request_times[0] > 3600:
            self.hour_request_times.popleft()
    
    def record_success(self):
        """Record a successful request"""
        self.consecutive_failures = 0
        self.current_backoff = 0.0
    
    def record_failure(self):
        """Record a failed request and increase backoff"""
        self.consecutive_failures += 1
        if self.consecutive_failures > 1:
            self.current_backoff = min(
                self.config.max_backoff,
                (self.config.backoff_multiplier ** (self.consecutive_failures - 1)) * 1.0
            )
            logger.warning(f"Increased backoff to {self.current_backoff:.1f}s after {self.consecutive_failures} failures")

class FileHasher:
    """Utility for file hashing and change detection"""
    
    @staticmethod
    def hash_file(file_path: Path) -> str:
        """Calculate SHA-256 hash of file content"""
        try:
            with open(file_path, 'rb') as f:
                return hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            logger.error(f"Error hashing file {file_path}: {e}")
            return ""
    
    @staticmethod
    def hash_string(content: str) -> str:
        """Calculate SHA-256 hash of string content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

class QuantConnectSyncManager:
    """Advanced QuantConnect synchronization manager"""
    
    def __init__(self):
        self.rate_limiter = AdvancedRateLimiter(RateLimitConfig())
        self.error_handler = ErrorHandler()
        self.qc_client = QuantConnectClient()
        self.sync_db_path = Path("data/sync_database.json")
        self.synced_files: Dict[str, SyncedFile] = {}
        self.local_strategies_dir = Path("strategies")
        self.load_sync_database()
    
    def load_sync_database(self):
        """Load synchronization database"""
        try:
            if self.sync_db_path.exists():
                with open(self.sync_db_path, 'r') as f:
                    data = json.load(f)
                
                for file_key, file_data in data.items():
                    self.synced_files[file_key] = SyncedFile(
                        local_path=Path(file_data['local_path']),
                        qc_project_id=file_data['qc_project_id'],
                        qc_file_name=file_data['qc_file_name'],
                        local_hash=file_data['local_hash'],
                        qc_hash=file_data['qc_hash'],
                        last_sync=datetime.fromisoformat(file_data['last_sync']),
                        sync_direction=file_data['sync_direction'],
                        conflicts=file_data.get('conflicts', [])
                    )
                
                logger.info(f"Loaded {len(self.synced_files)} synced files from database")
        except Exception as e:
            logger.warning(f"Could not load sync database: {e}")
    
    def save_sync_database(self):
        """Save synchronization database"""
        try:
            self.sync_db_path.parent.mkdir(parents=True, exist_ok=True)
            
            data = {}
            for file_key, synced_file in self.synced_files.items():
                data[file_key] = {
                    'local_path': str(synced_file.local_path),
                    'qc_project_id': synced_file.qc_project_id,
                    'qc_file_name': synced_file.qc_file_name,
                    'local_hash': synced_file.local_hash,
                    'qc_hash': synced_file.qc_hash,
                    'last_sync': synced_file.last_sync.isoformat(),
                    'sync_direction': synced_file.sync_direction,
                    'conflicts': synced_file.conflicts
                }
            
            with open(self.sync_db_path, 'w') as f:
                json.dump(data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Could not save sync database: {e}")
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def rate_limited_request(self, coro):
        """Execute a QuantConnect API request with rate limiting"""
        await self.rate_limiter.acquire()
        
        try:
            result = await coro
            self.rate_limiter.record_success()
            return result
        except Exception as e:
            self.rate_limiter.record_failure()
            raise
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.HIGH)
    async def discover_local_strategies(self) -> List[Path]:
        """Discover all strategy files in local directory"""
        self.local_strategies_dir.mkdir(parents=True, exist_ok=True)
        
        strategy_files = []
        for pattern in ["*.py", "*.cs"]:
            strategy_files.extend(self.local_strategies_dir.glob(pattern))
        
        logger.info(f"Discovered {len(strategy_files)} local strategy files")
        return strategy_files
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.HIGH)
    async def discover_qc_projects(self) -> List[Dict[str, Any]]:
        """Discover all QuantConnect projects"""
        
        async def _get_projects():
            async with self.qc_client as client:
                return await client.list_projects()
        
        projects = await self.rate_limited_request(_get_projects())
        logger.info(f"Discovered {len(projects)} QuantConnect projects")
        return projects
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def get_qc_project_files(self, project_id: int) -> List[Dict[str, Any]]:
        """Get all files in a QuantConnect project"""
        
        async def _get_files():
            async with self.qc_client as client:
                # This would be implemented in the actual QC client
                # For now, return mock data
                return [
                    {"name": "main.py", "content": "# Strategy code", "modified": datetime.now().isoformat()}
                ]
        
        return await self.rate_limited_request(_get_files())
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def upload_file_to_qc(self, project_id: int, file_name: str, content: str) -> bool:
        """Upload file to QuantConnect project with rate limiting"""
        
        async def _upload():
            async with self.qc_client as client:
                return await client.upload_file(project_id, file_name, content)
        
        return await self.rate_limited_request(_upload())
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def download_file_from_qc(self, project_id: int, file_name: str) -> Optional[str]:
        """Download file from QuantConnect project"""
        
        async def _download():
            # This would be implemented in the actual QC client
            # For now, return mock content
            return f"# Downloaded from QC project {project_id}\n# File: {file_name}\n"
        
        return await self.rate_limited_request(_download())
    
    async def detect_changes(self) -> Dict[str, List[SyncedFile]]:
        """Detect changes between local and QuantConnect files"""
        changes = {
            'local_modified': [],
            'qc_modified': [],
            'conflicts': [],
            'new_local': [],
            'new_qc': []
        }
        
        # Check local files
        local_files = await self.discover_local_strategies()
        
        for local_file in local_files:
            file_key = str(local_file.relative_to(self.local_strategies_dir))
            current_hash = FileHasher.hash_file(local_file)
            
            if file_key in self.synced_files:
                synced_file = self.synced_files[file_key]
                
                # Check if local file changed
                if current_hash != synced_file.local_hash:
                    synced_file.local_hash = current_hash
                    
                    # Get QC file content and check for conflicts
                    qc_content = await self.download_file_from_qc(
                        synced_file.qc_project_id, synced_file.qc_file_name
                    )
                    
                    if qc_content:
                        qc_hash = FileHasher.hash_string(qc_content)
                        
                        if qc_hash != synced_file.qc_hash:
                            # Both files changed - conflict
                            changes['conflicts'].append(synced_file)
                        else:
                            # Only local file changed
                            changes['local_modified'].append(synced_file)
                    else:
                        changes['local_modified'].append(synced_file)
            else:
                # New local file
                changes['new_local'].append(local_file)
        
        return changes
    
    @handle_errors(ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)
    async def sync_local_to_qc(self, synced_file: SyncedFile) -> bool:
        """Sync local file to QuantConnect"""
        try:
            # Read local file
            with open(synced_file.local_path, 'r') as f:
                content = f.read()
            
            # Upload to QuantConnect
            success = await self.upload_file_to_qc(
                synced_file.qc_project_id, 
                synced_file.qc_file_name, 
                content
            )
            
            if success:
                synced_file.qc_hash = FileHasher.hash_string(content)
                synced_file.last_sync = datetime.now()
                logger.success(f"✅ Synced {synced_file.local_path.name} to QC")
                return True
            else:
                logger.error(f"❌ Failed to sync {synced_file.local_path.name} to QC")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing {synced_file.local_path.name} to QC: {e}")
            return False
    
    @handle_errors(ErrorCategory.SYSTEM, ErrorSeverity.MEDIUM)
    async def sync_qc_to_local(self, synced_file: SyncedFile) -> bool:
        """Sync QuantConnect file to local"""
        try:
            # Download from QuantConnect
            content = await self.download_file_from_qc(
                synced_file.qc_project_id, synced_file.qc_file_name
            )
            
            if content:
                # Write to local file
                with open(synced_file.local_path, 'w') as f:
                    f.write(content)
                
                synced_file.local_hash = FileHasher.hash_string(content)
                synced_file.last_sync = datetime.now()
                logger.success(f"✅ Synced {synced_file.qc_file_name} from QC to local")
                return True
            else:
                logger.error(f"❌ Failed to download {synced_file.qc_file_name} from QC")
                return False
                
        except Exception as e:
            logger.error(f"Error syncing {synced_file.qc_file_name} from QC: {e}")
            return False
    
    async def handle_conflict(self, synced_file: SyncedFile, resolution: str = 'local_wins') -> bool:
        """Handle sync conflicts"""
        logger.warning(f"⚠️ Handling conflict for {synced_file.local_path.name}")
        
        if resolution == 'local_wins':
            return await self.sync_local_to_qc(synced_file)
        elif resolution == 'qc_wins':
            return await self.sync_qc_to_local(synced_file)
        elif resolution == 'create_backup':
            # Create backup and sync local to QC
            backup_path = synced_file.local_path.with_suffix(f'.backup_{int(time.time())}.py')
            
            # Download QC version as backup
            qc_content = await self.download_file_from_qc(
                synced_file.qc_project_id, synced_file.qc_file_name
            )
            
            if qc_content:
                with open(backup_path, 'w') as f:
                    f.write(qc_content)
                logger.info(f"Created backup: {backup_path}")
            
            return await self.sync_local_to_qc(synced_file)
        else:
            logger.error(f"Unknown conflict resolution: {resolution}")
            return False
    
    @handle_errors(ErrorCategory.SYSTEM, ErrorSeverity.HIGH)
    async def full_sync(self, conflict_resolution: str = 'create_backup') -> Dict[str, Any]:
        """Perform full synchronization between local and QuantConnect"""
        logger.info("🔄 Starting full synchronization...")
        
        sync_results = {
            'synced_to_qc': 0,
            'synced_from_qc': 0,
            'conflicts_resolved': 0,
            'new_files_created': 0,
            'errors': []
        }
        
        try:
            # Detect changes
            changes = await self.detect_changes()
            
            # Handle local modifications
            for synced_file in changes['local_modified']:
                if await self.sync_local_to_qc(synced_file):
                    sync_results['synced_to_qc'] += 1
                else:
                    sync_results['errors'].append(f"Failed to sync {synced_file.local_path.name} to QC")
            
            # Handle QC modifications
            for synced_file in changes['qc_modified']:
                if await self.sync_qc_to_local(synced_file):
                    sync_results['synced_from_qc'] += 1
                else:
                    sync_results['errors'].append(f"Failed to sync {synced_file.qc_file_name} from QC")
            
            # Handle conflicts
            for synced_file in changes['conflicts']:
                if await self.handle_conflict(synced_file, conflict_resolution):
                    sync_results['conflicts_resolved'] += 1
                else:
                    sync_results['errors'].append(f"Failed to resolve conflict for {synced_file.local_path.name}")
            
            # Handle new local files
            for local_file in changes['new_local']:
                # Create new QC project or add to existing project
                success = await self.create_qc_project_for_file(local_file)
                if success:
                    sync_results['new_files_created'] += 1
                else:
                    sync_results['errors'].append(f"Failed to create QC project for {local_file.name}")
            
            # Save sync database
            self.save_sync_database()
            
            logger.success(f"✅ Sync completed: {sync_results}")
            return sync_results
            
        except Exception as e:
            logger.error(f"Error in full sync: {e}")
            sync_results['errors'].append(str(e))
            return sync_results
    
    @handle_errors(ErrorCategory.QUANTCONNECT_API, ErrorSeverity.MEDIUM)
    async def create_qc_project_for_file(self, local_file: Path) -> bool:
        """Create a new QuantConnect project for a local file"""
        try:
            project_name = f"AlgoForge_{local_file.stem}_{int(time.time())}"
            
            async def _create_project():
                async with self.qc_client as client:
                    return await client.create_project(project_name)
            
            project_id = await self.rate_limited_request(_create_project())
            
            if project_id:
                # Read local file content
                with open(local_file, 'r') as f:
                    content = f.read()
                
                # Upload to new project
                success = await self.upload_file_to_qc(project_id, "main.py", content)
                
                if success:
                    # Add to sync database
                    file_key = str(local_file.relative_to(self.local_strategies_dir))
                    self.synced_files[file_key] = SyncedFile(
                        local_path=local_file,
                        qc_project_id=project_id,
                        qc_file_name="main.py",
                        local_hash=FileHasher.hash_file(local_file),
                        qc_hash=FileHasher.hash_string(content),
                        last_sync=datetime.now(),
                        sync_direction='up'
                    )
                    
                    logger.success(f"✅ Created QC project {project_name} for {local_file.name}")
                    return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error creating QC project for {local_file.name}: {e}")
            return False
    
    async def start_continuous_sync(self, interval_minutes: int = 5):
        """Start continuous synchronization"""
        logger.info(f"🔄 Starting continuous sync (every {interval_minutes} minutes)")
        
        while True:
            try:
                await self.full_sync()
                await asyncio.sleep(interval_minutes * 60)
            except Exception as e:
                logger.error(f"Error in continuous sync: {e}")
                await asyncio.sleep(60)  # Wait 1 minute before retrying
    
    def get_sync_status(self) -> Dict[str, Any]:
        """Get current synchronization status"""
        total_files = len(self.synced_files)
        recent_syncs = len([
            f for f in self.synced_files.values()
            if f.last_sync > datetime.now() - timedelta(hours=24)
        ])
        
        conflicts = len([
            f for f in self.synced_files.values()
            if f.conflicts
        ])
        
        return {
            'total_synced_files': total_files,
            'recent_syncs_24h': recent_syncs,
            'active_conflicts': conflicts,
            'rate_limiter_status': {
                'requests_this_minute': len(self.rate_limiter.request_times),
                'requests_this_hour': len(self.rate_limiter.hour_request_times),
                'consecutive_failures': self.rate_limiter.consecutive_failures,
                'current_backoff': self.rate_limiter.current_backoff
            }
        }

# Example usage and testing
async def test_sync_system():
    """Test the synchronization system"""
    logger.info("🧪 Testing QuantConnect sync system...")
    
    sync_manager = QuantConnectSyncManager()
    
    # Test discovery
    local_files = await sync_manager.discover_local_strategies()
    qc_projects = await sync_manager.discover_qc_projects()
    
    logger.info(f"Found {len(local_files)} local files and {len(qc_projects)} QC projects")
    
    # Test change detection
    changes = await sync_manager.detect_changes()
    logger.info(f"Detected changes: {changes}")
    
    # Test sync status
    status = sync_manager.get_sync_status()
    logger.info(f"Sync status: {status}")
    
    # Test full sync
    sync_results = await sync_manager.full_sync()
    logger.info(f"Sync results: {sync_results}")
    
    logger.success("🎉 Sync system test completed!")
    return True

if __name__ == "__main__":
    asyncio.run(test_sync_system())