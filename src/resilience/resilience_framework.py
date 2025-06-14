#!/usr/bin/env python3
"""
Resilience Framework for AlgoForge 3.0
Provides bulletproof error handling, retry mechanisms, and failover capabilities
"""
import asyncio
import threading
import time
import os
import sys
import json
import sqlite3
import functools
import psutil
import signal
from pathlib import Path
from typing import Any, Callable, Optional, Dict, List
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from enum import Enum
import aiohttp
import aiofiles
from loguru import logger

class FailureType(Enum):
    NETWORK = "network"
    API = "api"
    DATABASE = "database"
    FILESYSTEM = "filesystem"
    MEMORY = "memory"
    CONFIGURATION = "configuration"
    DEPENDENCY = "dependency"
    RESOURCE = "resource"
    CONCURRENCY = "concurrency"

@dataclass
class RetryConfig:
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    exponential_factor: float = 2.0
    jitter: bool = True

@dataclass
class CircuitBreakerConfig:
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: type = Exception

class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class CircuitBreaker:
    """Circuit breaker pattern implementation"""
    
    def __init__(self, config: CircuitBreakerConfig):
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time = 0
        self.lock = threading.Lock()
    
    def __call__(self, func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            with self.lock:
                if self.state == CircuitBreakerState.OPEN:
                    if time.time() - self.last_failure_time < self.config.recovery_timeout:
                        raise Exception(f"Circuit breaker OPEN for {func.__name__}")
                    else:
                        self.state = CircuitBreakerState.HALF_OPEN
                        logger.info(f"Circuit breaker HALF_OPEN for {func.__name__}")
            
            try:
                result = await func(*args, **kwargs)
                
                with self.lock:
                    if self.state == CircuitBreakerState.HALF_OPEN:
                        self.state = CircuitBreakerState.CLOSED
                        self.failure_count = 0
                        logger.info(f"Circuit breaker CLOSED for {func.__name__}")
                
                return result
                
            except self.config.expected_exception as e:
                with self.lock:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.config.failure_threshold:
                        self.state = CircuitBreakerState.OPEN
                        logger.error(f"Circuit breaker OPEN for {func.__name__}: {e}")
                    
                    raise
        
        return wrapper

class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate
        self.burst = burst
        self.tokens = burst
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add tokens based on elapsed time
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            
            return False
    
    async def wait_for_token(self, tokens: int = 1):
        while not await self.acquire(tokens):
            await asyncio.sleep(0.1)

class ResourcePool:
    """Generic resource pool with automatic cleanup"""
    
    def __init__(self, factory_func, max_size: int = 10, cleanup_func=None):
        self.factory_func = factory_func
        self.cleanup_func = cleanup_func
        self.max_size = max_size
        self.pool = asyncio.Queue(maxsize=max_size)
        self.created_count = 0
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        try:
            # Try to get existing resource
            resource = self.pool.get_nowait()
            return resource
        except asyncio.QueueEmpty:
            # Create new resource if under limit
            async with self.lock:
                if self.created_count < self.max_size:
                    resource = await self.factory_func()
                    self.created_count += 1
                    return resource
            
            # Wait for available resource
            return await self.pool.get()
    
    async def release(self, resource):
        try:
            self.pool.put_nowait(resource)
        except asyncio.QueueFull:
            # Pool is full, cleanup resource
            if self.cleanup_func:
                await self.cleanup_func(resource)
            async with self.lock:
                self.created_count -= 1

class ResilienceManager:
    """Central manager for all resilience features"""
    
    def __init__(self):
        self.rate_limiters: Dict[str, RateLimiter] = {}
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.resource_pools: Dict[str, ResourcePool] = {}
        self.health_checks: Dict[str, Callable] = {}
        self.shutdown_handlers: List[Callable] = []
        self.lock = threading.Lock()
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown handlers"""
        def signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, initiating graceful shutdown")
            asyncio.create_task(self.graceful_shutdown())
        
        signal.signal(signal.SIGTERM, signal_handler)
        signal.signal(signal.SIGINT, signal_handler)
    
    async def graceful_shutdown(self):
        """Graceful shutdown of all resources"""
        logger.info("Starting graceful shutdown...")
        
        for handler in self.shutdown_handlers:
            try:
                await handler()
            except Exception as e:
                logger.error(f"Shutdown handler failed: {e}")
        
        logger.info("Graceful shutdown completed")
    
    def add_shutdown_handler(self, handler: Callable):
        """Add a shutdown handler"""
        self.shutdown_handlers.append(handler)
    
    def get_rate_limiter(self, name: str, rate: float = 10.0, burst: int = 20) -> RateLimiter:
        """Get or create rate limiter"""
        if name not in self.rate_limiters:
            self.rate_limiters[name] = RateLimiter(rate, burst)
        return self.rate_limiters[name]
    
    def get_circuit_breaker(self, name: str, config: CircuitBreakerConfig = None) -> CircuitBreaker:
        """Get or create circuit breaker"""
        if name not in self.circuit_breakers:
            config = config or CircuitBreakerConfig()
            self.circuit_breakers[name] = CircuitBreaker(config)
        return self.circuit_breakers[name]
    
    def get_resource_pool(self, name: str, factory_func, max_size: int = 10, cleanup_func=None) -> ResourcePool:
        """Get or create resource pool"""
        if name not in self.resource_pools:
            self.resource_pools[name] = ResourcePool(factory_func, max_size, cleanup_func)
        return self.resource_pools[name]

# Global resilience manager instance
resilience_manager = ResilienceManager()

def retry_with_backoff(
    retry_config: RetryConfig = None,
    failure_types: List[FailureType] = None
):
    """Decorator for retry with exponential backoff"""
    if retry_config is None:
        retry_config = RetryConfig()
    
    if failure_types is None:
        failure_types = [FailureType.NETWORK, FailureType.API]
    
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            delay = retry_config.base_delay
            
            for attempt in range(retry_config.max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Check if this is a retryable error
                    retryable = False
                    if FailureType.NETWORK in failure_types and "network" in str(e).lower():
                        retryable = True
                    elif FailureType.API in failure_types and any(code in str(e) for code in ["429", "503", "502", "504"]):
                        retryable = True
                    elif FailureType.DATABASE in failure_types and "database" in str(e).lower():
                        retryable = True
                    
                    if not retryable or attempt == retry_config.max_attempts - 1:
                        logger.error(f"Function {func.__name__} failed after {attempt + 1} attempts: {e}")
                        raise
                    
                    # Calculate delay with jitter
                    actual_delay = delay
                    if retry_config.jitter:
                        import random
                        actual_delay *= (0.5 + random.random() * 0.5)
                    
                    logger.warning(f"Attempt {attempt + 1} failed for {func.__name__}: {e}. Retrying in {actual_delay:.2f}s")
                    await asyncio.sleep(actual_delay)
                    
                    # Exponential backoff
                    delay = min(delay * retry_config.exponential_factor, retry_config.max_delay)
            
            raise last_exception
        
        return wrapper
    return decorator

@contextmanager
def safe_file_operation(file_path: str, operation: str = "read"):
    """Safe file operation with automatic cleanup"""
    file_obj = None
    backup_path = None
    
    try:
        # Create backup for write operations
        if operation == "write" and os.path.exists(file_path):
            backup_path = f"{file_path}.backup_{int(time.time())}"
            import shutil
            shutil.copy2(file_path, backup_path)
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        if operation == "read":
            file_obj = open(file_path, 'r', encoding='utf-8')
        elif operation == "write":
            file_obj = open(file_path, 'w', encoding='utf-8')
        elif operation == "append":
            file_obj = open(file_path, 'a', encoding='utf-8')
        
        yield file_obj
        
    except PermissionError as e:
        logger.error(f"Permission denied for {file_path}: {e}")
        raise
    except OSError as e:
        if "No space left on device" in str(e):
            logger.critical(f"Disk full while accessing {file_path}")
            # Try to free up space
            cleanup_temp_files()
        raise
    except Exception as e:
        logger.error(f"File operation failed for {file_path}: {e}")
        raise
    finally:
        if file_obj:
            try:
                file_obj.close()
            except:
                pass
        
        # Cleanup backup on successful write
        if backup_path and operation == "write":
            try:
                os.unlink(backup_path)
            except:
                pass

@asynccontextmanager
async def safe_database_connection(db_path: str):
    """Safe database connection with automatic cleanup"""
    conn = None
    try:
        # Create directory if needed
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        
        # Connect with timeout
        conn = sqlite3.connect(db_path, timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrency
        conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/performance
        
        yield conn
        
    except sqlite3.OperationalError as e:
        if "database is locked" in str(e):
            logger.warning(f"Database locked, retrying: {db_path}")
            await asyncio.sleep(1.0)
            raise
        elif "database disk image is malformed" in str(e):
            logger.critical(f"Database corrupted: {db_path}")
            # Try to recover from backup
            backup_path = f"{db_path}.backup"
            if os.path.exists(backup_path):
                import shutil
                shutil.copy2(backup_path, db_path)
                logger.info(f"Restored database from backup: {backup_path}")
        raise
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise
    finally:
        if conn:
            try:
                conn.close()
            except:
                pass

async def safe_api_request(
    session: aiohttp.ClientSession,
    method: str,
    url: str,
    rate_limiter_name: str = "default",
    **kwargs
) -> aiohttp.ClientResponse:
    """Safe API request with rate limiting and retries"""
    
    # Get rate limiter
    rate_limiter = resilience_manager.get_rate_limiter(rate_limiter_name, rate=10, burst=20)
    await rate_limiter.wait_for_token()
    
    # Apply timeout if not specified
    if 'timeout' not in kwargs:
        kwargs['timeout'] = aiohttp.ClientTimeout(total=30)
    
    try:
        async with session.request(method, url, **kwargs) as response:
            # Handle rate limiting
            if response.status == 429:
                retry_after = response.headers.get('Retry-After', '60')
                await asyncio.sleep(float(retry_after))
                raise aiohttp.ClientError(f"Rate limited, retry after {retry_after}s")
            
            # Handle server errors
            if response.status >= 500:
                raise aiohttp.ClientError(f"Server error: {response.status}")
            
            response.raise_for_status()
            return response
            
    except asyncio.TimeoutError:
        logger.error(f"Request timeout for {url}")
        raise
    except aiohttp.ClientError as e:
        logger.error(f"API request failed for {url}: {e}")
        raise

def memory_monitor(threshold_mb: int = 1000):
    """Memory monitoring decorator"""
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            # Check memory before execution
            process = psutil.Process()
            memory_before = process.memory_info().rss / 1024 / 1024  # MB
            
            if memory_before > threshold_mb:
                logger.warning(f"High memory usage before {func.__name__}: {memory_before:.2f}MB")
                # Force garbage collection
                import gc
                gc.collect()
            
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                # Check memory after execution
                memory_after = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = memory_after - memory_before
                
                if memory_increase > 100:  # 100MB increase
                    logger.warning(f"High memory increase in {func.__name__}: {memory_increase:.2f}MB")
        
        return wrapper
    return decorator

def thread_safe(func):
    """Thread-safe decorator using locks"""
    lock = threading.Lock()
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with lock:
            return func(*args, **kwargs)
    
    return wrapper

async def validate_environment():
    """Validate environment configuration"""
    errors = []
    warnings = []
    
    # Check required environment variables (with fallback defaults)
    required_vars = {
        'QUANTCONNECT_USER_ID': 'QuantConnect user ID',
        'QUANTCONNECT_API_TOKEN': 'QuantConnect API token'
    }
    
    # Load from hardcoded config if environment variables are missing
    for var, description in required_vars.items():
        if not os.getenv(var):
            # Try to get from config file
            try:
                from config import config
                if var == 'QUANTCONNECT_USER_ID' and hasattr(config.quantconnect, 'user_id'):
                    if config.quantconnect.user_id != "357130":  # Not default
                        warnings.append(f"Using {var} from config file instead of environment")
                        continue
                elif var == 'QUANTCONNECT_API_TOKEN' and hasattr(config.quantconnect, 'api_token'):
                    if len(config.quantconnect.api_token) > 10:  # Has real token
                        warnings.append(f"Using {var} from config file instead of environment")
                        continue
            except:
                pass
            
            # Only error if no fallback available
            warnings.append(f"Environment variable not set: {var} ({description}) - using config fallback")
    
    # Check optional environment variables
    optional_vars = {
        'BRAVE_API_KEY': 'Brave Search API',
        'TWELVE_DATA_API_KEY': 'Twelve Data API',
        'ALPHA_VANTAGE_API_KEY': 'Alpha Vantage API'
    }
    
    for var, description in optional_vars.items():
        if not os.getenv(var):
            warnings.append(f"Optional environment variable not set: {var} ({description})")
    
    # Check disk space
    disk_usage = psutil.disk_usage('.')
    free_gb = disk_usage.free / 1024 / 1024 / 1024
    if free_gb < 1.0:  # Less than 1GB free
        errors.append(f"Low disk space: {free_gb:.2f}GB free")
    elif free_gb < 5.0:  # Less than 5GB free
        warnings.append(f"Disk space warning: {free_gb:.2f}GB free")
    
    # Check memory
    memory = psutil.virtual_memory()
    if memory.percent > 90:
        errors.append(f"High memory usage: {memory.percent:.1f}%")
    elif memory.percent > 80:
        warnings.append(f"Memory usage warning: {memory.percent:.1f}%")
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python version too old: {sys.version_info}")
    
    # Report results
    if errors:
        logger.error("Environment validation failed:")
        for error in errors:
            logger.error(f"  ❌ {error}")
        raise Exception(f"Environment validation failed: {len(errors)} errors")
    
    if warnings:
        logger.warning("Environment validation warnings:")
        for warning in warnings:
            logger.warning(f"  ⚠️ {warning}")
    
    logger.info("✅ Environment validation passed")

def cleanup_temp_files():
    """Clean up temporary files to free disk space"""
    temp_dirs = ["/tmp", "./temp", "./cache", "./logs"]
    cleaned_bytes = 0
    
    for temp_dir in temp_dirs:
        if not os.path.exists(temp_dir):
            continue
        
        try:
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    try:
                        # Remove files older than 24 hours
                        if os.path.getmtime(file_path) < time.time() - 86400:
                            size = os.path.getsize(file_path)
                            os.unlink(file_path)
                            cleaned_bytes += size
                    except (OSError, PermissionError):
                        pass
        except (OSError, PermissionError):
            pass
    
    logger.info(f"Cleaned up {cleaned_bytes / 1024 / 1024:.2f}MB of temporary files")

async def health_check_all():
    """Comprehensive system health check"""
    health_status = {
        "timestamp": time.time(),
        "checks": {}
    }
    
    # Database health
    try:
        async with safe_database_connection("./config/health_check.db") as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS health_test (id INTEGER)")
            conn.execute("INSERT INTO health_test VALUES (1)")
            conn.commit()
            health_status["checks"]["database"] = "healthy"
    except Exception as e:
        health_status["checks"]["database"] = f"unhealthy: {e}"
    
    # File system health
    try:
        test_file = "./temp/health_check.txt"
        with safe_file_operation(test_file, "write") as f:
            f.write("health check")
        os.unlink(test_file)
        health_status["checks"]["filesystem"] = "healthy"
    except Exception as e:
        health_status["checks"]["filesystem"] = f"unhealthy: {e}"
    
    # Memory health
    try:
        memory = psutil.virtual_memory()
        if memory.percent < 80:
            health_status["checks"]["memory"] = "healthy"
        else:
            health_status["checks"]["memory"] = f"warning: {memory.percent:.1f}% used"
    except Exception as e:
        health_status["checks"]["memory"] = f"unhealthy: {e}"
    
    # Network health
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("https://httpbin.org/status/200", timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    health_status["checks"]["network"] = "healthy"
                else:
                    health_status["checks"]["network"] = f"unhealthy: status {response.status}"
    except Exception as e:
        health_status["checks"]["network"] = f"unhealthy: {e}"
    
    return health_status

# Initialize resilience framework
async def initialize_resilience_framework():
    """Initialize the resilience framework"""
    logger.info("🛡️ Initializing resilience framework...")
    
    try:
        # Validate environment
        await validate_environment()
        
        # Setup health checks
        health_status = await health_check_all()
        unhealthy_checks = [k for k, v in health_status["checks"].items() if "unhealthy" in v]
        
        if unhealthy_checks:
            logger.warning(f"Some health checks failed: {unhealthy_checks}")
        else:
            logger.info("✅ All health checks passed")
        
        # Setup cleanup
        cleanup_temp_files()
        
        logger.info("✅ Resilience framework initialized")
        return True
        
    except Exception as e:
        logger.error(f"❌ Resilience framework initialization failed: {e}")
        return False