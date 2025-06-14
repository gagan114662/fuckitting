#!/usr/bin/env python3
"""
Alternative Data Source Manager
Switches to alternative APIs when primary sources fail or hit rate limits
"""
import asyncio
import aiohttp
import json
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from pathlib import Path
from loguru import logger
from dataclasses import dataclass
import time

@dataclass
class DataSource:
    name: str
    api_key_env: str
    base_url: str
    rate_limit_per_minute: int
    priority: int  # Lower is higher priority
    is_active: bool = True
    last_error: Optional[str] = None
    consecutive_failures: int = 0
    last_success: Optional[datetime] = None

class DataSourceManager:
    """Manages multiple data sources with automatic failover"""
    
    def __init__(self):
        self.data_sources = self._initialize_data_sources()
        self.current_source = None
        self.request_counts = {}  # Track requests per source
        self.failover_history = []
        
    def _initialize_data_sources(self) -> Dict[str, DataSource]:
        """Initialize available data sources in priority order"""
        sources = {
            'quantconnect': DataSource(
                name='quantconnect',
                api_key_env='QUANTCONNECT_API_TOKEN',
                base_url='https://www.quantconnect.com/api/v2',
                rate_limit_per_minute=30,
                priority=1
            ),
            'twelve_data': DataSource(
                name='twelve_data',
                api_key_env='TWELVE_DATA_API_KEY',
                base_url='https://api.twelvedata.com',
                rate_limit_per_minute=800,  # Free tier: 800 requests/day
                priority=2
            ),
            'alpha_vantage': DataSource(
                name='alpha_vantage',
                api_key_env='ALPHA_VANTAGE_API_KEY',
                base_url='https://www.alphavantage.co/query',
                rate_limit_per_minute=5,  # Free tier: 5 requests/minute
                priority=3
            ),
            'yahoo_finance': DataSource(
                name='yahoo_finance',
                api_key_env=None,  # No API key required
                base_url='https://query1.finance.yahoo.com/v8/finance/chart',
                rate_limit_per_minute=60,
                priority=4
            ),
            'polygon': DataSource(
                name='polygon',
                api_key_env='POLYGON_API_KEY',
                base_url='https://api.polygon.io',
                rate_limit_per_minute=200,  # Free tier
                priority=5
            ),
            'iex_cloud': DataSource(
                name='iex_cloud',
                api_key_env='IEX_CLOUD_API_KEY',
                base_url='https://cloud.iexapis.com/stable',
                rate_limit_per_minute=100,
                priority=6
            ),
            'finnhub': DataSource(
                name='finnhub',
                api_key_env='FINNHUB_API_KEY',
                base_url='https://finnhub.io/api/v1',
                rate_limit_per_minute=60,  # Free tier
                priority=7
            )
        }
        
        # Check which sources have valid API keys
        for source in sources.values():
            if source.api_key_env:
                api_key = os.getenv(source.api_key_env)
                source.is_active = bool(api_key)
                if not api_key:
                    logger.debug(f"Data source {source.name} inactive - no API key")
                else:
                    logger.info(f"✅ Data source {source.name} active with API key")
            else:
                source.is_active = True  # Sources like Yahoo Finance don't need keys
        
        return sources
    
    def switch_to_alternative_apis(self) -> bool:
        """Switch to alternative data sources"""
        logger.info("🔄 Switching to alternative data sources...")
        
        try:
            # Get currently failing source
            if self.current_source:
                failing_source = self.data_sources[self.current_source]
                failing_source.consecutive_failures += 1
                failing_source.last_error = "Rate limit exceeded or API unavailable"
                logger.warning(f"Marking {self.current_source} as failing (failures: {failing_source.consecutive_failures})")
            
            # Find best alternative source
            alternative_source = self._find_best_alternative()
            
            if alternative_source:
                old_source = self.current_source
                self.current_source = alternative_source.name
                
                # Record failover
                failover_record = {
                    'timestamp': datetime.now().isoformat(),
                    'from_source': old_source,
                    'to_source': alternative_source.name,
                    'reason': 'rate_limit_or_failure'
                }
                self.failover_history.append(failover_record)
                
                logger.success(f"✅ Switched to alternative data source: {alternative_source.name}")
                logger.info(f"   ├─ Rate limit: {alternative_source.rate_limit_per_minute}/min")
                logger.info(f"   ├─ Priority: {alternative_source.priority}")
                logger.info(f"   └─ Base URL: {alternative_source.base_url}")
                
                # Test the new source
                if asyncio.run(self._test_data_source(alternative_source)):
                    logger.success(f"✅ Confirmed {alternative_source.name} is working")
                    return True
                else:
                    logger.warning(f"⚠️ {alternative_source.name} test failed, trying next alternative")
                    return self.switch_to_alternative_apis()  # Recursive call to try next source
            else:
                logger.error("❌ No alternative data sources available")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error switching data sources: {e}")
            return False
    
    def _find_best_alternative(self) -> Optional[DataSource]:
        """Find the best available alternative data source"""
        # Sort sources by priority (lower number = higher priority)
        available_sources = [
            source for source in self.data_sources.values() 
            if source.is_active and source.name != self.current_source
        ]
        
        # Filter out sources with too many recent failures
        viable_sources = [
            source for source in available_sources
            if source.consecutive_failures < 3 or 
            (source.last_success and 
             datetime.now() - source.last_success < timedelta(hours=1))
        ]
        
        if not viable_sources:
            # If no viable sources, try any available source
            viable_sources = available_sources
        
        if not viable_sources:
            return None
        
        # Sort by priority and select best
        viable_sources.sort(key=lambda x: x.priority)
        return viable_sources[0]
    
    async def _test_data_source(self, source: DataSource) -> bool:
        """Test if a data source is working"""
        try:
            test_endpoints = {
                'quantconnect': '/projects/read',
                'twelve_data': '/time_series?symbol=AAPL&interval=1day&outputsize=1',
                'alpha_vantage': '?function=TIME_SERIES_INTRADAY&symbol=IBM&interval=5min&outputsize=compact',
                'yahoo_finance': '/AAPL?interval=1d&range=1d',
                'polygon': '/v2/aggs/ticker/AAPL/range/1/day/2023-01-01/2023-01-02',
                'iex_cloud': '/stock/aapl/quote'
            }
            
            endpoint = test_endpoints.get(source.name, '')
            if not endpoint:
                return True  # No test endpoint defined, assume working
            
            url = f"{source.base_url}{endpoint}"
            headers = {}
            
            # Add API key if required
            if source.api_key_env:
                api_key = os.getenv(source.api_key_env)
                if not api_key:
                    return False
                
                if source.name == 'quantconnect':
                    headers['Authorization'] = f'Basic {api_key}'
                elif source.name == 'twelve_data':
                    url += f'&apikey={api_key}'
                elif source.name == 'alpha_vantage':
                    url += f'&apikey={api_key}'
                elif source.name == 'polygon':
                    url += f'?apikey={api_key}'
                elif source.name == 'iex_cloud':
                    url += f'?token={api_key}'
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        source.last_success = datetime.now()
                        source.consecutive_failures = 0
                        return True
                    elif response.status == 429:  # Rate limited
                        source.last_error = "Rate limited"
                        return False
                    else:
                        source.last_error = f"HTTP {response.status}"
                        return False
                        
        except Exception as e:
            source.last_error = str(e)
            logger.debug(f"Test failed for {source.name}: {e}")
            return False
    
    async def get_market_data(self, symbol: str, interval: str = '1d', period: str = '1y') -> Optional[Dict[str, Any]]:
        """Get market data using current or alternative source"""
        if not self.current_source:
            # Initialize with best available source
            best_source = self._find_best_alternative()
            if not best_source:
                logger.error("No data sources available")
                return None
            self.current_source = best_source.name
        
        source = self.data_sources[self.current_source]
        
        try:
            data = await self._fetch_data_from_source(source, symbol, interval, period)
            if data:
                source.last_success = datetime.now()
                source.consecutive_failures = 0
                return data
            else:
                # Try switching to alternative
                if self.switch_to_alternative_apis():
                    return await self.get_market_data(symbol, interval, period)
                return None
                
        except Exception as e:
            logger.error(f"Error fetching data from {source.name}: {e}")
            if self.switch_to_alternative_apis():
                return await self.get_market_data(symbol, interval, period)
            return None
    
    async def _fetch_data_from_source(self, source: DataSource, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from specific source"""
        try:
            if source.name == 'yahoo_finance':
                return await self._fetch_yahoo_data(symbol, interval, period)
            elif source.name == 'twelve_data':
                return await self._fetch_twelve_data(symbol, interval, period)
            elif source.name == 'alpha_vantage':
                return await self._fetch_alpha_vantage_data(symbol, interval, period)
            elif source.name == 'polygon':
                return await self._fetch_polygon_data(symbol, interval, period)
            elif source.name == 'iex_cloud':
                return await self._fetch_iex_data(symbol, interval, period)
            else:
                logger.warning(f"Data fetching not implemented for {source.name}")
                return None
                
        except Exception as e:
            logger.error(f"Error fetching from {source.name}: {e}")
            return None
    
    async def _fetch_yahoo_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Yahoo Finance"""
        try:
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}"
            params = {
                'interval': interval,
                'range': period
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_yahoo_data(data)
                    else:
                        logger.warning(f"Yahoo Finance returned status {response.status}")
                        return None
                        
        except Exception as e:
            logger.error(f"Yahoo Finance fetch error: {e}")
            return None
    
    async def _fetch_twelve_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Twelve Data"""
        api_key = os.getenv('TWELVE_DATA_API_KEY')
        if not api_key:
            return None
        
        try:
            url = "https://api.twelvedata.com/time_series"
            params = {
                'symbol': symbol,
                'interval': interval,
                'apikey': api_key,
                'outputsize': '1000'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_twelve_data(data)
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"Twelve Data fetch error: {e}")
            return None
    
    async def _fetch_alpha_vantage_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Alpha Vantage"""
        api_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        if not api_key:
            return None
        
        try:
            url = "https://www.alphavantage.co/query"
            params = {
                'function': 'TIME_SERIES_DAILY',
                'symbol': symbol,
                'apikey': api_key,
                'outputsize': 'compact'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_alpha_vantage_data(data)
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"Alpha Vantage fetch error: {e}")
            return None
    
    async def _fetch_polygon_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from Polygon"""
        api_key = os.getenv('POLYGON_API_KEY')
        if not api_key:
            return None
        
        try:
            # Convert period to date range
            end_date = datetime.now()
            if period == '1y':
                start_date = end_date - timedelta(days=365)
            elif period == '6m':
                start_date = end_date - timedelta(days=180)
            else:
                start_date = end_date - timedelta(days=30)
            
            url = f"https://api.polygon.io/v2/aggs/ticker/{symbol}/range/1/day/{start_date.strftime('%Y-%m-%d')}/{end_date.strftime('%Y-%m-%d')}"
            params = {'apikey': api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_polygon_data(data)
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"Polygon fetch error: {e}")
            return None
    
    async def _fetch_iex_data(self, symbol: str, interval: str, period: str) -> Optional[Dict[str, Any]]:
        """Fetch data from IEX Cloud"""
        api_key = os.getenv('IEX_CLOUD_API_KEY')
        if not api_key:
            return None
        
        try:
            # Convert period to IEX range
            range_map = {'1d': '1d', '5d': '5d', '1m': '1m', '3m': '3m', '6m': '6m', '1y': '1y'}
            iex_range = range_map.get(period, '1y')
            
            url = f"https://cloud.iexapis.com/stable/stock/{symbol}/chart/{iex_range}"
            params = {'token': api_key}
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, params=params, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._normalize_iex_data(data)
                    else:
                        return None
                        
        except Exception as e:
            logger.error(f"IEX Cloud fetch error: {e}")
            return None
    
    def _normalize_yahoo_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Yahoo Finance data format"""
        try:
            chart = data['chart']['result'][0]
            timestamps = chart['timestamp']
            quotes = chart['indicators']['quote'][0]
            
            normalized = {
                'symbol': chart['meta']['symbol'],
                'source': 'yahoo_finance',
                'data': []
            }
            
            for i, ts in enumerate(timestamps):
                normalized['data'].append({
                    'date': datetime.fromtimestamp(ts).isoformat(),
                    'open': quotes['open'][i],
                    'high': quotes['high'][i],
                    'low': quotes['low'][i],
                    'close': quotes['close'][i],
                    'volume': quotes['volume'][i] if quotes['volume'][i] else 0
                })
            
            return normalized
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error normalizing Yahoo data: {e}")
            return None
    
    def _normalize_twelve_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Twelve Data format"""
        try:
            normalized = {
                'symbol': data['meta']['symbol'],
                'source': 'twelve_data',
                'data': []
            }
            
            for item in data['values']:
                normalized['data'].append({
                    'date': item['datetime'],
                    'open': float(item['open']),
                    'high': float(item['high']),
                    'low': float(item['low']),
                    'close': float(item['close']),
                    'volume': int(item['volume']) if item['volume'] else 0
                })
            
            return normalized
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error normalizing Twelve Data: {e}")
            return None
    
    def _normalize_alpha_vantage_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Alpha Vantage data format"""
        try:
            time_series = data['Time Series (Daily)']
            symbol = data['Meta Data']['2. Symbol']
            
            normalized = {
                'symbol': symbol,
                'source': 'alpha_vantage',
                'data': []
            }
            
            for date, values in time_series.items():
                normalized['data'].append({
                    'date': date,
                    'open': float(values['1. open']),
                    'high': float(values['2. high']),
                    'low': float(values['3. low']),
                    'close': float(values['4. close']),
                    'volume': int(values['5. volume'])
                })
            
            return normalized
            
        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error normalizing Alpha Vantage data: {e}")
            return None
    
    def _normalize_polygon_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize Polygon data format"""
        try:
            normalized = {
                'symbol': data['ticker'],
                'source': 'polygon',
                'data': []
            }
            
            for item in data['results']:
                normalized['data'].append({
                    'date': datetime.fromtimestamp(item['t'] / 1000).isoformat(),
                    'open': item['o'],
                    'high': item['h'],
                    'low': item['l'],
                    'close': item['c'],
                    'volume': item['v']
                })
            
            return normalized
            
        except (KeyError, TypeError) as e:
            logger.error(f"Error normalizing Polygon data: {e}")
            return None
    
    def _normalize_iex_data(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Normalize IEX Cloud data format"""
        try:
            normalized = {
                'symbol': data[0]['symbol'] if data else 'UNKNOWN',
                'source': 'iex_cloud',
                'data': []
            }
            
            for item in data:
                normalized['data'].append({
                    'date': item['date'],
                    'open': item['open'] or 0,
                    'high': item['high'] or 0,
                    'low': item['low'] or 0,
                    'close': item['close'] or 0,
                    'volume': item['volume'] or 0
                })
            
            return normalized
            
        except (KeyError, IndexError, TypeError) as e:
            logger.error(f"Error normalizing IEX data: {e}")
            return None
    
    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {
            'current_source': self.current_source,
            'sources': {},
            'failover_count': len(self.failover_history),
            'last_failover': self.failover_history[-1] if self.failover_history else None
        }
        
        for name, source in self.data_sources.items():
            status['sources'][name] = {
                'name': source.name,
                'is_active': source.is_active,
                'priority': source.priority,
                'consecutive_failures': source.consecutive_failures,
                'last_error': source.last_error,
                'last_success': source.last_success.isoformat() if source.last_success else None,
                'rate_limit_per_minute': source.rate_limit_per_minute
            }
        
        return status

async def main():
    """Test the data source manager"""
    manager = DataSourceManager()
    
    # Test switching sources
    print("🔄 Testing data source switching...")
    success = manager.switch_to_alternative_apis()
    print(f"Switch successful: {success}")
    
    # Test getting market data
    print("\n📊 Testing market data retrieval...")
    data = await manager.get_market_data('AAPL', '1d', '5d')
    if data:
        print(f"✅ Successfully retrieved data from {data['source']}")
        print(f"   Symbol: {data['symbol']}")
        print(f"   Data points: {len(data['data'])}")
    else:
        print("❌ Failed to retrieve market data")
    
    # Show status
    print("\n📈 Data source status:")
    status = manager.get_data_source_status()
    print(f"Current source: {status['current_source']}")
    print(f"Failover count: {status['failover_count']}")
    
    for name, source_status in status['sources'].items():
        active_icon = "✅" if source_status['is_active'] else "❌"
        print(f"  {active_icon} {name}: Priority {source_status['priority']}, Failures: {source_status['consecutive_failures']}")

if __name__ == "__main__":
    asyncio.run(main())