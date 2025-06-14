#!/usr/bin/env python3
"""
Real QuantConnect Backtest Storage System
Stores actual JSON responses from QuantConnect API for verification
"""
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from loguru import logger

@dataclass
class BacktestMetadata:
    """Metadata for backtest results"""
    backtest_id: str
    project_id: str
    strategy_name: str
    created_date: str
    completed_date: Optional[str]
    status: str
    algorithm_name: str
    api_response_hash: str
    file_path: str
    performance_summary: Dict[str, Any]

class BacktestStorage:
    """Manages storage of real QuantConnect backtest JSON responses"""
    
    def __init__(self, base_path: str = "data/backtests"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for organization
        (self.base_path / "raw_json").mkdir(exist_ok=True)
        (self.base_path / "processed").mkdir(exist_ok=True)
        (self.base_path / "metadata").mkdir(exist_ok=True)
        (self.base_path / "failed").mkdir(exist_ok=True)
        
        self.metadata_file = self.base_path / "backtest_registry.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, BacktestMetadata]:
        """Load backtest metadata registry"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: BacktestMetadata(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load metadata: {e}")
                return {}
        return {}
    
    def _save_metadata(self):
        """Save metadata registry"""
        try:
            data = {k: asdict(v) for k, v in self.metadata.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save metadata: {e}")
    
    def _calculate_hash(self, data: Dict) -> str:
        """Calculate hash of API response for integrity verification"""
        json_str = json.dumps(data, sort_keys=True)
        return hashlib.sha256(json_str.encode()).hexdigest()
    
    def store_raw_backtest(
        self, 
        backtest_response: Dict[str, Any],
        project_id: str,
        strategy_name: str
    ) -> str:
        """
        Store raw QuantConnect backtest JSON response
        
        Args:
            backtest_response: Raw JSON response from QuantConnect API
            project_id: QuantConnect project ID
            strategy_name: Name of the strategy
            
        Returns:
            File path where the backtest was stored
        """
        try:
            # Extract backtest information
            backtest_id = backtest_response.get('backtest', {}).get('backtestId', 'unknown')
            if backtest_id == 'unknown':
                # Fallback: look in different response structure
                backtest_id = backtest_response.get('backtestId', f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            
            # Create timestamp for file naming
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # Calculate hash for integrity
            response_hash = self._calculate_hash(backtest_response)
            
            # Create filename
            filename = f"{strategy_name}_{backtest_id}_{timestamp}.json"
            file_path = self.base_path / "raw_json" / filename
            
            # Store raw JSON with metadata header
            storage_data = {
                "metadata": {
                    "stored_at": datetime.now().isoformat(),
                    "backtest_id": backtest_id,
                    "project_id": project_id,
                    "strategy_name": strategy_name,
                    "response_hash": response_hash,
                    "quantconnect_api_version": "v2",
                    "storage_version": "1.0"
                },
                "raw_response": backtest_response
            }
            
            # Write to file
            with open(file_path, 'w') as f:
                json.dump(storage_data, f, indent=2)
            
            # Extract performance summary
            performance_summary = self._extract_performance_summary(backtest_response)
            
            # Create metadata entry
            metadata = BacktestMetadata(
                backtest_id=backtest_id,
                project_id=project_id,
                strategy_name=strategy_name,
                created_date=timestamp,
                completed_date=None,  # Will be updated when backtest completes
                status=backtest_response.get('backtest', {}).get('status', 'unknown'),
                algorithm_name=backtest_response.get('backtest', {}).get('name', strategy_name),
                api_response_hash=response_hash,
                file_path=str(file_path),
                performance_summary=performance_summary
            )
            
            # Store metadata
            self.metadata[backtest_id] = metadata
            self._save_metadata()
            
            logger.success(f"✅ REAL BACKTEST STORED: {filename}")
            logger.info(f"📁 Location: {file_path}")
            logger.info(f"🔐 Hash: {response_hash[:12]}...")
            
            return str(file_path)
            
        except Exception as e:
            logger.error(f"❌ Failed to store backtest: {e}")
            # Store in failed directory for debugging
            failed_file = self.base_path / "failed" / f"failed_{timestamp}.json"
            try:
                with open(failed_file, 'w') as f:
                    json.dump({
                        "error": str(e),
                        "timestamp": datetime.now().isoformat(),
                        "attempted_response": backtest_response
                    }, f, indent=2)
            except:
                pass
            raise
    
    def _extract_performance_summary(self, response: Dict) -> Dict[str, Any]:
        """Extract key performance metrics from backtest response"""
        try:
            # Try different possible response structures
            statistics = {}
            
            # Structure 1: response.backtest.statistics
            if 'backtest' in response and 'statistics' in response['backtest']:
                statistics = response['backtest']['statistics']
            
            # Structure 2: response.statistics
            elif 'statistics' in response:
                statistics = response['statistics']
            
            # Structure 3: response.result.statistics
            elif 'result' in response and 'statistics' in response['result']:
                statistics = response['result']['statistics']
            
            # Extract key metrics with safe defaults
            return {
                "total_return": statistics.get('Total Return', 0),
                "cagr": statistics.get('Compounding Annual Return', 0),
                "sharpe_ratio": statistics.get('Sharpe Ratio', 0),
                "max_drawdown": statistics.get('Drawdown', 0),
                "total_trades": statistics.get('Total Trades', 0),
                "win_rate": statistics.get('Win Rate', 0),
                "profit_loss_ratio": statistics.get('Profit-Loss Ratio', 0),
                "alpha": statistics.get('Alpha', 0),
                "beta": statistics.get('Beta', 0),
                "annual_variance": statistics.get('Annual Variance', 0),
                "information_ratio": statistics.get('Information Ratio', 0),
                "tracking_error": statistics.get('Tracking Error', 0),
                "treynor_ratio": statistics.get('Treynor Ratio', 0),
                "sortino_ratio": statistics.get('Sortino Ratio', 0)
            }
            
        except Exception as e:
            logger.warning(f"Could not extract performance summary: {e}")
            return {"extraction_error": str(e)}
    
    def update_backtest_completion(self, backtest_id: str, completion_response: Dict):
        """Update backtest when it completes"""
        if backtest_id in self.metadata:
            # Update completion info
            self.metadata[backtest_id].completed_date = datetime.now().strftime('%Y%m%d_%H%M%S')
            self.metadata[backtest_id].status = "completed"
            
            # Update performance summary with final results
            final_performance = self._extract_performance_summary(completion_response)
            self.metadata[backtest_id].performance_summary.update(final_performance)
            
            # Store completion response as separate file
            original_path = Path(self.metadata[backtest_id].file_path)
            completion_path = original_path.parent / f"completion_{original_path.name}"
            
            with open(completion_path, 'w') as f:
                json.dump({
                    "completion_data": completion_response,
                    "completed_at": datetime.now().isoformat(),
                    "original_backtest_file": str(original_path)
                }, f, indent=2)
            
            self._save_metadata()
            logger.success(f"✅ Updated backtest completion: {backtest_id}")
    
    def verify_backtest_integrity(self, backtest_id: str) -> bool:
        """Verify stored backtest hasn't been tampered with"""
        if backtest_id not in self.metadata:
            return False
        
        file_path = self.metadata[backtest_id].file_path
        stored_hash = self.metadata[backtest_id].api_response_hash
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                current_hash = self._calculate_hash(data['raw_response'])
                return current_hash == stored_hash
        except Exception as e:
            logger.error(f"Integrity check failed for {backtest_id}: {e}")
            return False
    
    def get_backtest_summary(self) -> Dict[str, Any]:
        """Get summary of all stored backtests"""
        return {
            "total_backtests": len(self.metadata),
            "completed_backtests": len([m for m in self.metadata.values() if m.completed_date]),
            "strategies": list(set(m.strategy_name for m in self.metadata.values())),
            "date_range": {
                "earliest": min((m.created_date for m in self.metadata.values()), default="none"),
                "latest": max((m.created_date for m in self.metadata.values()), default="none")
            },
            "average_performance": self._calculate_average_performance(),
            "storage_location": str(self.base_path)
        }
    
    def _calculate_average_performance(self) -> Dict[str, float]:
        """Calculate average performance across all backtests"""
        if not self.metadata:
            return {}
        
        completed = [m for m in self.metadata.values() if m.completed_date]
        if not completed:
            return {"note": "no_completed_backtests"}
        
        metrics = ["cagr", "sharpe_ratio", "max_drawdown", "win_rate", "total_trades"]
        averages = {}
        
        for metric in metrics:
            values = [
                m.performance_summary.get(metric, 0) 
                for m in completed 
                if isinstance(m.performance_summary.get(metric), (int, float))
            ]
            if values:
                averages[f"avg_{metric}"] = sum(values) / len(values)
        
        return averages
    
    def list_backtests(self, strategy_name: Optional[str] = None) -> List[Dict]:
        """List all stored backtests with metadata"""
        backtests = []
        for backtest_id, metadata in self.metadata.items():
            if strategy_name is None or metadata.strategy_name == strategy_name:
                backtests.append({
                    "backtest_id": backtest_id,
                    "strategy_name": metadata.strategy_name,
                    "created_date": metadata.created_date,
                    "status": metadata.status,
                    "file_path": metadata.file_path,
                    "performance_summary": metadata.performance_summary,
                    "integrity_verified": self.verify_backtest_integrity(backtest_id)
                })
        return sorted(backtests, key=lambda x: x["created_date"], reverse=True)
    
    def export_backtest_data(self, backtest_id: str, export_path: Optional[str] = None) -> str:
        """Export backtest data for external analysis"""
        if backtest_id not in self.metadata:
            raise ValueError(f"Backtest {backtest_id} not found")
        
        metadata = self.metadata[backtest_id]
        
        # Load the raw data
        with open(metadata.file_path, 'r') as f:
            data = json.load(f)
        
        # Create export package
        export_data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "backtest_id": backtest_id,
                "original_file": metadata.file_path,
                "integrity_verified": self.verify_backtest_integrity(backtest_id)
            },
            "metadata": asdict(metadata),
            "raw_quantconnect_response": data['raw_response'],
            "performance_analysis": metadata.performance_summary
        }
        
        # Determine export path
        if export_path is None:
            export_path = self.base_path / "processed" / f"export_{backtest_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        # Write export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Exported backtest {backtest_id} to {export_path}")
        return str(export_path)

# Initialize global storage instance
backtest_storage = BacktestStorage()

def store_quantconnect_backtest(response: Dict, project_id: str, strategy_name: str) -> str:
    """Convenient function to store a QuantConnect backtest response"""
    return backtest_storage.store_raw_backtest(response, project_id, strategy_name)

def get_stored_backtests_summary() -> Dict:
    """Get summary of all stored backtests"""
    return backtest_storage.get_backtest_summary()

if __name__ == "__main__":
    # Demo usage
    storage = BacktestStorage()
    summary = storage.get_backtest_summary()
    print("📊 Backtest Storage Summary:")
    print(json.dumps(summary, indent=2))