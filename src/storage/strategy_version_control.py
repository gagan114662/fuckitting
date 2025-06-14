#!/usr/bin/env python3
"""
Strategy Version Control System
Tracks all strategy versions with git-like functionality
"""
import json
import os
import hashlib
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from loguru import logger

@dataclass
class StrategyVersion:
    """Represents a specific version of a strategy"""
    version_id: str
    strategy_name: str
    version_number: str
    created_date: str
    author: str
    commit_message: str
    file_path: str
    code_hash: str
    parent_version: Optional[str]
    performance_metrics: Dict[str, Any]
    quantconnect_project_id: Optional[str]
    tags: List[str]

@dataclass
class StrategyBranch:
    """Represents a branch of strategy development"""
    branch_name: str
    head_version: str
    created_date: str
    description: str

class StrategyVersionControl:
    """Git-like version control for trading strategies"""
    
    def __init__(self, base_path: str = "data/strategies"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # Create directory structure
        (self.base_path / "versions").mkdir(exist_ok=True)
        (self.base_path / "branches").mkdir(exist_ok=True)
        (self.base_path / "backups").mkdir(exist_ok=True)
        (self.base_path / "exports").mkdir(exist_ok=True)
        
        self.versions_file = self.base_path / "version_registry.json"
        self.branches_file = self.base_path / "branches.json"
        
        self.versions = self._load_versions()
        self.branches = self._load_branches()
        
        # Ensure main branch exists
        if "main" not in self.branches:
            self.create_branch("main", "Main development branch")
    
    def _load_versions(self) -> Dict[str, StrategyVersion]:
        """Load version registry"""
        if self.versions_file.exists():
            try:
                with open(self.versions_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: StrategyVersion(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load versions: {e}")
                return {}
        return {}
    
    def _load_branches(self) -> Dict[str, StrategyBranch]:
        """Load branch registry"""
        if self.branches_file.exists():
            try:
                with open(self.branches_file, 'r') as f:
                    data = json.load(f)
                    return {
                        k: StrategyBranch(**v) for k, v in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load branches: {e}")
                return {}
        return {}
    
    def _save_versions(self):
        """Save version registry"""
        try:
            data = {k: asdict(v) for k, v in self.versions.items()}
            with open(self.versions_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save versions: {e}")
    
    def _save_branches(self):
        """Save branch registry"""
        try:
            data = {k: asdict(v) for k, v in self.branches.items()}
            with open(self.branches_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save branches: {e}")
    
    def _calculate_code_hash(self, code: str) -> str:
        """Calculate hash of strategy code"""
        return hashlib.sha256(code.encode()).hexdigest()
    
    def _generate_version_id(self, strategy_name: str) -> str:
        """Generate unique version ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        return f"{strategy_name}_{timestamp}_{os.urandom(4).hex()}"
    
    def commit_strategy(
        self,
        strategy_name: str,
        code: str,
        commit_message: str,
        author: str = "AlgoForge",
        branch: str = "main",
        quantconnect_project_id: Optional[str] = None,
        performance_metrics: Optional[Dict] = None,
        tags: Optional[List[str]] = None
    ) -> str:
        """
        Commit a new version of a strategy
        
        Args:
            strategy_name: Name of the strategy
            code: Strategy code content
            commit_message: Description of changes
            author: Author of the changes
            branch: Branch to commit to
            quantconnect_project_id: Associated QuantConnect project
            performance_metrics: Performance data from backtests
            tags: Tags for this version (e.g., ['production', 'high-sharpe'])
            
        Returns:
            Version ID of the committed strategy
        """
        try:
            # Validate branch exists
            if branch not in self.branches:
                raise ValueError(f"Branch '{branch}' does not exist")
            
            # Calculate version number
            existing_versions = [
                v for v in self.versions.values() 
                if v.strategy_name == strategy_name
            ]
            version_number = f"v{len(existing_versions) + 1}.0"
            
            # Generate version ID
            version_id = self._generate_version_id(strategy_name)
            
            # Calculate code hash
            code_hash = self._calculate_code_hash(code)
            
            # Check if this exact code already exists
            duplicate = self._find_duplicate_code(code_hash, strategy_name)
            if duplicate:
                logger.warning(f"Identical code already exists as version {duplicate}")
                return duplicate
            
            # Get parent version (current head of branch)
            parent_version = self.branches[branch].head_version if self.branches[branch].head_version else None
            
            # Create file path
            file_path = self.base_path / "versions" / f"{version_id}.py"
            
            # Store strategy code
            with open(file_path, 'w') as f:
                f.write(self._create_strategy_file_header(
                    strategy_name, version_id, version_number, commit_message, author
                ))
                f.write("\n\n")
                f.write(code)
            
            # Create version object
            version = StrategyVersion(
                version_id=version_id,
                strategy_name=strategy_name,
                version_number=version_number,
                created_date=datetime.now().isoformat(),
                author=author,
                commit_message=commit_message,
                file_path=str(file_path),
                code_hash=code_hash,
                parent_version=parent_version,
                performance_metrics=performance_metrics or {},
                quantconnect_project_id=quantconnect_project_id,
                tags=tags or []
            )
            
            # Store version
            self.versions[version_id] = version
            
            # Update branch head
            self.branches[branch].head_version = version_id
            
            # Save everything
            self._save_versions()
            self._save_branches()
            
            # Create backup
            self._create_backup(version_id)
            
            logger.success(f"✅ STRATEGY COMMITTED: {strategy_name} {version_number}")
            logger.info(f"📝 Version ID: {version_id}")
            logger.info(f"🌿 Branch: {branch}")
            logger.info(f"📁 File: {file_path}")
            logger.info(f"🔐 Hash: {code_hash[:12]}...")
            
            return version_id
            
        except Exception as e:
            logger.error(f"❌ Failed to commit strategy: {e}")
            raise
    
    def _create_strategy_file_header(self, name: str, version_id: str, version: str, message: str, author: str) -> str:
        """Create header for strategy file"""
        return f'''"""
{name} - {version}

Version ID: {version_id}
Created: {datetime.now().isoformat()}
Author: {author}
Commit: {message}

This file is under AlgoForge Strategy Version Control.
Original code follows below.
"""'''
    
    def _find_duplicate_code(self, code_hash: str, strategy_name: str) -> Optional[str]:
        """Find if identical code already exists"""
        for version_id, version in self.versions.items():
            if (version.code_hash == code_hash and 
                version.strategy_name == strategy_name):
                return version_id
        return None
    
    def _create_backup(self, version_id: str):
        """Create backup of strategy version"""
        if version_id not in self.versions:
            return
        
        version = self.versions[version_id]
        backup_dir = self.base_path / "backups" / version.strategy_name
        backup_dir.mkdir(parents=True, exist_ok=True)
        
        backup_path = backup_dir / f"{version_id}_backup.py"
        shutil.copy2(version.file_path, backup_path)
    
    def create_branch(self, branch_name: str, description: str = "") -> bool:
        """Create a new branch"""
        if branch_name in self.branches:
            logger.warning(f"Branch '{branch_name}' already exists")
            return False
        
        self.branches[branch_name] = StrategyBranch(
            branch_name=branch_name,
            head_version="",  # Empty until first commit
            created_date=datetime.now().isoformat(),
            description=description
        )
        
        self._save_branches()
        logger.success(f"✅ Created branch: {branch_name}")
        return True
    
    def list_versions(self, strategy_name: Optional[str] = None, branch: Optional[str] = None) -> List[Dict]:
        """List strategy versions with filters"""
        versions = []
        
        for version_id, version in self.versions.items():
            # Apply filters
            if strategy_name and version.strategy_name != strategy_name:
                continue
            
            if branch:
                # Check if this version is in the branch's history
                if not self._is_version_in_branch(version_id, branch):
                    continue
            
            versions.append({
                "version_id": version_id,
                "strategy_name": version.strategy_name,
                "version_number": version.version_number,
                "created_date": version.created_date,
                "author": version.author,
                "commit_message": version.commit_message,
                "performance_metrics": version.performance_metrics,
                "tags": version.tags,
                "quantconnect_project_id": version.quantconnect_project_id
            })
        
        return sorted(versions, key=lambda x: x["created_date"], reverse=True)
    
    def _is_version_in_branch(self, version_id: str, branch_name: str) -> bool:
        """Check if version is in branch history"""
        if branch_name not in self.branches:
            return False
        
        # Traverse back from branch head
        current = self.branches[branch_name].head_version
        visited = set()
        
        while current and current not in visited:
            if current == version_id:
                return True
            visited.add(current)
            current = self.versions[current].parent_version if current in self.versions else None
        
        return False
    
    def get_strategy_code(self, version_id: str) -> str:
        """Get strategy code for a specific version"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        with open(version.file_path, 'r') as f:
            return f.read()
    
    def tag_version(self, version_id: str, tag: str) -> bool:
        """Add tag to a version"""
        if version_id not in self.versions:
            return False
        
        if tag not in self.versions[version_id].tags:
            self.versions[version_id].tags.append(tag)
            self._save_versions()
            logger.info(f"🏷️ Tagged {version_id} with '{tag}'")
        
        return True
    
    def update_performance_metrics(self, version_id: str, metrics: Dict[str, Any]) -> bool:
        """Update performance metrics for a version"""
        if version_id not in self.versions:
            return False
        
        self.versions[version_id].performance_metrics.update(metrics)
        self._save_versions()
        logger.info(f"📊 Updated performance metrics for {version_id}")
        return True
    
    def export_strategy_version(self, version_id: str, export_path: Optional[str] = None) -> str:
        """Export a strategy version with all metadata"""
        if version_id not in self.versions:
            raise ValueError(f"Version {version_id} not found")
        
        version = self.versions[version_id]
        
        # Get strategy code
        code = self.get_strategy_code(version_id)
        
        # Create export package
        export_data = {
            "export_info": {
                "exported_at": datetime.now().isoformat(),
                "exporter": "AlgoForge Strategy Version Control",
                "version": "1.0"
            },
            "version_metadata": asdict(version),
            "strategy_code": code,
            "version_history": self._get_version_history(version_id)
        }
        
        # Determine export path
        if export_path is None:
            export_path = (self.base_path / "exports" / 
                          f"{version.strategy_name}_{version.version_number}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        
        # Write export file
        with open(export_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"📤 Exported strategy version {version_id} to {export_path}")
        return str(export_path)
    
    def _get_version_history(self, version_id: str) -> List[Dict]:
        """Get history of versions leading to this version"""
        history = []
        current = version_id
        visited = set()
        
        while current and current not in visited and current in self.versions:
            version = self.versions[current]
            history.append({
                "version_id": current,
                "version_number": version.version_number,
                "created_date": version.created_date,
                "commit_message": version.commit_message,
                "author": version.author
            })
            visited.add(current)
            current = version.parent_version
        
        return history
    
    def get_strategy_summary(self) -> Dict[str, Any]:
        """Get summary of all strategies and versions"""
        strategies = {}
        
        for version in self.versions.values():
            name = version.strategy_name
            if name not in strategies:
                strategies[name] = {
                    "total_versions": 0,
                    "latest_version": None,
                    "branches": set(),
                    "tags": set(),
                    "performance_summary": {}
                }
            
            strategies[name]["total_versions"] += 1
            
            # Update latest version
            if (strategies[name]["latest_version"] is None or 
                version.created_date > strategies[name]["latest_version"]["created_date"]):
                strategies[name]["latest_version"] = {
                    "version_id": version.version_id,
                    "version_number": version.version_number,
                    "created_date": version.created_date
                }
            
            # Collect tags
            strategies[name]["tags"].update(version.tags)
        
        # Convert sets to lists for JSON serialization
        for strategy in strategies.values():
            strategy["tags"] = list(strategy["tags"])
            strategy["branches"] = list(strategy["branches"])
        
        return {
            "total_strategies": len(strategies),
            "total_versions": len(self.versions),
            "total_branches": len(self.branches),
            "strategies": strategies,
            "storage_location": str(self.base_path)
        }

# Initialize global version control
strategy_version_control = StrategyVersionControl()

def commit_strategy_version(
    strategy_name: str,
    code: str,
    commit_message: str,
    **kwargs
) -> str:
    """Convenient function to commit a strategy version"""
    return strategy_version_control.commit_strategy(
        strategy_name, code, commit_message, **kwargs
    )

def get_strategy_versions_summary() -> Dict:
    """Get summary of all strategy versions"""
    return strategy_version_control.get_strategy_summary()

if __name__ == "__main__":
    # Demo usage
    vc = StrategyVersionControl()
    summary = vc.get_strategy_summary()
    print("📊 Strategy Version Control Summary:")
    print(json.dumps(summary, indent=2))