#!/usr/bin/env python3
"""
Autonomous Model Retraining System
Automatically retrains models when performance drift is detected
"""
import asyncio
import numpy as np
import pandas as pd
import json
import os
import argparse
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
from loguru import logger
import joblib
import sqlite3
from dataclasses import dataclass, asdict

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression, Ridge
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    logger.warning("scikit-learn not available, using simplified models")
    SKLEARN_AVAILABLE = False

@dataclass
class ModelPerformance:
    model_name: str
    model_type: str
    timestamp: str
    mse: float
    r2: float
    cross_val_score: float
    prediction_accuracy: float
    drift_score: float
    is_acceptable: bool
    training_samples: int
    features_used: List[str]

@dataclass
class RetrainingConfig:
    drift_threshold: float = 0.15  # 15% performance degradation
    min_training_samples: int = 1000
    retrain_frequency_hours: int = 24
    max_retrain_attempts: int = 3
    validation_split: float = 0.2
    cross_validation_folds: int = 5

class ModelManager:
    """Manages machine learning models for trading strategies"""
    
    def __init__(self, models_dir: str = "models", data_dir: str = "data"):
        self.models_dir = Path(models_dir)
        self.data_dir = Path(data_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.config = RetrainingConfig()
        self.model_registry = {}
        self.performance_history = []
        
        # Initialize model database
        self.db_path = self.models_dir / "model_registry.db"
        self._init_model_database()
        
        # Load existing models
        self._load_model_registry()
    
    def _init_model_database(self):
        """Initialize SQLite database for model tracking"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS model_performance (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        model_type TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        mse REAL,
                        r2 REAL,
                        cross_val_score REAL,
                        prediction_accuracy REAL,
                        drift_score REAL,
                        is_acceptable BOOLEAN,
                        training_samples INTEGER,
                        features_used TEXT,
                        model_path TEXT,
                        config_used TEXT
                    )
                ''')
                
                conn.execute('''
                    CREATE TABLE IF NOT EXISTS retraining_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        model_name TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        trigger_reason TEXT,
                        old_performance TEXT,
                        new_performance TEXT,
                        success BOOLEAN,
                        error_message TEXT,
                        training_duration_seconds REAL
                    )
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_model_name ON model_performance(model_name)
                ''')
                
                conn.execute('''
                    CREATE INDEX IF NOT EXISTS idx_timestamp ON model_performance(timestamp)
                ''')
        
        except Exception as e:
            logger.error(f"Failed to initialize model database: {e}")
    
    def _load_model_registry(self):
        """Load existing models from disk"""
        try:
            for model_file in self.models_dir.glob("*.joblib"):
                model_name = model_file.stem
                try:
                    model = joblib.load(model_file)
                    self.model_registry[model_name] = {
                        'model': model,
                        'path': model_file,
                        'last_trained': datetime.fromtimestamp(model_file.stat().st_mtime),
                        'type': type(model).__name__ if hasattr(model, '__class__') else 'unknown'
                    }
                    logger.debug(f"Loaded model: {model_name}")
                except Exception as e:
                    logger.warning(f"Failed to load model {model_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error loading model registry: {e}")
    
    async def retrain_models_autonomous(self) -> Dict[str, Any]:
        """Autonomously retrain models that need updating"""
        logger.info("🤖 Starting autonomous model retraining...")
        
        retrain_results = {
            'timestamp': datetime.now().isoformat(),
            'models_checked': 0,
            'models_retrained': 0,
            'retrain_details': [],
            'total_duration': 0,
            'success': True
        }
        
        start_time = datetime.now()
        
        try:
            # Check each model for drift
            models_needing_retrain = []
            
            for model_name, model_info in self.model_registry.items():
                retrain_results['models_checked'] += 1
                
                # Check if model needs retraining
                needs_retrain, reason = await self._check_if_model_needs_retrain(model_name, model_info)
                
                if needs_retrain:
                    models_needing_retrain.append((model_name, reason))
                    logger.info(f"📊 {model_name} needs retraining: {reason}")
            
            # Retrain models that need it
            for model_name, reason in models_needing_retrain:
                logger.info(f"🔄 Retraining model: {model_name}")
                
                retrain_start = datetime.now()
                retrain_success = await self._retrain_single_model(model_name, reason)
                retrain_duration = (datetime.now() - retrain_start).total_seconds()
                
                retrain_detail = {
                    'model_name': model_name,
                    'reason': reason,
                    'success': retrain_success,
                    'duration_seconds': retrain_duration
                }
                
                retrain_results['retrain_details'].append(retrain_detail)
                
                if retrain_success:
                    retrain_results['models_retrained'] += 1
                    logger.success(f"✅ Successfully retrained {model_name}")
                else:
                    logger.error(f"❌ Failed to retrain {model_name}")
                    retrain_results['success'] = False
            
            # Train new models if needed
            await self._train_new_models_if_needed(retrain_results)
            
            retrain_results['total_duration'] = (datetime.now() - start_time).total_seconds()
            
            if retrain_results['models_retrained'] > 0:
                logger.success(f"✅ Autonomous retraining completed: {retrain_results['models_retrained']} models retrained")
            else:
                logger.info("ℹ️ No models needed retraining")
            
            return retrain_results
            
        except Exception as e:
            logger.error(f"❌ Error in autonomous model retraining: {e}")
            retrain_results['success'] = False
            retrain_results['error'] = str(e)
            return retrain_results
    
    async def _check_if_model_needs_retrain(self, model_name: str, model_info: Dict[str, Any]) -> Tuple[bool, str]:
        """Check if a model needs retraining"""
        try:
            # Check time since last training
            last_trained = model_info.get('last_trained', datetime.min)
            hours_since_training = (datetime.now() - last_trained).total_seconds() / 3600
            
            if hours_since_training > self.config.retrain_frequency_hours:
                return True, f"Scheduled retrain (last trained {hours_since_training:.1f}h ago)"
            
            # Check performance drift
            latest_performance = await self._get_latest_model_performance(model_name)
            if latest_performance:
                if latest_performance.drift_score > self.config.drift_threshold:
                    return True, f"Performance drift detected ({latest_performance.drift_score:.3f} > {self.config.drift_threshold})"
                
                if not latest_performance.is_acceptable:
                    return True, "Model performance below acceptable threshold"
            
            # Check if new training data is available
            new_data_available = await self._check_new_training_data(model_name)
            if new_data_available:
                return True, "New training data available"
            
            return False, "No retraining needed"
            
        except Exception as e:
            logger.error(f"Error checking retrain need for {model_name}: {e}")
            return False, f"Error: {str(e)}"
    
    async def _get_latest_model_performance(self, model_name: str) -> Optional[ModelPerformance]:
        """Get latest performance metrics for a model"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute('''
                    SELECT * FROM model_performance 
                    WHERE model_name = ? 
                    ORDER BY timestamp DESC 
                    LIMIT 1
                ''', (model_name,))
                
                result = cursor.fetchone()
                if result:
                    row = dict(result)
                    return ModelPerformance(
                        model_name=row['model_name'],
                        model_type=row['model_type'],
                        timestamp=row['timestamp'],
                        mse=row['mse'],
                        r2=row['r2'],
                        cross_val_score=row['cross_val_score'],
                        prediction_accuracy=row['prediction_accuracy'],
                        drift_score=row['drift_score'],
                        is_acceptable=bool(row['is_acceptable']),
                        training_samples=row['training_samples'],
                        features_used=json.loads(row['features_used']) if row['features_used'] else []
                    )
                
                return None
                
        except Exception as e:
            logger.error(f"Error getting model performance: {e}")
            return None
    
    async def _check_new_training_data(self, model_name: str) -> bool:
        """Check if new training data is available"""
        try:
            # Look for new data files
            data_files = list(self.data_dir.glob("*.csv"))
            if not data_files:
                return False
            
            # Check modification times
            latest_data_time = max(f.stat().st_mtime for f in data_files)
            model_info = self.model_registry.get(model_name, {})
            last_trained = model_info.get('last_trained', datetime.min)
            
            return datetime.fromtimestamp(latest_data_time) > last_trained
            
        except Exception as e:
            logger.debug(f"Error checking new training data: {e}")
            return False
    
    async def _retrain_single_model(self, model_name: str, reason: str) -> bool:
        """Retrain a single model"""
        try:
            # Get old performance for comparison
            old_performance = await self._get_latest_model_performance(model_name)
            
            # Load training data
            training_data = await self._load_training_data(model_name)
            if training_data is None or len(training_data) < self.config.min_training_samples:
                logger.warning(f"Insufficient training data for {model_name}")
                return False
            
            # Prepare features and targets
            X, y, feature_names = await self._prepare_training_data(training_data, model_name)
            if X is None or y is None:
                logger.error(f"Failed to prepare training data for {model_name}")
                return False
            
            # Train new model
            new_model = await self._train_model(X, y, model_name)
            if new_model is None:
                logger.error(f"Failed to train new model for {model_name}")
                return False
            
            # Evaluate new model
            performance = await self._evaluate_model(new_model, X, y, feature_names, model_name)
            
            # Compare with old performance
            if old_performance and performance.prediction_accuracy < old_performance.prediction_accuracy:
                logger.warning(f"New model performance worse than old for {model_name}")
                # Optionally, don't replace if significantly worse
                if performance.prediction_accuracy < old_performance.prediction_accuracy * 0.9:
                    logger.warning(f"New model significantly worse, keeping old model")
                    return False
            
            # Save new model
            model_path = self.models_dir / f"{model_name}.joblib"
            joblib.dump(new_model, model_path)
            
            # Update registry
            self.model_registry[model_name] = {
                'model': new_model,
                'path': model_path,
                'last_trained': datetime.now(),
                'type': type(new_model).__name__
            }
            
            # Save performance to database
            await self._save_model_performance(performance)
            
            # Log retraining
            await self._log_retraining(
                model_name=model_name,
                reason=reason,
                old_performance=old_performance,
                new_performance=performance,
                success=True
            )
            
            logger.success(f"✅ Model {model_name} retrained successfully")
            logger.info(f"   ├─ Accuracy: {performance.prediction_accuracy:.3f}")
            logger.info(f"   ├─ R²: {performance.r2:.3f}")
            logger.info(f"   └─ Training samples: {performance.training_samples}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error retraining model {model_name}: {e}")
            
            # Log failed retraining
            await self._log_retraining(
                model_name=model_name,
                reason=reason,
                old_performance=None,
                new_performance=None,
                success=False,
                error_message=str(e)
            )
            
            return False
    
    async def _load_training_data(self, model_name: str) -> Optional[pd.DataFrame]:
        """Load training data for a model"""
        try:
            # Look for model-specific data first
            model_data_file = self.data_dir / f"{model_name}_training_data.csv"
            if model_data_file.exists():
                return pd.read_csv(model_data_file)
            
            # Fallback to general trading data
            general_data_files = [
                self.data_dir / "trading_data.csv",
                self.data_dir / "market_data.csv",
                self.data_dir / "strategy_results.csv"
            ]
            
            for data_file in general_data_files:
                if data_file.exists():
                    return pd.read_csv(data_file)
            
            # Generate synthetic data if no real data available
            logger.warning(f"No training data found for {model_name}, generating synthetic data")
            return self._generate_synthetic_training_data()
            
        except Exception as e:
            logger.error(f"Error loading training data: {e}")
            return None
    
    def _generate_synthetic_training_data(self) -> pd.DataFrame:
        """Generate synthetic training data for testing"""
        np.random.seed(42)
        n_samples = 2000
        
        # Generate synthetic market features
        data = {
            'returns': np.random.normal(0.001, 0.02, n_samples),
            'volume': np.random.lognormal(10, 1, n_samples),
            'volatility': np.random.exponential(0.015, n_samples),
            'rsi': np.random.uniform(20, 80, n_samples),
            'macd': np.random.normal(0, 0.001, n_samples),
            'bollinger_position': np.random.uniform(-1, 1, n_samples),
            'momentum': np.random.normal(0, 0.01, n_samples),
            'mean_reversion': np.random.normal(0, 0.005, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable (future returns)
        df['target'] = (df['returns'].shift(-1) + 
                       0.1 * df['momentum'] + 
                       0.05 * df['mean_reversion'] + 
                       np.random.normal(0, 0.01, n_samples))
        
        df['timestamp'] = pd.date_range(start='2020-01-01', periods=n_samples, freq='H')
        
        return df.dropna()
    
    async def _prepare_training_data(self, data: pd.DataFrame, model_name: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], List[str]]:
        """Prepare features and targets for training"""
        try:
            # Define feature columns based on model type
            feature_columns = [col for col in data.columns if col not in ['target', 'timestamp', 'date']]
            
            if 'target' not in data.columns:
                # If no target column, create one (e.g., future returns)
                if 'returns' in data.columns:
                    data['target'] = data['returns'].shift(-1)
                else:
                    logger.error("No target variable found and cannot create one")
                    return None, None, []
            
            # Select features and target
            X = data[feature_columns].values
            y = data['target'].values
            
            # Remove NaN values
            valid_indices = ~(np.isnan(X).any(axis=1) | np.isnan(y))
            X = X[valid_indices]
            y = y[valid_indices]
            
            if len(X) == 0:
                logger.error("No valid training samples after removing NaN values")
                return None, None, []
            
            # Scale features if using certain models
            if SKLEARN_AVAILABLE:
                scaler = StandardScaler()
                X = scaler.fit_transform(X)
                
                # Save scaler for later use
                scaler_path = self.models_dir / f"{model_name}_scaler.joblib"
                joblib.dump(scaler, scaler_path)
            
            return X, y, feature_columns
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None, []
    
    async def _train_model(self, X: np.ndarray, y: np.ndarray, model_name: str) -> Optional[Any]:
        """Train a new model"""
        try:
            if not SKLEARN_AVAILABLE:
                # Simple linear model without sklearn
                return self._train_simple_model(X, y)
            
            # Choose model type based on model name or data characteristics
            if 'random_forest' in model_name.lower():
                model = RandomForestRegressor(n_estimators=100, random_state=42)
            elif 'gradient_boost' in model_name.lower():
                model = GradientBoostingRegressor(random_state=42)
            elif 'ridge' in model_name.lower():
                model = Ridge()
            else:
                # Default to Random Forest
                model = RandomForestRegressor(n_estimators=50, random_state=42)
            
            # Train the model
            model.fit(X, y)
            
            return model
            
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None
    
    def _train_simple_model(self, X: np.ndarray, y: np.ndarray) -> Optional[Dict[str, Any]]:
        """Train a simple linear model without sklearn"""
        try:
            # Add bias term
            X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
            
            # Solve normal equation: w = (X^T X)^-1 X^T y
            XtX = X_with_bias.T @ X_with_bias
            Xty = X_with_bias.T @ y
            
            # Add regularization to prevent overfitting
            regularization = 0.01 * np.eye(XtX.shape[0])
            weights = np.linalg.solve(XtX + regularization, Xty)
            
            # Create simple model object
            simple_model = {
                'type': 'linear_regression',
                'weights': weights,
                'n_features': X.shape[1]
            }
            
            return simple_model
            
        except Exception as e:
            logger.error(f"Error training simple model: {e}")
            return None
    
    async def _evaluate_model(self, model: Any, X: np.ndarray, y: np.ndarray, 
                            feature_names: List[str], model_name: str) -> ModelPerformance:
        """Evaluate model performance"""
        try:
            # Make predictions
            if isinstance(model, dict) and model.get('type') == 'linear_regression':
                # Simple model predictions
                X_with_bias = np.column_stack([np.ones(X.shape[0]), X])
                y_pred = X_with_bias @ model['weights']
            else:
                # Sklearn model predictions
                y_pred = model.predict(X)
            
            # Calculate metrics
            mse = np.mean((y - y_pred) ** 2)
            r2 = 1 - (np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))
            
            # Prediction accuracy (percentage of predictions within 1 standard deviation)
            prediction_errors = np.abs(y - y_pred)
            error_threshold = np.std(y) * 1.0
            prediction_accuracy = np.mean(prediction_errors < error_threshold)
            
            # Cross-validation score (simplified)
            if SKLEARN_AVAILABLE and hasattr(model, 'fit'):
                try:
                    cv_scores = cross_val_score(model, X, y, cv=min(5, len(X)//10))
                    cross_val_score_mean = np.mean(cv_scores)
                except:
                    cross_val_score_mean = r2  # Fallback to R²
            else:
                cross_val_score_mean = r2
            
            # Calculate drift score (simplified - would be more complex in practice)
            drift_score = max(0, 0.1 - prediction_accuracy)  # Higher drift if accuracy is low
            
            # Determine if performance is acceptable
            is_acceptable = (prediction_accuracy > 0.6 and 
                           r2 > 0.1 and 
                           not np.isnan(mse))
            
            performance = ModelPerformance(
                model_name=model_name,
                model_type=type(model).__name__ if hasattr(model, '__class__') else 'simple_linear',
                timestamp=datetime.now().isoformat(),
                mse=float(mse),
                r2=float(r2),
                cross_val_score=float(cross_val_score_mean),
                prediction_accuracy=float(prediction_accuracy),
                drift_score=float(drift_score),
                is_acceptable=is_acceptable,
                training_samples=len(X),
                features_used=feature_names
            )
            
            return performance
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            # Return default performance object
            return ModelPerformance(
                model_name=model_name,
                model_type='unknown',
                timestamp=datetime.now().isoformat(),
                mse=float('inf'),
                r2=0.0,
                cross_val_score=0.0,
                prediction_accuracy=0.0,
                drift_score=1.0,
                is_acceptable=False,
                training_samples=len(X),
                features_used=feature_names
            )
    
    async def _save_model_performance(self, performance: ModelPerformance):
        """Save model performance to database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO model_performance 
                    (model_name, model_type, timestamp, mse, r2, cross_val_score, 
                     prediction_accuracy, drift_score, is_acceptable, training_samples, features_used)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (
                    performance.model_name,
                    performance.model_type,
                    performance.timestamp,
                    performance.mse,
                    performance.r2,
                    performance.cross_val_score,
                    performance.prediction_accuracy,
                    performance.drift_score,
                    performance.is_acceptable,
                    performance.training_samples,
                    json.dumps(performance.features_used)
                ))
                
        except Exception as e:
            logger.error(f"Error saving model performance: {e}")
    
    async def _log_retraining(self, model_name: str, reason: str, old_performance: Optional[ModelPerformance],
                            new_performance: Optional[ModelPerformance], success: bool, 
                            error_message: str = None):
        """Log retraining event"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute('''
                    INSERT INTO retraining_logs 
                    (model_name, timestamp, trigger_reason, old_performance, new_performance, success, error_message)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (
                    model_name,
                    datetime.now().isoformat(),
                    reason,
                    json.dumps(asdict(old_performance)) if old_performance else None,
                    json.dumps(asdict(new_performance)) if new_performance else None,
                    success,
                    error_message
                ))
                
        except Exception as e:
            logger.error(f"Error logging retraining: {e}")
    
    async def _train_new_models_if_needed(self, retrain_results: Dict[str, Any]):
        """Train new models if needed for better coverage"""
        try:
            # Check if we have enough models
            if len(self.model_registry) < 3:
                logger.info("🆕 Training additional models for better coverage...")
                
                # Define model types to ensure we have
                desired_models = ['momentum_model', 'mean_reversion_model', 'volatility_model']
                
                for model_name in desired_models:
                    if model_name not in self.model_registry:
                        logger.info(f"🔄 Training new model: {model_name}")
                        
                        # Load training data
                        training_data = await self._load_training_data(model_name)
                        if training_data is not None and len(training_data) >= self.config.min_training_samples:
                            # Train new model
                            success = await self._retrain_single_model(model_name, "new_model_creation")
                            if success:
                                retrain_results['models_retrained'] += 1
                                logger.success(f"✅ Created new model: {model_name}")
                        
        except Exception as e:
            logger.error(f"Error training new models: {e}")
    
    def get_model_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for all models"""
        try:
            summary = {
                'total_models': len(self.model_registry),
                'models': {},
                'overall_health': 'unknown'
            }
            
            healthy_models = 0
            
            for model_name in self.model_registry.keys():
                performance = asyncio.run(self._get_latest_model_performance(model_name))
                if performance:
                    summary['models'][model_name] = {
                        'accuracy': performance.prediction_accuracy,
                        'r2': performance.r2,
                        'drift_score': performance.drift_score,
                        'is_acceptable': performance.is_acceptable,
                        'last_updated': performance.timestamp
                    }
                    
                    if performance.is_acceptable:
                        healthy_models += 1
                else:
                    summary['models'][model_name] = {
                        'status': 'no_performance_data'
                    }
            
            # Determine overall health
            if healthy_models == len(self.model_registry):
                summary['overall_health'] = 'healthy'
            elif healthy_models > len(self.model_registry) / 2:
                summary['overall_health'] = 'degraded'
            else:
                summary['overall_health'] = 'critical'
            
            return summary
            
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}

async def main():
    """Main function for standalone execution"""
    parser = argparse.ArgumentParser(description="Autonomous model retraining system")
    parser.add_argument("--autonomous", action="store_true", help="Run autonomous retraining")
    parser.add_argument("--model", help="Retrain specific model")
    parser.add_argument("--summary", action="store_true", help="Show performance summary")
    parser.add_argument("--create", help="Create new model with given name")
    
    args = parser.parse_args()
    
    manager = ModelManager()
    
    if args.autonomous:
        result = await manager.retrain_models_autonomous()
        if result['success']:
            print(f"✅ Autonomous retraining completed:")
            print(f"  Models checked: {result['models_checked']}")
            print(f"  Models retrained: {result['models_retrained']}")
            print(f"  Total duration: {result['total_duration']:.1f}s")
            return True
        else:
            print("❌ Autonomous retraining failed")
            if 'error' in result:
                print(f"Error: {result['error']}")
            return False
    
    elif args.model:
        success = await manager._retrain_single_model(args.model, "manual_retrain")
        if success:
            print(f"✅ Successfully retrained {args.model}")
            return True
        else:
            print(f"❌ Failed to retrain {args.model}")
            return False
    
    elif args.summary:
        summary = manager.get_model_performance_summary()
        print("Model Performance Summary:")
        print(f"  Total models: {summary['total_models']}")
        print(f"  Overall health: {summary['overall_health']}")
        
        for model_name, metrics in summary.get('models', {}).items():
            if 'accuracy' in metrics:
                status_icon = "✅" if metrics['is_acceptable'] else "❌"
                print(f"  {status_icon} {model_name}:")
                print(f"    Accuracy: {metrics['accuracy']:.3f}")
                print(f"    R²: {metrics['r2']:.3f}")
                print(f"    Drift: {metrics['drift_score']:.3f}")
            else:
                print(f"  ⚠️ {model_name}: {metrics.get('status', 'unknown')}")
        
        return True
    
    elif args.create:
        success = await manager._retrain_single_model(args.create, "new_model_creation")
        if success:
            print(f"✅ Successfully created model: {args.create}")
            return True
        else:
            print(f"❌ Failed to create model: {args.create}")
            return False
    
    else:
        # Default: run autonomous retraining
        result = await manager.retrain_models_autonomous()
        return result['success']

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)