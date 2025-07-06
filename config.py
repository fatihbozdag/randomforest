"""
Configuration management for Water Quality Analysis System
"""

import os
from typing import Dict, Any, List
import json

class Config:
    """Centralized configuration for water quality analysis"""
    
    def __init__(self, config_path: str = None):
        self.config_path = config_path
        self.settings = self._load_default_config()
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
    
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration settings"""
        return {
            # Data processing settings
            "data": {
                "random_seed": 42,
                "test_size": 0.2,
                "validation_size": 0.1,
                "normalize_features": True,
                "handle_outliers": True,
                "outlier_threshold": 3.0
            },
            
            # Model settings
            "models": {
                "random_forest": {
                    "n_estimators": [100, 200, 300],
                    "max_depth": [10, 20, None],
                    "min_samples_split": [2, 5, 10],
                    "min_samples_leaf": [1, 2, 4],
                    "max_features": ["sqrt", "log2", None]
                },
                "gradient_boosting": {
                    "n_estimators": [100, 200, 300],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 5, 7],
                    "subsample": [0.8, 0.9, 1.0],
                    "colsample_bytree": [0.8, 0.9, 1.0]
                },
                "svr": {
                    "C": [0.1, 1, 10, 100],
                    "gamma": ["scale", "auto", 0.001, 0.01, 0.1],
                    "kernel": ["rbf", "poly", "linear"],
                    "epsilon": [0.01, 0.1, 0.2]
                },
                "neural_network": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 25), (100, 50)],
                    "activation": ["relu", "tanh"],
                    "alpha": [0.0001, 0.001, 0.01],
                    "learning_rate": ["constant", "adaptive"],
                    "max_iter": [500, 1000]
                },
                "linear_models": {
                    "ridge": {"alpha": [0.1, 1, 10, 100]},
                    "lasso": {"alpha": [0.001, 0.01, 0.1, 1]},
                    "elastic_net": {
                        "alpha": [0.001, 0.01, 0.1, 1],
                        "l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]
                    }
                }
            },
            
            # Cross-validation settings
            "cross_validation": {
                "cv_folds": 5,
                "n_repeats": 3,
                "scoring": ["neg_mean_squared_error", "r2", "neg_mean_absolute_error"]
            },
            
            # Optimization settings
            "optimization": {
                "n_iter": 50,
                "cv_folds": 3,
                "n_jobs": -1,
                "random_state": 42
            },
            
            # Visualization settings
            "visualization": {
                "figure_size": (12, 8),
                "dpi": 300,
                "style": "seaborn-v0_8",
                "color_palette": "viridis",
                "save_format": "png"
            },
            
            # Output settings
            "output": {
                "base_dir": "output",
                "models_dir": "models",
                "plots_dir": "plots",
                "reports_dir": "reports",
                "predictions_dir": "predictions"
            },
            
            # Feature engineering
            "feature_engineering": {
                "polynomial_degree": 2,
                "interaction_terms": True,
                "pca_components": None,
                "feature_selection": True,
                "correlation_threshold": 0.95
            },
            
            # Logging settings
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                "file": "water_quality_analysis.log"
            }
        }
    
    def _load_config(self, config_path: str):
        """Load configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                self._update_nested_dict(self.settings, user_config)
        except Exception as e:
            print(f"Warning: Could not load config file {config_path}: {e}")
    
    def _update_nested_dict(self, base_dict: Dict, update_dict: Dict):
        """Recursively update nested dictionary"""
        for key, value in update_dict.items():
            if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                self._update_nested_dict(base_dict[key], value)
            else:
                base_dict[key] = value
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.settings
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.settings
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        config[keys[-1]] = value
    
    def save(self, config_path: str = None):
        """Save configuration to JSON file"""
        if config_path is None:
            config_path = self.config_path
        if config_path:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)
            with open(config_path, 'w') as f:
                json.dump(self.settings, f, indent=2)
    
    def get_model_params(self, model_name: str) -> Dict[str, Any]:
        """Get hyperparameters for a specific model"""
        return self.settings["models"].get(model_name, {})
    
    def get_output_paths(self) -> Dict[str, str]:
        """Get output directory paths"""
        base_dir = self.settings["output"]["base_dir"]
        return {
            "base": base_dir,
            "models": os.path.join(base_dir, self.settings["output"]["models_dir"]),
            "plots": os.path.join(base_dir, self.settings["output"]["plots_dir"]),
            "reports": os.path.join(base_dir, self.settings["output"]["reports_dir"]),
            "predictions": os.path.join(base_dir, self.settings["output"]["predictions_dir"])
        }

# Global configuration instance
config = Config() 