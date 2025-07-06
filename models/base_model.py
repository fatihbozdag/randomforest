"""
Base model class for Water Quality Analysis System
"""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import joblib
import os
from datetime import datetime

class BaseWaterQualityModel(ABC):
    """
    Abstract base class for all water quality prediction models
    """
    
    def __init__(self, name: str = None, **kwargs):
        """
        Initialize base model
        
        Args:
            name: Model name
            **kwargs: Additional model parameters
        """
        self.name = name or self.__class__.__name__
        self.model = None
        self.is_fitted = False
        self.feature_names = None
        self.training_time = None
        self.prediction_time = None
        self.hyperparameters = kwargs
        self.feature_importance = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'BaseWaterQualityModel':
        """
        Fit the model to training data
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        pass
    
    @abstractmethod
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores
        
        Returns:
            Feature importance array or None if not available
        """
        pass
    
    def fit_predict(self, X_train: np.ndarray, y_train: np.ndarray, 
                   X_test: np.ndarray, **kwargs) -> np.ndarray:
        """
        Fit model and make predictions on test data
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            **kwargs: Additional fitting parameters
            
        Returns:
            Predictions on test data
        """
        self.fit(X_train, y_train, **kwargs)
        return self.predict(X_test)
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters
        
        Returns:
            Dictionary of model parameters
        """
        if hasattr(self.model, 'get_params'):
            return self.model.get_params()
        return self.hyperparameters
    
    def set_params(self, **params) -> 'BaseWaterQualityModel':
        """
        Set model parameters
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        if hasattr(self.model, 'set_params'):
            self.model.set_params(**params)
        else:
            self.hyperparameters.update(params)
        return self
    
    def save_model(self, filepath: str) -> None:
        """
        Save model to file
        
        Args:
            filepath: Path to save model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        model_data = {
            'model': self.model,
            'name': self.name,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'hyperparameters': self.hyperparameters,
            'feature_importance': self.feature_importance,
            'training_history': self.training_history,
            'saved_at': datetime.now().isoformat()
        }
        
        joblib.dump(model_data, filepath)
    
    def load_model(self, filepath: str) -> 'BaseWaterQualityModel':
        """
        Load model from file
        
        Args:
            filepath: Path to load model from
            
        Returns:
            Self for method chaining
        """
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.name = model_data['name']
        self.is_fitted = model_data['is_fitted']
        self.feature_names = model_data['feature_names']
        self.hyperparameters = model_data['hyperparameters']
        self.feature_importance = model_data['feature_importance']
        self.training_history = model_data.get('training_history', {})
        
        return self
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information
        
        Returns:
            Dictionary with model information
        """
        return {
            'name': self.name,
            'type': self.__class__.__name__,
            'is_fitted': self.is_fitted,
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names) if self.feature_names else None,
            'hyperparameters': self.hyperparameters,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time,
            'has_feature_importance': self.feature_importance is not None
        }
    
    def validate_input(self, X: np.ndarray, y: np.ndarray = None) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Validate input data
        
        Args:
            X: Input features
            y: Input targets (optional)
            
        Returns:
            Tuple of validated X and y
        """
        # Convert to numpy array if needed
        if isinstance(X, pd.DataFrame):
            X = X.values
        elif not isinstance(X, np.ndarray):
            X = np.array(X)
        
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
            elif not isinstance(y, np.ndarray):
                y = np.array(y)
        
        # Check for NaN values
        if np.isnan(X).any():
            raise ValueError("Input features contain NaN values")
        
        if y is not None and np.isnan(y).any():
            raise ValueError("Input targets contain NaN values")
        
        return X, y
    
    def _record_training_time(self, start_time: float) -> None:
        """
        Record training time
        
        Args:
            start_time: Start time from time.time()
        """
        import time
        self.training_time = time.time() - start_time
    
    def _record_prediction_time(self, start_time: float) -> None:
        """
        Record prediction time
        
        Args:
            start_time: Start time from time.time()
        """
        import time
        self.prediction_time = time.time() - start_time
    
    def __str__(self) -> str:
        """String representation of the model"""
        return f"{self.name} ({self.__class__.__name__})"
    
    def __repr__(self) -> str:
        """Detailed string representation of the model"""
        info = self.get_model_info()
        return (f"{self.__class__.__name__}(name='{info['name']}', "
                f"fitted={info['is_fitted']}, n_features={info['n_features']})") 