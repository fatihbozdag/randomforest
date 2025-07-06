"""
Support Vector Regression model for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

from .base_model import BaseWaterQualityModel
from config import config

class SVRModel(BaseWaterQualityModel):
    """
    Support Vector Regression model for water quality prediction
    """
    
    def __init__(self, name: str = "SVR", kernel: str = "rbf", **kwargs):
        """
        Initialize SVR model
        
        Args:
            name: Model name
            kernel: Kernel type ('rbf', 'poly', 'linear', 'sigmoid')
            **kwargs: SVR hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        # Set default hyperparameters based on kernel
        default_params = self._get_default_params(kernel)
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Remove any parameters that SVR doesn't accept
        svr_params = {k: v for k, v in default_params.items() if k != 'random_state'}
        
        # Initialize model
        self.model = SVR(**svr_params)
        
        # Additional attributes
        self.scaler = StandardScaler()
        self.support_vectors_ = None
        self.n_support_ = None
        self.dual_coef_ = None
    
    def _get_default_params(self, kernel: str) -> Dict[str, Any]:
        """Get default parameters based on kernel type"""
        base_params = {
            'kernel': kernel
        }
        
        if kernel == 'rbf':
            base_params.update({
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1
            })
        elif kernel == 'poly':
            base_params.update({
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1,
                'degree': 3,
                'coef0': 0.0
            })
        elif kernel == 'linear':
            base_params.update({
                'C': 1.0,
                'epsilon': 0.1
            })
        elif kernel == 'sigmoid':
            base_params.update({
                'C': 1.0,
                'gamma': 'scale',
                'epsilon': 0.1,
                'coef0': 0.0
            })
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")
        
        return base_params
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'SVRModel':
        """
        Fit the SVR model
        
        Args:
            X: Training features
            y: Training targets
            **kwargs: Additional fitting parameters
            
        Returns:
            Self for method chaining
        """
        start_time = time.time()
        
        # Validate input
        X, y = self.validate_input(X, y)
        
        # Store feature names if provided as DataFrame
        if hasattr(X, 'columns'):
            self.feature_names = list(X.columns)
        elif self.feature_names is None:
            self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        
        # Scale features (SVR is sensitive to feature scaling)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y, **kwargs)
        
        # Store additional information
        self.is_fitted = True
        self.support_vectors_ = self.model.support_vectors_
        self.n_support_ = self.model.n_support_
        self.dual_coef_ = self.model.dual_coef_
        
        # Store training information
        self.training_history = {
            'n_support_vectors': len(self.support_vectors_),
            'support_ratio': len(self.support_vectors_) / len(X),
            'kernel': self.model.kernel,
            'C': self.model.C,
            'epsilon': self.model.epsilon
        }
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted SVR model
        
        Args:
            X: Input features
            
        Returns:
            Predictions array
        """
        start_time = time.time()
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Validate input
        X, _ = self.validate_input(X)
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        predictions = self.model.predict(X_scaled)
        
        # Record prediction time
        self._record_prediction_time(start_time)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (only available for linear kernel)
        
        Returns:
            Feature importance array or None
        """
        if not self.is_fitted or self.model.kernel != 'linear':
            return None
        
        # For linear kernel, feature importance is the absolute value of coefficients
        return np.abs(self.model.coef_[0])
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.get_feature_importance()
        if importance is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_support_vectors_info(self) -> Dict[str, Any]:
        """
        Get information about support vectors
        
        Returns:
            Dictionary with support vector information
        """
        if not self.is_fitted:
            return {}
        
        return {
            'n_support_vectors': len(self.support_vectors_),
            'support_ratio': len(self.support_vectors_) / self.n_support_,
            'support_vectors_shape': self.support_vectors_.shape,
            'dual_coef_shape': self.dual_coef_.shape if self.dual_coef_ is not None else None
        }
    
    def tune_hyperparameters(self, X: np.ndarray, y: np.ndarray, 
                           param_grid: Dict[str, list] = None,
                           cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Tune hyperparameters using GridSearchCV
        
        Args:
            X: Training features
            y: Training targets
            param_grid: Parameter grid for tuning
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with tuning results
        """
        # Use default parameter grid if not provided
        if param_grid is None:
            param_grid = config.get_model_params('svr')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=SVR(random_state=config.get("data.random_seed", 42)),
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Perform grid search
        grid_search.fit(X_scaled, y)
        
        # Update model with best parameters
        self.model = grid_search.best_estimator_
        self.hyperparameters = grid_search.best_params_
        
        # Store tuning results
        tuning_results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_,
            'best_estimator': grid_search.best_estimator_
        }
        
        return tuning_results
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5, scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform cross-validation
        
        Args:
            X: Training features
            y: Training targets
            cv_folds: Number of cross-validation folds
            scoring: Scoring metric
            
        Returns:
            Dictionary with cross-validation results
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        cv_scores = cross_val_score(
            estimator=self.model,
            X=X_scaled,
            y=y,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate additional metrics
        y_pred = self.model.predict(X_scaled)
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        cv_results = {
            'cv_scores': cv_scores,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'n_support_vectors': len(self.support_vectors_) if self.is_fitted else None
        }
        
        return cv_results
    
    def get_partial_dependence(self, X: np.ndarray, feature_idx: int, 
                              percentiles: tuple = (0.05, 0.95)) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate partial dependence for a specific feature
        
        Args:
            X: Training features
            feature_idx: Index of feature to analyze
            percentiles: Percentiles for feature range
            
        Returns:
            Tuple of (feature_values, partial_dependence_values)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating partial dependence")
        
        # Get feature range
        feature_values = X[:, feature_idx]
        feature_min = np.percentile(feature_values, percentiles[0] * 100)
        feature_max = np.percentile(feature_values, percentiles[1] * 100)
        
        # Create grid of values
        grid_values = np.linspace(feature_min, feature_max, 50)
        
        # Calculate partial dependence
        partial_dependence = []
        for value in grid_values:
            X_temp = X.copy()
            X_temp[:, feature_idx] = value
            X_temp_scaled = self.scaler.transform(X_temp)
            predictions = self.model.predict(X_temp_scaled)
            partial_dependence.append(np.mean(predictions))
        
        return grid_values, np.array(partial_dependence)
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics
        
        Returns:
            Dictionary with complexity metrics
        """
        if not self.is_fitted:
            return {}
        
        complexity_metrics = {
            'kernel': self.model.kernel,
            'C': self.model.C,
            'epsilon': self.model.epsilon,
            'n_support_vectors': len(self.support_vectors_),
            'support_ratio': len(self.support_vectors_) / self.n_support_
        }
        
        if self.model.kernel == 'poly':
            complexity_metrics.update({
                'degree': self.model.degree,
                'coef0': self.model.coef0
            })
        elif self.model.kernel == 'rbf':
            complexity_metrics.update({
                'gamma': self.model.gamma
            })
        elif self.model.kernel == 'sigmoid':
            complexity_metrics.update({
                'gamma': self.model.gamma,
                'coef0': self.model.coef0
            })
        
        return complexity_metrics
    
    def get_kernel_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Get kernel matrix for input data
        
        Args:
            X: Input features
            
        Returns:
            Kernel matrix
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing kernel matrix")
        
        X_scaled = self.scaler.transform(X)
        return self.model.kernel(X_scaled, X_scaled)
    
    def get_decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Get decision function values
        
        Args:
            X: Input features
            
        Returns:
            Decision function values
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing decision function")
        
        X_scaled = self.scaler.transform(X)
        return self.model.decision_function(X_scaled)
    
    def plot_support_vectors(self, X: np.ndarray, y: np.ndarray, 
                           feature1: int = 0, feature2: int = 1, save_path: str = None) -> None:
        """
        Plot support vectors for 2D visualization
        
        Args:
            X: Training features
            y: Training targets
            feature1: First feature index for plotting
            feature2: Second feature index for plotting
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            print("Model must be fitted before plotting support vectors")
            return
        
        import matplotlib.pyplot as plt
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Get support vector indices
        support_indices = self.model.support_
        
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        plt.scatter(X_scaled[:, feature1], X_scaled[:, feature2], 
                   c=y, cmap='viridis', alpha=0.6, label='All points')
        
        # Plot support vectors
        plt.scatter(X_scaled[support_indices, feature1], X_scaled[support_indices, feature2],
                   c='red', s=100, marker='o', edgecolors='black', 
                   linewidth=2, label='Support Vectors')
        
        plt.xlabel(f'Feature {feature1}')
        plt.ylabel(f'Feature {feature2}')
        plt.title('Support Vectors Visualization')
        plt.colorbar(label='Target')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 