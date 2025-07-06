"""
Random Forest model for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

from .base_model import BaseWaterQualityModel
from config import config

class RandomForestModel(BaseWaterQualityModel):
    """
    Random Forest model for water quality prediction
    """
    
    def __init__(self, name: str = "RandomForest", **kwargs):
        """
        Initialize Random Forest model
        
        Args:
            name: Model name
            **kwargs: Random Forest hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        # Set default hyperparameters
        default_params = {
            'n_estimators': 100,
            'max_depth': None,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'random_state': config.get("data.random_seed", 42),
            'n_jobs': -1,
            'oob_score': True
        }
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Initialize model
        self.model = RandomForestRegressor(**default_params)
        
        # Additional Random Forest specific attributes
        self.oob_score_ = None
        self.feature_importance_ = None
        self.trees_info = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'RandomForestModel':
        """
        Fit the Random Forest model
        
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
        
        # Fit the model
        self.model.fit(X, y, **kwargs)
        
        # Store additional information
        self.is_fitted = True
        self.oob_score_ = self.model.oob_score_
        self.feature_importance = self.model.feature_importances_
        
        # Store training information
        self.training_history = {
            'oob_score': self.oob_score_,
            'n_trees': self.model.n_estimators,
            'feature_importance': dict(zip(self.feature_names, self.feature_importance))
        }
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Random Forest model
        
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
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Record prediction time
        self._record_prediction_time(start_time)
        
        return predictions
    
    def predict_with_uncertainty(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Get predictions from all trees
        predictions_per_tree = np.array([tree.predict(X) for tree in self.model.estimators_])
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(predictions_per_tree, axis=0)
        std_predictions = np.std(predictions_per_tree, axis=0)
        
        return mean_predictions, std_predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores
        
        Returns:
            Feature importance array
        """
        if self.is_fitted:
            return self.feature_importance
        return None
    
    def get_feature_importance_df(self) -> pd.DataFrame:
        """
        Get feature importance as DataFrame
        
        Returns:
            DataFrame with feature names and importance scores
        """
        if not self.is_fitted:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.feature_importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_oob_score(self) -> Optional[float]:
        """
        Get out-of-bag score
        
        Returns:
            OOB score if available
        """
        return self.oob_score_
    
    def get_trees_info(self) -> Dict[str, Any]:
        """
        Get information about individual trees
        
        Returns:
            Dictionary with tree information
        """
        if not self.is_fitted:
            return {}
        
        trees_info = {
            'n_trees': len(self.model.estimators_),
            'tree_depths': [tree.get_depth() for tree in self.model.estimators_],
            'tree_leaves': [tree.get_n_leaves() for tree in self.model.estimators_],
            'avg_depth': np.mean([tree.get_depth() for tree in self.model.estimators_]),
            'avg_leaves': np.mean([tree.get_n_leaves() for tree in self.model.estimators_])
        }
        
        return trees_info
    
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
            param_grid = config.get_model_params('random_forest')
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=RandomForestRegressor(random_state=config.get("data.random_seed", 42)),
            param_grid=param_grid,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1,
            verbose=1
        )
        
        # Perform grid search
        grid_search.fit(X, y)
        
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
        # Perform cross-validation
        cv_scores = cross_val_score(
            estimator=self.model,
            X=X,
            y=y,
            cv=cv_folds,
            scoring=scoring,
            n_jobs=-1
        )
        
        # Calculate additional metrics
        y_pred = self.model.predict(X)
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
            'oob_score': self.oob_score_ if self.is_fitted else None
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
            predictions = self.predict(X_temp)
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
        
        trees_info = self.get_trees_info()
        
        complexity_metrics = {
            'n_trees': trees_info['n_trees'],
            'avg_tree_depth': trees_info['avg_depth'],
            'avg_tree_leaves': trees_info['avg_leaves'],
            'total_leaves': sum(trees_info['tree_leaves']),
            'max_depth': max(trees_info['tree_depths']),
            'min_depth': min(trees_info['tree_depths']),
            'depth_std': np.std(trees_info['tree_depths']),
            'leaves_std': np.std(trees_info['tree_leaves'])
        }
        
        return complexity_metrics
    
    def get_feature_importance_confidence(self, X: np.ndarray, y: np.ndarray, 
                                        n_iterations: int = 10) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for feature importance using bootstrap
        
        Args:
            X: Training features
            y: Training targets
            n_iterations: Number of bootstrap iterations
            
        Returns:
            Dictionary with mean and confidence intervals
        """
        if not self.is_fitted:
            return {}
        
        n_samples = X.shape[0]
        n_features = X.shape[1]
        
        importance_bootstrap = np.zeros((n_iterations, n_features))
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Fit model on bootstrap sample
            rf_boot = RandomForestRegressor(
                n_estimators=50,  # Use fewer trees for speed
                random_state=i,
                n_jobs=1
            )
            rf_boot.fit(X_boot, y_boot)
            
            # Store feature importance
            importance_bootstrap[i, :] = rf_boot.feature_importances_
        
        # Calculate confidence intervals
        mean_importance = np.mean(importance_bootstrap, axis=0)
        std_importance = np.std(importance_bootstrap, axis=0)
        
        # 95% confidence intervals
        lower_ci = np.percentile(importance_bootstrap, 2.5, axis=0)
        upper_ci = np.percentile(importance_bootstrap, 97.5, axis=0)
        
        return {
            'mean': mean_importance,
            'std': std_importance,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        } 