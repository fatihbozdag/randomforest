"""
Gradient Boosting model for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Using scikit-learn GradientBoostingRegressor.")

from .base_model import BaseWaterQualityModel
from config import config

class GradientBoostingModel(BaseWaterQualityModel):
    """
    Gradient Boosting model for water quality prediction
    """
    
    def __init__(self, name: str = "GradientBoosting", use_xgboost: bool = True, **kwargs):
        """
        Initialize Gradient Boosting model
        
        Args:
            name: Model name
            use_xgboost: Whether to use XGBoost (if available)
            **kwargs: Gradient Boosting hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        self.use_xgboost = use_xgboost and XGBOOST_AVAILABLE
        
        if self.use_xgboost:
            # XGBoost default parameters
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'random_state': config.get("data.random_seed", 42),
                'n_jobs': -1,
                'eval_metric': 'rmse'
            }
        else:
            # Scikit-learn default parameters
            default_params = {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 6,
                'subsample': 0.8,
                'random_state': config.get("data.random_seed", 42)
            }
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Initialize model
        if self.use_xgboost:
            self.model = xgb.XGBRegressor(**default_params)
        else:
            self.model = GradientBoostingRegressor(**default_params)
        
        # Additional attributes
        self.learning_curves = {}
        self.feature_importance_ = None
        self.best_iteration = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs) -> 'GradientBoostingModel':
        """
        Fit the Gradient Boosting model
        
        Args:
            X: Training features
            y: Training targets
            X_val: Validation features (for early stopping)
            y_val: Validation targets (for early stopping)
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
        
        # If X_val not provided but we need early-stopping, create an internal 20% split
        if self.use_xgboost and (X_val is None or y_val is None):
            X, X_val, y, y_val = train_test_split(
                X, y, test_size=0.2, random_state=config.get("data.random_seed", 42))
        
        # Fit the model
        if self.use_xgboost:
            self._fit_xgboost(X, y, X_val, y_val, **kwargs)
        else:
            self._fit_sklearn(X, y, **kwargs)
        
        # Store training information
        self.training_history = {
            'learning_curves': self.learning_curves,
            'best_iteration': self.best_iteration,
            'feature_importance': dict(zip(self.feature_names, self.feature_importance_)) if hasattr(self, 'feature_importance_') else {}
        }
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def _fit_xgboost(self, X: np.ndarray, y: np.ndarray, 
                    X_val: np.ndarray = None, y_val: np.ndarray = None, **kwargs):
        """Fit XGBoost model"""
        fit_params = {'verbose': False}
        
        # Only use early stopping if validation data is provided
        if X_val is not None and y_val is not None:
            # Some XGBoost versions require `early_stopping_rounds` inside kwargs; to remain compatible
            # we keep evaluation set but skip the argument if it is not supported.
            fit_params['eval_set'] = [(X, y), (X_val, y_val)]
        
        self.model.fit(X, y, **fit_params, **kwargs)
        
        # Store results
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        # Safely capture best_iteration (may not exist if early stopping disabled)
        try:
            self.best_iteration = self.model.best_iteration
        except Exception:
            self.best_iteration = None
        
        # Store learning curves if early stopping was used
        if X_val is not None and y_val is not None:
            self.learning_curves = {
                'train': self.model.evals_result()['validation_0']['rmse'],
                'validation': self.model.evals_result()['validation_1']['rmse']
            }
    
    def _fit_sklearn(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit scikit-learn GradientBoostingRegressor"""
        self.model.fit(X, y, **kwargs)
        
        # Store results
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        # Store learning curves (staged predictions)
        train_scores = []
        for i, y_pred in enumerate(self.model.staged_predict(X)):
            score = mean_squared_error(y, y_pred)
            train_scores.append(score)
        
        self.learning_curves = {'train': train_scores}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Gradient Boosting model
        
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
    
    def predict_with_uncertainty(self, X: np.ndarray, n_estimators: int = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions with uncertainty estimates using staged predictions
        
        Args:
            X: Input features
            n_estimators: Number of estimators to use (for staged predictions)
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        if n_estimators is None:
            n_estimators = self.model.n_estimators
        
        # Get staged predictions
        predictions_staged = []
        for pred in self.model.staged_predict(X):
            predictions_staged.append(pred)
            if len(predictions_staged) >= n_estimators:
                break
        
        predictions_staged = np.array(predictions_staged)
        
        # Calculate mean and standard deviation
        mean_predictions = np.mean(predictions_staged, axis=0)
        std_predictions = np.std(predictions_staged, axis=0)
        
        return mean_predictions, std_predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores
        
        Returns:
            Feature importance array
        """
        if self.is_fitted:
            return self.feature_importance_
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
            'importance': self.feature_importance_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_learning_curves(self) -> Dict[str, list]:
        """
        Get learning curves
        
        Returns:
            Dictionary with learning curves
        """
        return self.learning_curves
    
    def get_best_iteration(self) -> Optional[int]:
        """
        Get best iteration (for XGBoost)
        
        Returns:
            Best iteration number
        """
        return self.best_iteration
    
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
            param_grid = config.get_model_params('gradient_boosting')
        
        # Create GridSearchCV object
        if self.use_xgboost:
            estimator = xgb.XGBRegressor(random_state=config.get("data.random_seed", 42))
        else:
            estimator = GradientBoostingRegressor(random_state=config.get("data.random_seed", 42))
        
        grid_search = GridSearchCV(
            estimator=estimator,
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
            'best_iteration': self.best_iteration
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
    
    def plot_learning_curves(self, save_path: str = None) -> None:
        """
        Plot learning curves
        
        Args:
            save_path: Path to save the plot
        """
        if not self.learning_curves:
            print("No learning curves available")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        
        for curve_name, curve_values in self.learning_curves.items():
            plt.plot(curve_values, label=curve_name.capitalize())
        
        plt.xlabel('Iteration')
        plt.ylabel('RMSE')
        plt.title('Learning Curves')
        plt.legend()
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics
        
        Returns:
            Dictionary with complexity metrics
        """
        if not self.is_fitted:
            return {}
        
        complexity_metrics = {
            'n_estimators': self.model.n_estimators,
            'learning_rate': self.model.learning_rate,
            'max_depth': self.model.max_depth,
            'subsample': getattr(self.model, 'subsample', 1.0),
            'best_iteration': self.best_iteration
        }
        
        if self.use_xgboost:
            complexity_metrics.update({
                'colsample_bytree': self.model.colsample_bytree,
                'reg_alpha': self.model.reg_alpha,
                'reg_lambda': self.model.reg_lambda
            })
        
        return complexity_metrics 