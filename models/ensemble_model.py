"""
Ensemble Methods for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import time
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression
import warnings

from .base_model import BaseWaterQualityModel
from .random_forest_model import RandomForestModel
from .gradient_boosting_model import GradientBoostingModel
from .svr_model import SVRModel
from .linear_models import LinearModels
from config import config

class EnsembleModel(BaseWaterQualityModel):
    """
    Ensemble Methods for water quality prediction
    """
    
    def __init__(self, name: str = "Ensemble", ensemble_type: str = "voting", **kwargs):
        """
        Initialize Ensemble Model
        
        Args:
            name: Model name
            ensemble_type: Type of ensemble ('voting', 'stacking', 'blending')
            **kwargs: Ensemble hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        self.ensemble_type = ensemble_type.lower()
        
        # Set default hyperparameters
        default_params = self._get_default_params(ensemble_type)
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Initialize ensemble
        self.model = None
        self.base_models = {}
        self.weights = None
        self.meta_model = None
        
        # Additional attributes
        self.individual_predictions = {}
        self.ensemble_weights = None
    
    def _get_default_params(self, ensemble_type: str) -> Dict[str, Any]:
        """Get default parameters based on ensemble type"""
        base_params = {
            'random_state': config.get("data.random_seed", 42)
        }
        
        if ensemble_type == 'voting':
            base_params.update({
                'weights': None,  # Equal weights by default
                'n_jobs': -1
            })
        elif ensemble_type == 'stacking':
            base_params.update({
                'cv': 5,
                'n_jobs': -1,
                'passthrough': False
            })
        elif ensemble_type == 'blending':
            base_params.update({
                'holdout_size': 0.2,
                'random_state': config.get("data.random_seed", 42)
            })
        else:
            raise ValueError(f"Unsupported ensemble type: {ensemble_type}")
        
        return base_params
    
    def add_base_model(self, model_name: str, model: BaseWaterQualityModel, weight: float = 1.0):
        """
        Add a base model to the ensemble
        
        Args:
            model_name: Name of the base model
            model: Base model instance
            weight: Weight for voting (only used for voting ensemble)
        """
        self.base_models[model_name] = {
            'model': model,
            'weight': weight
        }
    
    def create_default_ensemble(self):
        """Create a default ensemble with common models"""
        # Random Forest
        rf_model = RandomForestModel(
            n_estimators=100,
            max_depth=10,
            random_state=config.get("data.random_seed", 42)
        )
        self.add_base_model('RandomForest', rf_model, weight=1.0)
        
        # Gradient Boosting
        gb_model = GradientBoostingModel(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=config.get("data.random_seed", 42)
        )
        self.add_base_model('GradientBoosting', gb_model, weight=1.0)
        
        # SVR
        svr_model = SVRModel(
            kernel='rbf',
            C=1.0,
            gamma='scale'
        )
        self.add_base_model('SVR', svr_model, weight=1.0)
        
        # Linear Regression
        linear_model = LinearModels(
            model_type='linear'
        )
        self.add_base_model('LinearRegression', linear_model, weight=1.0)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'EnsembleModel':
        """
        Fit the Ensemble Model
        
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
        
        # Create default ensemble if no base models provided
        if not self.base_models:
            self.create_default_ensemble()
        
        # Fit ensemble based on type
        if self.ensemble_type == 'voting':
            self._fit_voting_ensemble(X, y, **kwargs)
        elif self.ensemble_type == 'stacking':
            self._fit_stacking_ensemble(X, y, **kwargs)
        elif self.ensemble_type == 'blending':
            self._fit_blending_ensemble(X, y, **kwargs)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        # Store training information
        self.training_history = {
            'ensemble_type': self.ensemble_type,
            'n_base_models': len(self.base_models),
            'base_model_names': list(self.base_models.keys()),
            'weights': self.ensemble_weights
        }
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def _fit_voting_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit voting ensemble"""
        # Prepare estimators for VotingRegressor
        estimators = []
        weights = []
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            # Fit the base model
            model.fit(X, y, **kwargs)
            
            estimators.append((name, model.model))
            weights.append(weight)
        
        # Create and fit voting regressor
        self.model = VotingRegressor(
            estimators=estimators,
            weights=weights if any(w != 1.0 for w in weights) else None,
            n_jobs=self.hyperparameters.get('n_jobs', -1)
        )
        
        self.model.fit(X, y)
        self.ensemble_weights = weights
        self.is_fitted = True
    
    def _fit_stacking_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit stacking ensemble"""
        # Prepare estimators for StackingRegressor
        estimators = []
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            estimators.append((name, model.model))
        
        # Create meta-model
        meta_model = LinearRegression()
        
        # Create and fit stacking regressor
        self.model = StackingRegressor(
            estimators=estimators,
            final_estimator=meta_model,
            cv=self.hyperparameters.get('cv', 5),
            n_jobs=self.hyperparameters.get('n_jobs', -1),
            passthrough=self.hyperparameters.get('passthrough', False)
        )
        
        self.model.fit(X, y)
        self.meta_model = meta_model
        self.is_fitted = True
    
    def _fit_blending_ensemble(self, X: np.ndarray, y: np.ndarray, **kwargs):
        """Fit blending ensemble"""
        from sklearn.model_selection import train_test_split
        
        # Split data for blending
        holdout_size = self.hyperparameters.get('holdout_size', 0.2)
        X_train, X_holdout, y_train, y_holdout = train_test_split(
            X, y, test_size=holdout_size, random_state=self.hyperparameters.get('random_state', 42)
        )
        
        # Fit base models on training data
        base_predictions = {}
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            
            # Fit on training data
            model.fit(X_train, y_train, **kwargs)
            
            # Get predictions on holdout set
            predictions = model.predict(X_holdout)
            base_predictions[name] = predictions
        
        # Create meta-features
        meta_features = np.column_stack(list(base_predictions.values()))
        
        # Fit meta-model
        self.meta_model = LinearRegression()
        self.meta_model.fit(meta_features, y_holdout)
        
        # Store base models and their predictions
        self.individual_predictions = base_predictions
        self.is_fitted = True
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Ensemble Model
        
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
        
        # Make predictions based on ensemble type
        if self.ensemble_type == 'voting':
            predictions = self.model.predict(X)
        elif self.ensemble_type == 'stacking':
            predictions = self.model.predict(X)
        elif self.ensemble_type == 'blending':
            predictions = self._predict_blending(X)
        else:
            raise ValueError(f"Unsupported ensemble type: {self.ensemble_type}")
        
        # Record prediction time
        self._record_prediction_time(start_time)
        
        return predictions
    
    def _predict_blending(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using blending ensemble"""
        # Get predictions from all base models
        base_predictions = {}
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            predictions = model.predict(X)
            base_predictions[name] = predictions
        
        # Create meta-features
        meta_features = np.column_stack(list(base_predictions.values()))
        
        # Make final prediction using meta-model
        final_predictions = self.meta_model.predict(meta_features)
        
        return final_predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance scores (average of base models)
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted:
            return None
        
        # Collect feature importance from all base models
        importance_scores = []
        weights = []
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            weight = model_info['weight']
            
            if model.is_fitted and model.get_feature_importance() is not None:
                importance_scores.append(model.get_feature_importance())
                weights.append(weight)
        
        if not importance_scores:
            return None
        
        # Calculate weighted average
        importance_scores = np.array(importance_scores)
        weights = np.array(weights)
        
        weighted_importance = np.average(importance_scores, axis=0, weights=weights)
        
        return weighted_importance
    
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
    
    def get_individual_predictions(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get predictions from individual base models
        
        Args:
            X: Input features
            
        Returns:
            Dictionary with model names and their predictions
        """
        if not self.is_fitted:
            return {}
        
        individual_predictions = {}
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            if model.is_fitted:
                predictions = model.predict(X)
                individual_predictions[name] = predictions
        
        return individual_predictions
    
    def get_ensemble_weights(self) -> Dict[str, float]:
        """
        Get ensemble weights
        
        Returns:
            Dictionary with model names and their weights
        """
        if not self.is_fitted:
            return {}
        
        weights = {}
        for name, model_info in self.base_models.items():
            weights[name] = model_info['weight']
        
        return weights
    
    def get_base_model_performance(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Dict[str, float]]:
        """
        Get performance metrics for individual base models
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Dictionary with model names and their performance metrics
        """
        if not self.is_fitted:
            return {}
        
        performance = {}
        
        for name, model_info in self.base_models.items():
            model = model_info['model']
            if model.is_fitted:
                y_pred = model.predict(X)
                
                mse = mean_squared_error(y, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y, y_pred)
                r2 = r2_score(y, y_pred)
                
                performance[name] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2
                }
        
        return performance
    
    def tune_ensemble_weights(self, X: np.ndarray, y: np.ndarray, 
                            cv_folds: int = 5) -> Dict[str, Any]:
        """
        Tune ensemble weights using cross-validation
        
        Args:
            X: Training features
            y: Training targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with tuning results
        """
        if not self.is_fitted:
            return {"error": "Model must be fitted before tuning weights"}
        
        # Get individual predictions
        individual_predictions = self.get_individual_predictions(X)
        
        if len(individual_predictions) < 2:
            return {"error": "Need at least 2 base models for weight tuning"}
        
        # Create parameter grid for weights
        weight_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        param_grid = {}
        
        for i, name in enumerate(individual_predictions.keys()):
            if i < len(individual_predictions) - 1:  # Leave last weight to be 1 - sum of others
                param_grid[f'weight_{name}'] = weight_values
        
        # Create custom scoring function
        def ensemble_score(estimator, X, y):
            weights = [estimator.get_params()[f'weight_{name}'] 
                      for name in individual_predictions.keys()[:-1]]
            weights.append(1 - sum(weights))  # Last weight
            
            # Normalize weights
            weights = np.array(weights)
            weights = weights / np.sum(weights)
            
            # Calculate weighted prediction
            weighted_pred = np.zeros(len(y))
            for i, (name, pred) in enumerate(individual_predictions.items()):
                weighted_pred += weights[i] * pred
            
            return -mean_squared_error(y, weighted_pred)
        
        # Perform grid search
        from sklearn.base import BaseEstimator
        
        class WeightOptimizer(BaseEstimator):
            def __init__(self, **params):
                self.set_params(**params)
            
            def fit(self, X, y):
                return self
            
            def predict(self, X):
                return np.zeros(len(X))
        
        grid_search = GridSearchCV(
            estimator=WeightOptimizer(),
            param_grid=param_grid,
            cv=cv_folds,
            scoring=ensemble_score,
            n_jobs=-1
        )
        
        # Create dummy X for grid search (we only need the predictions)
        dummy_X = np.zeros((len(y), 1))
        grid_search.fit(dummy_X, y)
        
        # Update weights
        best_weights = grid_search.best_params_
        for name in individual_predictions.keys():
            weight_key = f'weight_{name}'
            if weight_key in best_weights:
                self.base_models[name]['weight'] = best_weights[weight_key]
        
        return {
            'best_weights': best_weights,
            'best_score': grid_search.best_score_,
            'cv_results': grid_search.cv_results_
        }
    
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
        y_pred = self.predict(X)
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
            'ensemble_type': self.ensemble_type,
            'n_base_models': len(self.base_models)
        }
        
        return cv_results
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """
        Get model complexity metrics
        
        Returns:
            Dictionary with complexity metrics
        """
        if not self.is_fitted:
            return {}
        
        complexity_metrics = {
            'ensemble_type': self.ensemble_type,
            'n_base_models': len(self.base_models),
            'base_model_types': [type(model_info['model']).__name__ 
                               for model_info in self.base_models.values()],
            'total_parameters': sum(model_info['model'].get_model_complexity().get('total_parameters', 0)
                                  for model_info in self.base_models.values()
                                  if model_info['model'].is_fitted)
        }
        
        return complexity_metrics
    
    def plot_ensemble_comparison(self, X: np.ndarray, y: np.ndarray, save_path: str = None) -> None:
        """
        Plot comparison of ensemble vs individual models
        
        Args:
            X: Input features
            y: True targets
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            print("Model must be fitted before plotting comparison")
            return
        
        import matplotlib.pyplot as plt
        
        # Get individual model predictions
        individual_predictions = self.get_individual_predictions(X)
        ensemble_predictions = self.predict(X)
        
        # Calculate metrics
        models = list(individual_predictions.keys()) + ['Ensemble']
        predictions = list(individual_predictions.values()) + [ensemble_predictions]
        
        metrics = []
        for pred in predictions:
            mse = mean_squared_error(y, pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y, pred)
            r2 = r2_score(y, pred)
            metrics.append({'mse': mse, 'rmse': rmse, 'mae': mae, 'r2': r2})
        
        # Create comparison plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        metric_names = ['rmse', 'mae', 'r2', 'mse']
        metric_labels = ['RMSE', 'MAE', 'R²', 'MSE']
        
        for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
            ax = axes[i // 2, i % 2]
            values = [m[metric] for m in metrics]
            
            bars = ax.bar(models, values, alpha=0.7)
            ax.set_title(f'{label} Comparison')
            ax.set_ylabel(label)
            ax.tick_params(axis='x', rotation=45)
            
            # Highlight ensemble
            if i < len(bars):
                bars[-1].set_color('red')
                bars[-1].set_alpha(0.8)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 