"""
Linear Models for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import time
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

from .base_model import BaseWaterQualityModel
from config import config

class LinearModels(BaseWaterQualityModel):
    """
    Linear Models for water quality prediction
    """
    
    def __init__(self, name: str = "LinearModels", model_type: str = "linear", **kwargs):
        """
        Initialize Linear Models
        
        Args:
            name: Model name
            model_type: Type of linear model ('linear', 'ridge', 'lasso', 'elastic_net')
            **kwargs: Model hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        self.model_type = model_type.lower()
        
        # Set default hyperparameters based on model type
        default_params = self._get_default_params(model_type)
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Initialize model
        self.model = self._create_model(model_type, default_params)
        
        # Additional attributes
        self.scaler = StandardScaler()
        self.coef_ = None
        self.intercept_ = None
        self.feature_importance_ = None
    
    def _get_default_params(self, model_type: str) -> Dict[str, Any]:
        """Get default parameters based on model type"""
        base_params = {
            'random_state': config.get("data.random_seed", 42)
        }
        
        if model_type == 'linear':
            # LinearRegression has no hyperparameters to tune
            pass
        elif model_type == 'ridge':
            base_params.update({
                'alpha': 1.0,
                'solver': 'auto',
                'max_iter': 1000
            })
        elif model_type == 'lasso':
            base_params.update({
                'alpha': 1.0,
                'max_iter': 1000,
                'tol': 1e-4
            })
        elif model_type == 'elastic_net':
            base_params.update({
                'alpha': 1.0,
                'l1_ratio': 0.5,
                'max_iter': 1000,
                'tol': 1e-4
            })
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
        
        return base_params
    
    def _create_model(self, model_type: str, params: Dict[str, Any]):
        """Create the appropriate linear model"""
        if model_type == 'linear':
            return LinearRegression()
        elif model_type == 'ridge':
            return Ridge(**params)
        elif model_type == 'lasso':
            return Lasso(**params)
        elif model_type == 'elastic_net':
            return ElasticNet(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'LinearModels':
        """
        Fit the Linear Model
        
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
        
        # Scale features (important for regularized models)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y, **kwargs)
        
        # Store additional information
        self.is_fitted = True
        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        
        # Calculate feature importance (absolute values of coefficients)
        self.feature_importance = np.abs(self.coef_)
        
        # Store training information
        self.training_history = {
            'model_type': self.model_type,
            'coefficients': self.coef_.tolist(),
            'intercept': self.intercept_,
            'feature_importance': dict(zip(self.feature_names, self.feature_importance))
        }
        
        # Add model-specific information
        if hasattr(self.model, 'alpha'):
            self.training_history['alpha'] = self.model.alpha
        if hasattr(self.model, 'l1_ratio'):
            self.training_history['l1_ratio'] = self.model.l1_ratio
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Linear Model
        
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
        Get feature importance scores (absolute values of coefficients)
        
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
        importance = self.get_feature_importance()
        if importance is None:
            return pd.DataFrame()
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance,
            'coefficient': self.coef_
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """
        Get model coefficients and intercept
        
        Returns:
            Tuple of (coefficients, intercept)
        """
        if not self.is_fitted:
            return None, None
        
        return self.coef_, self.intercept_
    
    def get_coefficients_df(self) -> pd.DataFrame:
        """
        Get coefficients as DataFrame
        
        Returns:
            DataFrame with feature names and coefficients
        """
        if not self.is_fitted:
            return pd.DataFrame()
        
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': self.coef_,
            'abs_coefficient': np.abs(self.coef_)
        }).sort_values('abs_coefficient', ascending=False)
        
        return coef_df
    
    def get_model_equation(self) -> str:
        """
        Get the model equation as a string
        
        Returns:
            Model equation string
        """
        if not self.is_fitted:
            return "Model not fitted"
        
        equation = f"y = {self.intercept_:.4f}"
        
        for i, (feature, coef) in enumerate(zip(self.feature_names, self.coef_)):
            if coef >= 0:
                equation += f" + {coef:.4f} * {feature}"
            else:
                equation += f" - {abs(coef):.4f} * {feature}"
        
        return equation
    
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
            param_grid = config.get_model_params('linear_models').get(self.model_type, {})
        
        if not param_grid:
            return {"message": f"No hyperparameters to tune for {self.model_type}"}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create GridSearchCV object
        estimator = self._create_model(self.model_type, {'random_state': config.get("data.random_seed", 42)})
        
        grid_search = GridSearchCV(
            estimator=estimator,
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
            'model_type': self.model_type
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
        
        # For linear models, partial dependence is linear
        feature_values = X[:, feature_idx]
        feature_min = np.percentile(feature_values, percentiles[0] * 100)
        feature_max = np.percentile(feature_values, percentiles[1] * 100)
        
        # Create grid of values
        grid_values = np.linspace(feature_min, feature_max, 50)
        
        # For linear models, partial dependence is just the coefficient times the feature value
        # plus the intercept (assuming other features are at their mean)
        X_mean = np.mean(X, axis=0)
        partial_dependence = []
        
        for value in grid_values:
            X_temp = X_mean.copy()
            X_temp[feature_idx] = value
            X_temp_scaled = self.scaler.transform(X_temp.reshape(1, -1))
            prediction = self.model.predict(X_temp_scaled)[0]
            partial_dependence.append(prediction)
        
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
            'model_type': self.model_type,
            'n_features': len(self.coef_),
            'n_nonzero_coefficients': np.sum(self.coef_ != 0),
            'sparsity': 1 - (np.sum(self.coef_ != 0) / len(self.coef_)),
            'l1_norm': np.sum(np.abs(self.coef_)),
            'l2_norm': np.sqrt(np.sum(self.coef_**2))
        }
        
        # Add model-specific metrics
        if hasattr(self.model, 'alpha'):
            complexity_metrics['alpha'] = self.model.alpha
        if hasattr(self.model, 'l1_ratio'):
            complexity_metrics['l1_ratio'] = self.model.l1_ratio
        
        return complexity_metrics
    
    def get_residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate residuals (y_true - y_pred)
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Residuals array
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before calculating residuals")
        
        y_pred = self.predict(X)
        return y - y_pred
    
    def get_residual_analysis(self, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """
        Perform residual analysis
        
        Args:
            X: Input features
            y: True targets
            
        Returns:
            Dictionary with residual analysis results
        """
        residuals = self.get_residuals(X, y)
        
        analysis = {
            'mean_residual': np.mean(residuals),
            'std_residual': np.std(residuals),
            'min_residual': np.min(residuals),
            'max_residual': np.max(residuals),
            'median_residual': np.median(residuals),
            'skewness': pd.Series(residuals).skew(),
            'kurtosis': pd.Series(residuals).kurtosis(),
            'residuals': residuals
        }
        
        return analysis
    
    def plot_residuals(self, X: np.ndarray, y: np.ndarray, save_path: str = None) -> None:
        """
        Plot residual analysis
        
        Args:
            X: Input features
            y: True targets
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            print("Model must be fitted before plotting residuals")
            return
        
        import matplotlib.pyplot as plt
        
        residuals = self.get_residuals(X, y)
        y_pred = self.predict(X)
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 0].axhline(y=0, color='red', linestyle='--')
        axes[0, 0].set_xlabel('Predicted Values')
        axes[0, 0].set_ylabel('Residuals')
        axes[0, 0].set_title('Residuals vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Residuals histogram
        axes[0, 1].hist(residuals, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_xlabel('Residuals')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Residuals Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(y, y_pred, alpha=0.6)
        axes[1, 1].plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_feature_importance_confidence(self, X: np.ndarray, y: np.ndarray, 
                                        n_iterations: int = 100) -> Dict[str, np.ndarray]:
        """
        Calculate confidence intervals for coefficients using bootstrap
        
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
        
        coef_bootstrap = np.zeros((n_iterations, n_features))
        
        for i in range(n_iterations):
            # Bootstrap sample
            indices = np.random.choice(n_samples, size=n_samples, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            
            # Scale features
            X_boot_scaled = self.scaler.fit_transform(X_boot)
            
            # Fit model on bootstrap sample
            model_boot = self._create_model(self.model_type, self.hyperparameters)
            model_boot.fit(X_boot_scaled, y_boot)
            
            # Store coefficients
            coef_bootstrap[i, :] = model_boot.coef_
        
        # Calculate confidence intervals
        mean_coef = np.mean(coef_bootstrap, axis=0)
        std_coef = np.std(coef_bootstrap, axis=0)
        
        # 95% confidence intervals
        lower_ci = np.percentile(coef_bootstrap, 2.5, axis=0)
        upper_ci = np.percentile(coef_bootstrap, 97.5, axis=0)
        
        return {
            'mean': mean_coef,
            'std': std_coef,
            'lower_ci': lower_ci,
            'upper_ci': upper_ci
        } 