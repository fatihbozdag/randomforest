"""
Neural Network model for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple, List
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import warnings

from .base_model import BaseWaterQualityModel
from config import config

class NeuralNetworkModel(BaseWaterQualityModel):
    """
    Neural Network model for water quality prediction
    """
    
    def __init__(self, name: str = "NeuralNetwork", **kwargs):
        """
        Initialize Neural Network model
        
        Args:
            name: Model name
            **kwargs: Neural Network hyperparameters
        """
        super().__init__(name=name, **kwargs)
        
        # Set default hyperparameters
        default_params = {
            'hidden_layer_sizes': (100, 50),
            'activation': 'relu',
            'solver': 'adam',
            'alpha': 0.0001,
            'batch_size': 'auto',
            'learning_rate': 'adaptive',
            'learning_rate_init': 0.001,
            'max_iter': 1000,
            'shuffle': True,
            'random_state': config.get("data.random_seed", 42),
            'early_stopping': True,
            'validation_fraction': 0.1,
            'n_iter_no_change': 10,
            'tol': 1e-4
        }
        
        # Update with provided parameters
        default_params.update(kwargs)
        self.hyperparameters = default_params
        
        # Initialize model
        self.model = MLPRegressor(**default_params)
        
        # Additional attributes
        self.scaler = StandardScaler()
        self.loss_curve = None
        self.validation_scores = None
        self.coefs_ = None
        self.intercepts_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> 'NeuralNetworkModel':
        """
        Fit the Neural Network model
        
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
        
        # Scale features (neural networks are sensitive to feature scaling)
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit the model
        self.model.fit(X_scaled, y, **kwargs)
        
        # Store additional information
        self.is_fitted = True
        self.loss_curve = self.model.loss_curve_
        self.coefs_ = self.model.coefs_
        self.intercepts_ = self.model.intercepts_
        
        # Store training information
        self.training_history = {
            'loss_curve': self.loss_curve,
            'n_layers': len(self.coefs_),
            'layer_sizes': [coef.shape[0] for coef in self.coefs_],
            'activation': self.model.activation,
            'solver': self.model.solver,
            'alpha': self.model.alpha,
            'learning_rate': self.model.learning_rate,
            'n_iterations': self.model.n_iter_
        }
        
        # Record training time
        self._record_training_time(start_time)
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions using the fitted Neural Network model
        
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
        Get feature importance scores using connection weights
        
        Returns:
            Feature importance array
        """
        if not self.is_fitted or len(self.coefs_) == 0:
            return None
        
        # Calculate feature importance based on connection weights
        # Use the first layer weights and take the mean absolute value
        first_layer_weights = np.abs(self.coefs_[0])
        feature_importance = np.mean(first_layer_weights, axis=1)
        
        return feature_importance
    
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
    
    def get_network_architecture(self) -> Dict[str, Any]:
        """
        Get neural network architecture information
        
        Returns:
            Dictionary with architecture details
        """
        if not self.is_fitted:
            return {}
        
        architecture = {
            'n_layers': len(self.coefs_),
            'input_size': self.coefs_[0].shape[0],
            'output_size': self.coefs_[-1].shape[1],
            'hidden_layers': [coef.shape[0] for coef in self.coefs_[1:]],
            'total_parameters': sum(coef.size + intercept.size for coef, intercept in zip(self.coefs_, self.intercepts_)),
            'activation_function': self.model.activation,
            'solver': self.model.solver
        }
        
        return architecture
    
    def get_layer_weights(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get weights and biases for a specific layer
        
        Args:
            layer_idx: Layer index (0-based)
            
        Returns:
            Tuple of (weights, biases)
        """
        if not self.is_fitted or layer_idx >= len(self.coefs_):
            raise ValueError(f"Invalid layer index: {layer_idx}")
        
        return self.coefs_[layer_idx], self.intercepts_[layer_idx]
    
    def get_loss_curve(self) -> Optional[np.ndarray]:
        """
        Get training loss curve
        
        Returns:
            Loss curve array
        """
        return self.loss_curve
    
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
            param_grid = config.get_model_params('neural_network')
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create GridSearchCV object
        grid_search = GridSearchCV(
            estimator=MLPRegressor(random_state=config.get("data.random_seed", 42)),
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
            'n_iterations': self.model.n_iter_ if self.is_fitted else None
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
        
        architecture = self.get_network_architecture()
        
        complexity_metrics = {
            'n_layers': architecture['n_layers'],
            'total_parameters': architecture['total_parameters'],
            'input_size': architecture['input_size'],
            'output_size': architecture['output_size'],
            'hidden_layers': architecture['hidden_layers'],
            'activation_function': architecture['activation_function'],
            'solver': architecture['solver'],
            'alpha': self.model.alpha,
            'learning_rate': self.model.learning_rate,
            'n_iterations': self.model.n_iter_
        }
        
        return complexity_metrics
    
    def plot_loss_curve(self, save_path: str = None) -> None:
        """
        Plot training loss curve
        
        Args:
            save_path: Path to save the plot
        """
        if self.loss_curve is None:
            print("No loss curve available")
            return
        
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.loss_curve)
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Training Loss Curve')
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_network_architecture(self, save_path: str = None) -> None:
        """
        Plot neural network architecture visualization
        
        Args:
            save_path: Path to save the plot
        """
        if not self.is_fitted:
            print("Model must be fitted before plotting architecture")
            return
        
        import matplotlib.pyplot as plt
        
        architecture = self.get_network_architecture()
        layer_sizes = [architecture['input_size']] + architecture['hidden_layers'] + [architecture['output_size']]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot layers
        for i, size in enumerate(layer_sizes):
            x = i
            y_positions = np.linspace(-size/2, size/2, size)
            
            # Plot neurons
            ax.scatter([x] * size, y_positions, s=100, c='lightblue', edgecolors='black')
            
            # Add layer labels
            if i == 0:
                ax.text(x, max(y_positions) + 0.5, 'Input', ha='center', fontsize=12, fontweight='bold')
            elif i == len(layer_sizes) - 1:
                ax.text(x, max(y_positions) + 0.5, 'Output', ha='center', fontsize=12, fontweight='bold')
            else:
                ax.text(x, max(y_positions) + 0.5, f'Hidden {i}', ha='center', fontsize=12, fontweight='bold')
            
            # Add size labels
            ax.text(x, min(y_positions) - 0.5, str(size), ha='center', fontsize=10)
        
        # Plot connections (simplified)
        for i in range(len(layer_sizes) - 1):
            x1, x2 = i, i + 1
            y1_positions = np.linspace(-layer_sizes[i]/2, layer_sizes[i]/2, layer_sizes[i])
            y2_positions = np.linspace(-layer_sizes[i+1]/2, layer_sizes[i+1]/2, layer_sizes[i+1])
            
            # Plot a few sample connections
            for j in range(min(5, layer_sizes[i])):
                for k in range(min(5, layer_sizes[i+1])):
                    ax.plot([x1, x2], [y1_positions[j], y2_positions[k]], 
                           'gray', alpha=0.3, linewidth=0.5)
        
        ax.set_xlim(-0.5, len(layer_sizes) - 0.5)
        ax.set_ylim(min(layer_sizes) * -0.6, max(layer_sizes) * 0.6)
        ax.set_xlabel('Layers')
        ax.set_ylabel('Neurons')
        ax.set_title('Neural Network Architecture')
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_layer_activations(self, X: np.ndarray, layer_idx: int = None) -> np.ndarray:
        """
        Get activations for a specific layer or all layers
        
        Args:
            X: Input features
            layer_idx: Layer index (None for all layers)
            
        Returns:
            Layer activations
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before computing activations")
        
        X_scaled = self.scaler.transform(X)
        
        if layer_idx is None:
            # Return activations for all layers
            activations = []
            current_input = X_scaled
            
            for i, (coef, intercept) in enumerate(zip(self.coefs_, self.intercepts_)):
                if i < len(self.coefs_) - 1:  # Not the output layer
                    activation = np.dot(current_input, coef.T) + intercept
                    if self.model.activation == 'relu':
                        activation = np.maximum(0, activation)
                    elif self.model.activation == 'tanh':
                        activation = np.tanh(activation)
                    elif self.model.activation == 'logistic':
                        activation = 1 / (1 + np.exp(-activation))
                    
                    activations.append(activation)
                    current_input = activation
            
            return activations
        else:
            # Return activations for specific layer
            if layer_idx >= len(self.coefs_):
                raise ValueError(f"Invalid layer index: {layer_idx}")
            
            current_input = X_scaled
            for i in range(layer_idx + 1):
                coef, intercept = self.coefs_[i], self.intercepts_[i]
                activation = np.dot(current_input, coef.T) + intercept
                
                if i < len(self.coefs_) - 1:  # Not the output layer
                    if self.model.activation == 'relu':
                        activation = np.maximum(0, activation)
                    elif self.model.activation == 'tanh':
                        activation = np.tanh(activation)
                    elif self.model.activation == 'logistic':
                        activation = 1 / (1 + np.exp(-activation))
                
                current_input = activation
            
            return current_input 