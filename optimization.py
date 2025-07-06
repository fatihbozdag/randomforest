"""
Hyperparameter Optimization Module for Water Quality Analysis

This module provides comprehensive hyperparameter optimization capabilities including:
- Grid Search
- Random Search  
- Bayesian Optimization
- Hyperparameter Importance Analysis
- Automated optimization pipelines
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional, Union
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import uniform, randint
import optuna
from optuna.samplers import TPESampler
from optuna.visualization import plot_optimization_history, plot_param_importances
import joblib
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HyperparameterOptimizer:
    """
    Comprehensive hyperparameter optimization for water quality analysis models.
    
    Supports multiple optimization strategies and provides detailed analysis
    of hyperparameter importance and optimization history.
    """
    
    def __init__(self, 
                 cv_folds: int = 5,
                 n_jobs: int = -1,
                 random_state: int = 42,
                 verbose: int = 1):
        """
        Initialize the hyperparameter optimizer.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random seed for reproducibility
            verbose: Verbosity level
        """
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.optimization_results = {}
        self.best_params = {}
        self.best_scores = {}
        
    def grid_search(self, 
                   model, 
                   param_grid: Dict[str, List], 
                   X_train: np.ndarray, 
                   y_train: np.ndarray,
                   scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform grid search hyperparameter optimization.
        
        Args:
            model: Model instance to optimize
            param_grid: Dictionary of parameter names and lists of values to try
            X_train: Training features
            y_train: Training targets
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting Grid Search optimization...")
        
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            return_train_score=True
        )
        
        grid_search.fit(X_train, y_train)
        
        results = {
            'best_params': grid_search.best_params_,
            'best_score': grid_search.best_score_,
            'best_estimator': grid_search.best_estimator_,
            'cv_results': grid_search.cv_results_,
            'n_splits': self.cv_folds,
            'optimization_method': 'grid_search'
        }
        
        self.optimization_results['grid_search'] = results
        self.best_params['grid_search'] = grid_search.best_params_
        self.best_scores['grid_search'] = grid_search.best_score_
        
        logger.info(f"Grid Search completed. Best score: {grid_search.best_score_:.4f}")
        logger.info(f"Best parameters: {grid_search.best_params_}")
        
        return results
    
    def random_search(self, 
                     model, 
                     param_distributions: Dict[str, Any], 
                     X_train: np.ndarray, 
                     y_train: np.ndarray,
                     n_iter: int = 100,
                     scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform random search hyperparameter optimization.
        
        Args:
            model: Model instance to optimize
            param_distributions: Dictionary of parameter names and distributions
            X_train: Training features
            y_train: Training targets
            n_iter: Number of parameter settings sampled
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting Random Search optimization...")
        
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=self.cv_folds,
            scoring=scoring,
            n_jobs=self.n_jobs,
            verbose=self.verbose,
            random_state=self.random_state,
            return_train_score=True
        )
        
        random_search.fit(X_train, y_train)
        
        results = {
            'best_params': random_search.best_params_,
            'best_score': random_search.best_score_,
            'best_estimator': random_search.best_estimator_,
            'cv_results': random_search.cv_results_,
            'n_splits': self.cv_folds,
            'n_iter': n_iter,
            'optimization_method': 'random_search'
        }
        
        self.optimization_results['random_search'] = results
        self.best_params['random_search'] = random_search.best_params_
        self.best_scores['random_search'] = random_search.best_score_
        
        logger.info(f"Random Search completed. Best score: {random_search.best_score_:.4f}")
        logger.info(f"Best parameters: {random_search.best_params_}")
        
        return results
    
    def bayesian_optimization(self, 
                            model_class, 
                            param_space: Dict[str, Any], 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            n_trials: int = 100,
                            scoring: str = 'neg_mean_squared_error') -> Dict[str, Any]:
        """
        Perform Bayesian optimization using Optuna.
        
        Args:
            model_class: Model class (not instance)
            param_space: Dictionary defining parameter search space
            X_train: Training features
            y_train: Training targets
            n_trials: Number of optimization trials
            scoring: Scoring metric for optimization
            
        Returns:
            Dictionary containing optimization results
        """
        logger.info("Starting Bayesian Optimization...")
        
        def objective(trial):
            # Sample parameters from the search space
            params = {}
            for param_name, param_config in param_space.items():
                if param_config['type'] == 'categorical':
                    params[param_name] = trial.suggest_categorical(param_name, param_config['choices'])
                elif param_config['type'] == 'int':
                    params[param_name] = trial.suggest_int(param_name, param_config['low'], param_config['high'])
                elif param_config['type'] == 'float':
                    params[param_name] = trial.suggest_float(param_name, param_config['low'], param_config['high'], log=param_config.get('log', False))
            
            # Create model with sampled parameters
            model = model_class(**params)
            
            # Perform cross-validation
            scores = cross_val_score(
                model, X_train, y_train, 
                cv=self.cv_folds, 
                scoring=scoring, 
                n_jobs=self.n_jobs
            )
            
            return scores.mean()
        
        # Create study and optimize
        study = optuna.create_study(
            direction='maximize' if scoring.startswith('r2') else 'minimize',
            sampler=TPESampler(seed=self.random_state)
        )
        
        study.optimize(objective, n_trials=n_trials)
        
        # Get best parameters and create best model
        best_params = study.best_params
        best_model = model_class(**best_params)
        best_model.fit(X_train, y_train)
        
        results = {
            'best_params': best_params,
            'best_score': study.best_value,
            'best_estimator': best_model,
            'study': study,
            'n_trials': n_trials,
            'optimization_method': 'bayesian_optimization'
        }
        
        self.optimization_results['bayesian_optimization'] = results
        self.best_params['bayesian_optimization'] = best_params
        self.best_scores['bayesian_optimization'] = study.best_value
        
        logger.info(f"Bayesian Optimization completed. Best score: {study.best_value:.4f}")
        logger.info(f"Best parameters: {best_params}")
        
        return results
    
    def analyze_hyperparameter_importance(self, 
                                        optimization_method: str = 'bayesian_optimization',
                                        top_n: int = 10) -> pd.DataFrame:
        """
        Analyze hyperparameter importance from optimization results.
        
        Args:
            optimization_method: Method to analyze ('bayesian_optimization' or 'random_search')
            top_n: Number of top parameters to return
            
        Returns:
            DataFrame with hyperparameter importance scores
        """
        if optimization_method not in self.optimization_results:
            raise ValueError(f"No results found for {optimization_method}")
        
        if optimization_method == 'bayesian_optimization':
            study = self.optimization_results[optimization_method]['study']
            importance = optuna.importance.get_param_importances(study)
            
            # Convert to DataFrame
            importance_df = pd.DataFrame([
                {'parameter': param, 'importance': score}
                for param, score in importance.items()
            ])
            
        elif optimization_method == 'random_search':
            # For random search, we can analyze correlation with performance
            cv_results = self.optimization_results[optimization_method]['cv_results']
            
            # Create DataFrame with parameters and scores
            results_df = pd.DataFrame(cv_results)
            param_cols = [col for col in results_df.columns if col.startswith('param_')]
            
            importance_scores = {}
            for param in param_cols:
                param_name = param.replace('param_', '')
                correlation = abs(results_df[param].corr(results_df['mean_test_score']))
                importance_scores[param_name] = correlation
            
            importance_df = pd.DataFrame([
                {'parameter': param, 'importance': score}
                for param, score in importance_scores.items()
            ])
        
        # Sort by importance and return top N
        importance_df = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        return importance_df
    
    def compare_optimization_methods(self) -> pd.DataFrame:
        """
        Compare different optimization methods.
        
        Returns:
            DataFrame comparing optimization methods
        """
        comparison_data = []
        
        for method, results in self.optimization_results.items():
            comparison_data.append({
                'method': method,
                'best_score': results['best_score'],
                'n_trials': results.get('n_trials', results.get('n_iter', 'N/A')),
                'best_params': str(results['best_params'])
            })
        
        return pd.DataFrame(comparison_data)
    
    def plot_optimization_history(self, 
                                optimization_method: str = 'bayesian_optimization',
                                figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot optimization history.
        
        Args:
            optimization_method: Method to plot
            figsize: Figure size
        """
        if optimization_method not in self.optimization_results:
            raise ValueError(f"No results found for {optimization_method}")
        
        if optimization_method == 'bayesian_optimization':
            study = self.optimization_results[optimization_method]['study']
            fig = plot_optimization_history(study)
            fig.show()
        else:
            # For grid/random search, plot manually
            cv_results = self.optimization_results[optimization_method]['cv_results']
            
            plt.figure(figsize=figsize)
            plt.plot(cv_results['mean_test_score'])
            plt.title(f'{optimization_method.replace("_", " ").title()} Optimization History')
            plt.xlabel('Trial')
            plt.ylabel('Mean Test Score')
            plt.grid(True)
            plt.show()
    
    def plot_parameter_importance(self, 
                                optimization_method: str = 'bayesian_optimization',
                                top_n: int = 10,
                                figsize: Tuple[int, int] = (12, 8)) -> None:
        """
        Plot hyperparameter importance.
        
        Args:
            optimization_method: Method to analyze
            top_n: Number of top parameters to show
            figsize: Figure size
        """
        if optimization_method not in self.optimization_results:
            raise ValueError(f"No results found for {optimization_method}")
        
        if optimization_method == 'bayesian_optimization':
            study = self.optimization_results[optimization_method]['study']
            fig = plot_param_importances(study)
            fig.show()
        else:
            # For other methods, plot manually
            importance_df = self.analyze_hyperparameter_importance(optimization_method, top_n)
            
            plt.figure(figsize=figsize)
            sns.barplot(data=importance_df, x='importance', y='parameter')
            plt.title(f'{optimization_method.replace("_", " ").title()} Parameter Importance')
            plt.xlabel('Importance Score')
            plt.ylabel('Parameter')
            plt.tight_layout()
            plt.show()
    
    def save_optimization_results(self, 
                                filepath: str,
                                optimization_method: str = None) -> None:
        """
        Save optimization results to file.
        
        Args:
            filepath: Path to save results
            optimization_method: Specific method to save (None for all)
        """
        if optimization_method:
            if optimization_method not in self.optimization_results:
                raise ValueError(f"No results found for {optimization_method}")
            
            results = self.optimization_results[optimization_method]
            # Remove non-serializable objects
            if 'study' in results:
                del results['study']
            if 'best_estimator' in results:
                del results['best_estimator']
            
            joblib.dump(results, filepath)
        else:
            # Save all results
            all_results = {}
            for method, results in self.optimization_results.items():
                # Remove non-serializable objects
                if 'study' in results:
                    del results['study']
                if 'best_estimator' in results:
                    del results['best_estimator']
                all_results[method] = results
            
            joblib.dump(all_results, filepath)
        
        logger.info(f"Optimization results saved to {filepath}")
    
    def load_optimization_results(self, filepath: str) -> None:
        """
        Load optimization results from file.
        
        Args:
            filepath: Path to load results from
        """
        self.optimization_results = joblib.load(filepath)
        logger.info(f"Optimization results loaded from {filepath}")


class AutomatedOptimizer:
    """
    Automated hyperparameter optimization pipeline for water quality analysis.
    
    Provides pre-configured optimization strategies for different model types.
    """
    
    def __init__(self, cv_folds: int = 5, n_jobs: int = -1, random_state: int = 42):
        """
        Initialize the automated optimizer.
        
        Args:
            cv_folds: Number of cross-validation folds
            n_jobs: Number of parallel jobs
            random_state: Random seed
        """
        self.optimizer = HyperparameterOptimizer(cv_folds, n_jobs, random_state)
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.random_state = random_state
    
    def optimize_random_forest(self, 
                             X_train: np.ndarray, 
                             y_train: np.ndarray,
                             optimization_method: str = 'bayesian_optimization',
                             n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize Random Forest hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Optimization results
        """
        from models.random_forest_model import RandomForestModel
        
        if optimization_method == 'bayesian_optimization':
            param_space = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'max_depth': {'type': 'int', 'low': 3, 'high': 20},
                'min_samples_split': {'type': 'int', 'low': 2, 'high': 20},
                'min_samples_leaf': {'type': 'int', 'low': 1, 'high': 10},
                'max_features': {'type': 'categorical', 'choices': ['sqrt', 'log2', None]},
                'bootstrap': {'type': 'categorical', 'choices': [True, False]}
            }
            
            return self.optimizer.bayesian_optimization(
                RandomForestModel, param_space, X_train, y_train, n_trials
            )
        
        elif optimization_method == 'grid_search':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2', None]
            }
            
            model = RandomForestModel()
            return self.optimizer.grid_search(model, param_grid, X_train, y_train)
        
        elif optimization_method == 'random_search':
            param_distributions = {
                'n_estimators': randint(50, 500),
                'max_depth': randint(3, 20),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2', None],
                'bootstrap': [True, False]
            }
            
            model = RandomForestModel()
            return self.optimizer.random_search(model, param_distributions, X_train, y_train)
    
    def optimize_gradient_boosting(self, 
                                 X_train: np.ndarray, 
                                 y_train: np.ndarray,
                                 optimization_method: str = 'bayesian_optimization',
                                 n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize Gradient Boosting hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Optimization results
        """
        from models.gradient_boosting_model import GradientBoostingModel
        
        if optimization_method == 'bayesian_optimization':
            param_space = {
                'n_estimators': {'type': 'int', 'low': 50, 'high': 500},
                'learning_rate': {'type': 'float', 'low': 0.01, 'high': 0.3, 'log': True},
                'max_depth': {'type': 'int', 'low': 3, 'high': 10},
                'subsample': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'colsample_bytree': {'type': 'float', 'low': 0.6, 'high': 1.0},
                'reg_alpha': {'type': 'float', 'low': 0, 'high': 1, 'log': True},
                'reg_lambda': {'type': 'float', 'low': 0, 'high': 1, 'log': True}
            }
            
            return self.optimizer.bayesian_optimization(
                GradientBoostingModel, param_space, X_train, y_train, n_trials
            )
        
        elif optimization_method == 'grid_search':
            param_grid = {
                'n_estimators': [100, 200, 300],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            model = GradientBoostingModel()
            return self.optimizer.grid_search(model, param_grid, X_train, y_train)
        
        elif optimization_method == 'random_search':
            param_distributions = {
                'n_estimators': randint(50, 500),
                'learning_rate': uniform(0.01, 0.29),
                'max_depth': randint(3, 10),
                'subsample': uniform(0.6, 0.4),
                'colsample_bytree': uniform(0.6, 0.4),
                'reg_alpha': uniform(0, 1),
                'reg_lambda': uniform(0, 1)
            }
            
            model = GradientBoostingModel()
            return self.optimizer.random_search(model, param_distributions, X_train, y_train)
    
    def optimize_svr(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    optimization_method: str = 'bayesian_optimization',
                    n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize SVR hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Optimization results
        """
        from models.svr_model import SVRModel
        
        if optimization_method == 'bayesian_optimization':
            param_space = {
                'C': {'type': 'float', 'low': 0.1, 'high': 100, 'log': True},
                'epsilon': {'type': 'float', 'low': 0.01, 'high': 1.0, 'log': True},
                'kernel': {'type': 'categorical', 'choices': ['rbf', 'linear', 'poly']},
                'gamma': {'type': 'float', 'low': 0.001, 'high': 1.0, 'log': True}
            }
            
            return self.optimizer.bayesian_optimization(
                SVRModel, param_space, X_train, y_train, n_trials
            )
        
        elif optimization_method == 'grid_search':
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'epsilon': [0.01, 0.1, 0.5],
                'kernel': ['rbf', 'linear'],
                'gamma': [0.001, 0.01, 0.1, 'scale']
            }
            
            model = SVRModel()
            return self.optimizer.grid_search(model, param_grid, X_train, y_train)
        
        elif optimization_method == 'random_search':
            param_distributions = {
                'C': uniform(0.1, 99.9),
                'epsilon': uniform(0.01, 0.99),
                'kernel': ['rbf', 'linear', 'poly'],
                'gamma': uniform(0.001, 0.999)
            }
            
            model = SVRModel()
            return self.optimizer.random_search(model, param_distributions, X_train, y_train)
    
    def optimize_neural_network(self, 
                              X_train: np.ndarray, 
                              y_train: np.ndarray,
                              optimization_method: str = 'bayesian_optimization',
                              n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize Neural Network hyperparameters.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Optimization results
        """
        from models.neural_network_model import NeuralNetworkModel
        
        if optimization_method == 'bayesian_optimization':
            param_space = {
                'hidden_layer_sizes': {'type': 'categorical', 'choices': [(50,), (100,), (50, 50), (100, 50), (100, 100)]},
                'learning_rate_init': {'type': 'float', 'low': 0.001, 'high': 0.1, 'log': True},
                'alpha': {'type': 'float', 'low': 0.0001, 'high': 0.1, 'log': True},
                'dropout_rate': {'type': 'float', 'low': 0.0, 'high': 0.5},
                'batch_size': {'type': 'categorical', 'choices': [16, 32, 64, 128]}
            }
            
            return self.optimizer.bayesian_optimization(
                NeuralNetworkModel, param_space, X_train, y_train, n_trials
            )
        
        elif optimization_method == 'grid_search':
            param_grid = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50)],
                'learning_rate_init': [0.001, 0.01, 0.1],
                'alpha': [0.0001, 0.001, 0.01],
                'dropout_rate': [0.0, 0.2, 0.4],
                'batch_size': [32, 64, 128]
            }
            
            model = NeuralNetworkModel()
            return self.optimizer.grid_search(model, param_grid, X_train, y_train)
        
        elif optimization_method == 'random_search':
            param_distributions = {
                'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 50), (100, 100)],
                'learning_rate_init': uniform(0.001, 0.099),
                'alpha': uniform(0.0001, 0.0999),
                'dropout_rate': uniform(0.0, 0.5),
                'batch_size': [16, 32, 64, 128]
            }
            
            model = NeuralNetworkModel()
            return self.optimizer.random_search(model, param_distributions, X_train, y_train)
    
    def optimize_all_models(self, 
                          X_train: np.ndarray, 
                          y_train: np.ndarray,
                          optimization_method: str = 'bayesian_optimization',
                          n_trials: int = 50) -> Dict[str, Any]:
        """
        Optimize all model types.
        
        Args:
            X_train: Training features
            y_train: Training targets
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Dictionary with optimization results for all models
        """
        logger.info("Starting optimization for all models...")
        
        results = {}
        
        # Optimize each model type
        model_optimizers = [
            ('random_forest', self.optimize_random_forest),
            ('gradient_boosting', self.optimize_gradient_boosting),
            ('svr', self.optimize_svr),
            ('neural_network', self.optimize_neural_network)
        ]
        
        for model_name, optimizer_func in model_optimizers:
            logger.info(f"Optimizing {model_name}...")
            try:
                results[model_name] = optimizer_func(
                    X_train, y_train, optimization_method, n_trials
                )
            except Exception as e:
                logger.error(f"Error optimizing {model_name}: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        # Compare results
        comparison = self.optimizer.compare_optimization_methods()
        
        return {
            'model_results': results,
            'comparison': comparison
        } 