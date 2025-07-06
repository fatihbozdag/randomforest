"""
Model evaluation module for Water Quality Analysis System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold, LeaveOneOut
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score, 
    explained_variance_score, max_error, mean_poisson_deviance,
    mean_gamma_deviance, mean_absolute_percentage_error
)
from sklearn.metrics import make_scorer
import warnings
import time
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from models.base_model import BaseWaterQualityModel
from utils import (
    setup_logging, calculate_prediction_intervals, 
    calculate_confidence_intervals, classify_water_quality_batch
)
from config import config

class ModelEvaluator:
    """
    Comprehensive model evaluation system
    """
    
    def __init__(self, log_level: str = "INFO"):
        """
        Initialize model evaluator
        
        Args:
            log_level: Logging level
        """
        self.logger = setup_logging(log_level)
        self.metrics = {}
        self.cv_scores = {}
        self.residual_analysis = {}
        self.model_comparisons = {}
        self.evaluation_history = []
        
        # Define custom metrics
        self.custom_metrics = {
            'mape': mean_absolute_percentage_error,
            'explained_variance': explained_variance_score,
            'max_error': max_error
        }
    
    def evaluate_model(self, model: BaseWaterQualityModel, X: np.ndarray, y: np.ndarray, 
                      cv_folds: int = 5, cv_strategy: str = 'kfold', 
                      confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Comprehensive model evaluation
        
        Args:
            model: Model to evaluate
            X: Input features
            y: True targets
            cv_folds: Number of cross-validation folds
            cv_strategy: Cross-validation strategy ('kfold', 'stratified', 'loo')
            confidence_level: Confidence level for intervals
            
        Returns:
            Dictionary with comprehensive evaluation results
        """
        self.logger.info(f"Evaluating model: {model.name}")
        start_time = time.time()
        
        # Validate inputs
        if not model.is_fitted:
            raise ValueError(f"Model {model.name} must be fitted before evaluation")
        
        # Perform cross-validation
        cv_results = self._perform_cross_validation(model, X, y, cv_folds, cv_strategy)
        
        # Calculate comprehensive metrics
        metrics_results = self._calculate_comprehensive_metrics(model, X, y)
        
        # Perform residual analysis
        residual_results = self._perform_residual_analysis(model, X, y)
        
        # Calculate prediction intervals
        prediction_intervals = self._calculate_prediction_intervals(model, X, y, confidence_level)
        
        # Compile results
        evaluation_results = {
            'model_name': model.name,
            'model_type': type(model).__name__,
            'evaluation_time': time.time() - start_time,
            'cv_results': cv_results,
            'metrics': metrics_results,
            'residual_analysis': residual_results,
            'prediction_intervals': prediction_intervals,
            'model_complexity': model.get_model_complexity(),
            'feature_importance': self._get_feature_importance_summary(model)
        }
        
        # Store results
        self.metrics[model.name] = evaluation_results
        self.evaluation_history.append({
            'timestamp': time.time(),
            'model_name': model.name,
            'results': evaluation_results
        })
        
        self.logger.info(f"Evaluation completed for {model.name}")
        return evaluation_results
    
    def _perform_cross_validation(self, model: BaseWaterQualityModel, X: np.ndarray, y: np.ndarray,
                                cv_folds: int, cv_strategy: str) -> Dict[str, Any]:
        """Perform cross-validation with different strategies"""
        
        # Select cross-validation strategy
        if cv_strategy == 'kfold':
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=config.get("data.random_seed", 42))
        elif cv_strategy == 'stratified':
            # For regression, we'll create bins for stratification
            y_binned = pd.cut(y, bins=cv_folds, labels=False)
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=config.get("data.random_seed", 42))
            y_cv = y_binned
        elif cv_strategy == 'loo':
            cv = LeaveOneOut()
            y_cv = y
        else:
            raise ValueError(f"Unsupported CV strategy: {cv_strategy}")
        
        # Define scoring metrics
        scoring_metrics = {
            'neg_mse': 'neg_mean_squared_error',
            'neg_rmse': make_scorer(lambda y_true, y_pred: -np.sqrt(mean_squared_error(y_true, y_pred))),
            'neg_mae': 'neg_mean_absolute_error',
            'r2': 'r2',
            'explained_variance': 'explained_variance'
        }
        
        cv_results = {}
        
        for metric_name, scorer in scoring_metrics.items():
            try:
                scores = cross_val_score(
                    estimator=model.model,
                    X=X,
                    y=y_cv if cv_strategy == 'stratified' else y,
                    cv=cv,
                    scoring=scorer,
                    n_jobs=-1
                )
                
                cv_results[metric_name] = {
                    'scores': scores,
                    'mean': scores.mean(),
                    'std': scores.std(),
                    'min': scores.min(),
                    'max': scores.max()
                }
            except Exception as e:
                self.logger.warning(f"Could not calculate {metric_name}: {e}")
                cv_results[metric_name] = None
        
        return cv_results
    
    def _calculate_comprehensive_metrics(self, model: BaseWaterQualityModel, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Calculate comprehensive set of regression metrics"""
        
        y_pred = model.predict(X)
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = mean_squared_error(y, y_pred)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = mean_absolute_error(y, y_pred)
        metrics['r2'] = r2_score(y, y_pred)
        
        # Additional metrics
        metrics['explained_variance'] = explained_variance_score(y, y_pred)
        metrics['max_error'] = max_error(y, y_pred)
        
        # Percentage errors
        try:
            metrics['mape'] = mean_absolute_percentage_error(y, y_pred)
        except:
            metrics['mape'] = np.nan
        
        # Custom metrics
        metrics['mean_absolute_deviation'] = np.mean(np.abs(y - np.mean(y)))
        metrics['relative_mae'] = metrics['mae'] / metrics['mean_absolute_deviation']
        
        # Calculate MAPE manually if sklearn version doesn't support it
        if np.isnan(metrics['mape']):
            mape = np.mean(np.abs((y - y_pred) / np.where(y != 0, y, 1))) * 100
            metrics['mape'] = mape
        
        # Additional statistical metrics
        residuals = y - y_pred
        metrics['residual_mean'] = np.mean(residuals)
        metrics['residual_std'] = np.std(residuals)
        
        metrics['residual_skewness'] = stats.skew(residuals)
        metrics['residual_kurtosis'] = stats.kurtosis(residuals)
        
        return metrics
    
    def _perform_residual_analysis(self, model: BaseWaterQualityModel, X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform comprehensive residual analysis"""
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Basic residual statistics
        residual_stats = {
            'mean': np.mean(residuals),
            'std': np.std(residuals),
            'min': np.min(residuals),
            'max': np.max(residuals),
            'median': np.median(residuals),
            'skewness': stats.skew(residuals),
            'kurtosis': stats.kurtosis(residuals)
        }
        
        # Normality tests
        normality_tests = {
            'shapiro_wilk': stats.shapiro(residuals),
            'anderson_darling': stats.anderson(residuals),
            'ks_test': stats.kstest(residuals, 'norm', args=(residual_stats['mean'], residual_stats['std']))
        }
        
        # Heteroscedasticity test (Breusch-Pagan test)
        try:
            # Simple test for heteroscedasticity
            residual_squared = residuals ** 2
            correlation = np.corrcoef(y_pred, residual_squared)[0, 1]
            heteroscedasticity_p = 1 - stats.norm.cdf(abs(correlation) * np.sqrt(len(residuals) - 2))
        except:
            heteroscedasticity_p = np.nan
        
        # Autocorrelation test (Durbin-Watson)
        try:
            dw_statistic = np.sum(np.diff(residuals) ** 2) / np.sum(residuals ** 2)
        except:
            dw_statistic = np.nan
        
        residual_analysis = {
            'residuals': residuals,
            'predictions': y_pred,
            'actual': y,
            'statistics': residual_stats,
            'normality_tests': normality_tests,
            'heteroscedasticity_p': heteroscedasticity_p,
            'durbin_watson': dw_statistic,
            'is_normal': normality_tests['shapiro_wilk'][1] > 0.05,
            'is_homoscedastic': heteroscedasticity_p > 0.05 if not np.isnan(heteroscedasticity_p) else None
        }
        
        return residual_analysis
    
    def _calculate_prediction_intervals(self, model: BaseWaterQualityModel, X: np.ndarray, y: np.ndarray,
                                      confidence_level: float) -> Dict[str, Any]:
        """Calculate prediction intervals and confidence intervals"""
        
        y_pred = model.predict(X)
        residuals = y - y_pred
        
        # Calculate prediction intervals using residual standard deviation
        residual_std = np.std(residuals)
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        prediction_intervals = {
            'lower_bound': y_pred - z_score * residual_std,
            'upper_bound': y_pred + z_score * residual_std,
            'confidence_level': confidence_level,
            'residual_std': residual_std,
            'z_score': z_score
        }
        
        # Calculate confidence intervals for mean prediction
        n = len(y)
        standard_error = residual_std / np.sqrt(n)
        confidence_intervals = {
            'lower_bound': y_pred - z_score * standard_error,
            'upper_bound': y_pred + z_score * standard_error,
            'confidence_level': confidence_level,
            'standard_error': standard_error
        }
        
        return {
            'prediction_intervals': prediction_intervals,
            'confidence_intervals': confidence_intervals
        }
    
    def _get_feature_importance_summary(self, model: BaseWaterQualityModel) -> Dict[str, Any]:
        """Get feature importance summary"""
        
        feature_importance = model.get_feature_importance()
        if feature_importance is None:
            return {'available': False}
        
        # Get top features
        feature_names = model.feature_names if model.feature_names else [f'feature_{i}' for i in range(len(feature_importance))]
        
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': feature_importance
        }).sort_values('importance', ascending=False)
        
        return {
            'available': True,
            'top_features': importance_df.head(10).to_dict('records'),
            'total_features': len(feature_importance),
            'importance_range': (feature_importance.min(), feature_importance.max()),
            'importance_mean': feature_importance.mean(),
            'importance_std': feature_importance.std()
        }
    
    def compare_models(self, models_dict: Dict[str, BaseWaterQualityModel], 
                      X: np.ndarray, y: np.ndarray, cv_folds: int = 5) -> Dict[str, Any]:
        """
        Compare multiple models comprehensively
        
        Args:
            models_dict: Dictionary of model_name: model pairs
            X: Input features
            y: True targets
            cv_folds: Number of cross-validation folds
            
        Returns:
            Dictionary with comparison results
        """
        self.logger.info(f"Comparing {len(models_dict)} models")
        
        comparison_results = {
            'models': {},
            'summary': {},
            'ranking': {},
            'statistical_tests': {}
        }
        
        # Evaluate each model
        for model_name, model in models_dict.items():
            self.logger.info(f"Evaluating {model_name}")
            evaluation_results = self.evaluate_model(model, X, y, cv_folds)
            comparison_results['models'][model_name] = evaluation_results
        
        # Create comparison summary
        comparison_results['summary'] = self._create_comparison_summary(comparison_results['models'])
        
        # Create model rankings
        comparison_results['ranking'] = self._create_model_rankings(comparison_results['models'])
        
        # Perform statistical significance tests
        comparison_results['statistical_tests'] = self._perform_statistical_tests(comparison_results['models'], X, y)
        
        # Store comparison results
        self.model_comparisons[time.time()] = comparison_results
        
        return comparison_results
    
    def _create_comparison_summary(self, models_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Create summary comparison of models"""
        
        summary = {
            'metrics_comparison': {},
            'cv_comparison': {},
            'complexity_comparison': {},
            'best_models': {}
        }
        
        # Compare metrics
        metrics_to_compare = ['rmse', 'mae', 'r2', 'mape', 'explained_variance']
        
        for metric in metrics_to_compare:
            metric_values = {}
            for model_name, results in models_results.items():
                if metric in results['metrics']:
                    metric_values[model_name] = results['metrics'][metric]
            
            if metric_values:
                summary['metrics_comparison'][metric] = {
                    'values': metric_values,
                    'best': min(metric_values.items(), key=lambda x: x[1]) if metric in ['rmse', 'mae', 'mape'] else max(metric_values.items(), key=lambda x: x[1]),
                    'worst': max(metric_values.items(), key=lambda x: x[1]) if metric in ['rmse', 'mae', 'mape'] else min(metric_values.items(), key=lambda x: x[1])
                }
        
        # Compare CV results
        cv_metrics = ['neg_mse', 'neg_mae', 'r2']
        for metric in cv_metrics:
            cv_values = {}
            for model_name, results in models_results.items():
                if metric in results['cv_results'] and results['cv_results'][metric]:
                    cv_values[model_name] = results['cv_results'][metric]['mean']
            
            if cv_values:
                summary['cv_comparison'][metric] = {
                    'values': cv_values,
                    'best': max(cv_values.items(), key=lambda x: x[1]) if metric.startswith('neg_') else max(cv_values.items(), key=lambda x: x[1]),
                    'worst': min(cv_values.items(), key=lambda x: x[1]) if metric.startswith('neg_') else min(cv_values.items(), key=lambda x: x[1])
                }
        
        # Compare model complexity
        complexity_metrics = ['total_parameters', 'n_features']
        for metric in complexity_metrics:
            complexity_values = {}
            for model_name, results in models_results.items():
                if metric in results['model_complexity']:
                    complexity_values[model_name] = results['model_complexity'][metric]
            
            if complexity_values:
                summary['complexity_comparison'][metric] = {
                    'values': complexity_values,
                    'most_complex': max(complexity_values.items(), key=lambda x: x[1]),
                    'least_complex': min(complexity_values.items(), key=lambda x: x[1])
                }
        
        # Identify best models by different criteria
        summary['best_models'] = {
            'best_rmse': min([(name, results['metrics']['rmse']) for name, results in models_results.items()], key=lambda x: x[1])[0],
            'best_r2': max([(name, results['metrics']['r2']) for name, results in models_results.items()], key=lambda x: x[1])[0],
            'best_cv_score': max([(name, results['cv_results']['r2']['mean']) for name, results in models_results.items() if 'r2' in results['cv_results'] and results['cv_results']['r2']], key=lambda x: x[1])[0]
        }
        
        return summary
    
    def _create_model_rankings(self, models_results: Dict[str, Dict]) -> Dict[str, List[str]]:
        """Create model rankings by different criteria"""
        
        rankings = {}
        
        # Rank by RMSE (lower is better)
        rmse_ranking = sorted(
            [(name, results['metrics']['rmse']) for name, results in models_results.items()],
            key=lambda x: x[1]
        )
        rankings['by_rmse'] = [name for name, _ in rmse_ranking]
        
        # Rank by R² (higher is better)
        r2_ranking = sorted(
            [(name, results['metrics']['r2']) for name, results in models_results.items()],
            key=lambda x: x[1],
            reverse=True
        )
        rankings['by_r2'] = [name for name, _ in r2_ranking]
        
        # Rank by CV R² score
        cv_r2_ranking = sorted(
            [(name, results['cv_results']['r2']['mean']) for name, results in models_results.items() 
             if 'r2' in results['cv_results'] and results['cv_results']['r2']],
            key=lambda x: x[1],
            reverse=True
        )
        rankings['by_cv_r2'] = [name for name, _ in cv_r2_ranking]
        
        # Rank by complexity (lower is better)
        complexity_ranking = sorted(
            [(name, results['model_complexity'].get('total_parameters', 0)) for name, results in models_results.items()],
            key=lambda x: x[1]
        )
        rankings['by_complexity'] = [name for name, _ in complexity_ranking]
        
        return rankings
    
    def _perform_statistical_tests(self, models_results: Dict[str, Dict], X: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
        """Perform statistical significance tests between models"""
        
        statistical_tests = {}
        
        # Get predictions from all models
        predictions = {}
        for model_name, results in models_results.items():
            # We need to get the model instance to make predictions
            # This is a limitation - we should store model instances
            predictions[model_name] = results['residual_analysis']['predictions']
        
        # Perform paired t-tests between all model pairs
        model_names = list(predictions.keys())
        t_test_results = {}
        
        for i, model1 in enumerate(model_names):
            for j, model2 in enumerate(model_names[i+1:], i+1):
                pred1 = predictions[model1]
                pred2 = predictions[model2]
                
                # Calculate squared errors
                se1 = (y - pred1) ** 2
                se2 = (y - pred2) ** 2
                
                # Paired t-test
                t_stat, p_value = stats.ttest_rel(se1, se2)
                
                t_test_results[f"{model1}_vs_{model2}"] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'model1_better': t_stat < 0  # Negative t-stat means model1 has lower MSE
                }
        
        statistical_tests['paired_t_tests'] = t_test_results
        
        return statistical_tests
    
    def generate_evaluation_report(self, output_path: str = None) -> str:
        """
        Generate comprehensive evaluation report
        
        Args:
            output_path: Path to save the report
            
        Returns:
            Report content as string
        """
        
        if not self.metrics:
            return "No evaluation results available"
        
        report = []
        report.append("=" * 80)
        report.append("WATER QUALITY ANALYSIS - MODEL EVALUATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Individual model results
        for model_name, results in self.metrics.items():
            report.append(f"MODEL: {model_name}")
            report.append("-" * 40)
            
            # Metrics
            report.append("PERFORMANCE METRICS:")
            metrics = results['metrics']
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report.append(f"  {metric.upper()}: {value:.6f}")
                else:
                    report.append(f"  {metric.upper()}: {value}")
            
            # CV Results
            report.append("\nCROSS-VALIDATION RESULTS:")
            cv_results = results['cv_results']
            for metric, cv_data in cv_results.items():
                if cv_data:
                    report.append(f"  {metric.upper()}: {cv_data['mean']:.6f} ± {cv_data['std']:.6f}")
            
            # Residual Analysis
            report.append("\nRESIDUAL ANALYSIS:")
            residual_stats = results['residual_analysis']['statistics']
            report.append(f"  Mean: {residual_stats['mean']:.6f}")
            report.append(f"  Std: {residual_stats['std']:.6f}")
            report.append(f"  Skewness: {residual_stats['skewness']:.6f}")
            report.append(f"  Kurtosis: {residual_stats['kurtosis']:.6f}")
            report.append(f"  Normal Distribution: {results['residual_analysis']['is_normal']}")
            
            report.append("\n" + "=" * 80)
        
        # Model comparisons
        if self.model_comparisons:
            latest_comparison = max(self.model_comparisons.keys())
            comparison = self.model_comparisons[latest_comparison]
            
            report.append("\nMODEL COMPARISON SUMMARY")
            report.append("=" * 80)
            
            # Best models
            best_models = comparison['summary']['best_models']
            report.append("BEST MODELS BY CRITERIA:")
            for criterion, model in best_models.items():
                report.append(f"  {criterion}: {model}")
            
            # Rankings
            rankings = comparison['ranking']
            report.append("\nMODEL RANKINGS:")
            for criterion, ranking in rankings.items():
                report.append(f"  {criterion}: {' > '.join(ranking)}")
        
        report_content = "\n".join(report)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report_content)
        
        return report_content
    
    def plot_model_comparison(self, save_path: str = None) -> None:
        """Plot comprehensive model comparison"""
        
        if not self.metrics:
            print("No evaluation results available for plotting")
            return
        
        # Prepare data for plotting
        model_names = list(self.metrics.keys())
        metrics_to_plot = ['rmse', 'mae', 'r2', 'mape']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, metric in enumerate(metrics_to_plot):
            values = [self.metrics[name]['metrics'][metric] for name in model_names]
            
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.upper()} Comparison')
            axes[i].set_ylabel(metric.upper())
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_residual_analysis(self, model_name: str, save_path: str = None) -> None:
        """Plot residual analysis for a specific model"""
        
        if model_name not in self.metrics:
            print(f"No evaluation results available for model: {model_name}")
            return
        
        results = self.metrics[model_name]
        residual_analysis = results['residual_analysis']
        
        residuals = residual_analysis['residuals']
        predictions = residual_analysis['predictions']
        actual = residual_analysis['actual']
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Residuals vs Predicted
        axes[0, 0].scatter(predictions, residuals, alpha=0.6)
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
        stats.probplot(residuals, dist="norm", plot=axes[1, 0])
        axes[1, 0].set_title('Q-Q Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Actual vs Predicted
        axes[1, 1].scatter(actual, predictions, alpha=0.6)
        axes[1, 1].plot([actual.min(), actual.max()], [actual.min(), actual.max()], 'r--', lw=2)
        axes[1, 1].set_xlabel('Actual Values')
        axes[1, 1].set_ylabel('Predicted Values')
        axes[1, 1].set_title('Actual vs Predicted')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show() 