#!/usr/bin/env python3
"""
Main Analysis Pipeline for Water Quality Analysis

This module provides a comprehensive command-line interface to run the entire
water quality analysis pipeline including data processing, model training,
evaluation, visualization, and hyperparameter optimization.
"""

import argparse
import sys
import os
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import joblib

# Import project modules
from data_processor import WaterQualityDataProcessor
from models import (
    RandomForestModel, GradientBoostingModel, SVRModel, 
    NeuralNetworkModel, LinearModels, EnsembleModel
)
from evaluation import ModelEvaluator
from visualization import create_rf_visualizations, create_model_visualizations
from official_bcg_analysis import official_bcg_analysis, classify_water_quality_bcg
from optimization import HyperparameterOptimizer, AutomatedOptimizer
from config import Config
from utils import setup_logging, export_results

warnings.filterwarnings('ignore')

class WaterQualityAnalysisPipeline:
    """
    Comprehensive water quality analysis pipeline.
    
    Orchestrates the entire analysis process from data loading to
    final model deployment and reporting.
    """
    
    def __init__(self, config: Config = None):
        """
        Initialize the analysis pipeline.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
        self.data_processor = None
        self.models = {}
        self.evaluator = None
        self.visualizer = None
        self.optimizer = None
        self.results = {}
        self.output_paths = self.config.get_output_paths()
        
        # Setup logging
        setup_logging(level=self.config.get('logging.level', 'INFO'))
        self.logger = logging.getLogger(__name__)
        
        # Create output directories
        self._create_output_directories()
    
    def _create_output_directories(self):
        """Create necessary output directories."""
        directories = [
            self.output_paths["base"],
            self.output_paths["models"],
            self.output_paths["reports"],
            self.output_paths["plots"],
            self.output_paths["predictions"]
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def load_and_process_data(self, 
                            train_path: str = "train.xlsx",
                            test_path: str = "test.xlsx") -> Dict[str, Any]:
        """
        Load and process the water quality data.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            
        Returns:
            Dictionary containing processed data
        """
        self.logger.info("Loading and processing data...")
        
        try:
            self.data_processor = WaterQualityDataProcessor(
                random_seed=self.config.get("data.random_seed", 42),
                log_level=self.config.get("logging.level", "INFO")
            )
            
            # Load and process data
            self.data_processor.load_data(train_path, test_path)
            self.data_processor.preprocess_data()
            
            # Get processed data splits
            X_train, X_val, X_test, y_train, y_val, y_test = self.data_processor.get_data_splits()
            
            processed_data = {
                'X_train': X_train,
                'X_val': X_val, 
                'X_test': X_test,
                'y_train': y_train,
                'y_val': y_val,
                'y_test': y_test
            }
            
            self.logger.info("Data processing completed successfully")
            return processed_data
            
        except Exception as e:
            self.logger.error(f"Error in data processing: {str(e)}")
            raise
    
    def train_models(self, 
                    X_train: np.ndarray, 
                    y_train: np.ndarray,
                    models_to_train: List[str] = None) -> Dict[str, Any]:
        """
        Train multiple machine learning models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_train: List of model names to train (None for all)
            
        Returns:
            Dictionary containing trained models
        """
        self.logger.info("Training models...")
        
        if models_to_train is None:
            models_to_train = [
                'random_forest', 'gradient_boosting', 'svr', 
                'neural_network', 'linear_models', 'ensemble'
            ]
        
        trained_models = {}
        
        for model_name in models_to_train:
            try:
                self.logger.info(f"Training {model_name}...")
                
                if model_name == 'random_forest':
                    model = RandomForestModel()
                elif model_name == 'gradient_boosting':
                    model = GradientBoostingModel()
                elif model_name == 'svr':
                    model = SVRModel()
                elif model_name == 'neural_network':
                    model = NeuralNetworkModel()
                elif model_name == 'linear_models':
                    model = LinearModels(model_type='ridge', alpha=10.0)
                elif model_name == 'ensemble':
                    model = EnsembleModel()
                else:
                    self.logger.warning(f"Unknown model: {model_name}")
                    continue
                
                # Train the model
                model.fit(X_train, y_train)
                trained_models[model_name] = model
                
                self.logger.info(f"{model_name} training completed")
                
            except Exception as e:
                self.logger.error(f"Error training {model_name}: {str(e)}")
                continue
        
        self.models = trained_models
        self.logger.info(f"Successfully trained {len(trained_models)} models")
        return trained_models
    
    def evaluate_models(self, 
                       X_test: np.ndarray, 
                       y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate all trained models.
        
        Args:
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary containing evaluation results
        """
        self.logger.info("Evaluating models...")
        
        if not self.models:
            raise ValueError("No models available for evaluation")
        
        self.evaluator = ModelEvaluator(
            log_level=self.config.get("logging.level", "INFO")
        )
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                self.logger.info(f"Evaluating {model_name}...")
                
                results = self.evaluator.evaluate_model(
                    model, X_test, y_test, cv_folds=self.config.get("cross_validation.cv_folds", 5)
                )
                evaluation_results[model_name] = results
                
            except Exception as e:
                self.logger.error(f"Error evaluating {model_name}: {str(e)}")
                continue
        
        self.results['evaluation'] = evaluation_results
        self.logger.info("Model evaluation completed")
        return evaluation_results
    
    def create_visualizations(self, 
                            X_train: np.ndarray, 
                            y_train: np.ndarray,
                            X_test: np.ndarray, 
                            y_test: np.ndarray) -> None:
        """
        Create comprehensive visualizations.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
        """
        self.logger.info("Creating visualizations...")
        
        # Call the specific Random Forest visualization function directly
        create_rf_visualizations() # This function now handles loading data and plotting
        
        # Previous visualization calls related to WaterQualityVisualizer are removed
        # as they are now handled within create_rf_visualizations or are no longer needed
        
        self.logger.info("Visualizations created successfully")
    
    def optimize_hyperparameters(self, 
                               X_train: np.ndarray, 
                               y_train: np.ndarray,
                               models_to_optimize: List[str] = None,
                               optimization_method: str = 'bayesian_optimization',
                               n_trials: int = 100) -> Dict[str, Any]:
        """
        Optimize hyperparameters for selected models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_optimize: List of model names to optimize (None for all)
            optimization_method: Optimization method to use
            n_trials: Number of trials for Bayesian optimization
            
        Returns:
            Dictionary containing optimization results
        """
        self.logger.info("Starting hyperparameter optimization...")
        
        if models_to_optimize is None:
            models_to_optimize = ['random_forest', 'gradient_boosting', 'svr', 'neural_network']
        
        self.optimizer = AutomatedOptimizer(
            cv_folds=self.config.get("cross_validation.cv_folds", 5),
            n_jobs=self.config.get("optimization.n_jobs", -1),
            random_state=self.config.get("data.random_seed", 42)
        )
        
        optimization_results = {}
        
        for model_name in models_to_optimize:
            try:
                self.logger.info(f"Optimizing {model_name}...")
                
                if model_name == 'random_forest':
                    results = self.optimizer.optimize_random_forest(
                        X_train, y_train, optimization_method, n_trials
                    )
                elif model_name == 'gradient_boosting':
                    results = self.optimizer.optimize_gradient_boosting(
                        X_train, y_train, optimization_method, n_trials
                    )
                elif model_name == 'svr':
                    results = self.optimizer.optimize_svr(
                        X_train, y_train, optimization_method, n_trials
                    )
                elif model_name == 'neural_network':
                    results = self.optimizer.optimize_neural_network(
                        X_train, y_train, optimization_method, n_trials
                    )
                else:
                    self.logger.warning(f"Unknown model for optimization: {model_name}")
                    continue
                
                optimization_results[model_name] = results
                
                # Update model with optimized parameters
                if 'best_estimator' in results:
                    self.models[f"{model_name}_optimized"] = results['best_estimator']
                
            except Exception as e:
                self.logger.error(f"Error optimizing {model_name}: {str(e)}")
                continue
        
        self.results['optimization'] = optimization_results
        self.logger.info("Hyperparameter optimization completed")
        return optimization_results
    
    def save_models_and_results(self) -> None:
        """Save trained models and results."""
        self.logger.info("Saving models and results...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save models
        for model_name, model in self.models.items():
            model_dir = os.path.join(self.output_paths["models"], model_name)
            Path(model_dir).mkdir(parents=True, exist_ok=True)
            model_path = os.path.join(model_dir, f"model_{timestamp}.joblib")
            joblib.dump(model, model_path)
            # Save metrics for this model if available
            if self.results.get('evaluation') and model_name in self.results['evaluation']:
                metrics_path = os.path.join(model_dir, f"metrics_{timestamp}.json")
                export_results(self.results['evaluation'][model_name], metrics_path, format="json")
            self.logger.info(f"Saved model + metrics in: {model_dir}")
        
        # Save results
        results_path = os.path.join(
            self.output_paths["reports"],
            f"analysis_results_{timestamp}.json"
        )
        export_results(self.results, results_path, format="json")
        self.logger.info(f"Saved results: {results_path}")
        
        # Save optimization results
        if self.results.get('optimization'):
            opt_path = os.path.join(
                self.output_paths["reports"],
                f"optimization_results_{timestamp}.joblib"
            )
            self.optimizer.optimizer.save_optimization_results(opt_path)
            self.logger.info(f"Saved optimization results: {opt_path}")
    
    def generate_report(self) -> str:
        """
        Generate a comprehensive analysis report.
        
        Returns:
            Path to the generated report
        """
        self.logger.info("Generating analysis report...")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_paths["reports"], f"report_{timestamp}.txt")
        
        with open(report_path, 'w') as f:
            f.write("WATER QUALITY ANALYSIS REPORT\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Data summary
            f.write("DATA SUMMARY\n")
            f.write("-" * 20 + "\n")
            if self.data_processor:
                f.write(f"Training samples: {self.data_processor.X_train.shape[0]}\n")
                f.write(f"Test samples: {self.data_processor.X_test.shape[0]}\n")
                f.write(f"Features: {self.data_processor.X_train.shape[1]}\n\n")
            
            # Model performance summary
            f.write("MODEL PERFORMANCE SUMMARY\n")
            f.write("-" * 30 + "\n")
            if self.results.get('evaluation'):
                for model_name, results in self.results['evaluation'].items():
                    f.write(f"\n{model_name.upper()}:\n")
                    r2_val = results.get('r2')
                    rmse_val = results.get('rmse')
                    mae_val = results.get('mae')
                    mape_val = results.get('mape')

                    f.write(f"  R² Score: {r2_val:.4f}\n" if isinstance(r2_val, (int, float)) else f"  R² Score: N/A\n")
                    f.write(f"  RMSE: {rmse_val:.4f}\n" if isinstance(rmse_val, (int, float)) else f"  RMSE: N/A\n")
                    f.write(f"  MAE: {mae_val:.4f}\n" if isinstance(mae_val, (int, float)) else f"  MAE: N/A\n")
                    f.write(f"  MAPE: {mape_val:.4f}\n" if isinstance(mape_val, (int, float)) else f"  MAPE: N/A\n")
            
            # Optimization summary
            if self.results.get('optimization'):
                f.write("\n\nHYPERPARAMETER OPTIMIZATION SUMMARY\n")
                f.write("-" * 40 + "\n")
                for model_name, results in self.results['optimization'].items():
                    f.write(f"\n{model_name.upper()}:\n")
                    f.write(f"  Best Score: {results.get('best_score', 'N/A'):.4f}\n")
                    f.write(f"  Best Parameters: {results.get('best_params', 'N/A')}\n")
        
        self.logger.info(f"Report generated: {report_path}")
        return report_path
    
    def run_full_pipeline(self, 
                         train_path: str = "train.xlsx",
                         test_path: str = "test.xlsx",
                         models_to_train: List[str] = None,
                         optimize_hyperparameters: bool = False,
                         optimization_method: str = 'bayesian_optimization',
                         n_trials: int = 100) -> Dict[str, Any]:
        """
        Run the complete analysis pipeline.
        
        Args:
            train_path: Path to training data
            test_path: Path to test data
            models_to_train: List of models to train (None for all)
            optimize_hyperparameters: Whether to perform hyperparameter optimization
            optimization_method: Optimization method to use
            n_trials: Number of trials for optimization
            
        Returns:
            Dictionary containing all results
        """
        self.logger.info("Starting full analysis pipeline...")
        
        try:
            # Step 1: Load and process data
            processed_data = self.load_and_process_data(train_path, test_path)
            
            # Step 2: Train models
            trained_models = self.train_models(
                processed_data['X_train'], 
                processed_data['y_train'],
                models_to_train
            )
            
            # Step 3: Evaluate models
            evaluation_results = self.evaluate_models(
                processed_data['X_test'], 
                processed_data['y_test']
            )
            
            # Use the **full external test set** (47 rows) instead of the internal hold-out
            X_test_external = self.data_processor.get_test_data_processed()
            stations_external = self.data_processor.test_data['station'].values

            pred_df = pd.DataFrame({'Station': stations_external})

            for m_name, m in self.models.items():
                try:
                    preds = m.predict(X_test_external)
                    pred_df[f'Pred_{m_name}'] = preds
                    pred_df[f'BCG_{m_name}'] = [classify_water_quality_bcg(p) for p in preds]
                except Exception as e:
                    self.logger.error(f'Prediction failed for {m_name}: {e}')

            # Save combined predictions
            ts_stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            Path(self.output_paths['predictions']).mkdir(parents=True, exist_ok=True)
            pred_csv = os.path.join(self.output_paths['predictions'], f'predictions_{ts_stamp}.csv')
            pred_xlsx = os.path.join(self.output_paths['predictions'], f'predictions_{ts_stamp}.xlsx')
            pred_df.to_csv(pred_csv, index=False)
            pred_df.to_excel(pred_xlsx, index=False)
            self.logger.info(f'Saved combined predictions: {pred_csv}')

            # Generate visuals for each model
            for m_name in self.models.keys():
                create_model_visualizations(pred_df, m_name)

            # Run official BCG analysis to generate bcg_predictions.xlsx
            self.logger.info("Running official BCG analysis...")
            official_bcg_analysis()
            self.logger.info("Official BCG analysis completed.")

            # Step 4: Create visualizations
            self.create_visualizations(
                processed_data['X_train'], 
                processed_data['y_train'],
                processed_data['X_test'], 
                processed_data['y_test']
            )
            
            # Step 5: Hyperparameter optimization (optional)
            if optimize_hyperparameters:
                optimization_results = self.optimize_hyperparameters(
                    processed_data['X_train'], 
                    processed_data['y_train'],
                    models_to_train,
                    optimization_method,
                    n_trials
                )
                
                # Re-evaluate optimized models
                if optimization_results:
                    self.evaluate_models(
                        processed_data['X_test'], 
                        processed_data['y_test']
                    )
            
            # Step 6: Save results
            self.save_models_and_results()
            
            # Step 7: Generate report
            report_path = self.generate_report()
            
            self.logger.info("Full analysis pipeline completed successfully!")
            
            return {
                'processed_data': processed_data,
                'trained_models': trained_models,
                'evaluation_results': evaluation_results,
                'optimization_results': self.results.get('optimization'),
                'report_path': report_path
            }
            
        except Exception as e:
            self.logger.error(f"Error in pipeline: {str(e)}")
            raise


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description="Water Quality Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline with all models
  python main.py --full-pipeline
  
  # Run with specific models only
  python main.py --models random_forest gradient_boosting --full-pipeline
  
  # Run with hyperparameter optimization
  python main.py --full-pipeline --optimize --optimization-method bayesian_optimization
  
  # Run only data processing and visualization
  python main.py --data-only --visualize
        """
    )
    
    # Data options
    parser.add_argument('--train-path', default='train.xlsx',
                       help='Path to training data (default: train.xlsx)')
    parser.add_argument('--test-path', default='test.xlsx',
                       help='Path to test data (default: test.xlsx)')
    
    # Pipeline options
    parser.add_argument('--full-pipeline', action='store_true',
                       help='Run the complete analysis pipeline')
    parser.add_argument('--data-only', action='store_true',
                       help='Run only data processing')
    parser.add_argument('--models', nargs='+',
                       choices=['random_forest', 'gradient_boosting', 'svr', 
                               'neural_network', 'linear_models', 'ensemble'],
                       help='Specific models to train')
    
    # Optimization options
    parser.add_argument('--optimize', action='store_true',
                       help='Perform hyperparameter optimization')
    parser.add_argument('--optimization-method', 
                       choices=['grid_search', 'random_search', 'bayesian_optimization'],
                       default='bayesian_optimization',
                       help='Optimization method (default: bayesian_optimization)')
    parser.add_argument('--n-trials', type=int, default=100,
                       help='Number of optimization trials (default: 100)')
    
    # Output options
    parser.add_argument('--visualize', action='store_true',
                       help='Create visualizations')
    parser.add_argument('--save-models', action='store_true',
                       help='Save trained models')
    parser.add_argument('--generate-report', action='store_true',
                       help='Generate analysis report')
    
    # Configuration options
    parser.add_argument('--config-file', help='Path to configuration file')
    parser.add_argument('--output-dir', help='Output directory')
    parser.add_argument('--log-level', 
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level (default: INFO)')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config = Config()
        if args.config_file:
            config.load_from_file(args.config_file)
        if args.output_dir:
            config.OUTPUT_DIR = args.output_dir
        
        # Create pipeline
        pipeline = WaterQualityAnalysisPipeline(config)
        
        if args.full_pipeline:
            # Run complete pipeline
            results = pipeline.run_full_pipeline(
                train_path=args.train_path,
                test_path=args.test_path,
                models_to_train=args.models,
                optimize_hyperparameters=args.optimize,
                optimization_method=args.optimization_method,
                n_trials=args.n_trials
            )
            
            logger.info("Full pipeline completed successfully!")
            logger.info(f"Results saved to: {pipeline.output_paths['reports']}")
            
        elif args.data_only:
            # Run only data processing
            processed_data = pipeline.load_and_process_data(
                args.train_path, args.test_path
            )
            
            if args.visualize:
                pipeline.create_visualizations(
                    processed_data['X_train'], processed_data['y_train'],
                    processed_data['X_test'], processed_data['y_test']
                )
            
            logger.info("Data processing completed!")
            
        else:
            # Show help if no action specified
            parser.print_help()
            return 1
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 