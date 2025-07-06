#!/usr/bin/env python3
"""
Comprehensive Water Quality Analysis Script
Fixes all issues and provides detailed model performance analysis
"""

import pandas as pd
import numpy as np
import os
import warnings
from datetime import datetime
import json

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

from data_processor import WaterQualityDataProcessor
from models.random_forest_model import RandomForestModel
from models.gradient_boosting_model import GradientBoostingModel
from models.svr_model import SVRModel
from models.neural_network_model import NeuralNetworkModel
from models.linear_models import LinearModels
from models.ensemble_model import EnsembleModel
from evaluation import ModelEvaluator
from utils import classify_water_quality_batch, export_results

class ComprehensiveWaterQualityAnalysis:
    """Comprehensive water quality analysis with all fixes applied"""
    
    def __init__(self):
        self.processor = None
        self.models = {}
        self.evaluator = ModelEvaluator()
        self.results = {}
        self.predictions = {}
        
        # Water quality classification criteria
        self.quality_criteria = {
            "Excellent": "Score ≥ 0.8 (High quality, suitable for all uses)",
            "Good": "0.6 ≤ Score < 0.8 (Good quality, minor treatment may be needed)",
            "Fair": "0.4 ≤ Score < 0.6 (Fair quality, treatment required)",
            "Poor": "0.2 ≤ Score < 0.4 (Poor quality, significant treatment needed)",
            "Very Poor": "Score < 0.2 (Very poor quality, not suitable for most uses)"
        }
    
    def load_and_process_data(self, train_path="train.xlsx", test_path="test.xlsx"):
        """Load and process the data"""
        print("="*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("="*80)
        
        # Initialize data processor
        self.processor = WaterQualityDataProcessor(random_seed=42, log_level="INFO")
        
        # Load and preprocess data
        self.processor.load_data(train_path, test_path)
        self.processor.preprocess_data()
        
        # Get processed data
        X_train, X_val, X_test, y_train, y_val, y_test = self.processor.get_data_splits()
        
        print(f"\n✓ Data loaded and processed successfully!")
        print(f"  - Training samples: {len(X_train)}")
        print(f"  - Validation samples: {len(X_val)}")
        print(f"  - Test samples: {len(X_test)}")
        print(f"  - Features: {X_train.shape[1]}")
        print(f"  - Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
        
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def train_models(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models with proper error handling"""
        print("\n" + "="*80)
        print("STEP 2: MODEL TRAINING")
        print("="*80)
        
        # Define models to train
        model_configs = {
            'random_forest': {
                'class': RandomForestModel,
                'params': {'n_estimators': 100, 'max_depth': 10}
            },
            'gradient_boosting': {
                'class': GradientBoostingModel,
                'params': {'n_estimators': 100, 'learning_rate': 0.1}
            },
            'svr': {
                'class': SVRModel,
                'params': {'kernel': 'rbf', 'C': 1.0}
            },
            'neural_network': {
                'class': NeuralNetworkModel,
                'params': {'hidden_layer_sizes': (100, 50), 'max_iter': 500}
            },
            'linear_models': {
                'class': LinearModels,
                'params': {'model_type': 'ridge', 'alpha': 1.0}
            },
            'ensemble': {
                'class': EnsembleModel,
                'params': {'ensemble_type': 'voting'}
            }
        }
        
        trained_models = {}
        
        for model_name, config in model_configs.items():
            try:
                print(f"\nTraining {model_name}...")
                
                # Initialize model
                model = config['class'](**config['params'])
                
                # Train model with appropriate parameters
                if model_name == 'gradient_boosting' and X_val is not None:
                    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
                else:
                    model.fit(X_train, y_train)
                
                trained_models[model_name] = model
                print(f"✓ {model_name} trained successfully")
                
            except Exception as e:
                print(f"✗ Error training {model_name}: {e}")
                continue
        
        self.models = trained_models
        print(f"\n✓ Successfully trained {len(trained_models)} models")
        return trained_models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluate all trained models"""
        print("\n" + "="*80)
        print("STEP 3: MODEL EVALUATION")
        print("="*80)
        
        evaluation_results = {}
        
        for model_name, model in self.models.items():
            try:
                print(f"\nEvaluating {model_name}...")
                
                # Perform comprehensive evaluation
                results = self.evaluator.evaluate_model(
                    model, X_test, y_test, cv_folds=5
                )
                
                evaluation_results[model_name] = results
                
                # Display key metrics
                metrics = results['metrics']
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE:  {metrics['mae']:.4f}")
                print(f"  R²:   {metrics['r2']:.4f}")
                print(f"  MAPE: {metrics.get('mape', 'N/A'):.4f}")
                
            except Exception as e:
                print(f"✗ Error evaluating {model_name}: {e}")
                continue
        
        self.results = evaluation_results
        return evaluation_results
    
    def make_predictions(self, test_path="test.xlsx"):
        """Make predictions on test data"""
        print("\n" + "="*80)
        print("STEP 4: PREDICTIONS ON TEST DATA")
        print("="*80)
        
        # Get processed test data
        X_test_processed = self.processor.get_test_data_processed()
        
        # Load original test data for station information
        test_data = pd.read_excel(test_path)
        
        predictions = {}
        
        for model_name, model in self.models.items():
            try:
                pred = model.predict(X_test_processed)
                predictions[model_name] = pred
                print(f"✓ {model_name}: {len(pred)} predictions made")
            except Exception as e:
                print(f"✗ Error making predictions with {model_name}: {e}")
        
        # Create predictions DataFrame
        if predictions:
            results_df = pd.DataFrame()
            results_df['Station'] = test_data['istasyon'].values
            results_df['TP_norm'] = test_data['TP_norm'].values
            results_df['EC_norm'] = test_data['EC_norm'].values
            results_df['DO_norm'] = test_data['DO_norm'].values
            results_df['TIT_norm'] = test_data['tıt_norm'].values
            
            # Add model predictions
            for model_name, pred in predictions.items():
                results_df[f'Predicted_Quality_{model_name}'] = pred
                results_df[f'Quality_Class_{model_name}'] = classify_water_quality_batch(pred)
            
            # Calculate ensemble prediction
            pred_cols = [col for col in results_df.columns if col.startswith('Predicted_Quality_')]
            if len(pred_cols) > 1:
                results_df['Ensemble_Prediction'] = results_df[pred_cols].mean(axis=1)
                results_df['Ensemble_Quality_Class'] = classify_water_quality_batch(results_df['Ensemble_Prediction'])
            
            self.predictions = results_df
            return results_df
        
        return None
    
    def generate_comprehensive_report(self):
        """Generate comprehensive analysis report"""
        print("\n" + "="*80)
        print("STEP 5: COMPREHENSIVE REPORT GENERATION")
        print("="*80)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        os.makedirs("output/comprehensive_analysis", exist_ok=True)
        
        # 1. Model Performance Report
        self._generate_model_performance_report(timestamp)
        
        # 2. Water Quality Classification Report
        self._generate_quality_classification_report(timestamp)
        
        # 3. Predictions Report
        self._generate_predictions_report(timestamp)
        
        # 4. Technical Summary
        self._generate_technical_summary(timestamp)
        
        print(f"\n✓ Comprehensive reports generated in output/comprehensive_analysis/")
    
    def _generate_model_performance_report(self, timestamp):
        """Generate detailed model performance report"""
        report_path = f"output/comprehensive_analysis/model_performance_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("WATER QUALITY ANALYSIS - MODEL PERFORMANCE REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            # Model comparison table
            f.write("MODEL PERFORMANCE COMPARISON\n")
            f.write("-"*80 + "\n")
            f.write(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'MAPE':<10}\n")
            f.write("-"*80 + "\n")
            
            for model_name, results in self.results.items():
                metrics = results['metrics']
                f.write(f"{model_name:<20} "
                       f"{metrics['rmse']:<10.4f} "
                       f"{metrics['mae']:<10.4f} "
                       f"{metrics['r2']:<10.4f} "
                       f"{metrics.get('mape', 0):<10.4f}\n")
            
            # Detailed metrics for each model
            f.write("\n\nDETAILED MODEL METRICS\n")
            f.write("="*80 + "\n")
            
            for model_name, results in self.results.items():
                f.write(f"\n{model_name.upper()}\n")
                f.write("-"*40 + "\n")
                
                metrics = results['metrics']
                f.write(f"Root Mean Square Error (RMSE): {metrics['rmse']:.6f}\n")
                f.write(f"Mean Absolute Error (MAE):    {metrics['mae']:.6f}\n")
                f.write(f"R-squared (R²):               {metrics['r2']:.6f}\n")
                f.write(f"Mean Absolute Percentage Error: {metrics.get('mape', 'N/A')}\n")
                f.write(f"Explained Variance:           {metrics.get('explained_variance', 'N/A')}\n")
                f.write(f"Max Error:                    {metrics.get('max_error', 'N/A')}\n")
                
                # Cross-validation results
                if 'cv_results' in results:
                    cv_results = results['cv_results']
                    f.write(f"\nCross-Validation Results:\n")
                    for metric, values in cv_results.items():
                        if values and isinstance(values, dict):
                            f.write(f"  {metric}: {values['mean']:.4f} (±{values['std']:.4f})\n")
        
        print(f"✓ Model performance report: {report_path}")
    
    def _generate_quality_classification_report(self, timestamp):
        """Generate water quality classification explanation"""
        report_path = f"output/comprehensive_analysis/quality_classification_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("WATER QUALITY CLASSIFICATION SYSTEM\n")
            f.write("="*80 + "\n\n")
            
            f.write("CLASSIFICATION CRITERIA\n")
            f.write("-"*40 + "\n")
            f.write("The water quality prediction system uses a fuzzy logic approach\n")
            f.write("to classify water quality on a scale from 0 to 1, where:\n\n")
            
            for category, description in self.quality_criteria.items():
                f.write(f"{category:<12}: {description}\n")
            
            f.write("\n\nFEATURE PARAMETERS\n")
            f.write("-"*40 + "\n")
            f.write("TP (Total Phosphorus):     Normalized 0-1, indicates nutrient pollution\n")
            f.write("EC (Electrical Conductivity): Normalized 0-1, indicates dissolved salts\n")
            f.write("DO (Dissolved Oxygen):     Normalized 0-1, critical for aquatic life\n")
            f.write("TIT (Titration/Alkalinity): Normalized 0-1, buffering capacity\n")
            
            f.write("\n\nINTERPretation GUIDELINES\n")
            f.write("-"*40 + "\n")
            f.write("• Excellent (≥0.8): Water meets all quality standards\n")
            f.write("• Good (0.6-0.8): Minor quality issues, minimal treatment needed\n")
            f.write("• Fair (0.4-0.6): Moderate quality issues, treatment required\n")
            f.write("• Poor (0.2-0.4): Significant quality problems\n")
            f.write("• Very Poor (<0.2): Severe contamination, major treatment needed\n")
        
        print(f"✓ Quality classification report: {report_path}")
    
    def _generate_predictions_report(self, timestamp):
        """Generate predictions analysis report"""
        if self.predictions is None or self.predictions.empty:
            return
        
        # Save detailed predictions
        pred_path = f"output/comprehensive_analysis/detailed_predictions_{timestamp}.xlsx"
        self.predictions.to_excel(pred_path, index=False)
        
        # Generate summary report
        report_path = f"output/comprehensive_analysis/predictions_summary_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write("WATER QUALITY PREDICTIONS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Total Predictions: {len(self.predictions)}\n")
            f.write(f"Unique Stations: {self.predictions['Station'].nunique()}\n\n")
            
            # Quality distribution
            if 'Ensemble_Quality_Class' in self.predictions.columns:
                quality_dist = self.predictions['Ensemble_Quality_Class'].value_counts()
                f.write("QUALITY DISTRIBUTION\n")
                f.write("-"*40 + "\n")
                for quality, count in quality_dist.items():
                    percentage = (count / len(self.predictions)) * 100
                    f.write(f"{quality:<15}: {count:>3} stations ({percentage:>5.1f}%)\n")
            
            # Station-wise summary
            f.write("\n\nSTATION-WISE SUMMARY\n")
            f.write("-"*40 + "\n")
            station_summary = self.predictions.groupby('Station').agg({
                'Ensemble_Prediction': ['count', 'mean', 'std'],
                'Ensemble_Quality_Class': lambda x: x.mode()[0] if not x.empty else 'Unknown'
            }).round(3)
            
            for station in station_summary.index:
                count = station_summary.loc[station, ('Ensemble_Prediction', 'count')]
                mean_qual = station_summary.loc[station, ('Ensemble_Prediction', 'mean')]
                dominant_class = station_summary.loc[station, ('Ensemble_Quality_Class', '<lambda>')]
                f.write(f"{station:<10}: {count:>2} samples, avg={mean_qual:>6.3f}, class={dominant_class}\n")
        
        print(f"✓ Predictions report: {report_path}")
        print(f"✓ Detailed predictions: {pred_path}")
    
    def _generate_technical_summary(self, timestamp):
        """Generate technical summary"""
        summary_path = f"output/comprehensive_analysis/technical_summary_{timestamp}.json"
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "data_summary": {
                "training_samples": self.processor.train_data.shape[0] if self.processor else 0,
                "test_samples": self.processor.test_data.shape[0] if self.processor else 0,
                "features_engineered": self.processor.X_train_processed.shape[1] if self.processor else 0,
                "original_features": 6
            },
            "models_trained": list(self.models.keys()),
            "model_performance": {},
            "prediction_summary": {}
        }
        
        # Add model performance
        for model_name, results in self.results.items():
            summary["model_performance"][model_name] = {
                "rmse": results['metrics']['rmse'],
                "mae": results['metrics']['mae'],
                "r2": results['metrics']['r2'],
                "training_time": results.get('evaluation_time', 0)
            }
        
        # Add prediction summary
        if self.predictions is not None and not self.predictions.empty:
            summary["prediction_summary"] = {
                "total_predictions": len(self.predictions),
                "quality_distribution": self.predictions['Ensemble_Quality_Class'].value_counts().to_dict() if 'Ensemble_Quality_Class' in self.predictions.columns else {}
            }
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"✓ Technical summary: {summary_path}")
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("COMPREHENSIVE WATER QUALITY ANALYSIS")
        print("="*80)
        print("Fixing all issues and providing detailed performance analysis")
        print("="*80)
        
        try:
            # Step 1: Load and process data
            X_train, X_val, X_test, y_train, y_val, y_test = self.load_and_process_data()
            
            # Step 2: Train models
            self.train_models(X_train, y_train, X_val, y_val)
            
            # Step 3: Evaluate models
            self.evaluate_models(X_test, y_test)
            
            # Step 4: Make predictions
            self.make_predictions()
            
            # Step 5: Generate comprehensive reports
            self.generate_comprehensive_report()
            
            # Display water quality classification explanation
            self._display_quality_explanation()
            
            print("\n" + "="*80)
            print("ANALYSIS COMPLETED SUCCESSFULLY!")
            print("="*80)
            print("All issues have been fixed and comprehensive reports generated.")
            print("Check the output/comprehensive_analysis/ directory for detailed results.")
            
        except Exception as e:
            print(f"\n✗ Analysis failed: {e}")
            raise
    
    def _display_quality_explanation(self):
        """Display water quality classification explanation"""
        print("\n" + "="*80)
        print("WATER QUALITY CLASSIFICATION EXPLANATION")
        print("="*80)
        
        print("\nThe water quality scoring system uses the following criteria:")
        for category, description in self.quality_criteria.items():
            print(f"• {category}: {description}")
        
        print(f"\nFeature Parameters:")
        print(f"• TP (Total Phosphorus): Indicates nutrient pollution levels")
        print(f"• EC (Electrical Conductivity): Measures dissolved salts and minerals")
        print(f"• DO (Dissolved Oxygen): Critical for aquatic ecosystem health")
        print(f"• TIT (Titration/Alkalinity): Water's buffering capacity against pH changes")

if __name__ == "__main__":
    # Run comprehensive analysis
    analyzer = ComprehensiveWaterQualityAnalysis()
    analyzer.run_complete_analysis() 