#!/usr/bin/env python3
"""
Simple Model Evaluation Script
Provides proper model performance metrics without complex evaluation issues
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

from data_processor import WaterQualityDataProcessor
from models.random_forest_model import RandomForestModel
from models.svr_model import SVRModel
from models.neural_network_model import NeuralNetworkModel
from models.linear_models import LinearModels
from utils import classify_water_quality_batch

def simple_model_evaluation():
    """Simple but comprehensive model evaluation"""
    print("SIMPLE MODEL EVALUATION WITH PERFORMANCE METRICS")
    print("="*80)
    
    # Step 1: Load and process data
    print("\n1. Loading and processing data...")
    processor = WaterQualityDataProcessor(random_seed=42, log_level="ERROR")
    processor.load_data("train.xlsx", "test.xlsx")
    processor.preprocess_data()
    
    X_train, X_val, X_test, y_train, y_val, y_test = processor.get_data_splits()
    print(f"✓ Data loaded: {len(X_train)} train, {len(X_test)} test, {X_train.shape[1]} features")
    
    # Step 2: Train models
    print("\n2. Training models...")
    models = {}
    
    # Random Forest
    try:
        rf = RandomForestModel(n_estimators=100, max_depth=10)
        rf.fit(X_train, y_train)
        models['Random Forest'] = rf
        print("✓ Random Forest trained")
    except Exception as e:
        print(f"✗ Random Forest failed: {e}")
    
    # SVR
    try:
        svr = SVRModel(kernel='rbf', C=1.0)
        svr.fit(X_train, y_train)
        models['SVR'] = svr
        print("✓ SVR trained")
    except Exception as e:
        print(f"✗ SVR failed: {e}")
    
    # Neural Network
    try:
        nn = NeuralNetworkModel(hidden_layer_sizes=(100, 50), max_iter=500)
        nn.fit(X_train, y_train)
        models['Neural Network'] = nn
        print("✓ Neural Network trained")
    except Exception as e:
        print(f"✗ Neural Network failed: {e}")
    
    # Linear Models
    try:
        lm = LinearModels(model_type='ridge', alpha=1.0)
        lm.fit(X_train, y_train)
        models['Linear Models'] = lm
        print("✓ Linear Models trained")
    except Exception as e:
        print(f"✗ Linear Models failed: {e}")
    
    # Step 3: Evaluate models
    print(f"\n3. Evaluating {len(models)} models...")
    results = {}
    
    print(f"\n{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'CV-RMSE':<10}")
    print("-"*60)
    
    for name, model in models.items():
        try:
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Cross-validation RMSE
            cv_scores = cross_val_score(model.model, X_train, y_train, 
                                      cv=5, scoring='neg_mean_squared_error')
            cv_rmse = np.sqrt(-cv_scores.mean())
            
            results[name] = {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred
            }
            
            print(f"{name:<15} {rmse:<8.4f} {mae:<8.4f} {r2:<8.4f} {cv_rmse:<10.4f}")
            
        except Exception as e:
            print(f"{name:<15} ERROR: {e}")
    
    # Step 4: Make predictions on test data
    print(f"\n4. Making predictions on test data...")
    X_test_processed = processor.get_test_data_processed()
    test_data = pd.read_excel("test.xlsx")
    
    predictions_df = pd.DataFrame()
    predictions_df['Station'] = test_data['istasyon'].values
    predictions_df['TP_norm'] = test_data['TP_norm'].values
    predictions_df['EC_norm'] = test_data['EC_norm'].values
    predictions_df['DO_norm'] = test_data['DO_norm'].values
    predictions_df['TIT_norm'] = test_data['tıt_norm'].values
    
    model_predictions = {}
    for name, model in models.items():
        try:
            pred = model.predict(X_test_processed)
            predictions_df[f'Pred_{name.replace(" ", "_")}'] = pred
            predictions_df[f'Class_{name.replace(" ", "_")}'] = classify_water_quality_batch(pred)
            model_predictions[name] = pred
            print(f"✓ {name}: {len(pred)} predictions made")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    # Ensemble prediction
    if len(model_predictions) > 1:
        ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
        predictions_df['Ensemble_Prediction'] = ensemble_pred
        predictions_df['Ensemble_Class'] = classify_water_quality_batch(ensemble_pred)
    
    # Step 5: Generate reports
    print(f"\n5. Generating reports...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output/simple_evaluation", exist_ok=True)
    
    # Save predictions
    pred_file = f"output/simple_evaluation/predictions_{timestamp}.xlsx"
    predictions_df.to_excel(pred_file, index=False)
    
    # Generate performance report
    report_file = f"output/simple_evaluation/performance_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("WATER QUALITY ANALYSIS - MODEL PERFORMANCE REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("DATASET SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Test samples: {len(X_test)}\n")
        f.write(f"Features: {X_train.shape[1]}\n")
        f.write(f"Target range: [{y_train.min():.3f}, {y_train.max():.3f}]\n\n")
        
        f.write("MODEL PERFORMANCE COMPARISON\n")
        f.write("-"*80 + "\n")
        f.write(f"{'Model':<20} {'RMSE':<10} {'MAE':<10} {'R²':<10} {'CV-RMSE':<10}\n")
        f.write("-"*80 + "\n")
        
        for name, metrics in results.items():
            f.write(f"{name:<20} "
                   f"{metrics['rmse']:<10.4f} "
                   f"{metrics['mae']:<10.4f} "
                   f"{metrics['r2']:<10.4f} "
                   f"{metrics['cv_rmse']:<10.4f}\n")
        
        f.write("\n\nWATER QUALITY CLASSIFICATION CRITERIA\n")
        f.write("-"*50 + "\n")
        f.write("Excellent (≥0.8): High quality, suitable for all uses\n")
        f.write("Good (0.6-0.8):   Good quality, minor treatment may be needed\n")
        f.write("Fair (0.4-0.6):   Fair quality, treatment required\n")
        f.write("Poor (0.2-0.4):   Poor quality, significant treatment needed\n")
        f.write("Very Poor (<0.2): Very poor quality, not suitable for most uses\n\n")
        
        f.write("FEATURE PARAMETERS\n")
        f.write("-"*30 + "\n")
        f.write("TP:  Total Phosphorus (nutrient pollution indicator)\n")
        f.write("EC:  Electrical Conductivity (dissolved salts/minerals)\n")
        f.write("DO:  Dissolved Oxygen (critical for aquatic life)\n")
        f.write("TIT: Titration/Alkalinity (pH buffering capacity)\n\n")
        
        # Prediction summary
        if 'Ensemble_Class' in predictions_df.columns:
            quality_dist = predictions_df['Ensemble_Class'].value_counts()
            f.write("PREDICTION SUMMARY\n")
            f.write("-"*30 + "\n")
            f.write(f"Total predictions: {len(predictions_df)}\n")
            f.write(f"Unique stations: {predictions_df['Station'].nunique()}\n\n")
            f.write("Quality distribution:\n")
            for quality, count in quality_dist.items():
                percentage = (count / len(predictions_df)) * 100
                f.write(f"  {quality:<12}: {count:>3} ({percentage:>5.1f}%)\n")
    
    print(f"✓ Performance report: {report_file}")
    print(f"✓ Predictions file: {pred_file}")
    
    # Display summary
    print(f"\n" + "="*80)
    print("ANALYSIS SUMMARY")
    print("="*80)
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"Best performing model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        print(f"\nModel rankings by RMSE:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    if 'Ensemble_Class' in predictions_df.columns:
        quality_dist = predictions_df['Ensemble_Class'].value_counts()
        print(f"\nWater quality distribution in test data:")
        for quality, count in quality_dist.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"  {quality}: {count} stations ({percentage:.1f}%)")
    
    print(f"\nAll results saved to: output/simple_evaluation/")
    return results, predictions_df

if __name__ == "__main__":
    results, predictions = simple_model_evaluation() 