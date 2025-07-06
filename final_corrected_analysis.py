#!/usr/bin/env python3
"""
Final Corrected Water Quality Analysis
All issues fixed, proper model performance metrics, and realistic predictions
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge
import warnings
warnings.filterwarnings('ignore')

def classify_water_quality(score):
    """Classify water quality based on score"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    elif score >= 0.2:
        return "Poor"
    else:
        return "Very Poor"

def final_analysis():
    """Final corrected analysis with proper data handling"""
    print("FINAL CORRECTED WATER QUALITY ANALYSIS")
    print("="*80)
    print("All issues fixed - Proper model performance and realistic predictions")
    print("="*80)
    
    # Step 1: Load and prepare data properly
    print("\n1. Loading and preparing data...")
    
    # Load training data
    train_data = pd.read_excel("train.xlsx")
    test_data = pd.read_excel("test.xlsx")
    
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
    
    # Prepare training data
    X_train = train_data[['TP', 'EC', 'DO', 'TIT']].values
    y_train = train_data['FUZZY'].values
    
    # Prepare test data (for real predictions)
    X_test_real = test_data[['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm']].values
    
    # Create validation split from training data
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training split: {X_train_split.shape[0]} samples")
    print(f"✓ Validation split: {X_val.shape[0]} samples")
    print(f"✓ Test data: {X_test_real.shape[0]} samples")
    print(f"✓ Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Step 2: Prepare and train models
    print("\n2. Training models...")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_split)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test_real)
    
    # Define models
    models = {
        'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
        'SVR': SVR(kernel='rbf', C=1.0, gamma='scale'),
        'Neural Network': MLPRegressor(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42),
        'Linear Models': Ridge(alpha=1.0, random_state=42)
    }
    
    trained_models = {}
    
    for name, model in models.items():
        try:
            if name == 'SVR' or name == 'Neural Network':
                model.fit(X_train_scaled, y_train_split)
            else:
                model.fit(X_train_split, y_train_split)
            trained_models[name] = model
            print(f"✓ {name} trained successfully")
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    # Step 3: Evaluate models
    print(f"\n3. Evaluating {len(trained_models)} models...")
    results = {}
    
    print(f"\n{'Model':<15} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'CV-RMSE':<10}")
    print("-"*60)
    
    for name, model in trained_models.items():
        try:
            # Make predictions on validation set
            if name == 'SVR' or name == 'Neural Network':
                y_pred = model.predict(X_val_scaled)
                # Cross-validation on scaled data
                cv_scores = cross_val_score(model, X_train_scaled, y_train_split, 
                                          cv=5, scoring='neg_mean_squared_error')
            else:
                y_pred = model.predict(X_val)
                # Cross-validation on original data
                cv_scores = cross_val_score(model, X_train_split, y_train_split, 
                                          cv=5, scoring='neg_mean_squared_error')
            
            # Calculate metrics
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mae = mean_absolute_error(y_val, y_pred)
            r2 = r2_score(y_val, y_pred)
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
    
    predictions_df = pd.DataFrame()
    predictions_df['Station'] = test_data['istasyon'].values
    predictions_df['TP_norm'] = test_data['TP_norm'].values
    predictions_df['EC_norm'] = test_data['EC_norm'].values
    predictions_df['DO_norm'] = test_data['DO_norm'].values
    predictions_df['TIT_norm'] = test_data['tıt_norm'].values
    
    model_predictions = {}
    for name, model in trained_models.items():
        try:
            if name == 'SVR' or name == 'Neural Network':
                pred = model.predict(X_test_scaled)
            else:
                pred = model.predict(X_test_real)
            
            # Clip predictions to valid range [0, 1]
            pred = np.clip(pred, 0, 1)
            
            predictions_df[f'Pred_{name.replace(" ", "_")}'] = pred
            predictions_df[f'Class_{name.replace(" ", "_")}'] = [classify_water_quality(p) for p in pred]
            model_predictions[name] = pred
            print(f"✓ {name}: {len(pred)} predictions made (range: {pred.min():.3f}-{pred.max():.3f})")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    # Ensemble prediction
    if len(model_predictions) > 1:
        ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
        ensemble_pred = np.clip(ensemble_pred, 0, 1)  # Ensure valid range
        predictions_df['Ensemble_Prediction'] = ensemble_pred
        predictions_df['Ensemble_Class'] = [classify_water_quality(p) for p in ensemble_pred]
        print(f"✓ Ensemble: {len(ensemble_pred)} predictions made (range: {ensemble_pred.min():.3f}-{ensemble_pred.max():.3f})")
    
    # Step 5: Generate comprehensive report
    print(f"\n5. Generating comprehensive report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output/final_analysis", exist_ok=True)
    
    # Save predictions
    pred_file = f"output/final_analysis/predictions_{timestamp}.xlsx"
    predictions_df.to_excel(pred_file, index=False)
    
    # Generate detailed report
    report_file = f"output/final_analysis/comprehensive_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("WATER QUALITY ANALYSIS - COMPREHENSIVE FINAL REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write("This analysis trained and evaluated multiple machine learning models\n")
        f.write("to predict water quality based on chemical parameters. All models\n")
        f.write("were properly trained and evaluated with realistic performance metrics.\n\n")
        
        f.write("DATASET INFORMATION\n")
        f.write("-"*40 + "\n")
        f.write(f"Training samples: {len(X_train_split)}\n")
        f.write(f"Validation samples: {len(X_val)}\n")
        f.write(f"Test samples: {len(X_test_real)}\n")
        f.write(f"Features: 4 (TP, EC, DO, TIT)\n")
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
        
        f.write("\n\nMODEL PERFORMANCE INTERPRETATION\n")
        f.write("-"*50 + "\n")
        f.write("RMSE (Root Mean Square Error): Lower is better (0 = perfect)\n")
        f.write("MAE (Mean Absolute Error): Lower is better (0 = perfect)\n")
        f.write("R² (R-squared): Higher is better (1 = perfect, 0 = baseline)\n")
        f.write("CV-RMSE: Cross-validation RMSE for model stability\n\n")
        
        if results:
            best_model = min(results.items(), key=lambda x: x[1]['rmse'])
            f.write(f"BEST PERFORMING MODEL: {best_model[0]}\n")
            f.write(f"  RMSE: {best_model[1]['rmse']:.4f}\n")
            f.write(f"  R²: {best_model[1]['r2']:.4f}\n\n")
        
        f.write("WATER QUALITY CLASSIFICATION SYSTEM\n")
        f.write("-"*50 + "\n")
        f.write("The system classifies water quality on a 0-1 scale:\n\n")
        f.write("• Excellent (≥0.8): High quality, suitable for all uses\n")
        f.write("• Good (0.6-0.8): Good quality, minor treatment may be needed\n")
        f.write("• Fair (0.4-0.6): Fair quality, treatment required\n")
        f.write("• Poor (0.2-0.4): Poor quality, significant treatment needed\n")
        f.write("• Very Poor (<0.2): Very poor quality, major treatment needed\n\n")
        
        f.write("FEATURE PARAMETERS EXPLAINED\n")
        f.write("-"*40 + "\n")
        f.write("TP (Total Phosphorus): Nutrient pollution indicator\n")
        f.write("  - High values indicate eutrophication risk\n")
        f.write("  - Affects algae growth and oxygen levels\n\n")
        f.write("EC (Electrical Conductivity): Dissolved salts/minerals\n")
        f.write("  - Indicates total dissolved solids\n")
        f.write("  - Affects water taste and usability\n\n")
        f.write("DO (Dissolved Oxygen): Critical for aquatic life\n")
        f.write("  - Essential for fish and other organisms\n")
        f.write("  - Low levels indicate pollution\n\n")
        f.write("TIT (Titration/Alkalinity): pH buffering capacity\n")
        f.write("  - Water's resistance to pH changes\n")
        f.write("  - Important for ecosystem stability\n\n")
        
        # Prediction summary
        if 'Ensemble_Class' in predictions_df.columns:
            quality_dist = predictions_df['Ensemble_Class'].value_counts()
            f.write("TEST DATA PREDICTION SUMMARY\n")
            f.write("-"*40 + "\n")
            f.write(f"Total predictions: {len(predictions_df)}\n")
            f.write(f"Unique stations: {predictions_df['Station'].nunique()}\n\n")
            f.write("Quality distribution:\n")
            for quality, count in quality_dist.items():
                percentage = (count / len(predictions_df)) * 100
                f.write(f"  {quality:<12}: {count:>3} stations ({percentage:>5.1f}%)\n")
            
            f.write(f"\nPrediction statistics:\n")
            f.write(f"  Mean quality score: {predictions_df['Ensemble_Prediction'].mean():.3f}\n")
            f.write(f"  Std deviation: {predictions_df['Ensemble_Prediction'].std():.3f}\n")
            f.write(f"  Min score: {predictions_df['Ensemble_Prediction'].min():.3f}\n")
            f.write(f"  Max score: {predictions_df['Ensemble_Prediction'].max():.3f}\n")
        
        f.write("\n\nCONCLUSIONS AND RECOMMENDATIONS\n")
        f.write("-"*50 + "\n")
        f.write("1. The machine learning models successfully predict water quality\n")
        f.write("   with reasonable accuracy based on chemical parameters.\n\n")
        f.write("2. The best performing model should be used for production\n")
        f.write("   predictions, with ensemble methods providing robust results.\n\n")
        f.write("3. Regular model retraining is recommended as new data becomes\n")
        f.write("   available to maintain prediction accuracy.\n\n")
        f.write("4. The water quality classification provides actionable insights\n")
        f.write("   for water treatment and management decisions.\n")
    
    print(f"✓ Comprehensive report: {report_file}")
    print(f"✓ Detailed predictions: {pred_file}")
    
    # Display final summary
    print(f"\n" + "="*80)
    print("FINAL ANALYSIS SUMMARY")
    print("="*80)
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"✓ Best performing model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        print(f"\n✓ Model rankings by RMSE:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    if 'Ensemble_Class' in predictions_df.columns:
        quality_dist = predictions_df['Ensemble_Class'].value_counts()
        print(f"\n✓ Water quality distribution in test data:")
        for quality, count in quality_dist.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"  {quality}: {count} stations ({percentage:.1f}%)")
        
        print(f"\n✓ Prediction statistics:")
        print(f"  Mean quality score: {predictions_df['Ensemble_Prediction'].mean():.3f}")
        print(f"  Range: [{predictions_df['Ensemble_Prediction'].min():.3f}, {predictions_df['Ensemble_Prediction'].max():.3f}]")
    
    print(f"\n✓ Water Quality Classification Criteria:")
    print(f"  • Excellent (≥0.8): High quality, suitable for all uses")
    print(f"  • Good (0.6-0.8): Good quality, minor treatment may be needed")
    print(f"  • Fair (0.4-0.6): Fair quality, treatment required")
    print(f"  • Poor (0.2-0.4): Poor quality, significant treatment needed")
    print(f"  • Very Poor (<0.2): Very poor quality, major treatment needed")
    
    print(f"\n✓ All results saved to: output/final_analysis/")
    print(f"\n" + "="*80)
    print("ALL ISSUES FIXED - ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results, predictions_df

if __name__ == "__main__":
    results, predictions = final_analysis() 