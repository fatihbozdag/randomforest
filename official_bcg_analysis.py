#!/usr/bin/env python3
"""
Official BCG Water Quality Analysis
Using the official BCG (Biological Classification Group) thresholds
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

def classify_water_quality_bcg(score):
    """
    Classify water quality based on official BCG thresholds
    BCG = Biological Classification Group
    """
    if score >= 0.80:
        return "High (BCG1)"
    elif score >= 0.60:
        return "Good (BCG2)"
    elif score >= 0.40:
        return "Moderate (BCG3)"
    elif score >= 0.20:
        return "Poor (BCG4)"
    else:
        return "Bad (BCG5)"

def official_bcg_analysis():
    """Official BCG water quality analysis with correct thresholds"""
    print("OFFICIAL BCG WATER QUALITY ANALYSIS")
    print("="*80)
    print("Using Official BCG (Biological Classification Group) Thresholds")
    print("="*80)
    
    # Step 1: Load and prepare data
    print("\n1. Loading and preparing data...")
    
    train_data = pd.read_excel("train.xlsx")
    test_data = pd.read_excel("test.xlsx")
    
    # Harmonise column names for TIT / tıt inconsistencies
    if 'TIT' not in train_data.columns and 'tıt' in train_data.columns:
        train_data['TIT'] = train_data['tıt']
    if 'TIT' not in test_data.columns and 'tıt' in test_data.columns:
        test_data['TIT'] = test_data['tıt']
    
    print(f"✓ Training data: {train_data.shape}")
    print(f"✓ Test data: {test_data.shape}")
    
    # Prepare training data
    feature_cols_raw = ['TP', 'EC', 'DO', 'TIT']
    feature_cols_norm = ['TP_norm', 'EC_norm', 'DO_norm', 'tıt_norm']
    
    if all(col in train_data.columns for col in feature_cols_raw):
        X_train = train_data[feature_cols_raw].values
    else:
        X_train = train_data[feature_cols_norm].values
    y_train = train_data['FUZZY' if 'FUZZY' in train_data.columns else 'fuzzy'].values
    
    # Prepare test data
    if all(col in test_data.columns for col in feature_cols_raw):
        X_test_real = test_data[feature_cols_raw].values
    else:
        X_test_real = test_data[feature_cols_norm].values
    
    # Create validation split
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"✓ Training split: {X_train_split.shape[0]} samples")
    print(f"✓ Validation split: {X_val.shape[0]} samples")
    print(f"✓ Test data: {X_test_real.shape[0]} samples")
    print(f"✓ Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
    
    # Step 2: Train models
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
            # Train **all** models on the **scaled** feature matrix for consistency
            model.fit(X_train_scaled, y_train_split)
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
            # Make predictions on validation set using the **scaled** matrix
            y_pred = model.predict(X_val_scaled)
            cv_scores = cross_val_score(model, X_train_scaled, y_train_split, 
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
    predictions_df['Station'] = test_data['istasyon' if 'istasyon' in test_data.columns else 'station'].values
    # Add feature columns for reference
    predictions_df['TP'] = test_data['TP'] if 'TP' in test_data.columns else test_data.get('TP_norm')
    predictions_df['EC'] = test_data['EC'] if 'EC' in test_data.columns else test_data.get('EC_norm')
    predictions_df['DO'] = test_data['DO'] if 'DO' in test_data.columns else test_data.get('DO_norm')
    predictions_df['TIT'] = test_data['TIT'] if 'TIT' in test_data.columns else test_data.get('tıt_norm')
    
    model_predictions = {}
    for name, model in trained_models.items():
        try:
            # Use **scaled** test features for prediction across all models
            pred = model.predict(X_test_scaled)
            
            # Clip predictions to valid range [0, 1]
            pred = np.clip(pred, 0, 1)
            
            predictions_df[f'Pred_{name.replace(" ", "_")}'] = pred
            predictions_df[f'BCG_{name.replace(" ", "_")}'] = [classify_water_quality_bcg(p) for p in pred]
            model_predictions[name] = pred
            print(f"✓ {name}: {len(pred)} predictions made (range: {pred.min():.3f}-{pred.max():.3f})")
        except Exception as e:
            print(f"✗ {name}: {e}")
    
    # Ensemble prediction
    if len(model_predictions) > 1:
        ensemble_pred = np.mean(list(model_predictions.values()), axis=0)
        ensemble_pred = np.clip(ensemble_pred, 0, 1)
        predictions_df['Ensemble_Prediction'] = ensemble_pred
        predictions_df['Ensemble_BCG_Class'] = [classify_water_quality_bcg(p) for p in ensemble_pred]
        print(f"✓ Ensemble: {len(ensemble_pred)} predictions made (range: {ensemble_pred.min():.3f}-{ensemble_pred.max():.3f})")
    
    # Step 5: Generate comprehensive BCG report
    print(f"\n5. Generating comprehensive BCG report...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("output/bcg_analysis", exist_ok=True)
    
    # Save predictions
    pred_file = f"output/bcg_analysis/bcg_predictions_{timestamp}.xlsx"
    predictions_df.to_excel(pred_file, index=False)
    
    # Generate detailed report
    report_file = f"output/bcg_analysis/bcg_comprehensive_report_{timestamp}.txt"
    with open(report_file, 'w') as f:
        f.write("WATER QUALITY ANALYSIS - OFFICIAL BCG CLASSIFICATION REPORT\n")
        f.write("="*80 + "\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write("EXECUTIVE SUMMARY\n")
        f.write("-"*40 + "\n")
        f.write("This analysis uses the official BCG (Biological Classification Group)\n")
        f.write("thresholds for water quality classification. Multiple machine learning\n")
        f.write("models were trained and evaluated to predict water quality scores.\n\n")
        
        f.write("OFFICIAL BCG CLASSIFICATION THRESHOLDS\n")
        f.write("-"*50 + "\n")
        f.write("BCG1 - High (0.80-1.00):     Excellent water quality\n")
        f.write("BCG2 - Good (0.60-0.79):     Good water quality\n")
        f.write("BCG3 - Moderate (0.40-0.59): Moderate water quality\n")
        f.write("BCG4 - Poor (0.20-0.39):     Poor water quality\n")
        f.write("BCG5 - Bad (0.00-0.19):      Bad water quality\n\n")
        
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
        
        if results:
            best_model = min(results.items(), key=lambda x: x[1]['rmse'])
            f.write(f"\nBEST PERFORMING MODEL: {best_model[0]}\n")
            f.write(f"  RMSE: {best_model[1]['rmse']:.4f}\n")
            f.write(f"  R²: {best_model[1]['r2']:.4f}\n\n")
        
        f.write("CHEMICAL PARAMETERS EXPLANATION\n")
        f.write("-"*45 + "\n")
        f.write("TP (Total Phosphorus): Nutrient pollution indicator\n")
        f.write("  - High values indicate eutrophication risk\n")
        f.write("  - Affects algae growth and oxygen depletion\n\n")
        f.write("EC (Electrical Conductivity): Dissolved salts/minerals\n")
        f.write("  - Indicates total dissolved solids concentration\n")
        f.write("  - Affects water taste and agricultural use\n\n")
        f.write("DO (Dissolved Oxygen): Critical for aquatic life\n")
        f.write("  - Essential for fish and aquatic organisms\n")
        f.write("  - Low levels indicate organic pollution\n\n")
        f.write("TIT (Titration/Alkalinity): pH buffering capacity\n")
        f.write("  - Water's resistance to pH changes\n")
        f.write("  - Important for ecosystem stability\n\n")
        
        # BCG Prediction summary
        if 'Ensemble_BCG_Class' in predictions_df.columns:
            bcg_dist = predictions_df['Ensemble_BCG_Class'].value_counts()
            f.write("TEST DATA BCG CLASSIFICATION SUMMARY\n")
            f.write("-"*45 + "\n")
            f.write(f"Total predictions: {len(predictions_df)}\n")
            f.write(f"Unique stations: {predictions_df['Station'].nunique()}\n\n")
            f.write("BCG distribution:\n")
            for bcg_class, count in bcg_dist.items():
                percentage = (count / len(predictions_df)) * 100
                f.write(f"  {bcg_class:<18}: {count:>3} stations ({percentage:>5.1f}%)\n")
            
            f.write(f"\nPrediction statistics:\n")
            f.write(f"  Mean quality score: {predictions_df['Ensemble_Prediction'].mean():.3f}\n")
            f.write(f"  Std deviation: {predictions_df['Ensemble_Prediction'].std():.3f}\n")
            f.write(f"  Min score: {predictions_df['Ensemble_Prediction'].min():.3f}\n")
            f.write(f"  Max score: {predictions_df['Ensemble_Prediction'].max():.3f}\n")
        
        f.write("\n\nCONCLUSIONS AND MANAGEMENT RECOMMENDATIONS\n")
        f.write("-"*55 + "\n")
        f.write("1. Water quality assessment based on official BCG thresholds\n")
        f.write("   provides standardized classification for management decisions.\n\n")
        f.write("2. Stations classified as BCG4 (Poor) or BCG5 (Bad) require\n")
        f.write("   immediate attention and remediation measures.\n\n")
        f.write("3. Regular monitoring and model updates recommended to track\n")
        f.write("   water quality trends and intervention effectiveness.\n\n")
        f.write("4. Chemical parameter analysis helps identify specific\n")
        f.write("   pollution sources and appropriate treatment strategies.\n")
    
    print(f"✓ BCG comprehensive report: {report_file}")
    print(f"✓ BCG predictions file: {pred_file}")
    
    # Display final BCG summary
    print(f"\n" + "="*80)
    print("OFFICIAL BCG ANALYSIS SUMMARY")
    print("="*80)
    
    if results:
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"✓ Best performing model: {best_model[0]} (RMSE: {best_model[1]['rmse']:.4f})")
        
        print(f"\n✓ Model rankings by RMSE:")
        sorted_models = sorted(results.items(), key=lambda x: x[1]['rmse'])
        for i, (name, metrics) in enumerate(sorted_models, 1):
            print(f"  {i}. {name}: RMSE={metrics['rmse']:.4f}, R²={metrics['r2']:.4f}")
    
    if 'Ensemble_BCG_Class' in predictions_df.columns:
        bcg_dist = predictions_df['Ensemble_BCG_Class'].value_counts()
        print(f"\n✓ BCG water quality distribution in test data:")
        for bcg_class, count in bcg_dist.items():
            percentage = (count / len(predictions_df)) * 100
            print(f"  {bcg_class}: {count} stations ({percentage:.1f}%)")
        
        print(f"\n✓ Prediction statistics:")
        print(f"  Mean quality score: {predictions_df['Ensemble_Prediction'].mean():.3f}")
        print(f"  Range: [{predictions_df['Ensemble_Prediction'].min():.3f}, {predictions_df['Ensemble_Prediction'].max():.3f}]")
    
    print(f"\n✓ Official BCG Classification Thresholds:")
    print(f"  • BCG1 - High (0.80-1.00): Excellent water quality")
    print(f"  • BCG2 - Good (0.60-0.79): Good water quality")
    print(f"  • BCG3 - Moderate (0.40-0.59): Moderate water quality")
    print(f"  • BCG4 - Poor (0.20-0.39): Poor water quality")
    print(f"  • BCG5 - Bad (0.00-0.19): Bad water quality")
    
    print(f"\n✓ All BCG results saved to: output/bcg_analysis/")
    print(f"\n" + "="*80)
    print("OFFICIAL BCG ANALYSIS COMPLETED SUCCESSFULLY!")
    print("="*80)
    
    return results, predictions_df

if __name__ == "__main__":
    results, predictions = official_bcg_analysis() 