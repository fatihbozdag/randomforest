#!/usr/bin/env python3
"""
Script to load trained models and make predictions on test data
"""

import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from data_processor import WaterQualityDataProcessor
from utils import classify_water_quality_batch

def load_latest_models():
    """Load the most recent trained models"""
    models_dir = "output/models"
    models = {}
    
    # Find the latest model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
    
    # Group by model type and get the latest
    model_types = ['random_forest', 'neural_network', 'linear_models']
    
    for model_type in model_types:
        type_files = [f for f in model_files if f.startswith(model_type)]
        if type_files:
            latest_file = max(type_files, key=lambda x: os.path.getctime(os.path.join(models_dir, x)))
            model_path = os.path.join(models_dir, latest_file)
            try:
                models[model_type] = joblib.load(model_path)
                print(f"Loaded {model_type} from {latest_file}")
            except Exception as e:
                print(f"Error loading {model_type}: {e}")
    
    return models

def make_predictions():
    """Make predictions on test data using trained models"""
    print("Starting prediction process...")
    
    # Initialize data processor
    processor = WaterQualityDataProcessor(random_seed=42)
    
    # Load and process data
    print("Loading and processing data...")
    processor.load_data("train.xlsx", "test.xlsx")
    processor.preprocess_data()
    
    # Get processed test data
    X_test_processed = processor.get_test_data_processed()
    
    # Load trained models
    models = load_latest_models()
    
    if not models:
        print("No trained models found!")
        return
    
    # Make predictions with each model
    predictions = {}
    
    for model_name, model in models.items():
        try:
            print(f"Making predictions with {model_name}...")
            pred = model.predict(X_test_processed)
            predictions[model_name] = pred
            print(f"✓ {model_name}: {len(pred)} predictions made")
        except Exception as e:
            print(f"Error making predictions with {model_name}: {e}")
    
    # Create results DataFrame
    if predictions:
        # Load original test data to get station names
        test_data = pd.read_excel("test.xlsx")
        
        results_df = pd.DataFrame()
        results_df['Station'] = test_data['istasyon'].values
        results_df['TP_norm'] = test_data['TP_norm'].values
        results_df['EC_norm'] = test_data['EC_norm'].values
        results_df['DO_norm'] = test_data['DO_norm'].values
        results_df['TIT_norm'] = test_data['tıt_norm'].values
        
        # Add predictions from each model
        for model_name, pred in predictions.items():
            results_df[f'Predicted_Quality_{model_name}'] = pred
            results_df[f'Quality_Class_{model_name}'] = classify_water_quality_batch(pred)
        
        # Calculate ensemble prediction (average of all models)
        pred_cols = [col for col in results_df.columns if col.startswith('Predicted_Quality_')]
        if len(pred_cols) > 1:
            results_df['Ensemble_Prediction'] = results_df[pred_cols].mean(axis=1)
            results_df['Ensemble_Quality_Class'] = classify_water_quality_batch(results_df['Ensemble_Prediction'])
        
        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"output/predictions/test_predictions_{timestamp}.xlsx"
        
        os.makedirs("output/predictions", exist_ok=True)
        results_df.to_excel(output_file, index=False)
        
        print(f"\n✓ Predictions saved to: {output_file}")
        print(f"✓ Total predictions: {len(results_df)}")
        print(f"✓ Models used: {list(predictions.keys())}")
        
        # Display summary
        print("\n" + "="*60)
        print("PREDICTION SUMMARY")
        print("="*60)
        print(f"{'Station':<12} {'TP':<6} {'EC':<6} {'DO':<6} {'TIT':<6} {'Predicted':<10} {'Quality':<15}")
        print("-"*60)
        
        for idx, row in results_df.head(10).iterrows():
            if 'Ensemble_Prediction' in results_df.columns:
                pred_val = row['Ensemble_Prediction']
                quality = row['Ensemble_Quality_Class']
            else:
                pred_col = pred_cols[0]
                pred_val = row[pred_col]
                quality = row[pred_col.replace('Predicted_Quality_', 'Quality_Class_')]
            
            print(f"{row['Station']:<12} {row['TP_norm']:<6.3f} {row['EC_norm']:<6.3f} "
                  f"{row['DO_norm']:<6.3f} {row['TIT_norm']:<6.3f} {pred_val:<10.3f} {quality:<15}")
        
        if len(results_df) > 10:
            print(f"... and {len(results_df) - 10} more stations")
        
        return results_df
    
    else:
        print("No predictions could be made!")
        return None

if __name__ == "__main__":
    results = make_predictions()
    if results is not None:
        print("\n✓ Prediction process completed successfully!")
    else:
        print("\n✗ Prediction process failed!") 