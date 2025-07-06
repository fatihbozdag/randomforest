#!/usr/bin/env python3
"""
Script to view and analyze the predictions from the Excel file
"""

import pandas as pd
import numpy as np
import os

def view_predictions():
    """View the latest predictions"""
    
    # Find the latest predictions file
    predictions_dir = "output/predictions"
    if not os.path.exists(predictions_dir):
        print("No predictions directory found!")
        return
    
    pred_files = [f for f in os.listdir(predictions_dir) if f.endswith('.xlsx')]
    if not pred_files:
        print("No prediction files found!")
        return
    
    # Get the latest file
    latest_file = max(pred_files, key=lambda x: os.path.getctime(os.path.join(predictions_dir, x)))
    file_path = os.path.join(predictions_dir, latest_file)
    
    print(f"Reading predictions from: {latest_file}")
    print("="*80)
    
    # Read the Excel file
    df = pd.read_excel(file_path)
    
    print(f"Total predictions: {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print("\n" + "="*80)
    print("WATER QUALITY PREDICTIONS")
    print("="*80)
    
    # Display all predictions in a nice format
    print(f"{'Station':<10} {'TP':<6} {'EC':<6} {'DO':<6} {'TIT':<6} {'RF':<8} {'NN':<8} {'LM':<8} {'Ensemble':<10} {'Quality':<12}")
    print("-"*80)
    
    for idx, row in df.iterrows():
        station = row['Station']
        tp = row['TP_norm']
        ec = row['EC_norm']
        do = row['DO_norm']
        tit = row['TIT_norm']
        
        # Model predictions
        rf_pred = row['Predicted_Quality_random_forest']
        nn_pred = row['Predicted_Quality_neural_network']
        lm_pred = row['Predicted_Quality_linear_models']
        ensemble_pred = row['Ensemble_Prediction']
        quality = row['Ensemble_Quality_Class']
        
        print(f"{station:<10} {tp:<6.3f} {ec:<6.3f} {do:<6.3f} {tit:<6.3f} "
              f"{rf_pred:<8.3f} {nn_pred:<8.3f} {lm_pred:<8.3f} {ensemble_pred:<10.3f} {quality:<12}")
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    # Summary statistics
    pred_cols = ['Predicted_Quality_random_forest', 'Predicted_Quality_neural_network', 
                 'Predicted_Quality_linear_models', 'Ensemble_Prediction']
    
    for col in pred_cols:
        model_name = col.replace('Predicted_Quality_', '').replace('_', ' ').title()
        values = df[col]
        print(f"\n{model_name}:")
        print(f"  Mean: {values.mean():.3f}")
        print(f"  Std:  {values.std():.3f}")
        print(f"  Min:  {values.min():.3f}")
        print(f"  Max:  {values.max():.3f}")
    
    # Quality distribution
    print("\n" + "="*80)
    print("QUALITY DISTRIBUTION")
    print("="*80)
    
    quality_counts = df['Ensemble_Quality_Class'].value_counts()
    for quality, count in quality_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{quality:<15}: {count:>3} stations ({percentage:>5.1f}%)")
    
    # Station-wise analysis
    print("\n" + "="*80)
    print("STATION-WISE ANALYSIS")
    print("="*80)
    
    station_summary = df.groupby('Station').agg({
        'Ensemble_Prediction': ['count', 'mean', 'std'],
        'Ensemble_Quality_Class': lambda x: x.mode()[0] if not x.empty else 'Unknown'
    }).round(3)
    
    station_summary.columns = ['Count', 'Mean_Quality', 'Std_Quality', 'Dominant_Class']
    print(station_summary)
    
    return df

if __name__ == "__main__":
    predictions = view_predictions()
    if predictions is not None:
        print(f"\n✓ Successfully analyzed {len(predictions)} predictions!")
    else:
        print("\n✗ Failed to analyze predictions!") 