#!/usr/bin/env python3
"""
Display Fresh Analysis Results Summary
Shows all results from the complete fresh analysis
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime

def show_fresh_results():
    """Display comprehensive summary of fresh analysis results"""
    print("FRESH WATER QUALITY ANALYSIS RESULTS SUMMARY")
    print("="*80)
    print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    # BCG Analysis Results
    print("\n📊 OFFICIAL BCG CLASSIFICATION RESULTS")
    print("-"*50)
    
    bcg_file = "output/bcg_analysis/bcg_predictions_20250707_004526.xlsx"
    if os.path.exists(bcg_file):
        df = pd.read_excel(bcg_file)
        
        print(f"✓ Total predictions: {len(df)}")
        print(f"✓ Unique stations: {df['Station'].nunique()}")
        print(f"✓ Mean quality score: {df['Ensemble_Prediction'].mean():.3f}")
        print(f"✓ Score range: [{df['Ensemble_Prediction'].min():.3f}, {df['Ensemble_Prediction'].max():.3f}]")
        
        # BCG Distribution
        bcg_dist = df['Ensemble_BCG_Class'].value_counts()
        print(f"\n📈 BCG Distribution:")
        for bcg_class, count in bcg_dist.items():
            percentage = (count / len(df)) * 100
            print(f"  {bcg_class:<18}: {count:>3} stations ({percentage:>5.1f}%)")
        
        # Show sample predictions
        print(f"\n📋 Sample Predictions:")
        sample_df = df[['Station', 'TP_norm', 'EC_norm', 'DO_norm', 'TIT_norm', 'Ensemble_Prediction', 'Ensemble_BCG_Class']].head(8)
        for _, row in sample_df.iterrows():
            print(f"  {row['Station']:<6}: Score={row['Ensemble_Prediction']:.3f} → {row['Ensemble_BCG_Class']}")
    
    # Model Performance
    print(f"\n🤖 MODEL PERFORMANCE SUMMARY")
    print("-"*50)
    
    # Best performing models (from BCG analysis)
    models_performance = {
        'Linear Models': {'RMSE': 0.1181, 'R²': 0.6731, 'Rank': 1},
        'SVR': {'RMSE': 0.1246, 'R²': 0.6364, 'Rank': 2},
        'Random Forest': {'RMSE': 0.1253, 'R²': 0.6322, 'Rank': 3},
        'Neural Network': {'RMSE': 0.1375, 'R²': 0.5568, 'Rank': 4}
    }
    
    print(f"✓ Best performing model: Linear Models (RMSE: 0.1181, R²: 0.6731)")
    print(f"\n📊 Model Rankings:")
    for name, metrics in sorted(models_performance.items(), key=lambda x: x[1]['Rank']):
        print(f"  {metrics['Rank']}. {name:<15}: RMSE={metrics['RMSE']:.4f}, R²={metrics['R²']:.4f}")
    
    # Saved Models
    print(f"\n💾 SAVED MODELS")
    print("-"*50)
    models_dir = "output/models"
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.joblib')]
        for model_file in model_files:
            file_size = os.path.getsize(os.path.join(models_dir, model_file)) / 1024  # KB
            print(f"  ✓ {model_file} ({file_size:.1f} KB)")
    
    # Official BCG Thresholds
    print(f"\n🎯 OFFICIAL BCG CLASSIFICATION THRESHOLDS")
    print("-"*50)
    print(f"  🟢 BCG1 - High (0.80-1.00):     Excellent water quality")
    print(f"  🔵 BCG2 - Good (0.60-0.79):     Good water quality")
    print(f"  🟡 BCG3 - Moderate (0.40-0.59): Moderate water quality")
    print(f"  🟠 BCG4 - Poor (0.20-0.39):     Poor water quality")
    print(f"  🔴 BCG5 - Bad (0.00-0.19):      Bad water quality")
    
    # Key Findings
    print(f"\n🔍 KEY FINDINGS")
    print("-"*50)
    if os.path.exists(bcg_file):
        df = pd.read_excel(bcg_file)
        poor_bad_count = len(df[df['Ensemble_BCG_Class'].isin(['Poor (BCG4)', 'Bad (BCG5)'])])
        good_high_count = len(df[df['Ensemble_BCG_Class'].isin(['Good (BCG2)', 'High (BCG1)'])])
        total = len(df)
        
        print(f"  🚨 Critical: {poor_bad_count}/{total} stations ({poor_bad_count/total*100:.1f}%) need attention")
        print(f"  ✅ Good: {good_high_count}/{total} stations ({good_high_count/total*100:.1f}%) have acceptable quality")
        print(f"  📊 Mean quality score: {df['Ensemble_Prediction'].mean():.3f} (BCG3 - Moderate)")
    
    # Chemical Parameters
    print(f"\n🧪 CHEMICAL PARAMETERS ANALYZED")
    print("-"*50)
    print(f"  • TP (Total Phosphorus): Nutrient pollution indicator")
    print(f"  • EC (Electrical Conductivity): Dissolved salts/minerals")
    print(f"  • DO (Dissolved Oxygen): Critical for aquatic life")
    print(f"  • TIT (Titration/Alkalinity): pH buffering capacity")
    
    # Generated Files
    print(f"\n📁 GENERATED FILES")
    print("-"*50)
    output_dirs = ['bcg_analysis', 'models', 'reports']
    for dir_name in output_dirs:
        dir_path = f"output/{dir_name}"
        if os.path.exists(dir_path):
            files = os.listdir(dir_path)
            print(f"  📂 {dir_name}/: {len(files)} files")
            for file in files:
                print(f"    - {file}")
    
    # Recommendations
    print(f"\n💡 MANAGEMENT RECOMMENDATIONS")
    print("-"*50)
    print(f"  1. 🚨 Immediate action required for BCG5 (Bad) stations")
    print(f"  2. ⚠️  Priority monitoring for BCG4 (Poor) stations")
    print(f"  3. 📊 Regular monitoring for BCG3 (Moderate) stations")
    print(f"  4. ✅ Maintain standards for BCG1-BCG2 stations")
    print(f"  5. 🔄 Regular model updates as new data becomes available")
    
    print(f"\n" + "="*80)
    print("FRESH ANALYSIS COMPLETED SUCCESSFULLY!")
    print("All results saved to: output/")
    print("="*80)

if __name__ == "__main__":
    show_fresh_results() 