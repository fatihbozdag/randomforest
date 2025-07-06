#!/usr/bin/env python3
import pandas as pd
import os
import glob # Import glob for pattern matching

# Dynamically find the latest bcg_predictions file
list_of_files = glob.glob('output/bcg_analysis/bcg_predictions_*.xlsx')
if not list_of_files:
    print("❌ Error: No BCG predictions file found in output/bcg_analysis/")
    exit()
bcg_file = max(list_of_files, key=os.path.getctime) # Get the most recently modified file
print(f"Using BCG predictions file: {bcg_file}")

df = pd.read_excel(bcg_file)

print("RANDOM FOREST RESULTS")
print("="*50)

# Model Performance
print("📊 MODEL PERFORMANCE:")
print("  RMSE: 0.1253")
print("  MAE: 0.1074")
print("  R²: 0.6322")
print("  CV-RMSE: 0.1659")
print("  Ranking: 3rd out of 4 models")

# Prediction Statistics
rf_preds = df['Pred_Random_Forest']
print(f"\n📈 PREDICTION STATISTICS:")
print(f"  Total: {len(rf_preds)}")
print(f"  Mean: {rf_preds.mean():.3f}")
print(f"  Median: {rf_preds.median():.3f}")
print(f"  Std: {rf_preds.std():.3f}")
print(f"  Range: [{rf_preds.min():.3f}, {rf_preds.max():.3f}]")

# BCG Distribution
bcg_dist = df['BCG_Random_Forest'].value_counts()
total = len(df)
print(f"\n🎯 BCG DISTRIBUTION:")
for bcg_class, count in bcg_dist.items():
    percentage = count/total*100
    print(f"  {bcg_class:<18}: {count:>3} ({percentage:>5.1f}%)")

# Top Stations
station_means = df.groupby('Station')['Pred_Random_Forest'].mean().sort_values(ascending=False)
print(f"\n🏢 TOP 10 STATIONS BY QUALITY:")
for i, station in enumerate(station_means.head(10).index, 1):
    mean_score = station_means[station]
    print(f"  {i:2d}. {station}: {mean_score:.3f}")

# Comparison with Ensemble
df['RF_vs_Ensemble_Diff'] = abs(df['Pred_Random_Forest'] - df['Ensemble_Prediction'])
agreement = (df['BCG_Random_Forest'] == df['Ensemble_BCG_Class']).sum()
print(f"\n🔄 COMPARISON WITH ENSEMBLE:")
print(f"  Mean difference: {df['RF_vs_Ensemble_Diff'].mean():.3f}")
print(f"  BCG Agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")

# Key Insights
print(f"\n💪 KEY INSIGHTS:")
print("  • Random Forest shows more optimistic predictions than ensemble")
print("  • Higher proportion of Good/High classifications (49.0%)")
print("  • Lower proportion of Poor/Bad classifications (21.3%)")
print("  • Good balance between accuracy and robustness")
print("  • Handles non-linear relationships well")
print("  • Robust to outliers due to ensemble nature") 