#!/usr/bin/env python3
"""
Visualization script for Random Forest model results
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob # Import glob for pattern matching

def create_model_visualizations(predictions_df, model_name):
    """
    Create basic histogram and BCG distribution plots for a given model.
    Saves images into output/Model/<model_name>/<plot>.png
    """
    os.makedirs(f"output/Model/{model_name}", exist_ok=True)
    sns.set_style("whitegrid")

    pred_col = f"Pred_{model_name}"
    bcg_col = f"BCG_{model_name}"
    if pred_col not in predictions_df.columns:
        print(f"⚠️ {pred_col} not found in predictions file, skipping visuals for {model_name}")
        return

    # Histogram
    plt.figure(figsize=(10,6))
    sns.histplot(predictions_df[pred_col], bins=10, kde=True, color='skyblue')
    plt.title(f"Distribution of {model_name} Predictions")
    plt.xlabel('Predicted Water Quality Score')
    plt.ylabel('Count')
    plt.xlim(0,1)
    plt.tight_layout()
    hist_path = f"output/Model/{model_name}/{model_name}_prediction_distribution.png"
    plt.savefig(hist_path)
    plt.close()

    # BCG distribution bar
    if bcg_col in predictions_df.columns:
        plt.figure(figsize=(10,6))
        bcg_order = ['High (BCG1)', 'Good (BCG2)', 'Moderate (BCG3)', 'Poor (BCG4)', 'Bad (BCG5)']
        sns.countplot(data=predictions_df, x=bcg_col, order=bcg_order, palette='viridis')
        plt.title(f"{model_name} BCG Classification Distribution")
        plt.xlabel('BCG Class')
        plt.ylabel('Count')
        plt.tight_layout()
        bar_path = f"output/Model/{model_name}/{model_name}_bcg_distribution.png"
        plt.savefig(bar_path)
        plt.close()

    # Scatter TP vs prediction if TP present
    tp_col = 'TP' if 'TP' in predictions_df.columns else ('TP_norm' if 'TP_norm' in predictions_df.columns else None)
    if tp_col:
        plt.figure(figsize=(10,6))
        sns.scatterplot(data=predictions_df, x=tp_col, y=pred_col, hue=bcg_col if bcg_col in predictions_df.columns else None)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.title(f"{tp_col} vs {model_name} Prediction")
        plt.tight_layout()
        scatter_path = f"output/Model/{model_name}/{model_name}_tp_vs_pred.png"
        plt.savefig(scatter_path)
        plt.close()

    print(f"Visuals saved to output/Model/{model_name}/")

def create_rf_visualizations():
    """
    Generates and displays visualizations for Random Forest model results.
    """
    print("GENERATING RANDOM FOREST VISUALIZATIONS")
    print("="*80)

    # Dynamically find the latest bcg_predictions file
    list_of_files = glob.glob('output/bcg_analysis/bcg_predictions_*.xlsx')
    if not list_of_files:
        print("❌ Error: No BCG predictions file found in output/bcg_analysis/")
        return
    bcg_file = max(list_of_files, key=os.path.getctime) # Get the most recently modified file
    print(f"Using BCG predictions file: {bcg_file}")

    if not os.path.exists(bcg_file):
        print(f"❌ Error: BCG predictions file not found at {bcg_file}")
        return

    df = pd.read_excel(bcg_file)

    # Set style for plots
    sns.set_style("whitegrid")

    # 1. Histogram of Random Forest Predictions
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Pred_Random_Forest'], bins=10, kde=True, color='skyblue')
    plt.title('Distribution of Random Forest Water Quality Predictions')
    plt.xlabel('Predicted Water Quality Score')
    plt.ylabel('Number of Stations')
    plt.xlim(0, 1) # Scores are normalized between 0 and 1
    plt.grid(axis='y', alpha=0.75)
    plt.tight_layout()
    plt.savefig('output/rf_prediction_distribution.png')
    plt.show()
    print("Generated: output/rf_prediction_distribution.png")

    # 2. Bar Chart of BCG Classification Distribution
    plt.figure(figsize=(10, 6))
    bcg_order = ['High (BCG1)', 'Good (BCG2)', 'Moderate (BCG3)', 'Poor (BCG4)', 'Bad (BCG5)']
    sns.countplot(data=df, x='BCG_Random_Forest', order=bcg_order, palette='viridis')
    plt.title('Random Forest BCG Classification Distribution')
    plt.xlabel('Water Quality Class (BCG)')
    plt.ylabel('Number of Stations')
    plt.tight_layout()
    plt.savefig('output/rf_bcg_distribution.png')
    plt.show()
    print("Generated: output/rf_bcg_distribution.png")

    # 3. Scatter Plot: TP vs. Pred_Random_Forest colored by BCG Class
    tp_col = 'TP_norm' if 'TP_norm' in df.columns else ('TP' if 'TP' in df.columns else None)
    if tp_col:
        plt.figure(figsize=(12, 7))
        sns.scatterplot(data=df, x=tp_col, y='Pred_Random_Forest', hue='BCG_Random_Forest', 
                        s=100, alpha=0.7, palette='coolwarm', hue_order=bcg_order)
        plt.title(f'{tp_col} vs. Random Forest Prediction (Colored by BCG Class)')
        plt.xlabel(f'Total Phosphorus ({tp_col})')
        plt.ylabel('Predicted Water Quality Score')
        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.legend(title='BCG Class', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig('output/rf_tp_vs_prediction.png')
        plt.show()
        print("Generated: output/rf_tp_vs_prediction.png")
    else:
        print('⚠️ TP column not found for scatter plot, skipping.')

    # 4. Bar chart of Top 10 Stations by Mean Quality Score
    plt.figure(figsize=(12, 7))
    station_means = df.groupby('Station')['Pred_Random_Forest'].mean().sort_values(ascending=False).head(10)
    sns.barplot(x=station_means.index, y=station_means.values, palette='crest')
    plt.title('Top 10 Stations by Mean Random Forest Predicted Quality Score')
    plt.xlabel('Station')
    plt.ylabel('Mean Predicted Water Quality Score')
    plt.ylim(0, 1) # Scores are normalized between 0 and 1
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('output/rf_top_stations.png')
    plt.show()
    print("Generated: output/rf_top_stations.png")
    
    print("="*80)
    print("RANDOM FOREST VISUALIZATIONS GENERATED!")
    print("="*80)

    create_model_visualizations(df, 'Random_Forest')

if __name__ == "__main__":
    # Ensure the output directory exists
    os.makedirs('output', exist_ok=True)
    create_rf_visualizations() 