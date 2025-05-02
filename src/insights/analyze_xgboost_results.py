"""
XGBoost Timeframes Model Results Analysis
-----------------------------------------
This script analyzes the results from the XGBoost timeframes model, focusing on:
1. Feature importance comparison between San Juan and Iquitos
2. Error analysis across time periods
3. Climate pattern visualization
4. Performance metrics summarization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
import json
from pathlib import Path
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Set matplotlib style
plt.style.use('ggplot')
sns.set_context("talk")

# Define functions for analysis
def load_model(city):
    """Load XGBoost model for specified city."""
    model_path = f"models/{city}/xgboost_timeframes_{city}_latest/model.json"
    try:
        model = xgb.Booster()
        model.load_model(model_path)
        return model
    except Exception as e:
        print(f"Error loading model for {city}: {str(e)}")
        return None

def load_feature_names(city):
    """Load feature names for specified city."""
    feature_path = f"models/{city}/xgboost_timeframes_{city}_latest/feature_names.txt"
    try:
        features = []
        with open(feature_path, 'r') as f:
            for line in f:
                # Extract feature name by removing the number and dot prefix
                if '. ' in line:
                    feature_name = line.strip().split('. ', 1)[1]
                else:
                    feature_name = line.strip()
                features.append(feature_name)
        return features
    except Exception as e:
        print(f"Error loading feature names for {city}: {str(e)}")
        return []

def load_metrics(city):
    """Load model metrics for specified city."""
    metrics_path = f"models/{city}/xgboost_timeframes_{city}_latest/metrics.json"
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading metrics for {city}: {str(e)}")
        return {}

def load_data():
    """Load the dengue fever dataset."""
    try:
        data = pd.read_csv('data/processed/cleaned_data_merged.csv')
        data['week_start_date'] = pd.to_datetime(data['week_start_date'])
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def get_feature_importance(model, feature_names):
    """Extract feature importance from the model."""
    if model is None or not feature_names:
        return pd.DataFrame()
    
    # Get feature importance (use gain as the importance metric)
    importance = model.get_score(importance_type='gain')
    
    # Create a DataFrame with all features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': [importance.get(f, 0) for f in feature_names]
    })
    
    # Normalize importance
    if importance_df['importance'].sum() > 0:
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    else:
        importance_df['importance_normalized'] = 0
    
    # Sort by importance
    return importance_df.sort_values('importance', ascending=False)

def categorize_features(features):
    """Group features into logical categories."""
    categories = {
        'Temperature': [f for f in features if any(t in f.lower() for t in ['temp', 'air_temp', 'tdtr'])],
        'Precipitation': [f for f in features if any(p in f.lower() for p in ['precip', 'rain'])],
        'Humidity': [f for f in features if any(h in f.lower() for h in ['humid', 'dew'])],
        'Vegetation': [f for f in features if 'ndvi' in f.lower()],
        'Temporal': [f for f in features if any(c in f.lower() for c in ['sin', 'cos', 'week', 'month'])],
        'Engineered': [f for f in features if any(e in f.lower() for e in ['index', 'suitability', 'interaction'])]
    }
    
    # Add a "Lagged" category that overlaps with others
    categories['Lagged Features'] = [f for f in features if 'lag' in f.lower()]
    
    return categories

def plot_feature_importance_comparison(sj_importance, iq_importance, top_n=15):
    """Plot feature importance comparison between cities."""
    plt.figure(figsize=(15, 10))
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    # San Juan plot
    sj_top = sj_importance.head(top_n)
    sj_top = sj_top.sort_values('importance_normalized')
    ax1.barh(sj_top['feature'], sj_top['importance_normalized'], color='skyblue')
    ax1.set_xlabel('Relative Importance')
    ax1.set_title('San Juan: Top Feature Importance')
    ax1.grid(axis='x', linestyle='--', alpha=0.6)
    
    # Iquitos plot
    iq_top = iq_importance.head(top_n)
    iq_top = iq_top.sort_values('importance_normalized')
    ax2.barh(iq_top['feature'], iq_top['importance_normalized'], color='lightgreen')
    ax2.set_xlabel('Relative Importance')
    ax2.set_title('Iquitos: Top Feature Importance')
    ax2.grid(axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'feature_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Feature importance comparison saved to '{results_dir}/feature_importance_comparison.png'")
    
    return fig

def plot_category_importance(sj_importance, iq_importance, sj_features, iq_features):
    """Plot feature importance by category."""
    # Categorize features
    sj_categories = categorize_features(sj_features)
    iq_categories = categorize_features(iq_features)
    
    # Calculate importance by category
    category_data = []
    
    for category, features in sj_categories.items():
        # Calculate total importance for this category
        sj_cat_importance = sj_importance[sj_importance['feature'].isin(features)]['importance'].sum()
        iq_cat_importance = iq_importance[iq_importance['feature'].isin(
            iq_categories.get(category, []))]['importance'].sum()
        
        category_data.append({
            'Category': category,
            'San Juan': sj_cat_importance,
            'Iquitos': iq_cat_importance,
            'Feature Count': len(features)
        })
    
    # Create DataFrame
    category_df = pd.DataFrame(category_data)
    
    # Normalize importances
    if category_df['San Juan'].sum() > 0:
        category_df['San Juan'] = category_df['San Juan'] / category_df['San Juan'].sum()
    
    if category_df['Iquitos'].sum() > 0:
        category_df['Iquitos'] = category_df['Iquitos'] / category_df['Iquitos'].sum()
    
    # Plot
    plt.figure(figsize=(14, 8))
    
    # Create a grouped bar chart
    width = 0.35
    x = np.arange(len(category_df))
    
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.bar(x - width/2, category_df['San Juan'], width, label='San Juan', color='skyblue')
    ax.bar(x + width/2, category_df['Iquitos'], width, label='Iquitos', color='lightgreen')
    
    # Add labels and title
    ax.set_xlabel('Feature Category')
    ax.set_ylabel('Relative Importance')
    ax.set_title('Feature Category Importance: San Juan vs Iquitos')
    ax.set_xticks(x)
    ax.set_xticklabels(category_df['Category'], rotation=45, ha='right')
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'category_importance_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Category importance comparison saved to '{results_dir}/category_importance_comparison.png'")
    
    return fig

def plot_common_features_correlation(sj_importance, iq_importance):
    """Plot correlation between feature importance across cities."""
    # Find common features
    common_features = set(sj_importance['feature']).intersection(set(iq_importance['feature']))
    
    if not common_features:
        print("No common features found between cities")
        return None
    
    # Filter to common features
    sj_common = sj_importance[sj_importance['feature'].isin(common_features)]
    iq_common = iq_importance[iq_importance['feature'].isin(common_features)]
    
    # Create a merged DataFrame
    merged_importance = pd.merge(
        sj_common, 
        iq_common, 
        on='feature', 
        suffixes=('_sj', '_iq')
    )
    
    # Create scatter plot
    plt.figure(figsize=(12, 10))
    
    # Use log scale if importance values span multiple orders of magnitude
    if ((merged_importance['importance_normalized_sj'].max() / 
         merged_importance['importance_normalized_sj'].min() > 100) or
        (merged_importance['importance_normalized_iq'].max() / 
         merged_importance['importance_normalized_iq'].min() > 100)):
        plt.xscale('log')
        plt.yscale('log')
    
    # Create scatter plot
    plt.scatter(
        merged_importance['importance_normalized_sj'],
        merged_importance['importance_normalized_iq'],
        alpha=0.7,
        s=100
    )
    
    # Add feature labels
    for i, feature in enumerate(merged_importance['feature']):
        plt.annotate(
            feature,
            (merged_importance['importance_normalized_sj'].iloc[i],
             merged_importance['importance_normalized_iq'].iloc[i]),
            xytext=(5, 5),
            textcoords='offset points',
            fontsize=9
        )
    
    # Add diagonal line (equal importance)
    max_val = max(
        merged_importance['importance_normalized_sj'].max(),
        merged_importance['importance_normalized_iq'].max()
    )
    min_val = min(
        merged_importance['importance_normalized_sj'].min(),
        merged_importance['importance_normalized_iq'].min()
    ) / 2  # Extend a bit below min value
    
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    # Calculate correlation
    corr = merged_importance['importance_normalized_sj'].corr(
        merged_importance['importance_normalized_iq']
    )
    
    # Add labels and title
    plt.xlabel('San Juan Feature Importance')
    plt.ylabel('Iquitos Feature Importance')
    plt.title(f'Feature Importance Correlation (r = {corr:.3f})\nSan Juan vs Iquitos')
    plt.grid(True, alpha=0.3)
    
    # Save the figure
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'feature_importance_correlation.png', dpi=300, bbox_inches='tight')
    print(f"Feature importance correlation saved to '{results_dir}/feature_importance_correlation.png'")
    
    return plt.gcf()

def plot_performance_comparison(sj_metrics, iq_metrics):
    """Plot performance metrics comparison between cities."""
    # Metrics to compare
    metrics = ['rmse', 'mae', 'r2', 'outbreak_precision', 'outbreak_recall', 'outbreak_f1']
    metric_names = ['RMSE', 'MAE', 'R²', 'Outbreak Precision', 'Outbreak Recall', 'Outbreak F1']
    
    # Extract metrics
    sj_values = [sj_metrics.get(m, 0) for m in metrics]
    iq_values = [iq_metrics.get(m, 0) for m in metrics]
    
    # Create the figure - skip R² for Iquitos as it's extremely negative
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for i, (metric, name) in enumerate(zip(metrics, metric_names)):
        ax = axes[i]
        
        # For R², handle the extreme negative value for Iquitos
        if metric == 'r2':
            # Just show San Juan
            ax.bar(['San Juan'], [sj_metrics.get(metric, 0)], color='skyblue')
            ax.set_title(f"{name} (Iquitos: {iq_metrics.get(metric, 0):.1f})")
            ax.text(0, sj_metrics.get(metric, 0)/2, f"{sj_metrics.get(metric, 0):.3f}", 
                   ha='center', va='center', fontsize=12)
        else:
            ax.bar(['San Juan', 'Iquitos'], 
                 [sj_metrics.get(metric, 0), iq_metrics.get(metric, 0)],
                 color=['skyblue', 'lightgreen'])
            ax.set_title(name)
            
            # Add value labels
            for j, v in enumerate([sj_metrics.get(metric, 0), iq_metrics.get(metric, 0)]):
                ax.text(j, v/2, f"{v:.3f}", ha='center', va='center', fontsize=12)
        
        # Add grid and clean up
        ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'performance_metrics_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Performance metrics comparison saved to '{results_dir}/performance_metrics_comparison.png'")
    
    return fig

def plot_climate_importance(sj_importance, iq_importance, climate_vars=['temp', 'precip', 'humid']):
    """Plot importance of climate variables with different lag periods."""
    # Create figure
    fig, axes = plt.subplots(len(climate_vars), 2, figsize=(16, 4*len(climate_vars)), sharey='row')
    
    for i, climate_var in enumerate(climate_vars):
        for j, (city, importance) in enumerate([('San Juan', sj_importance), ('Iquitos', iq_importance)]):
            ax = axes[i, j]
            
            # Filter features containing this climate variable and with lag
            lag_features = importance[
                (importance['feature'].str.contains(climate_var, case=False)) & 
                (importance['feature'].str.contains('lag', case=False))
            ].copy()
            
            if not lag_features.empty:
                # Extract lag period from feature name
                lag_features['lag'] = lag_features['feature'].str.extract(r'lag_(\d+)').astype(int)
                
                # Sort by lag period
                lag_features = lag_features.sort_values('lag')
                
                # Plot
                ax.bar(lag_features['lag'], lag_features['importance_normalized'], 
                       color='skyblue' if j == 0 else 'lightgreen', alpha=0.7)
                ax.set_title(f"{city}: {climate_var.title()} Variables by Lag Period")
                ax.set_xlabel('Lag Period (Weeks)')
                
                if j == 0:
                    ax.set_ylabel('Relative Importance')
                
                ax.grid(axis='y', linestyle='--', alpha=0.3)
            else:
                ax.text(0.5, 0.5, f"No lag features for {climate_var}", 
                       ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    
    # Save the figure
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    plt.savefig(results_dir / 'climate_lag_importance.png', dpi=300, bbox_inches='tight')
    print(f"Climate lag importance saved to '{results_dir}/climate_lag_importance.png'")
    
    return fig

def generate_performance_summary(sj_metrics, iq_metrics):
    """Generate a comprehensive performance summary."""
    # Create a summary DataFrame
    metrics = ['rmse', 'mae', 'r2', 'outbreak_precision', 'outbreak_recall', 
              'outbreak_f1', 'outbreak_accuracy', 'weighted_rmse', 'bias', 'correlation']
    
    metric_names = ['RMSE', 'MAE', 'R²', 'Outbreak Precision', 'Outbreak Recall',
                   'Outbreak F1', 'Outbreak Accuracy', 'Weighted RMSE', 'Prediction Bias', 'Correlation']
    
    summary = pd.DataFrame({
        'Metric': metric_names,
        'San Juan': [sj_metrics.get(m, 'N/A') for m in metrics],
        'Iquitos': [iq_metrics.get(m, 'N/A') for m in metrics]
    })
    
    # Save to CSV
    results_dir = Path('results/insights')
    results_dir.mkdir(exist_ok=True, parents=True)
    summary.to_csv(results_dir / 'model_performance_summary.csv', index=False)
    print(f"Performance summary saved to '{results_dir}/model_performance_summary.csv'")
    
    # Return also for display
    return summary

def generate_city_specific_insights(city, metrics, importance):
    """Generate city-specific insights based on model performance and features."""
    insights = []
    
    # Performance insights
    r2 = metrics.get('r2', 0)
    if r2 < 0:
        insights.append(f"- Model for {city} has a negative R² ({r2:.4f}), indicating poor performance")
    
    # Feature insights
    if not importance.empty:
        top_features = importance.head(5)['feature'].tolist()
        insights.append(f"- Top predictive features for {city}: {', '.join(top_features)}")
        
        # Climate-specific insights
        temp_features = importance[importance['feature'].str.contains('temp', case=False)]
        precip_features = importance[importance['feature'].str.contains('precip', case=False)]
        
        if not temp_features.empty and not precip_features.empty:
            temp_importance = temp_features['importance'].sum()
            precip_importance = precip_features['importance'].sum()
            
            if temp_importance > precip_importance:
                insights.append(f"- Temperature factors are more important than precipitation for {city}")
            else:
                insights.append(f"- Precipitation factors are more important than temperature for {city}")
        
        # Lag insights
        lag_features = importance[importance['feature'].str.contains('lag', case=False)]
        if not lag_features.empty:
            # Extract lag numbers
            lag_features['lag'] = lag_features['feature'].str.extract(r'lag_(\d+)').astype(int)
            
            # Find most important lag
            max_lag = lag_features.loc[lag_features['importance'].idxmax(), 'lag']
            insights.append(f"- Most important lag period for {city}: {max_lag} weeks")
    
    # Outbreak detection insights
    prec = metrics.get('outbreak_precision', 0)
    rec = metrics.get('outbreak_recall', 0)
    
    if prec < 0.3 and rec > 0.8:
        insights.append(f"- Model has low precision ({prec:.2f}) but high recall ({rec:.2f}), predicting too many outbreaks")
    
    return "\n".join(insights)

def main():
    """Main function to run the analysis."""
    print("Starting XGBoost Timeframes Model Analysis...")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/insights")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load models and data
    sj_model = load_model('sj')
    iq_model = load_model('iq')
    
    sj_features = load_feature_names('sj')
    iq_features = load_feature_names('iq')
    
    sj_metrics = load_metrics('sj')
    iq_metrics = load_metrics('iq')
    
    # Get feature importance
    sj_importance = get_feature_importance(sj_model, sj_features)
    iq_importance = get_feature_importance(iq_model, iq_features)
    
    # Generate visualizations
    if not sj_importance.empty and not iq_importance.empty:
        print("\nGenerating feature importance visualizations...")
        plot_feature_importance_comparison(sj_importance, iq_importance)
        plot_category_importance(sj_importance, iq_importance, sj_features, iq_features)
        plot_common_features_correlation(sj_importance, iq_importance)
        plot_climate_importance(sj_importance, iq_importance)
    
    # Generate performance comparison
    if sj_metrics and iq_metrics:
        print("\nGenerating performance comparison...")
        plot_performance_comparison(sj_metrics, iq_metrics)
        summary = generate_performance_summary(sj_metrics, iq_metrics)
        print("\nPerformance Summary:")
        print(summary)
    
    # Generate city-specific insights
    print("\nCity-Specific Insights:")
    print("\nSan Juan:")
    print(generate_city_specific_insights('San Juan', sj_metrics, sj_importance))
    print("\nIquitos:")
    print(generate_city_specific_insights('Iquitos', iq_metrics, iq_importance))
    
    print("\nAnalysis complete! All results saved to 'results/insights/' directory.")

if __name__ == "__main__":
    main()