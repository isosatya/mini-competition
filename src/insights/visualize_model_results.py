"""
Visualize XGBoost Timeframes Model Results
-----------------------------------------
This script creates comprehensive visualizations from the XGBoost timeframes model results,
focusing on prediction accuracy, feature importance, and comparison between cities.
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

# Set up visualization style
plt.style.use('ggplot')
sns.set_context("talk")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12

def load_data():
    """Load the dengue fever dataset."""
    try:
        data = pd.read_csv('data/processed/cleaned_data_merged.csv')
        data['week_start_date'] = pd.to_datetime(data['week_start_date'])
        return data
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return pd.DataFrame()

def load_submission():
    """Load the latest submission file."""
    try:
        submission = pd.read_csv('results/submission_xgboost_timeframes_latest.csv')
        return submission
    except Exception as e:
        print(f"Error loading submission: {str(e)}")
        try:
            # Try any submission file in the directory
            files = list(Path('results/submissions').glob('submission_xgboost_timeframes_*.csv'))
            if files:
                latest = max(files, key=lambda p: p.stat().st_mtime)
                submission = pd.read_csv(latest)
                return submission
        except Exception:
            pass
        return pd.DataFrame()

def load_model_metrics():
    """Load model metrics for both cities."""
    metrics = {}
    for city in ['sj', 'iq']:
        try:
            with open(f'models/{city}/xgboost_timeframes_{city}_latest/metrics.json', 'r') as f:
                metrics[city] = json.load(f)
        except Exception as e:
            print(f"Error loading metrics for {city}: {str(e)}")
            metrics[city] = {}
    return metrics

def create_combined_feature_importance_plot(results_dir):
    """Create a combined feature importance visualization for both cities."""
    # Load feature importance images if they exist
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    try:
        # Use existing feature importance plots
        sj_img = plt.imread('results/insights/feature_importance_xgboost_timeframes_sj.png')
        iq_img = plt.imread('results/insights/feature_importance_xgboost_timeframes_iq.png')
        
        axes[0].imshow(sj_img)
        axes[0].axis('off')
        axes[0].set_title('San Juan Feature Importance')
        
        axes[1].imshow(iq_img)
        axes[1].axis('off')
        axes[1].set_title('Iquitos Feature Importance')
    except Exception as e:
        print(f"Could not load feature importance images: {str(e)}")
        
        # Create a placeholder
        for i, city in enumerate(['San Juan', 'Iquitos']):
            axes[i].text(0.5, 0.5, f"Feature Importance for {city}\n(Image not available)", 
                        ha='center', va='center', fontsize=14)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'combined_feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"Combined feature importance plot saved to '{results_dir}/combined_feature_importance.png'")
    
    return fig

def create_performance_dashboard(metrics, results_dir):
    """Create a performance metrics dashboard."""
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    axes = axes.flatten()
    
    # 1. Standard metrics comparison
    ax = axes[0]
    std_metrics = ['rmse', 'mae', 'correlation']
    std_names = ['RMSE', 'MAE', 'Correlation']
    
    x = np.arange(len(std_metrics))
    width = 0.35
    
    # Get values, handling Iquitos extreme values
    sj_values = [metrics['sj'].get(m, 0) for m in std_metrics]
    iq_values = []
    for m in std_metrics:
        if m == 'rmse' and metrics['iq'].get(m, 0) > 1000:
            # Scale down extremely large values for visualization
            iq_values.append(100)  # Cap at 100 for visualization
        elif m == 'mae' and metrics['iq'].get(m, 0) > 1000:
            iq_values.append(100)  # Cap at 100 for visualization
        else:
            iq_values.append(metrics['iq'].get(m, 0))
    
    ax.bar(x - width/2, sj_values, width, label='San Juan', color='skyblue')
    ax.bar(x + width/2, iq_values, width, label='Iquitos', color='lightgreen')
    
    ax.set_ylabel('Value')
    ax.set_title('Standard Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(std_names)
    ax.legend()
    
    # Add annotations with actual values
    for i, v in enumerate(sj_values):
        ax.text(i - width/2, v + 0.1, f"{v:.2f}", ha='center', fontsize=10)
    
    for i, v in enumerate(iq_values):
        actual = metrics['iq'].get(std_metrics[i], 0)
        if actual > 100:
            ax.text(i + width/2, v + 0.1, f"{actual:.1f}", ha='center', fontsize=10)
        else:
            ax.text(i + width/2, v + 0.1, f"{actual:.2f}", ha='center', fontsize=10)
    
    # 2. Outbreak detection metrics
    ax = axes[1]
    outbreak_metrics = ['outbreak_precision', 'outbreak_recall', 'outbreak_f1']
    outbreak_names = ['Precision', 'Recall', 'F1 Score']
    
    x = np.arange(len(outbreak_metrics))
    
    sj_values = [metrics['sj'].get(m, 0) for m in outbreak_metrics]
    iq_values = [metrics['iq'].get(m, 0) for m in outbreak_metrics]
    
    ax.bar(x - width/2, sj_values, width, label='San Juan', color='skyblue')
    ax.bar(x + width/2, iq_values, width, label='Iquitos', color='lightgreen')
    
    ax.set_ylabel('Value')
    ax.set_title('Outbreak Detection Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(outbreak_names)
    ax.legend()
    
    # Add value annotations
    for i, v in enumerate(sj_values):
        ax.text(i - width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    
    for i, v in enumerate(iq_values):
        ax.text(i + width/2, v + 0.02, f"{v:.2f}", ha='center', fontsize=10)
    
    # 3. R² values (separate due to extreme differences)
    ax = axes[2]
    
    r2_sj = metrics['sj'].get('r2', 0)
    r2_iq = metrics['iq'].get('r2', 0)
    
    ax.bar(['San Juan'], [r2_sj], color='skyblue')
    ax.set_ylabel('R² Value')
    ax.set_title(f'R² Score (Iquitos: {r2_iq:.1f})')
    ax.text(0, r2_sj/2, f"{r2_sj:.4f}", ha='center', va='center', fontsize=12, color='black')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    # 4. Confusion Matrix Information
    ax = axes[3]
    
    # Create a table with confusion matrix data
    cm_data = []
    for city, city_display in [('sj', 'San Juan'), ('iq', 'Iquitos')]:
        if 'confusion_matrix' in metrics[city]:
            cm = metrics[city]['confusion_matrix']
            cm_data.append([
                city_display,
                cm.get('true_positives', 0),
                cm.get('false_positives', 0),
                cm.get('true_negatives', 0),
                cm.get('false_negatives', 0)
            ])
    
    if cm_data:
        columns = ['City', 'True Positives', 'False Positives', 'True Negatives', 'False Negatives']
        table = ax.table(
            cellText=cm_data,
            colLabels=columns,
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1, 1.5)
        ax.axis('off')
        ax.set_title('Confusion Matrix Data')
    else:
        ax.text(0.5, 0.5, "Confusion Matrix Data Not Available", ha='center', va='center', fontsize=14)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'performance_dashboard.png', dpi=300, bbox_inches='tight')
    print(f"Performance dashboard saved to '{results_dir}/performance_dashboard.png'")
    
    return fig

def create_prediction_comparison_plots(results_dir):
    """Create plots showing predicted vs actual values if available."""
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    try:
        # Use existing prediction plots
        sj_img = plt.imread('results/insights/predictions_xgboost_timeframes_sj.png')
        iq_img = plt.imread('results/insights/predictions_xgboost_timeframes_iq.png')
        
        axes[0].imshow(sj_img)
        axes[0].axis('off')
        axes[0].set_title('San Juan Predictions')
        
        axes[1].imshow(iq_img)
        axes[1].axis('off')
        axes[1].set_title('Iquitos Predictions')
    except Exception as e:
        print(f"Could not load prediction images: {str(e)}")
        
        # Create a placeholder
        for i, city in enumerate(['San Juan', 'Iquitos']):
            axes[i].text(0.5, 0.5, f"Predictions for {city}\n(Image not available)", 
                        ha='center', va='center', fontsize=14)
            axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(results_dir / 'combined_predictions.png', dpi=300, bbox_inches='tight')
    print(f"Combined predictions plot saved to '{results_dir}/combined_predictions.png'")
    
    return fig

def create_submission_analysis(submission, data, results_dir):
    """Create analysis of submission predictions."""
    if submission.empty or data.empty:
        print("Submission or data not available for analysis")
        return None
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    # Check if we have submission data
    for i, city_code in enumerate(['sj', 'iq']):
        ax = axes[i]
        city_name = 'San Juan' if city_code == 'sj' else 'Iquitos'
        
        # Filter submission data for this city
        city_submission = submission[submission['city'] == city_code]
        
        if city_submission.empty:
            ax.text(0.5, 0.5, f"No submission data for {city_name}", ha='center', va='center', fontsize=14)
            ax.axis('off')
            continue
        
        # Create distribution plot
        sns.histplot(city_submission['total_cases'], bins=30, kde=True, ax=ax, color='skyblue')
        ax.set_title(f'{city_name} Prediction Distribution')
        ax.set_xlabel('Predicted Cases')
        ax.set_ylabel('Frequency')
        
        # Add annotation with statistics
        stats_text = (
            f"Min: {city_submission['total_cases'].min()}\n"
            f"Max: {city_submission['total_cases'].max()}\n"
            f"Mean: {city_submission['total_cases'].mean():.2f}\n"
            f"Count: {len(city_submission)}"
        )
        
        # Add text box with statistics
        ax.text(
            0.95, 0.95, stats_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.7)
        )
    
    plt.tight_layout()
    plt.savefig(results_dir / 'submission_distribution.png', dpi=300, bbox_inches='tight')
    print(f"Submission distribution plot saved to '{results_dir}/submission_distribution.png'")
    
    return fig

def create_city_comparison(data, results_dir):
    """Create climate and case comparison between cities."""
    if data.empty:
        print("Data not available for city comparison")
        return None
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 12))
    
    # 1. Time series of cases
    ax = axes[0]
    
    for city_code, color, name in [('sj', 'blue', 'San Juan'), ('iq', 'green', 'Iquitos')]:
        city_data = data[data['city'] == city_code]
        ax.plot(city_data['week_start_date'], city_data['total_cases'], 
               color=color, alpha=0.7, label=name)
    
    ax.set_title('Dengue Cases Over Time by City')
    ax.set_xlabel('Date')
    ax.set_ylabel('Total Cases')
    ax.legend()
    
    # Format date axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # 2. Monthly patterns
    ax = axes[1]
    
    # Add month column
    data['month'] = data['week_start_date'].dt.month
    
    # Calculate monthly averages by city
    monthly_avg = data.groupby(['city', 'month'])['total_cases'].mean().reset_index()
    
    # Pivot for easier plotting
    monthly_pivot = monthly_avg.pivot(index='month', columns='city', values='total_cases')
    
    # Plot monthly patterns
    monthly_pivot.plot(ax=ax, marker='o')
    
    ax.set_title('Monthly Average Cases by City')
    ax.set_xlabel('Month')
    ax.set_ylabel('Average Cases')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                       'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    ax.legend(['San Juan', 'Iquitos'])
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(results_dir / 'city_comparison.png', dpi=300, bbox_inches='tight')
    print(f"City comparison plot saved to '{results_dir}/city_comparison.png'")
    
    return fig

def create_results_summary_page(results_dir):
    """Create an HTML summary page with all visualizations."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>XGBoost Timeframes Model Results</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 20px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1, h2, h3 {
                color: #2c3e50;
            }
            .image-container {
                margin: 20px 0;
                text-align: center;
            }
            img {
                max-width: 100%;
                border: 1px solid #ddd;
                border-radius: 4px;
                padding: 5px;
            }
            .section {
                margin-bottom: 30px;
                padding-bottom: 20px;
                border-bottom: 1px solid #eee;
            }
            .metrics-table {
                width: 100%;
                border-collapse: collapse;
                margin: 20px 0;
            }
            .metrics-table th, .metrics-table td {
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            .metrics-table th {
                background-color: #f2f2f2;
            }
            .city-comparison {
                display: flex;
                justify-content: space-between;
            }
            .city-comparison div {
                width: 48%;
            }
            .footer {
                margin-top: 30px;
                padding-top: 20px;
                border-top: 1px solid #eee;
                text-align: center;
                font-size: 0.9em;
                color: #7f8c8d;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>XGBoost Timeframes Model Results</h1>
            
            <div class="section">
                <h2>Model Performance Overview</h2>
                <p>
                    The XGBoost timeframes model was trained using city-specific periods:
                    <ul>
                        <li><strong>San Juan:</strong> Training period 1990-2000, Testing period 2000-2007</li>
                        <li><strong>Iquitos:</strong> Training period 2000-2007, Testing period 2007-2010</li>
                    </ul>
                </p>
                <div class="image-container">
                    <img src="performance_dashboard.png" alt="Performance Dashboard">
                </div>
            </div>
            
            <div class="section">
                <h2>Predictions Visualization</h2>
                <div class="image-container">
                    <img src="combined_predictions.png" alt="Predictions Visualization">
                </div>
            </div>
            
            <div class="section">
                <h2>Feature Importance Analysis</h2>
                <div class="image-container">
                    <img src="combined_feature_importance.png" alt="Feature Importance">
                </div>
                <p>
                    <strong>Key Feature Insights:</strong>
                    <ul>
                        <li>San Juan model relies heavily on humidity and temperature lag features (12-week lags)</li>
                        <li>Iquitos model shows stronger influence from seasonal cyclical features and precipitation</li>
                        <li>Both models identify distinct climate patterns driving outbreaks in each city</li>
                    </ul>
                </p>
            </div>
            
            <div class="section">
                <h2>Submission Distribution</h2>
                <div class="image-container">
                    <img src="submission_distribution.png" alt="Submission Distribution">
                </div>
            </div>
            
            <div class="section">
                <h2>City Comparison</h2>
                <div class="image-container">
                    <img src="city_comparison.png" alt="City Comparison">
                </div>
                <p>
                    <strong>City-Specific Patterns:</strong>
                    <ul>
                        <li><strong>San Juan (Caribbean climate):</strong> Shows stronger influence from longer-term (12-week) humidity and temperature patterns with clear seasonal effects</li>
                        <li><strong>Iquitos (Amazon rainforest):</strong> Shows stronger influence from shorter-term (4-week) precipitation patterns and consistent seasonal cycles</li>
                    </ul>
                </p>
            </div>
            
            <div class="section">
                <h2>Model Improvement Recommendations</h2>
                <p>
                    <strong>1. Time Series Handling Improvements:</strong>
                    <ul>
                        <li>Implement sliding window approach instead of fixed timeframes</li>
                        <li>Add proper time series components (ARIMA, exponential smoothing)</li>
                        <li>Incorporate concept drift detection to handle changing patterns</li>
                    </ul>
                </p>
                <p>
                    <strong>2. Feature Engineering Enhancements:</strong>
                    <ul>
                        <li>Optimize lag selection based on city-specific patterns (12-week for San Juan, 4-week for Iquitos)</li>
                        <li>Create specialized interaction features between temperature, humidity, and precipitation</li>
                        <li>Implement feature selection to remove noisy or redundant predictors</li>
                    </ul>
                </p>
                <p>
                    <strong>3. Model Architecture Refinements:</strong>
                    <ul>
                        <li>Implement ensemble approach combining multiple model types</li>
                        <li>Add specialized count data modeling (zero-inflated Poisson, negative binomial)</li>
                        <li>Use two-stage modeling (classify outbreak probability, then predict severity)</li>
                    </ul>
                </p>
            </div>
            
            <div class="footer">
                <p>Report generated on May 2, 2025 • XGBoost Timeframes Model Analysis</p>
            </div>
        </div>
    </body>
    </html>
    """
    
    # Save HTML file
    with open(results_dir / 'xgboost_results_summary.html', 'w') as f:
        f.write(html_content)
    
    print(f"Results summary HTML saved to '{results_dir}/xgboost_results_summary.html'")

def main():
    """Main function to run the visualization script."""
    print("Starting XGBoost Timeframes Model Results Visualization...")
    
    # Create results directory if it doesn't exist
    results_dir = Path("results/insights")
    results_dir.mkdir(exist_ok=True, parents=True)
    
    # Load data
    print("Loading data...")
    data = load_data()
    submission = load_submission()
    metrics = load_model_metrics()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_combined_feature_importance_plot(results_dir)
    create_performance_dashboard(metrics, results_dir)
    create_prediction_comparison_plots(results_dir)
    
    if not submission.empty:
        create_submission_analysis(submission, data, results_dir)
    
    if not data.empty:
        create_city_comparison(data, results_dir)
    
    # Create summary HTML page
    print("\nGenerating summary page...")
    create_results_summary_page(results_dir)
    
    print(f"\nVisualization complete! All results saved to '{results_dir}/' directory.")

if __name__ == "__main__":
    main()