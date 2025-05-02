"""
Enhanced Evaluation Framework Module

This module provides a comprehensive evaluation framework for dengue prediction models,
with special focus on metrics that penalize missing outbreak peaks.
It also includes visualization tools for comparing model performances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import sys
import json
import glob
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
import shap
import joblib

# Import custom modules
try:
    from src.time_series_validation import OutbreakDetectionMetrics
except ImportError:
    # If the module is not importable, define the class here
    class OutbreakDetectionMetrics:
        """Custom metrics for outbreak detection."""
        
        @staticmethod
        def define_outbreak(y, threshold_percentile=75):
            """Define outbreaks based on a percentile threshold."""
            threshold = np.percentile(y, threshold_percentile)
            return (y > threshold).astype(int)
        
        @staticmethod
        def outbreak_f1_score(y_true, y_pred, threshold_percentile=75):
            """Calculate F1 score for outbreak detection."""
            outbreak_true = OutbreakDetectionMetrics.define_outbreak(y_true, threshold_percentile)
            outbreak_pred = OutbreakDetectionMetrics.define_outbreak(y_pred, threshold_percentile)
            
            return f1_score(outbreak_true, outbreak_pred)
        
        @staticmethod
        def outbreak_weighted_rmse(y_true, y_pred, threshold_percentile=75, penalty_weight=2.0):
            """Calculate weighted RMSE that penalizes missed outbreaks more heavily."""
            outbreak_mask = OutbreakDetectionMetrics.define_outbreak(y_true, threshold_percentile)
            
            # Create weights (higher for outbreak periods)
            weights = np.ones_like(y_true, dtype=float)
            weights[outbreak_mask == 1] = penalty_weight
            
            # Calculate weighted squared errors
            squared_errors = (y_true - y_pred) ** 2
            weighted_squared_errors = squared_errors * weights
            
            # Return weighted RMSE
            return np.sqrt(np.mean(weighted_squared_errors))
        
        @staticmethod
        def outbreak_detection_summary(y_true, y_pred, threshold_percentile=75):
            """Provide a comprehensive summary of outbreak detection metrics."""
            outbreak_true = OutbreakDetectionMetrics.define_outbreak(y_true, threshold_percentile)
            outbreak_pred = OutbreakDetectionMetrics.define_outbreak(y_pred, threshold_percentile)
            
            # Basic metrics
            f1 = f1_score(outbreak_true, outbreak_pred)
            weighted_rmse = OutbreakDetectionMetrics.outbreak_weighted_rmse(
                y_true, y_pred, threshold_percentile
            )
            
            # Standard metrics during outbreak periods only
            outbreak_mask = (outbreak_true == 1)
            
            if np.sum(outbreak_mask) > 0:
                outbreak_rmse = np.sqrt(mean_squared_error(
                    y_true[outbreak_mask], y_pred[outbreak_mask]
                ))
                outbreak_mae = mean_absolute_error(
                    y_true[outbreak_mask], y_pred[outbreak_mask]
                )
            else:
                outbreak_rmse = np.nan
                outbreak_mae = np.nan
            
            return {
                'outbreak_f1': f1,
                'weighted_rmse': weighted_rmse,
                'outbreak_rmse': outbreak_rmse,
                'outbreak_mae': outbreak_mae
            }

# Suppress warnings
warnings.filterwarnings('ignore')

# Debugging setup
DEBUG = True

def debug_print(message):
    """Print debug messages if DEBUG is True."""
    if DEBUG:
        print(f"[DEBUG] {message}")

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
debug_print(f"Added {project_root} to Python path")

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def load_submission_files():
    """
    Load all submission files from the processed directory.
    
    Returns:
        dict: Dictionary of submission DataFrames
    """
    print_section("Loading Submission Files")
    
    # Initialize data paths
    data_dir = Path("data/processed")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    # Find all submission files
    submission_files = glob.glob(str(data_dir / "submission_*.csv"))
    
    if not submission_files:
        raise FileNotFoundError(f"No submission files found in {data_dir}")
    
    # Load submissions
    submissions = {}
    
    for file_path in submission_files:
        file_name = Path(file_path).name
        model_name = file_name.replace("submission_", "").replace(".csv", "")
        
        debug_print(f"Loading submission for {model_name}...")
        submissions[model_name] = pd.read_csv(file_path)
    
    print(f"Loaded {len(submissions)} submission files.")
    
    return submissions

def load_validation_data():
    """
    Load validation data for evaluation.
    
    Returns:
        pd.DataFrame: Validation data with ground truth
    """
    print_section("Loading Validation Data")
    
    # Initialize data paths
    data_dir = Path("data/processed")
    
    # Define file path for validation data
    validation_path = data_dir / "validation_data.csv"
    
    # Check if validation data exists
    if not validation_path.exists():
        print("Validation data not found. Creating validation set from training data...")
        
        # Load training data
        train_path = data_dir / "enhanced_featured_train_data.csv"
        
        if not train_path.exists():
            train_path = data_dir / "featured_train_data.csv"
        
        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at {train_path}")
        
        train_data = pd.read_csv(train_path)
        
        # Create validation set (last 26 weeks of data)
        train_data['week_start_date'] = pd.to_datetime(train_data['week_start_date'])
        
        # Sort by date
        train_data = train_data.sort_values(['city', 'week_start_date'])
        
        # Create validation set for each city
        validation_data = []
        
        for city in train_data['city'].unique():
            city_data = train_data[train_data['city'] == city]
            # Take last 26 weeks (half a year) as validation
            city_validation = city_data.iloc[-26:]
            validation_data.append(city_validation)
        
        # Combine validation data
        validation_data = pd.concat(validation_data, ignore_index=True)
        
        # Save validation data
        validation_data.to_csv(validation_path, index=False)
        
        print(f"Created validation set with {len(validation_data)} rows.")
    else:
        # Load existing validation data
        validation_data = pd.read_csv(validation_path)
        print(f"Loaded validation data with {len(validation_data)} rows.")
    
    return validation_data

def evaluate_model(submission, validation_data, model_name):
    """
    Evaluate a model's performance against validation data.
    
    Args:
        submission (pd.DataFrame): Model predictions
        validation_data (pd.DataFrame): Validation data with ground truth
        model_name (str): Name of the model
    
    Returns:
        dict: Evaluation metrics
    """
    print_section(f"Evaluating {model_name}")
    
    # Merge submission with validation data
    merged = pd.merge(
        validation_data[['city', 'year', 'weekofyear', 'total_cases']],
        submission[['city', 'year', 'weekofyear', 'total_cases']],
        on=['city', 'year', 'weekofyear'],
        suffixes=('_true', '_pred')
    )
    
    # Calculate standard metrics
    rmse = np.sqrt(mean_squared_error(merged['total_cases_true'], merged['total_cases_pred']))
    mae = mean_absolute_error(merged['total_cases_true'], merged['total_cases_pred'])
    r2 = r2_score(merged['total_cases_true'], merged['total_cases_pred'])
    
    # Calculate outbreak-specific metrics
    outbreak_metrics = OutbreakDetectionMetrics.outbreak_detection_summary(
        merged['total_cases_true'], merged['total_cases_pred']
    )
    
    # Combine metrics
    metrics = {
        'model': model_name,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        **outbreak_metrics,
        'mean_absolute_percentage_error': np.mean(np.abs((merged['total_cases_true'] - merged['total_cases_pred']) / 
                                             np.maximum(merged['total_cases_true'], 1))) * 100,
        'dengue_competition_metric': np.mean(np.abs((merged['total_cases_true'] - merged['total_cases_pred']) / 
                                             np.maximum(merged['total_cases_true'], 1)))
    }
    
    # Calculate city-specific metrics
    city_metrics = {}
    
    for city in merged['city'].unique():
        city_data = merged[merged['city'] == city]
        
        city_metrics[city] = {
            'rmse': np.sqrt(mean_squared_error(city_data['total_cases_true'], city_data['total_cases_pred'])),
            'mae': mean_absolute_error(city_data['total_cases_true'], city_data['total_cases_pred']),
            'r2': r2_score(city_data['total_cases_true'], city_data['total_cases_pred']),
            **OutbreakDetectionMetrics.outbreak_detection_summary(
                city_data['total_cases_true'], city_data['total_cases_pred']
            ),
            'mean_absolute_percentage_error': np.mean(np.abs((city_data['total_cases_true'] - city_data['total_cases_pred']) / 
                                                 np.maximum(city_data['total_cases_true'], 1))) * 100,
            'dengue_competition_metric': np.mean(np.abs((city_data['total_cases_true'] - city_data['total_cases_pred']) / 
                                              np.maximum(city_data['total_cases_true'], 1)))
        }
    
    # Print metrics
    print(f"Overall Metrics:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    print(f"Outbreak F1: {metrics['outbreak_f1']:.2f}")
    print(f"Weighted RMSE: {metrics['weighted_rmse']:.2f}")
    print(f"Outbreak RMSE: {metrics['outbreak_rmse']:.2f}")
    print(f"Dengue Competition Metric: {metrics['dengue_competition_metric']:.2f}")
    
    # Print city-specific metrics
    for city, city_metric in city_metrics.items():
        print(f"\nMetrics for {city}:")
        print(f"RMSE: {city_metric['rmse']:.2f}")
        print(f"MAE: {city_metric['mae']:.2f}")
        print(f"R²: {city_metric['r2']:.2f}")
        print(f"Outbreak F1: {city_metric['outbreak_f1']:.2f}")
        print(f"Dengue Competition Metric: {city_metric['dengue_competition_metric']:.2f}")
    
    # Create merged data for plotting
    metrics['merged_data'] = merged
    metrics['city_metrics'] = city_metrics
    
    return metrics

def plot_predictions_vs_actual(metrics, model_name):
    """
    Plot predictions vs. actual values.
    
    Args:
        metrics (dict): Evaluation metrics with merged data
        model_name (str): Name of the model
    
    Returns:
        tuple: (figure, axes)
    """
    merged = metrics['merged_data']
    
    # Create figure
    fig, axes = plt.subplots(len(merged['city'].unique()), 1, figsize=(12, 5 * len(merged['city'].unique())))
    
    # Convert to list if only one city
    if len(merged['city'].unique()) == 1:
        axes = [axes]
    
    # Plot for each city
    for i, city in enumerate(sorted(merged['city'].unique())):
        city_data = merged[merged['city'] == city].copy()
        
        # Create a date field for better plotting
        city_data['date'] = pd.to_datetime(city_data['year'].astype(str) + '-' + 
                                        city_data['weekofyear'].astype(str) + '-1', format='%Y-%W-%w')
        city_data = city_data.sort_values('date')
        
        # Plot actual and predicted values
        axes[i].plot(city_data['date'], city_data['total_cases_true'], 'b-', label='Actual')
        axes[i].plot(city_data['date'], city_data['total_cases_pred'], 'r--', label='Predicted')
        
        # Add outbreak indicators
        outbreak_mask = OutbreakDetectionMetrics.define_outbreak(city_data['total_cases_true'])
        outbreak_dates = city_data.loc[outbreak_mask == 1, 'date']
        outbreak_cases = city_data.loc[outbreak_mask == 1, 'total_cases_true']
        
        axes[i].scatter(outbreak_dates, outbreak_cases, c='purple', s=100, alpha=0.6, label='Outbreaks')
        
        # Add labels and title
        axes[i].set_title(f'{city} - Actual vs. Predicted Cases')
        axes[i].set_xlabel('Date')
        axes[i].set_ylabel('Total Cases')
        axes[i].legend()
        axes[i].grid(True, alpha=0.3)
        
        # Set date formatting
        fig.autofmt_xdate()
        
        # Add metrics as text
        city_metric = metrics['city_metrics'][city]
        metrics_text = '\n'.join([
            f'RMSE: {city_metric["rmse"]:.2f}',
            f'MAE: {city_metric["mae"]:.2f}',
            f'R²: {city_metric["r2"]:.2f}',
            f'Outbreak F1: {city_metric["outbreak_f1"]:.2f}',
            f'Weighted RMSE: {city_metric["weighted_rmse"]:.2f}'
        ])
        
        axes[i].text(0.02, 0.95, metrics_text, transform=axes[i].transAxes,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
    
    # Add overall title
    fig.suptitle(f'Model: {model_name}', fontsize=16)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    
    return fig, axes

def compare_models(all_metrics):
    """
    Compare models based on evaluation metrics.
    
    Args:
        all_metrics (dict): Dictionary of evaluation metrics for each model
    
    Returns:
        pd.DataFrame: Comparison DataFrame
    """
    print_section("Comparing Models")
    
    # Initialize comparison data
    comparison_data = []
    
    # Add metrics for each model
    for model_name, metrics in all_metrics.items():
        comparison_data.append({
            'model': model_name,
            'rmse': metrics['rmse'],
            'mae': metrics['mae'],
            'r2': metrics['r2'],
            'outbreak_f1': metrics['outbreak_f1'],
            'weighted_rmse': metrics['weighted_rmse'],
            'outbreak_rmse': metrics['outbreak_rmse'],
            'outbreak_mae': metrics['outbreak_mae'],
            'dengue_competition_metric': metrics['dengue_competition_metric']
        })
    
    # Convert to DataFrame
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by outbreak F1 score (descending)
    comparison_df = comparison_df.sort_values('outbreak_f1', ascending=False)
    
    # Print comparison table
    print(comparison_df)
    
    return comparison_df

def plot_model_comparison(comparison_df):
    """
    Plot model comparison.
    
    Args:
        comparison_df (pd.DataFrame): Comparison DataFrame
    
    Returns:
        tuple: (figure, axes)
    """
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    # Plot RMSE and MAE
    metrics = ['rmse', 'mae', 'outbreak_f1', 'dengue_competition_metric']
    titles = ['RMSE (lower is better)', 'MAE (lower is better)', 
              'Outbreak F1 Score (higher is better)', 'Dengue Competition Metric (lower is better)']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        # Sort by the current metric
        if metric in ['outbreak_f1', 'r2']:
            # Higher is better for these metrics
            sorted_df = comparison_df.sort_values(metric, ascending=False)
            bar_color = 'green'
        else:
            # Lower is better for these metrics
            sorted_df = comparison_df.sort_values(metric)
            bar_color = 'skyblue'
        
        # Plot bar chart
        axes[i].bar(sorted_df['model'], sorted_df[metric], color=bar_color)
        axes[i].set_title(title)
        axes[i].set_xlabel('Model')
        axes[i].set_ylabel(metric.upper())
        
        # Add data labels
        for j, v in enumerate(sorted_df[metric]):
            axes[i].text(j, v + 0.05 * max(sorted_df[metric]), f'{v:.2f}', 
                       ha='center', va='bottom', rotation=0)
        
        # Rotate x-axis labels
        axes[i].set_xticklabels(sorted_df['model'], rotation=45, ha='right')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, axes

def analyze_feature_importance(all_models):
    """
    Analyze feature importance across models.
    
    Args:
        all_models (dict): Dictionary of trained models
    
    Returns:
        dict: Feature importance analysis
    """
    print_section("Analyzing Feature Importance")
    
    # Check if any models are available
    if not all_models:
        print("No models available for feature importance analysis.")
        return None
    
    feature_importance = {}
    
    # Analyze each model
    for model_name, model_dict in all_models.items():
        print(f"Analyzing {model_name}...")
        
        # Skip if feature importance is not available
        if 'feature_importance' not in model_dict:
            continue
        
        # Get feature importance
        feature_importance[model_name] = model_dict['feature_importance']
    
    # Combine feature importance across models
    combined_importance = pd.DataFrame()
    
    for model_name, importance_df in feature_importance.items():
        # Rename importance column
        importance_df = importance_df.rename(columns={'importance': model_name})
        
        # Merge with combined importance
        if combined_importance.empty:
            combined_importance = importance_df
        else:
            combined_importance = pd.merge(
                combined_importance, 
                importance_df, 
                on='feature', 
                how='outer'
            )
    
    # Fill NaN values with 0
    combined_importance = combined_importance.fillna(0)
    
    # Calculate average importance
    if len(feature_importance) > 0:
        combined_importance['avg_importance'] = combined_importance.drop('feature', axis=1).mean(axis=1)
        
        # Sort by average importance
        combined_importance = combined_importance.sort_values('avg_importance', ascending=False)
    
    # Print top features
    print("Top 20 Features:")
    print(combined_importance.head(20))
    
    return combined_importance

def plot_feature_importance(combined_importance):
    """
    Plot feature importance.
    
    Args:
        combined_importance (pd.DataFrame): Combined feature importance
    
    Returns:
        tuple: (figure, axes)
    """
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Get top 20 features
    top_features = combined_importance.head(20)
    
    # Create bar plot
    sns.barplot(x='avg_importance', y='feature', data=top_features, ax=ax)
    
    # Add labels and title
    ax.set_title('Top 20 Features by Average Importance')
    ax.set_xlabel('Average Importance')
    ax.set_ylabel('Feature')
    
    # Adjust layout
    plt.tight_layout()
    
    return fig, ax

def analyze_shap_values(model, X_train, model_name):
    """
    Analyze SHAP values for a model.
    
    Args:
        model: Trained model
        X_train (pd.DataFrame): Training data
        model_name (str): Name of the model
    
    Returns:
        tuple: (figure, explainer)
    """
    print_section(f"Analyzing SHAP Values for {model_name}")
    
    try:
        # Create SHAP explainer based on model type
        if hasattr(model, 'predict_proba'):
            explainer = shap.Explainer(model, X_train)
        else:
            explainer = shap.Explainer(model)
        
        # Calculate SHAP values
        shap_values = explainer(X_train)
        
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot SHAP summary
        shap.summary_plot(shap_values, X_train, show=False)
        
        # Add title
        plt.title(f'SHAP Values for {model_name}')
        
        # Adjust layout
        plt.tight_layout()
        
        return fig, explainer
    except Exception as e:
        print(f"Error calculating SHAP values: {e}")
        return None, None

def save_evaluation_results(all_metrics, comparison_df, combined_importance=None):
    """
    Save evaluation results.
    
    Args:
        all_metrics (dict): Dictionary of evaluation metrics for each model
        comparison_df (pd.DataFrame): Comparison DataFrame
        combined_importance (pd.DataFrame, optional): Combined feature importance
    """
    print_section("Saving Evaluation Results")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Create evaluation directory
    eval_dir = processed_dir / "evaluation"
    eval_dir.mkdir(exist_ok=True)
    
    # Save comparison DataFrame
    comparison_df.to_csv(eval_dir / "model_comparison.csv", index=False)
    
    # Save combined feature importance if available
    if combined_importance is not None:
        combined_importance.to_csv(eval_dir / "feature_importance.csv", index=False)
    
    # Save detailed metrics for each model
    for model_name, metrics in all_metrics.items():
        # Create model directory
        model_dir = eval_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save metrics
        metrics_copy = metrics.copy()
        
        # Remove non-serializable items
        if 'merged_data' in metrics_copy:
            del metrics_copy['merged_data']
        
        # Save metrics as JSON
        with open(model_dir / "metrics.json", 'w') as f:
            json.dump(metrics_copy, f, indent=4)
        
        # Create and save plot
        if 'merged_data' in metrics:
            fig, _ = plot_predictions_vs_actual(metrics, model_name)
            fig.savefig(model_dir / "predictions_vs_actual.png")
            plt.close(fig)
    
    # Create and save model comparison plot
    fig, _ = plot_model_comparison(comparison_df)
    fig.savefig(eval_dir / "model_comparison.png")
    plt.close(fig)
    
    # Save feature importance plot if available
    if combined_importance is not None:
        fig, _ = plot_feature_importance(combined_importance)
        fig.savefig(eval_dir / "feature_importance.png")
        plt.close(fig)
    
    print(f"Evaluation results saved to {eval_dir}")

def main():
    """Main function to run the enhanced evaluation framework."""
    print_section("Starting Enhanced Evaluation Framework")
    
    # Load all submission files
    submissions = load_submission_files()
    
    # Load validation data
    validation_data = load_validation_data()
    
    # Evaluate each model
    all_metrics = {}
    
    for model_name, submission in submissions.items():
        metrics = evaluate_model(submission, validation_data, model_name)
        all_metrics[model_name] = metrics
    
    # Compare models
    comparison_df = compare_models(all_metrics)
    
    # Try to load trained models for feature importance analysis
    try:
        trained_models = {}
        models_dir = Path("models")
        
        # Look for model files
        model_files = glob.glob(str(models_dir / "*.joblib"))
        
        for model_file in model_files:
            try:
                model_name = Path(model_file).stem
                print(f"Loading model {model_name}...")
                trained_models[model_name] = joblib.load(model_file)
            except Exception as e:
                print(f"Error loading model {model_file}: {e}")
        
        # Analyze feature importance if models are available
        if trained_models:
            combined_importance = analyze_feature_importance(trained_models)
        else:
            combined_importance = None
    except Exception as e:
        print(f"Error loading trained models: {e}")
        combined_importance = None
    
    # Save evaluation results
    save_evaluation_results(all_metrics, comparison_df, combined_importance)
    
    print_section("Evaluation Complete")
    print("1. Review evaluation metrics")
    print("2. Compare model performances")
    print("3. Identify best model(s) based on outbreak detection metrics")
    print("4. Analyze feature importance to guide further improvements")

if __name__ == "__main__":
    main()