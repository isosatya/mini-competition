"""
Time Series Validation Module

This module provides functions for proper time series validation,
taking special care to prevent data leakage and simulate real forecasting scenarios.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, f1_score
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import os
import sys
import warnings

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

class OutbreakDetectionMetrics:
    """
    Custom metrics for outbreak detection.
    
    This class implements metrics that are specifically designed to evaluate
    a model's ability to detect and accurately predict outbreaks.
    """
    
    @staticmethod
    def define_outbreak(y, threshold_percentile=75):
        """
        Define outbreaks based on a percentile threshold.
        
        Args:
            y (pd.Series or np.array): Case counts
            threshold_percentile (float): Percentile above which counts are considered outbreaks
        
        Returns:
            np.array: Binary array where 1 indicates an outbreak
        """
        threshold = np.percentile(y, threshold_percentile)
        return (y > threshold).astype(int)
    
    @staticmethod
    def outbreak_f1_score(y_true, y_pred, threshold_percentile=75):
        """
        Calculate F1 score for outbreak detection.
        
        Args:
            y_true (pd.Series or np.array): Actual case counts
            y_pred (pd.Series or np.array): Predicted case counts
            threshold_percentile (float): Percentile above which counts are considered outbreaks
        
        Returns:
            float: F1 score for outbreak detection
        """
        outbreak_true = OutbreakDetectionMetrics.define_outbreak(y_true, threshold_percentile)
        outbreak_pred = OutbreakDetectionMetrics.define_outbreak(y_pred, threshold_percentile)
        
        return f1_score(outbreak_true, outbreak_pred)
    
    @staticmethod
    def outbreak_weighted_rmse(y_true, y_pred, threshold_percentile=75, penalty_weight=2.0):
        """
        Calculate weighted RMSE that penalizes missed outbreaks more heavily.
        
        Args:
            y_true (pd.Series or np.array): Actual case counts
            y_pred (pd.Series or np.array): Predicted case counts
            threshold_percentile (float): Percentile above which counts are considered outbreaks
            penalty_weight (float): How much to penalize errors during outbreaks
        
        Returns:
            float: Weighted RMSE
        """
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
        """
        Provide a comprehensive summary of outbreak detection metrics.
        
        Args:
            y_true (pd.Series or np.array): Actual case counts
            y_pred (pd.Series or np.array): Predicted case counts
            threshold_percentile (float): Percentile above which counts are considered outbreaks
        
        Returns:
            dict: Dictionary of outbreak detection metrics
        """
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

class TimeSeriesValidator:
    """
    Class for time series validation that prevents data leakage.
    """
    
    def __init__(self, n_splits=5, test_size=None, gap=0):
        """
        Initialize the validator.
        
        Args:
            n_splits (int): Number of splits for time series cross-validation
            test_size (int, optional): Size of the test set
            gap (int, optional): Gap between train and test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        
        # Initialize cross-validator
        self.cv = TimeSeriesSplit(
            n_splits=n_splits,
            test_size=test_size,
            gap=gap
        )
    
    def get_splits(self, X, y, groups=None):
        """
        Generate train/test splits.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            groups (pd.Series, optional): Groups for grouped time series
        
        Returns:
            generator: Train/test indices
        """
        if groups is None:
            return self.cv.split(X)
        else:
            # Implement grouped time series split
            unique_groups = np.unique(groups)
            
            for train_idx, test_idx in self.cv.split(unique_groups):
                train_groups = unique_groups[train_idx]
                test_groups = unique_groups[test_idx]
                
                train_mask = np.isin(groups, train_groups)
                test_mask = np.isin(groups, test_groups)
                
                yield np.where(train_mask)[0], np.where(test_mask)[0]
    
    def validate(self, X, y, model, groups=None, fit_params=None, predict_params=None):
        """
        Validate a model using time series cross-validation.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            model: Model with fit and predict methods
            groups (pd.Series, optional): Groups for grouped time series
            fit_params (dict, optional): Additional parameters for model.fit
            predict_params (dict, optional): Additional parameters for model.predict
        
        Returns:
            dict: Validation results
        """
        if fit_params is None:
            fit_params = {}
        
        if predict_params is None:
            predict_params = {}
        
        # Initialize results
        results = {
            'fold': [],
            'train_size': [],
            'test_size': [],
            'rmse': [],
            'mae': [],
            'r2': [],
            'outbreak_f1': [],
            'weighted_rmse': [],
            'outbreak_rmse': [],
            'outbreak_mae': []
        }
        
        # Validate for each fold
        for fold, (train_idx, test_idx) in enumerate(self.get_splits(X, y, groups)):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            model.fit(X_train, y_train, **fit_params)
            
            # Make predictions
            y_pred = model.predict(X_test, **predict_params)
            
            # Calculate standard metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Calculate outbreak metrics
            outbreak_metrics = OutbreakDetectionMetrics.outbreak_detection_summary(
                y_test, y_pred
            )
            
            # Store results
            results['fold'].append(fold)
            results['train_size'].append(len(train_idx))
            results['test_size'].append(len(test_idx))
            results['rmse'].append(rmse)
            results['mae'].append(mae)
            results['r2'].append(r2)
            results['outbreak_f1'].append(outbreak_metrics['outbreak_f1'])
            results['weighted_rmse'].append(outbreak_metrics['weighted_rmse'])
            results['outbreak_rmse'].append(outbreak_metrics['outbreak_rmse'])
            results['outbreak_mae'].append(outbreak_metrics['outbreak_mae'])
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Calculate average metrics
        avg_metrics = {
            'rmse': results_df['rmse'].mean(),
            'mae': results_df['mae'].mean(),
            'r2': results_df['r2'].mean(),
            'outbreak_f1': results_df['outbreak_f1'].mean(),
            'weighted_rmse': results_df['weighted_rmse'].mean(),
            'outbreak_rmse': results_df['outbreak_rmse'].mean(),
            'outbreak_mae': results_df['outbreak_mae'].mean()
        }
        
        return {
            'fold_results': results_df,
            'avg_metrics': avg_metrics
        }
    
    def plot_validation_results(self, results, title=None, figsize=(14, 6)):
        """
        Plot validation results.
        
        Args:
            results (dict): Validation results from validate()
            title (str, optional): Plot title
            figsize (tuple, optional): Figure size
        """
        results_df = results['fold_results']
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Plot standard metrics
        results_df[['rmse', 'mae']].plot(ax=axes[0])
        axes[0].set_title('Standard Metrics by Fold')
        axes[0].set_xlabel('Fold')
        axes[0].set_ylabel('Value')
        axes[0].grid(True)
        
        # Plot outbreak metrics
        results_df[['outbreak_f1', 'weighted_rmse', 'outbreak_rmse']].plot(ax=axes[1])
        axes[1].set_title('Outbreak Metrics by Fold')
        axes[1].set_xlabel('Fold')
        axes[1].set_ylabel('Value')
        axes[1].grid(True)
        
        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=16)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
        else:
            fig.tight_layout()
        
        return fig

class BlockingTimeSeriesSplit:
    """
    Custom time series split that respects grouped time series.
    This is especially useful for multi-city dengue prediction where
    we want to keep cities together in train/test splits.
    """
    
    def __init__(self, n_splits=5, test_size=None):
        """
        Initialize the splitter.
        
        Args:
            n_splits (int): Number of splits
            test_size (int, optional): Size of the test set
        """
        self.n_splits = n_splits
        self.test_size = test_size
    
    def split(self, X, y=None, groups=None):
        """
        Generate indices to split data into training and test sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series, optional): Target
            groups (pd.Series): Group labels (e.g., city)
        
        Returns:
            generator: Train/test indices
        """
        # Validate inputs
        if groups is None:
            raise ValueError("Groups must be provided for BlockingTimeSeriesSplit")
        
        # Get unique time points and groups
        unique_times = np.unique(X.index.get_level_values(0))
        unique_groups = np.unique(groups)
        
        # Determine test size
        if self.test_size is None:
            test_size = len(unique_times) // self.n_splits
        else:
            test_size = self.test_size
        
        # Generate splits
        for i in range(self.n_splits):
            # Calculate test start and end indices
            test_end = len(unique_times) - i * test_size
            test_start = test_end - test_size
            
            if test_start < 0:
                break
            
            # Get training and test time points
            test_times = unique_times[test_start:test_end]
            train_times = unique_times[:test_start]
            
            # Create masks for training and test sets
            train_mask = np.isin(X.index.get_level_values(0), train_times)
            test_mask = np.isin(X.index.get_level_values(0), test_times)
            
            # Get indices
            train_indices = np.where(train_mask)[0]
            test_indices = np.where(test_mask)[0]
            
            yield train_indices, test_indices

def forecast_evaluation(y_true, y_pred, plot=True, figsize=(12, 6), title=None):
    """
    Evaluate forecast accuracy and visualize results.
    
    Args:
        y_true (pd.Series): Actual values
        y_pred (pd.Series): Predicted values
        plot (bool): Whether to create visualizations
        figsize (tuple): Figure size for plots
        title (str, optional): Plot title
    
    Returns:
        dict: Evaluation metrics
    """
    # Calculate standard metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Calculate outbreak metrics
    outbreak_metrics = OutbreakDetectionMetrics.outbreak_detection_summary(y_true, y_pred)
    
    # Combine metrics
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        **outbreak_metrics
    }
    
    # Create visualization if requested
    if plot:
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot actual and predicted values
        ax.plot(y_true.index, y_true.values, label='Actual', color='black')
        ax.plot(y_true.index, y_pred.values, label='Predicted', color='blue', linestyle='--')
        
        # Add outbreak indicators
        outbreak_mask = OutbreakDetectionMetrics.define_outbreak(y_true)
        outbreak_indices = y_true.index[outbreak_mask == 1]
        
        if len(outbreak_indices) > 0:
            ax.scatter(outbreak_indices, y_true.loc[outbreak_indices], 
                       color='red', marker='o', s=50, label='Outbreaks')
        
        # Add metrics text
        metrics_text = '\n'.join([
            f'RMSE: {rmse:.2f}',
            f'MAE: {mae:.2f}',
            f'R²: {r2:.2f}',
            f'Outbreak F1: {outbreak_metrics["outbreak_f1"]:.2f}',
            f'Weighted RMSE: {outbreak_metrics["weighted_rmse"]:.2f}'
        ])
        
        ax.text(0.02, 0.95, metrics_text, transform=ax.transAxes,
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
        
        # Add labels and title
        ax.set_xlabel('Time')
        ax.set_ylabel('Case Count')
        
        if title:
            ax.set_title(title)
        else:
            ax.set_title('Forecast Evaluation')
        
        # Add legend
        ax.legend()
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        metrics['figure'] = fig
    
    return metrics

def validate_model_with_expanding_window(X, y, model, initial_window=52, step=4, max_windows=10,
                                       fit_params=None, predict_params=None, city=None):
    """
    Validate a model using an expanding window approach.
    
    Args:
        X (pd.DataFrame): Features
        y (pd.Series): Target
        model: Model with fit and predict methods
        initial_window (int): Initial training window size (in weeks)
        step (int): Step size for expanding window (in weeks)
        max_windows (int): Maximum number of windows to evaluate
        fit_params (dict, optional): Additional parameters for model.fit
        predict_params (dict, optional): Additional parameters for model.predict
        city (str, optional): City name for plot titles
    
    Returns:
        dict: Validation results
    """
    if fit_params is None:
        fit_params = {}
    
    if predict_params is None:
        predict_params = {}
    
    # Ensure data is ordered by time
    if not isinstance(X.index, pd.DatetimeIndex) and 'week_start_date' in X.columns:
        X = X.set_index('week_start_date')
        y = y.set_index(X.index)
    
    # Initialize results
    results = {
        'window': [],
        'train_size': [],
        'test_size': [],
        'rmse': [],
        'mae': [],
        'r2': [],
        'outbreak_f1': [],
        'weighted_rmse': [],
        'outbreak_rmse': [],
        'outbreak_mae': []
    }
    
    # Get total length
    n = len(X)
    
    # Validate for each window
    for i in range(max_windows):
        # Calculate train/test split point
        train_end = initial_window + i * step
        
        if train_end >= n - step:
            break
        
        test_end = min(train_end + step, n)
        
        # Split data
        X_train, X_test = X.iloc[:train_end], X.iloc[train_end:test_end]
        y_train, y_test = y.iloc[:train_end], y.iloc[train_end:test_end]
        
        # Fit model
        model.fit(X_train, y_train, **fit_params)
        
        # Make predictions
        y_pred = model.predict(X_test, **predict_params)
        
        # Calculate standard metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate outbreak metrics
        outbreak_metrics = OutbreakDetectionMetrics.outbreak_detection_summary(
            y_test, y_pred
        )
        
        # Store results
        results['window'].append(i + 1)
        results['train_size'].append(len(X_train))
        results['test_size'].append(len(X_test))
        results['rmse'].append(rmse)
        results['mae'].append(mae)
        results['r2'].append(r2)
        results['outbreak_f1'].append(outbreak_metrics['outbreak_f1'])
        results['weighted_rmse'].append(outbreak_metrics['weighted_rmse'])
        results['outbreak_rmse'].append(outbreak_metrics['outbreak_rmse'])
        results['outbreak_mae'].append(outbreak_metrics['outbreak_mae'])
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Calculate average metrics
    avg_metrics = {
        'rmse': results_df['rmse'].mean(),
        'mae': results_df['mae'].mean(),
        'r2': results_df['r2'].mean(),
        'outbreak_f1': results_df['outbreak_f1'].mean(),
        'weighted_rmse': results_df['weighted_rmse'].mean(),
        'outbreak_rmse': results_df['outbreak_rmse'].mean(),
        'outbreak_mae': results_df['outbreak_mae'].mean()
    }
    
    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot standard metrics
    results_df[['rmse', 'mae']].plot(ax=axes[0])
    axes[0].set_title('Standard Metrics by Window')
    axes[0].set_xlabel('Window')
    axes[0].set_ylabel('Value')
    axes[0].grid(True)
    
    # Plot outbreak metrics
    results_df[['outbreak_f1', 'weighted_rmse']].plot(ax=axes[1])
    axes[1].set_title('Outbreak Metrics by Window')
    axes[1].set_xlabel('Window')
    axes[1].set_ylabel('Value')
    axes[1].grid(True)
    
    # Add title
    if city:
        fig.suptitle(f'Expanding Window Validation for {city}', fontsize=16)
    else:
        fig.suptitle('Expanding Window Validation', fontsize=16)
    
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    
    return {
        'fold_results': results_df,
        'avg_metrics': avg_metrics,
        'figure': fig
    }

def save_validation_results(results, model_name, city=None):
    """
    Save validation results to file.
    
    Args:
        results (dict): Validation results
        model_name (str): Name of the model
        city (str, optional): City name
    """
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Create validation directory
    validation_dir = models_dir / "validation"
    validation_dir.mkdir(exist_ok=True)
    
    # Create base filename
    if city:
        base_filename = f"{model_name}_{city}_validation"
    else:
        base_filename = f"{model_name}_validation"
    
    # Save results DataFrame
    results['fold_results'].to_csv(validation_dir / f"{base_filename}_results.csv", index=False)
    
    # Save average metrics
    with open(validation_dir / f"{base_filename}_metrics.txt", 'w') as f:
        f.write(f"Validation Metrics for {model_name}")
        if city:
            f.write(f" ({city})")
        f.write("\n\n")
        
        for metric, value in results['avg_metrics'].items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save figure if available
    if 'figure' in results:
        results['figure'].savefig(validation_dir / f"{base_filename}_plot.png")
        plt.close(results['figure'])

def main():
    """Example usage of time series validation."""
    print_section("Time Series Validation Example")
    
    # This function would normally be used by other modules
    print("This module provides time series validation functions for other modules.")
    print("To use it, import it in your model training script and call the appropriate functions.")
    print("\nExample usage:")
    print("from time_series_validation import TimeSeriesValidator, save_validation_results")
    print("validator = TimeSeriesValidator(n_splits=5)")
    print("results = validator.validate(X, y, model)")
    print("save_validation_results(results, 'my_model', 'my_city')")

if __name__ == "__main__":
    main()