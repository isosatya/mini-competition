"""
Enhanced Data Preprocessing Module

This module provides improved preprocessing methods for the Dengue Fever prediction data,
including better imputation, outlier handling, and stationarity transformations.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import sys
import warnings
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from scipy import stats
from statsmodels.tsa.stattools import adfuller

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

def load_raw_data():
    """
    Load the raw training and test data.
    
    Returns:
        tuple: (train_data, test_features)
    
    Raises:
        FileNotFoundError: If raw data files are missing
    """
    debug_print("Starting raw data loading")
    
    # Initialize data paths
    data_dir = Path("data/raw")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Raw data directory not found: {data_dir}")
    
    # Define file paths
    train_path = data_dir / "dengue_features_train.csv"
    labels_path = data_dir / "dengue_labels_train.csv"
    test_path = data_dir / "dengue_features_test.csv"
    
    # Check if files exist
    for path in [train_path, labels_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    debug_print("Loading raw training features...")
    train_features = pd.read_csv(train_path)
    debug_print(f"Training features loaded. Shape: {train_features.shape}")
    
    debug_print("Loading training labels...")
    train_labels = pd.read_csv(labels_path)
    debug_print(f"Training labels loaded. Shape: {train_labels.shape}")
    
    debug_print("Loading test features...")
    test_features = pd.read_csv(test_path)
    debug_print(f"Test features loaded. Shape: {test_features.shape}")
    
    # Merge training features and labels
    train_data = pd.merge(
        train_features,
        train_labels,
        on=['city', 'year', 'weekofyear'],
        how='left'
    )
    
    debug_print(f"Merged training data. Shape: {train_data.shape}")
    
    return train_data, test_features

def analyze_data(df):
    """
    Analyze data to identify missing values, outliers, and other issues.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    
    Returns:
        dict: Analysis results
    """
    print_section("Data Analysis")
    
    # Create analysis dictionary
    analysis = {}
    
    # Basic information
    analysis['shape'] = df.shape
    analysis['dtypes'] = df.dtypes
    
    # Missing values
    missing = df.isnull().sum()
    analysis['missing'] = missing
    analysis['missing_percent'] = (missing / len(df)) * 100
    
    # Print missing values
    print("Missing Values Analysis:")
    missing_df = pd.DataFrame({
        'Count': missing,
        'Percent': (missing / len(df)) * 100
    }).sort_values('Count', ascending=False)
    
    print(missing_df[missing_df['Count'] > 0])
    
    # Check for duplicate rows
    duplicates = df.duplicated().sum()
    analysis['duplicates'] = duplicates
    print(f"\nDuplicate rows: {duplicates}")
    
    # Analyze numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    
    # Calculate summary statistics
    analysis['numeric_summary'] = df[numeric_cols].describe()
    
    # Print summary statistics
    print("\nNumeric Column Summary:")
    print(analysis['numeric_summary'])
    
    # Create a processed directory to store analysis plots
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Create plots directory
    plots_dir = processed_dir / "plots"
    plots_dir.mkdir(exist_ok=True)
    
    # Plot histograms for selected columns (skip date-related columns)
    skip_cols = ['year', 'weekofyear']
    plot_cols = [col for col in numeric_cols if col not in skip_cols]
    
    # Plot histograms in a grid
    if len(plot_cols) > 0:
        rows = (len(plot_cols) + 2) // 3  # 3 columns per row
        fig, axes = plt.subplots(rows, 3, figsize=(15, rows * 4))
        axes = axes.flatten()
        
        for i, col in enumerate(plot_cols):
            df[col].hist(ax=axes[i], bins=30)
            axes[i].set_title(f'Histogram: {col}')
            axes[i].set_xlabel(col)
            axes[i].set_ylabel('Frequency')
            
        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
            
        plt.tight_layout()
        plt.savefig(plots_dir / "histograms.png")
        plt.close()
    
    # Plot correlation matrix
    plt.figure(figsize=(12, 10))
    corr = df[numeric_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix')
    plt.savefig(plots_dir / "correlation_matrix.png")
    plt.close()
    
    return analysis

def test_stationarity(df, column, city=None):
    """
    Test stationarity of a time series column.
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Column to test
        city (str, optional): City to filter for
    
    Returns:
        dict: Stationarity test results
    """
    # Filter by city if provided
    if city:
        series = df[df['city'] == city][column]
    else:
        series = df[column]
    
    # Drop missing values
    series = series.dropna()
    
    # Run Augmented Dickey-Fuller test
    try:
        result = adfuller(series)
        
        return {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'is_stationary': result[1] < 0.05  # True if p-value < 0.05
        }
    except:
        return {
            'adf_statistic': None,
            'p_value': None,
            'critical_values': None,
            'is_stationary': None
        }

def make_stationary(df, column, city=None, method='diff'):
    """
    Transform a time series column to make it stationary.
    
    Args:
        df (pd.DataFrame): DataFrame
        column (str): Column to transform
        city (str, optional): City to filter for
        method (str): Transformation method ('diff', 'log_diff', or 'percent_change')
    
    Returns:
        pd.Series: Transformed series
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Filter by city if provided
    if city:
        city_mask = df_copy['city'] == city
    else:
        city_mask = pd.Series(True, index=df_copy.index)
    
    # Apply transformation
    if method == 'diff':
        transformed = df_copy.loc[city_mask, column].diff()
    elif method == 'log_diff':
        # Add a small constant to avoid log(0)
        transformed = np.log(df_copy.loc[city_mask, column] + 1).diff()
    elif method == 'percent_change':
        transformed = df_copy.loc[city_mask, column].pct_change()
    else:
        raise ValueError(f"Unknown transformation method: {method}")
    
    return transformed

def impute_missing_values(df, numeric_cols, method='knn'):
    """
    Impute missing values in numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame
        numeric_cols (list): List of numeric columns
        method (str): Imputation method ('knn', 'seasonal', or 'mean')
    
    Returns:
        pd.DataFrame: DataFrame with imputed values
    """
    print_section("Imputing Missing Values")
    
    # Create a copy to avoid modifying the original
    df_imputed = df.copy()
    
    # Get missing value info
    missing = df[numeric_cols].isnull().sum()
    missing_cols = missing[missing > 0].index.tolist()
    
    if not missing_cols:
        print("No missing values to impute.")
        return df_imputed
    
    print(f"Imputing {len(missing_cols)} columns with missing values using {method} method.")
    
    if method == 'knn':
        # KNN imputation
        imputer = KNNImputer(n_neighbors=5)
        
        # Impute each city separately
        for city in df['city'].unique():
            city_mask = df_imputed['city'] == city
            
            # Impute only columns with missing values
            if df_imputed.loc[city_mask, missing_cols].isnull().sum().sum() > 0:
                df_imputed.loc[city_mask, missing_cols] = imputer.fit_transform(
                    df_imputed.loc[city_mask, missing_cols]
                )
    
    elif method == 'seasonal':
        # Seasonal imputation
        for city in df['city'].unique():
            city_mask = df_imputed['city'] == city
            
            for col in missing_cols:
                # Group by week of year and calculate mean
                seasonal_means = df_imputed.loc[city_mask].groupby('weekofyear')[col].transform('mean')
                
                # Impute missing values with seasonal means
                df_imputed.loc[city_mask, col] = df_imputed.loc[city_mask, col].fillna(seasonal_means)
                
                # If still missing, use forward fill followed by backward fill
                if df_imputed.loc[city_mask, col].isnull().sum() > 0:
                    df_imputed.loc[city_mask, col] = df_imputed.loc[city_mask, col].fillna(method='ffill')
                    df_imputed.loc[city_mask, col] = df_imputed.loc[city_mask, col].fillna(method='bfill')
    
    else:  # Default to mean imputation
        for city in df['city'].unique():
            city_mask = df_imputed['city'] == city
            
            for col in missing_cols:
                # Calculate mean for the city
                city_mean = df_imputed.loc[city_mask, col].mean()
                
                # Impute missing values with mean
                df_imputed.loc[city_mask, col] = df_imputed.loc[city_mask, col].fillna(city_mean)
    
    # Check if any missing values remain
    remaining_missing = df_imputed[numeric_cols].isnull().sum().sum()
    
    if remaining_missing > 0:
        print(f"Warning: {remaining_missing} missing values remain. Using forward fill as fallback.")
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='ffill')
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(method='bfill')
        df_imputed[numeric_cols] = df_imputed[numeric_cols].fillna(0)  # Fill any remaining with 0
    
    print("Imputation complete.")
    
    return df_imputed

def detect_and_handle_outliers(df, numeric_cols, method='isolation_forest'):
    """
    Detect and handle outliers in numeric columns.
    
    Args:
        df (pd.DataFrame): DataFrame
        numeric_cols (list): List of numeric columns
        method (str): Outlier detection method ('isolation_forest', 'zscore', or 'iqr')
    
    Returns:
        pd.DataFrame: DataFrame with handled outliers
    """
    print_section("Detecting and Handling Outliers")
    
    # Create a copy to avoid modifying the original
    df_cleaned = df.copy()
    
    if method == 'isolation_forest':
        # Isolation Forest for outlier detection
        for city in df['city'].unique():
            city_mask = df_cleaned['city'] == city
            
            # Skip if not enough data
            if df_cleaned.loc[city_mask, numeric_cols].shape[0] < 20:
                print(f"Skipping outlier detection for {city} due to insufficient data.")
                continue
            
            # Initialize Isolation Forest
            clf = IsolationForest(
                contamination=0.05,  # Assume 5% of data points are outliers
                random_state=42
            )
            
            # Fit and predict
            outlier_labels = clf.fit_predict(df_cleaned.loc[city_mask, numeric_cols])
            
            # Identify outliers (-1 label)
            outliers = (outlier_labels == -1)
            outlier_count = outliers.sum()
            
            print(f"Detected {outlier_count} outliers in {city} data.")
            
            # Only handle if outliers are detected
            if outlier_count > 0:
                # Replace outliers with rolling median (window size 5)
                for col in numeric_cols:
                    # Get outlier indices
                    outlier_indices = df_cleaned.loc[city_mask].index[outliers]
                    
                    # Calculate rolling median
                    rolling_median = df_cleaned.loc[city_mask, col].rolling(window=5, center=True, min_periods=1).median()
                    
                    # Replace outliers with rolling median
                    df_cleaned.loc[outlier_indices, col] = rolling_median.loc[outlier_indices]
    
    elif method == 'zscore':
        # Z-score method
        for city in df['city'].unique():
            city_mask = df_cleaned['city'] == city
            
            for col in numeric_cols:
                # Calculate z-scores
                z_scores = np.abs(stats.zscore(df_cleaned.loc[city_mask, col], nan_policy='omit'))
                
                # Identify outliers (z-score > 3)
                outliers = z_scores > 3
                outlier_count = outliers.sum()
                
                print(f"Detected {outlier_count} outliers in {city} {col}.")
                
                # Only handle if outliers are detected
                if outlier_count > 0:
                    # Calculate rolling median
                    rolling_median = df_cleaned.loc[city_mask, col].rolling(window=5, center=True, min_periods=1).median()
                    
                    # Get outlier indices
                    outlier_indices = df_cleaned.loc[city_mask].index[outliers]
                    
                    # Replace outliers with rolling median
                    df_cleaned.loc[outlier_indices, col] = rolling_median.loc[outlier_indices]
    
    else:  # IQR method
        # IQR method
        for city in df['city'].unique():
            city_mask = df_cleaned['city'] == city
            
            for col in numeric_cols:
                # Calculate Q1, Q3, and IQR
                Q1 = df_cleaned.loc[city_mask, col].quantile(0.25)
                Q3 = df_cleaned.loc[city_mask, col].quantile(0.75)
                IQR = Q3 - Q1
                
                # Identify outliers
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = (df_cleaned.loc[city_mask, col] < lower_bound) | (df_cleaned.loc[city_mask, col] > upper_bound)
                outlier_count = outliers.sum()
                
                print(f"Detected {outlier_count} outliers in {city} {col}.")
                
                # Only handle if outliers are detected
                if outlier_count > 0:
                    # Calculate rolling median
                    rolling_median = df_cleaned.loc[city_mask, col].rolling(window=5, center=True, min_periods=1).median()
                    
                    # Get outlier indices
                    outlier_indices = df_cleaned.loc[city_mask].index[outliers]
                    
                    # Replace outliers with rolling median
                    df_cleaned.loc[outlier_indices, col] = rolling_median.loc[outlier_indices]
    
    print("Outlier handling complete.")
    
    return df_cleaned

def apply_transformations(df, numeric_cols):
    """
    Apply transformations to make columns more normally distributed.
    
    Args:
        df (pd.DataFrame): DataFrame
        numeric_cols (list): List of numeric columns
    
    Returns:
        pd.DataFrame: DataFrame with transformed columns
    """
    print_section("Applying Transformations")
    
    # Create a copy to avoid modifying the original
    df_transformed = df.copy()
    
    # Test stationarity for each numeric column by city
    stationarity_results = {}
    
    for city in df['city'].unique():
        stationarity_results[city] = {}
        
        for col in numeric_cols:
            if col in ['year', 'weekofyear']:
                continue
                
            test_result = test_stationarity(df, col, city)
            stationarity_results[city][col] = test_result
            
            # Print result if test was successful
            if test_result['p_value'] is not None:
                status = "Stationary" if test_result['is_stationary'] else "Non-stationary"
                print(f"{city} - {col}: {status} (p-value: {test_result['p_value']:.5f})")
    
    # Apply transformations to non-stationary series
    transformed_cols = []
    
    for city in df['city'].unique():
        city_mask = df_transformed['city'] == city
        
        for col in numeric_cols:
            if col in ['year', 'weekofyear']:
                continue
                
            # Skip if stationarity test failed
            if (city not in stationarity_results or 
                col not in stationarity_results[city] or 
                stationarity_results[city][col]['is_stationary'] is None):
                continue
                
            # Apply transformation if non-stationary
            if not stationarity_results[city][col]['is_stationary']:
                # Choose transformation method based on column nature
                if "precip" in col:
                    # For precipitation, use log difference (handles zeros better)
                    new_col = f"{col}_log_diff"
                    df_transformed.loc[city_mask, new_col] = make_stationary(
                        df_transformed, col, city, 'log_diff'
                    )
                elif "total_cases" in col:
                    # For case counts, use log difference
                    new_col = f"{col}_log_diff"
                    df_transformed.loc[city_mask, new_col] = make_stationary(
                        df_transformed, col, city, 'log_diff'
                    )
                else:
                    # For other columns, use regular differencing
                    new_col = f"{col}_diff"
                    df_transformed.loc[city_mask, new_col] = make_stationary(
                        df_transformed, col, city, 'diff'
                    )
                
                transformed_cols.append(new_col)
                print(f"Transformed {city} - {col} to {new_col}")
    
    print(f"Added {len(transformed_cols)} transformed columns.")
    
    return df_transformed

def prepare_cleaned_data(train_data, test_features):
    """
    Prepare cleaned and enhanced data for modeling.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_features (pd.DataFrame): Test features
    
    Returns:
        tuple: (cleaned_train_data, cleaned_test_features)
    """
    print_section("Preparing Cleaned Data")
    
    # Ensure date columns are datetime
    train_data['week_start_date'] = pd.to_datetime(train_data['week_start_date'])
    test_features['week_start_date'] = pd.to_datetime(test_features['week_start_date'])
    
    # Add is_test column for identification
    train_data['is_test'] = False
    test_features['is_test'] = True
    
    # Merge train and test for consistent preprocessing
    combined = pd.concat([train_data, test_features], ignore_index=True)
    
    # Get numeric columns for preprocessing
    # Skip city (categorical) and week_start_date (datetime)
    numeric_cols = combined.select_dtypes(include=['number']).columns.tolist()
    
    # Analyze data
    analysis = analyze_data(combined)
    
    # Impute missing values
    combined_imputed = impute_missing_values(combined, numeric_cols, method='seasonal')
    
    # Handle outliers
    combined_cleaned = detect_and_handle_outliers(combined_imputed, numeric_cols, method='isolation_forest')
    
    # Apply transformations
    combined_transformed = apply_transformations(combined_cleaned, numeric_cols)
    
    # Split back into train and test
    cleaned_train_data = combined_transformed[combined_transformed['is_test'] == False].copy()
    cleaned_test_features = combined_transformed[combined_transformed['is_test'] == True].copy()
    
    # Remove is_test column from training data
    cleaned_train_data.drop('is_test', axis=1, inplace=True)
    
    # Sort data
    cleaned_train_data.sort_values(['city', 'week_start_date'], inplace=True)
    cleaned_test_features.sort_values(['city', 'week_start_date'], inplace=True)
    
    print(f"Cleaned training data shape: {cleaned_train_data.shape}")
    print(f"Cleaned test data shape: {cleaned_test_features.shape}")
    
    return cleaned_train_data, cleaned_test_features

def save_cleaned_data(train_data, test_features):
    """
    Save cleaned data to files.
    
    Args:
        train_data (pd.DataFrame): Cleaned training data
        test_features (pd.DataFrame): Cleaned test features
    """
    print_section("Saving Cleaned Data")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save data
    train_path = processed_dir / "enhanced_cleaned_train_data.csv"
    test_path = processed_dir / "enhanced_cleaned_test_features.csv"
    
    train_data.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"Cleaned training data saved to {train_path}")
    print(f"Cleaned test features saved to {test_path}")

def main():
    """Main function to run the enhanced preprocessing."""
    print_section("Starting Enhanced Preprocessing")
    
    # Load raw data
    train_data, test_features = load_raw_data()
    
    # Prepare cleaned data
    cleaned_train_data, cleaned_test_features = prepare_cleaned_data(train_data, test_features)
    
    # Save cleaned data
    save_cleaned_data(cleaned_train_data, cleaned_test_features)
    
    print_section("Enhanced Preprocessing Complete")
    print("Next steps:")
    print("1. Proceed with feature engineering")
    print("2. Train models on enhanced cleaned data")

if __name__ == "__main__":
    main()