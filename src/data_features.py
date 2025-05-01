"""
Feature Engineering Module

This module handles feature engineering for the Dengue Fever prediction model.
It creates new features from the cleaned data to improve model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys

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

def load_cleaned_data():
    """
    Load the cleaned training and test data.
    
    Returns:
        tuple: (train_data, test_features)
    
    Raises:
        FileNotFoundError: If cleaned data files are missing
    """
    debug_print("Starting cleaned data loading")
    
    # Initialize data paths
    data_dir = Path("data/processed")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    # Define file paths
    train_path = data_dir / "cleaned_train_data.csv"
    test_path = data_dir / "cleaned_test_features.csv"
    
    # Check if files exist
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    debug_print("Loading cleaned training data...")
    train_data = pd.read_csv(train_path)
    debug_print(f"Training data loaded. Shape: {train_data.shape}")
    
    debug_print("Loading cleaned test features...")
    test_features = pd.read_csv(test_path)
    debug_print(f"Test features loaded. Shape: {test_features.shape}")
    
    return train_data, test_features

def add_temporal_features(df):
    """
    Add temporal features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
    
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    print_section("Adding Temporal Features")
    
    # Convert week_start_date to datetime if it's not already
    if 'week_start_date' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['week_start_date']):
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    
    # Add temporal features
    df['dayofyear'] = df['week_start_date'].dt.dayofyear
    df['quarter'] = df['week_start_date'].dt.quarter
    df['is_month_start'] = df['week_start_date'].dt.is_month_start.astype(int)
    df['is_month_end'] = df['week_start_date'].dt.is_month_end.astype(int)
    
    # Add cyclic features for weekofyear
    df['weekofyear_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['weekofyear_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
    
    return df

def add_weather_features(df):
    """
    Add weather-related features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
    
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    print_section("Adding Weather Features")
    
    # Temperature features
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    if temp_cols:
        df['temp_avg'] = df[temp_cols].mean(axis=1)
    
    # Humidity features
    humidity_cols = [col for col in df.columns if 'humidity' in col.lower()]
    if humidity_cols:
        df['humidity_avg'] = df[humidity_cols].mean(axis=1)
    
    # Precipitation features
    precip_cols = [col for col in df.columns if 'precip' in col.lower()]
    if precip_cols:
        df['precip_total'] = df[precip_cols].sum(axis=1)
        df['precip_days'] = (df[precip_cols] > 0).sum(axis=1)
    
    # Air pressure features
    pressure_cols = [col for col in df.columns if 'pressure' in col.lower()]
    if pressure_cols:
        df['pressure_avg'] = df[pressure_cols].mean(axis=1)
    
    return df

def add_weather_lag_features(df, lags=[1, 2, 3, 4]):
    """
    Add lagged weather features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
        lags (list): List of lag periods to create
    
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    print_section("Adding Weather Lag Features")
    
    # Sort by city and date to ensure correct lagging
    df = df.sort_values(['city', 'week_start_date'])
    
    # Weather features to create lags for
    weather_features = [
        'temp_avg',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg'
    ]
    
    # Create lag features for each city separately
    for city in df['city'].unique():
        city_mask = df['city'] == city
        for feature in weather_features:
            if feature in df.columns:
                for lag in lags:
                    df.loc[city_mask, f'{feature}_lag_{lag}'] = df.loc[city_mask, feature].shift(lag)
    
    return df

def add_ndvi_features(df):
    """
    Add NDVI-related features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
    
    Returns:
        pd.DataFrame: DataFrame with added NDVI features
    """
    print_section("Adding NDVI Features")
    
    # Get all NDVI columns
    ndvi_cols = [col for col in df.columns if 'ndvi' in col.lower()]
    
    if ndvi_cols:
        # Calculate mean NDVI
        df['ndvi_mean'] = df[ndvi_cols].mean(axis=1)
        
        # Calculate NDVI standard deviation
        df['ndvi_std'] = df[ndvi_cols].std(axis=1)
        
        # Calculate NDVI range
        df['ndvi_range'] = df[ndvi_cols].max(axis=1) - df[ndvi_cols].min(axis=1)
        
        # Calculate NDVI trend (slope)
        df['ndvi_trend'] = df[ndvi_cols].apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if not x.isna().any() else np.nan,
            axis=1
        )
    
    return df

def save_featured_data(train_data, test_features):
    """
    Save the featured datasets to the processed directory.
    
    Args:
        train_data (pd.DataFrame): Featured training data
        test_features (pd.DataFrame): Featured test features
    """
    print_section("Saving Featured Data")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save featured data
    train_path = processed_dir / "featured_train_data.csv"
    test_path = processed_dir / "featured_test_features.csv"
    
    train_data.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"Featured training data saved to {train_path}")
    print(f"Featured test features saved to {test_path}")

def main():
    """Main function to run the feature engineering process."""
    print_section("Starting Feature Engineering")
    
    # Load data
    train_data, test_features = load_cleaned_data()
    
    # Add features
    train_data = add_temporal_features(train_data)
    test_features = add_temporal_features(test_features)
    
    train_data = add_weather_features(train_data)
    test_features = add_weather_features(test_features)
    
    train_data = add_weather_lag_features(train_data)
    test_features = add_weather_lag_features(test_features)
    
    train_data = add_ndvi_features(train_data)
    test_features = add_ndvi_features(test_features)
    
    # Save featured data
    save_featured_data(train_data, test_features)
    
    print_section("Next Steps")
    print("1. Review the featured data")
    print("2. Use the featured data for model training")
    print("3. If needed, adjust the feature engineering process")

if __name__ == "__main__":
    main() 