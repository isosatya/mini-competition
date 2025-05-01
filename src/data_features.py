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
    Add temporal features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with date column
    
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    print_section("Adding Temporal Features")
    
    # Create a copy to avoid modifying the original
    df_temp = df.copy()
    
    # Convert date to datetime
    df_temp['week_start_date'] = pd.to_datetime(df_temp['week_start_date'])
    
    # Add temporal features
    df_temp['year'] = df_temp['week_start_date'].dt.year
    df_temp['month'] = df_temp['week_start_date'].dt.month
    df_temp['weekofyear'] = df_temp['week_start_date'].dt.isocalendar().week
    
    # Add cyclic features for week of year
    df_temp['weekofyear_sin'] = np.sin(2 * np.pi * df_temp['weekofyear'] / 52)
    df_temp['weekofyear_cos'] = np.cos(2 * np.pi * df_temp['weekofyear'] / 52)
    
    return df_temp

def add_weather_features(df):
    """
    Add basic weather features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with weather columns
    
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    print_section("Adding Weather Features")
    
    # Create a copy to avoid modifying the original
    df_weather = df.copy()
    
    # Initialize all required weather features with 0
    required_features = [
        'temp_avg',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg'
    ]
    
    for feature in required_features:
        if feature not in df_weather.columns:
            df_weather[feature] = 0
    
    # Calculate average temperature
    temp_cols = [col for col in df.columns if 'temp' in col.lower() and 'lag' not in col.lower()]
    if temp_cols:
        df_weather['temp_avg'] = df[temp_cols].mean(axis=1)
    
    # Calculate average humidity
    humidity_cols = [col for col in df.columns if 'humidity' in col.lower() and 'lag' not in col.lower()]
    if humidity_cols:
        df_weather['humidity_avg'] = df[humidity_cols].mean(axis=1)
    
    # Calculate precipitation features
    precip_cols = [col for col in df.columns if 'precip' in col.lower() and 'lag' not in col.lower()]
    if precip_cols:
        df_weather['precip_total'] = df[precip_cols].sum(axis=1)
        df_weather['precip_days'] = (df[precip_cols] > 0).sum(axis=1)
    
    # Calculate average pressure
    pressure_cols = [col for col in df.columns if 'pressure' in col.lower() and 'lag' not in col.lower()]
    if pressure_cols:
        df_weather['pressure_avg'] = df[pressure_cols].mean(axis=1)
    
    return df_weather

def add_weather_lag_features(df):
    """
    Add lag features for weather data.
    
    Args:
        df (pd.DataFrame): DataFrame with weather features
    
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    print_section("Adding Weather Lag Features")
    
    # Define weather features for lagging
    weather_features = [
        'temp_avg',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg'
    ]
    
    # Create a copy to avoid modifying the original
    df_lagged = df.copy()
    
    # Ensure all required features exist
    for feature in weather_features:
        if feature not in df_lagged.columns:
            df_lagged[feature] = 0
    
    # Group by city to maintain city order
    for city in df['city'].unique():
        city_mask = df['city'] == city
        
        # Create lag features for each weather feature
        for feature in weather_features:
            for lag in [1, 2, 3, 4]:
                lag_col = f'{feature}_lag_{lag}'
                df_lagged.loc[city_mask, lag_col] = df_lagged.loc[city_mask, feature].shift(lag)
    
    # Fill NaN values with 0
    df_lagged = df_lagged.fillna(0)
    
    return df_lagged

def add_ndvi_features(df):
    """
    Add NDVI-related features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with NDVI columns
    
    Returns:
        pd.DataFrame: DataFrame with added NDVI features
    """
    print_section("Adding NDVI Features")
    
    # Create a copy to avoid modifying the original
    df_ndvi = df.copy()
    
    # Calculate NDVI statistics
    ndvi_cols = [col for col in df.columns if 'ndvi' in col.lower()]
    
    if ndvi_cols:
        # Calculate statistics for each city separately
        for city in df['city'].unique():
            city_mask = df['city'] == city
            for col in ndvi_cols:
                df_ndvi.loc[city_mask, f'{col}_mean'] = df.loc[city_mask, col].rolling(window=4, min_periods=1).mean()
                df_ndvi.loc[city_mask, f'{col}_std'] = df.loc[city_mask, col].rolling(window=4, min_periods=1).std()
    
    return df_ndvi

def prepare_features(df):
    """
    Prepare all features for the model.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with all features
    """
    print_section("Preparing All Features")
    
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Add temporal features
    df_features = add_temporal_features(df_features)
    
    # Add weather features first
    df_features = add_weather_features(df_features)
    
    # Add weather lag features
    df_features = add_weather_lag_features(df_features)
    
    # Add NDVI features
    df_features = add_ndvi_features(df_features)
    
    return df_features

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
    
    # Load cleaned data
    train_data, test_features = load_cleaned_data()
    
    # Prepare features for training data
    print("Preparing features for training data...")
    featured_train_data = prepare_features(train_data)
    
    # Prepare features for test data
    print("Preparing features for test data...")
    featured_test_features = prepare_features(test_features)
    
    # Save featured data
    save_featured_data(featured_train_data, featured_test_features)
    
    print_section("Next Steps")
    print("1. Review the featured data")
    print("2. Proceed with model training")
    print("3. If needed, adjust feature engineering")

if __name__ == "__main__":
    main() 