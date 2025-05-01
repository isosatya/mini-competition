"""
Feature Engineering Module

This module handles the creation of new features from the cleaned data.
It includes functions for creating temporal, weather, and derived features
that can improve model performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from data_investigation import print_section, debug_print

def load_cleaned_data():
    """
    Load the cleaned data from the processed directory.
    
    Returns:
        tuple: (train_data, test_features)
    
    Raises:
        FileNotFoundError: If cleaned data files are missing
    """
    print_section("Loading Cleaned Data")
    
    # Check if files exist
    train_path = Path("data/processed/cleaned_train_data.csv")
    test_path = Path("data/processed/cleaned_test_features.csv")
    
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError("Please run data_clean.py first to create cleaned data files")
    
    # Load data
    train_data = pd.read_csv(train_path)
    test_features = pd.read_csv(test_path)
    
    print(f"Loaded cleaned training data: {train_data.shape}")
    print(f"Loaded cleaned test features: {test_features.shape}")
    
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
    
    if 'week_start_date' in df.columns:
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df['week_start_date']):
            df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        
        # Add temporal features
        df['dayofyear'] = df['week_start_date'].dt.dayofyear
        df['quarter'] = df['week_start_date'].dt.quarter
        df['is_month_start'] = df['week_start_date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['week_start_date'].dt.is_month_end.astype(int)
        
        # Add cyclic week features
        df['weekofyear'] = df['week_start_date'].dt.isocalendar().week
        df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
        
        print("Added temporal features: dayofyear, quarter, is_month_start, is_month_end, week_sin, week_cos")
    
    return df

def add_weather_features(df):
    """
    Add derived weather features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
    
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    print_section("Adding Weather Features")
    
    # Temperature features
    temp_cols = [col for col in df.columns if 'temp' in col.lower()]
    if temp_cols:
        df['temp_mean'] = df[temp_cols].mean(axis=1)
        df['temp_std'] = df[temp_cols].std(axis=1)
        df['temp_range'] = df[temp_cols].max(axis=1) - df[temp_cols].min(axis=1)
        print("Added temperature features: mean, std, range")
    
    # Precipitation features
    precip_cols = [col for col in df.columns if 'precip' in col.lower()]
    if precip_cols:
        df['precip_total'] = df[precip_cols].sum(axis=1)
        df['precip_days'] = (df[precip_cols] > 0).sum(axis=1)
        print("Added precipitation features: total, days")
    
    # Humidity features
    humidity_cols = [col for col in df.columns if 'humidity' in col.lower()]
    if humidity_cols:
        df['humidity_mean'] = df[humidity_cols].mean(axis=1)
        print("Added humidity features: mean")
    
    return df

def add_ndvi_features(df):
    """
    Add derived NDVI features to the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
    
    Returns:
        pd.DataFrame: DataFrame with added NDVI features
    """
    print_section("Adding NDVI Features")
    
    ndvi_cols = [col for col in df.columns if 'ndvi' in col.lower()]
    if ndvi_cols:
        df['ndvi_mean'] = df[ndvi_cols].mean(axis=1)
        df['ndvi_std'] = df[ndvi_cols].std(axis=1)
        df['ndvi_range'] = df[ndvi_cols].max(axis=1) - df[ndvi_cols].min(axis=1)
        print("Added NDVI features: mean, std, range")
    
    return df

def add_lag_features(df, target_col='total_cases', lags=[1, 2, 3, 4]):
    """
    Add lagged features for the target variable.
    
    Args:
        df (pd.DataFrame): DataFrame to add features to
        target_col (str): Target column to create lags for
        lags (list): List of lag periods to create
    
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    print_section("Adding Lag Features")
    
    if target_col not in df.columns:
        print(f"Target column {target_col} not found, skipping lag features")
        return df
    
    # Sort by city and date to ensure correct lagging
    df = df.sort_values(['city', 'week_start_date'])
    
    # Create lag features for each city separately
    for city in df['city'].unique():
        city_mask = df['city'] == city
        for lag in lags:
            df.loc[city_mask, f'{target_col}_lag_{lag}'] = df.loc[city_mask, target_col].shift(lag)
    
    print(f"Added lag features for lags: {lags}")
    return df

def save_featured_data(train_data, test_features):
    """
    Save the data with new features to the processed directory.
    
    Args:
        train_data (pd.DataFrame): Training data with new features
        test_features (pd.DataFrame): Test features with new features
    """
    print_section("Saving Featured Data")
    
    # Save featured data
    train_data.to_csv('data/processed/featured_train_data.csv', index=False)
    test_features.to_csv('data/processed/featured_test_features.csv', index=False)
    
    print("Saved featured data to data/processed/")

def main():
    """Main function to run the feature engineering process."""
    print_section("Starting Feature Engineering")
    
    # Load cleaned data
    train_data, test_features = load_cleaned_data()
    
    # Add features
    train_data = add_temporal_features(train_data)
    test_features = add_temporal_features(test_features)
    
    train_data = add_weather_features(train_data)
    test_features = add_weather_features(test_features)
    
    train_data = add_ndvi_features(train_data)
    test_features = add_ndvi_features(test_features)
    
    train_data = add_lag_features(train_data)
    
    # Save featured data
    save_featured_data(train_data, test_features)
    
    print_section("Next Steps")
    print("1. The data is now ready for model training")
    print("2. Use the featured data files in data/processed/ for training")
    print("3. Consider the following features for your model:")
    print("   - Temporal features: year, month, weekofyear, dayofyear, quarter")
    print("   - Weather features: temperature statistics, precipitation totals")
    print("   - NDVI features: vegetation index statistics")
    print("   - Lag features: previous weeks' case counts")

if __name__ == "__main__":
    main() 