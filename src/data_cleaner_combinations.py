"""
Data Cleaning and Feature Combination Module

This module handles feature combinations and date conversions for the cleaned data.
It combines related features to reduce redundancy and improve model performance.
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

def combine_temperature_features(df):
    """
    Combine temperature-related features to reduce redundancy.
    
    Args:
        df (pd.DataFrame): DataFrame with temperature features
    
    Returns:
        pd.DataFrame: DataFrame with combined temperature features
    """
    print_section("Combining Temperature Features")
    
    # Check if all required columns exist
    reanalysis_cols = [
        'reanalysis_max_air_temp_k',
        'reanalysis_min_air_temp_k',
        'reanalysis_avg_temp_k',
        'reanalysis_air_temp_k'
    ]
    
    station_cols = [
        'station_max_temp_c',
        'station_min_temp_c',
        'station_avg_temp_c'
    ]
    
    # Create combined features if all columns exist
    if all(col in df.columns for col in reanalysis_cols):
        print("Combining reanalysis temperature features...")
        # Calculate mean temperature from all reanalysis sources
        df['reanalysis_temp_mean'] = df[reanalysis_cols].mean(axis=1)
        # Calculate temperature range
        df['reanalysis_temp_range'] = df['reanalysis_max_air_temp_k'] - df['reanalysis_min_air_temp_k']
        # Drop original columns
        df = df.drop(columns=reanalysis_cols)
        print("Reanalysis temperature features combined")
    
    if all(col in df.columns for col in station_cols):
        print("Combining station temperature features...")
        # Calculate mean temperature from all station sources
        df['station_temp_mean'] = df[station_cols].mean(axis=1)
        # Calculate temperature range
        df['station_temp_range'] = df['station_max_temp_c'] - df['station_min_temp_c']
        # Drop original columns
        df = df.drop(columns=station_cols)
        print("Station temperature features combined")
    
    return df

def convert_dates(df):
    """
    Convert date columns to datetime format.
    
    Args:
        df (pd.DataFrame): DataFrame with date columns
    
    Returns:
        pd.DataFrame: DataFrame with converted date columns
    """
    print_section("Converting Dates")
    
    if 'week_start_date' in df.columns:
        print("Converting week_start_date to datetime...")
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        print("Date conversion completed")
    
    return df

def save_combined_data(train_data, test_features):
    """
    Save the combined datasets to the processed directory.
    
    Args:
        train_data (pd.DataFrame): Combined training data
        test_features (pd.DataFrame): Combined test features
    """
    print_section("Saving Combined Data")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save combined data
    train_path = processed_dir / "combined_train_data.csv"
    test_path = processed_dir / "combined_test_features.csv"
    
    train_data.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"Combined training data saved to {train_path}")
    print(f"Combined test features saved to {test_path}")

def main():
    """Main function to run the feature combination process."""
    print_section("Starting Feature Combination")
    
    # Load data
    train_data, test_features = load_cleaned_data()
    
    # Convert dates
    train_data = convert_dates(train_data)
    test_features = convert_dates(test_features)
    
    # Combine features
    train_data = combine_temperature_features(train_data)
    test_features = combine_temperature_features(test_features)
    
    # Save combined data
    save_combined_data(train_data, test_features)
    
    print_section("Next Steps")
    print("1. Review the combined features")
    print("2. Use the combined data for model training")
    print("3. If needed, adjust the feature combinations")

if __name__ == "__main__":
    main() 