"""
Data Cleaning Module

This module handles the cleaning and preprocessing of the raw data.
It implements the cleaning strategies suggested by data_investigation.py
and saves the cleaned data to the processed directory.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from data_investigation import load_raw_data, print_section, debug_print

def create_processed_directory():
    """Create the processed data directory if it doesn't exist."""
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    debug_print(f"Created/verified processed directory: {processed_dir}")

def handle_missing_values(df, name="Dataset"):
    """
    Handle missing values in the dataset based on feature type.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
        name (str): Name of the dataset for printing
    
    Returns:
        pd.DataFrame: Cleaned DataFrame
    """
    print_section(f"Handling Missing Values for {name}")
    
    # Create a copy to avoid modifying the original
    df_clean = df.copy()
    
    # Handle missing values based on feature type
    for col in df_clean.columns:
        missing_percentage = (df_clean[col].isnull().sum() / len(df_clean)) * 100
        
        if missing_percentage > 50:
            print(f"Dropping column {col} (>{missing_percentage:.1f}% missing)")
            df_clean = df_clean.drop(columns=[col])
            continue
        
        # Skip if no missing values
        if missing_percentage == 0:
            continue
            
        # Handle based on column type and name
        if col == 'city':
            print(f"Using mode imputation for {col} (categorical)")
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        elif 'temp' in col.lower():
            print(f"Using linear interpolation for {col}")
            df_clean[col] = df_clean[col].interpolate(method='linear')
        elif 'precip' in col.lower():
            print(f"Using zero imputation for {col}")
            df_clean[col] = df_clean[col].fillna(0)
        elif 'ndvi' in col.lower():
            print(f"Using seasonal mean imputation for {col}")
            # Group by year and week to get seasonal means
            if 'year' in df_clean.columns and 'weekofyear' in df_clean.columns:
                seasonal_means = df_clean.groupby(['year', 'weekofyear'])[col].transform('mean')
                df_clean[col] = df_clean[col].fillna(seasonal_means)
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
        elif df_clean[col].dtype in ['object', 'category']:
            print(f"Using mode imputation for {col} (categorical)")
            df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])
        else:
            print(f"Using mean imputation for {col}")
            df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
    
    return df_clean

def clean_temporal_features(df):
    """
    Clean and standardize temporal features.
    
    Args:
        df (pd.DataFrame): DataFrame to clean
    
    Returns:
        pd.DataFrame: DataFrame with cleaned temporal features
    """
    print_section("Cleaning Temporal Features")
    
    if 'week_start_date' in df.columns:
        # Convert to datetime
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        
        # Extract temporal features
        df['year'] = df['week_start_date'].dt.year
        df['month'] = df['week_start_date'].dt.month
        df['weekofyear'] = df['week_start_date'].dt.isocalendar().week
        
        print("Added temporal features:")
        print(f"- year: {df['year'].min()} to {df['year'].max()}")
        print(f"- month: {df['month'].min()} to {df['month'].max()}")
        print(f"- weekofyear: {df['weekofyear'].min()} to {df['weekofyear'].max()}")
        print("\nNote: weekofyear will be transformed to cyclic features in data_features.py")
    
    return df

def merge_train_data(train_features, train_labels):
    """
    Merge training features and labels.
    
    Args:
        train_features (pd.DataFrame): Training features
        train_labels (pd.DataFrame): Training labels
    
    Returns:
        pd.DataFrame: Merged training data
    """
    print_section("Merging Training Data")
    
    # Ensure we have the correct columns for merging
    merge_cols = ['city', 'year', 'weekofyear']
    for col in merge_cols:
        if col not in train_features.columns or col not in train_labels.columns:
            raise ValueError(f"Missing merge column: {col}")
    
    # Merge the data
    merged_data = pd.merge(
        train_features,
        train_labels,
        on=merge_cols,
        how='inner'
    )
    
    print(f"Merged training data shape: {merged_data.shape}")
    return merged_data

def save_cleaned_data(train_data, test_features):
    """
    Save the cleaned data to the processed directory.
    
    Args:
        train_data (pd.DataFrame): Cleaned training data
        test_features (pd.DataFrame): Cleaned test features
    """
    print_section("Saving Cleaned Data")
    
    # Create processed directory
    create_processed_directory()
    
    # Define file paths
    train_path = 'data/processed/cleaned_train_data.csv'
    test_path = 'data/processed/cleaned_test_features.csv'
    
    # Save cleaned data
    train_data.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print("\nCreated files:")
    print(f"1. Training data: {train_path}")
    print(f"   - Shape: {train_data.shape}")
    print(f"   - Columns: {len(train_data.columns)}")
    print(f"2. Test features: {test_path}")
    print(f"   - Shape: {test_features.shape}")
    print(f"   - Columns: {len(test_features.columns)}")
    
    print("\nFiles are ready for feature engineering in data_features.py")

def main():
    """Main function to run the data cleaning process."""
    print_section("Starting Data Cleaning")
    
    # Load raw data
    train_features, train_labels, test_features = load_raw_data()
    
    # Clean temporal features
    train_features = clean_temporal_features(train_features)
    test_features = clean_temporal_features(test_features)
    
    # Handle missing values
    train_features = handle_missing_values(train_features, "Training Features")
    train_labels = handle_missing_values(train_labels, "Training Labels")
    test_features = handle_missing_values(test_features, "Test Features")
    
    # Merge training data
    train_data = merge_train_data(train_features, train_labels)
    
    # Save cleaned data
    save_cleaned_data(train_data, test_features)
    
    print_section("Next Steps")
    print("1. Run data_features.py to add new features")
    print("2. Proceed with model training using the processed data")

if __name__ == "__main__":
    main() 