"""
Data Investigation Module

This module provides functions for initial data exploration and analysis.
It focuses on understanding the raw data structure, missing values, and basic statistics
without modifying the data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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

# Configure matplotlib
plt.ion()  # Turn on interactive mode
plt.style.use('seaborn-v0_8')  # Use a nice style
debug_print("Matplotlib configured")

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def load_raw_data():
    """
    Load the raw training and test data.
    
    Returns:
        tuple: (train_features, train_labels, test_features)
    
    Raises:
        FileNotFoundError: If any of the required data files are missing
    """
    debug_print("Starting data loading")
    
    # Initialize data paths
    data_dir = Path("data/raw")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    # Define file paths
    train_features_path = data_dir / "Training_Data_Features.csv"
    train_labels_path = data_dir / "Training_Data_Labels.csv"
    test_features_path = data_dir / "Test_Data_Features.csv"
    
    # Check if files exist
    for path in [train_features_path, train_labels_path, test_features_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    debug_print("Loading training features...")
    train_features = pd.read_csv(train_features_path)
    debug_print(f"Training features loaded. Shape: {train_features.shape}")
    
    debug_print("Loading training labels...")
    train_labels = pd.read_csv(train_labels_path)
    debug_print(f"Training labels loaded. Shape: {train_labels.shape}")
    
    debug_print("Loading test features...")
    test_features = pd.read_csv(test_features_path)
    debug_print(f"Test features loaded. Shape: {test_features.shape}")
    
    return train_features, train_labels, test_features

def analyze_data_structure(df, name="Dataset"):
    """
    Analyze the basic structure of the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for printing
    """
    print_section(f"Data Structure Analysis for {name}")
    
    print("\nBasic Information:")
    print(f"Shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")
    print("\nColumn Types:")
    print(df.dtypes.value_counts())
    
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nColumn Descriptions:")
    for col in df.columns:
        print(f"\n{col}:")
        print(f"  Type: {df[col].dtype}")
        print(f"  Unique values: {df[col].nunique()}")
        if df[col].dtype in ['int64', 'float64']:
            print(f"  Min: {df[col].min()}")
            print(f"  Max: {df[col].max()}")
            print(f"  Mean: {df[col].mean()}")
            print(f"  Std: {df[col].std()}")
    
    # Add recommendations for temporal features
    if 'weekofyear' in df.columns:
        print("\nRecommendations for Temporal Features:")
        print("- Consider converting weekofyear to cyclic features using sine and cosine transformations")
        print("  This helps the model understand the cyclical nature of weeks (e.g., week 52 is close to week 1)")
        print("  Use: week_sin = sin(2π * weekofyear / 52)")
        print("       week_cos = cos(2π * weekofyear / 52)")

def analyze_missing_values(df, name="Dataset"):
    """
    Analyze missing values in the dataset and suggest handling strategies.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for printing
    """
    print_section(f"Missing Values Analysis for {name}")
    
    # Calculate missing values statistics
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    # Create a DataFrame with the results
    missing_stats = pd.DataFrame({
        'Missing Values': missing_values,
        'Percentage': missing_percentage
    })
    
    # Filter out columns with no missing values
    missing_stats = missing_stats[missing_stats['Missing Values'] > 0]
    
    if len(missing_stats) == 0:
        print("No missing values found in the dataset.")
        return
    
    print("\nMissing Values Summary:")
    print(missing_stats)
    
    # Visualize missing values
    plt.figure(figsize=(12, 6))
    plt.bar(missing_stats.index, missing_stats['Percentage'])
    plt.xticks(rotation=45, ha='right')
    plt.title(f'Percentage of Missing Values in {name}')
    plt.ylabel('Percentage')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    
    # Suggest handling strategies
    print("\nSuggested Handling Strategies:")
    for col in missing_stats.index:
        percentage = missing_stats.loc[col, 'Percentage']
        
        if 'temp' in col.lower():
            if percentage > 50:
                print(f"- {col}: Consider dropping the column (>{percentage:.1f}% missing)")
            else:
                print(f"- {col}: Use linear interpolation for temperature data (<{percentage:.1f}% missing)")
        elif 'precip' in col.lower():
            if percentage > 50:
                print(f"- {col}: Consider dropping the column (>{percentage:.1f}% missing)")
            else:
                print(f"- {col}: Use zero imputation for precipitation data (<{percentage:.1f}% missing)")
        elif 'ndvi' in col.lower():
            if percentage > 50:
                print(f"- {col}: Consider dropping the column (>{percentage:.1f}% missing)")
            else:
                print(f"- {col}: Use seasonal mean imputation for NDVI data (<{percentage:.1f}% missing)")
        else:
            if percentage > 50:
                print(f"- {col}: Consider dropping the column (>{percentage:.1f}% missing)")
            else:
                print(f"- {col}: Use mean imputation (<{percentage:.1f}% missing)")

def main():
    """Main function to run the data investigation."""
    print_section("Starting Data Investigation")
    
    # Load data
    train_features, train_labels, test_features = load_raw_data()
    
    # Analyze data structure
    analyze_data_structure(train_features, "Training Features")
    analyze_data_structure(train_labels, "Training Labels")
    analyze_data_structure(test_features, "Test Features")
    
    # Analyze missing values
    analyze_missing_values(train_features, "Training Features")
    analyze_missing_values(train_labels, "Training Labels")
    analyze_missing_values(test_features, "Test Features")
    
    print_section("Next Steps")
    print("1. Run data_clean.py to process and clean the data")
    print("2. Run data_features.py to add new features")
    print("3. Proceed with model training using the processed data")

if __name__ == "__main__":
    main() 