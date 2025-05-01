"""
Data Cleaner Investigation Module

This module investigates the cleaned data to ensure the cleaning process
was successful and to identify any remaining issues.
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

def analyze_cleaning_results(df, name="Dataset"):
    """
    Analyze the results of the cleaning process.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for printing
    """
    print_section(f"Cleaning Results Analysis for {name}")
    
    # Check for remaining missing values
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    print("\nMissing Values After Cleaning:")
    if missing_values.sum() == 0:
        print("No missing values found - cleaning successful!")
    else:
        print("Remaining missing values:")
        for col in missing_values[missing_values > 0].index:
            print(f"- {col}: {missing_values[col]} ({missing_percentage[col]:.2f}%)")
    
    # Analyze temporal features
    if 'week_start_date' in df.columns:
        print("\nTemporal Features:")
        print(f"- Date range: {df['week_start_date'].min()} to {df['week_start_date'].max()}")
        print(f"- Years: {df['year'].unique()}")
        print(f"- Months: {df['month'].unique()}")
        print(f"- Weeks: {df['weekofyear'].unique()}")
    
    # Analyze categorical features
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    if len(categorical_cols) > 0:
        print("\nCategorical Features:")
        for col in categorical_cols:
            print(f"- {col}: {df[col].nunique()} unique values")
            print(f"  Most common: {df[col].mode()[0]} ({df[col].value_counts().iloc[0] / len(df):.2%})")
    
    # Analyze numerical features
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        print("\nNumerical Features:")
        for col in numerical_cols:
            if col not in ['year', 'month', 'weekofyear']:  # Skip temporal features
                print(f"- {col}:")
                print(f"  Min: {df[col].min():.2f}")
                print(f"  Max: {df[col].max():.2f}")
                print(f"  Mean: {df[col].mean():.2f}")
                print(f"  Std: {df[col].std():.2f}")
                print(f"  Missing: {df[col].isnull().sum()}")

def analyze_feature_distributions(df, name="Dataset"):
    """
    Analyze and visualize feature distributions.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for printing
    """
    print_section(f"Feature Distributions for {name}")
    
    # Create directory for plots
    plot_dir = Path("data/processed/png")
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot numerical feature distributions
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if col not in ['year', 'month', 'weekofyear']:  # Skip temporal features
            plt.figure(figsize=(10, 4))
            plt.hist(df[col].dropna(), bins=50)
            plt.title(f'Distribution of {col}')
            plt.xlabel(col)
            plt.ylabel('Frequency')
            plt.tight_layout()
            
            # Save plot
            filename = f"{name.lower().replace(' ', '_')}_{col}_distribution.png"
            plt.savefig(plot_dir / filename)
            plt.close()
            print(f"Saved distribution plot for {col} to {plot_dir / filename}")
    
    # Plot categorical feature distributions
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        df[col].value_counts().plot(kind='bar')
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        filename = f"{name.lower().replace(' ', '_')}_{col}_distribution.png"
        plt.savefig(plot_dir / filename)
        plt.close()
        print(f"Saved distribution plot for {col} to {plot_dir / filename}")
    
    print(f"\nAll distribution plots saved to {plot_dir}")

def main():
    """Main function to run the data cleaner investigation."""
    print_section("Starting Data Cleaner Investigation")
    
    # Load cleaned data
    train_data, test_features = load_cleaned_data()
    
    # Analyze cleaning results
    analyze_cleaning_results(train_data, "Training Data")
    analyze_cleaning_results(test_features, "Test Features")
    
    # Analyze feature distributions
    analyze_feature_distributions(train_data, "Training Data")
    analyze_feature_distributions(test_features, "Test Features")
    
    print_section("Next Steps")
    print("1. Review the cleaning results and distributions")
    print("2. If satisfied, proceed with data_features.py")
    print("3. If issues are found, adjust data_cleaner.py")

if __name__ == "__main__":
    main() 