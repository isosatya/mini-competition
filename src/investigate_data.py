import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

import os
import sys
from pathlib import Path

# Add parent directory to Python path to make lib module importable
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from libs.core_classes import DataLoader, DataCleaner



def load_and_prepare_data():
    """
    Load and prepare the training and test data.
    Returns:
        tuple: (train_features, train_labels, test_features)
    """
    # Initialize data paths
    data_dir = Path("data/raw")
    train_features_path = data_dir / "Training_Data_Features.csv"
    train_labels_path = data_dir / "Training_Data_Labels.csv"
    test_features_path = data_dir / "Test_Data_Features.csv"

    # Load data
    train_features = pd.read_csv(train_features_path)
    train_labels = pd.read_csv(train_labels_path)
    test_features = pd.read_csv(test_features_path)

    return train_features, train_labels, test_features

def basic_data_analysis(df, name="Dataset"):
    """
    Perform basic data analysis on a DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        name (str): Name of the dataset for printing
    """
    print(f"\n=== Basic Analysis for {name} ===")
    print(f"Shape: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    
    print("\nData Types:")
    print(df.dtypes)
    
    print("\nMissing Values:")
    print(df.isnull().sum())
    
    print("\nBasic Statistics:")
    print(df.describe())

def plot_feature_distributions(df, target_col=None):
    """
    Plot distributions of numerical features.
    
    Args:
        df (pd.DataFrame): DataFrame to plot
        target_col (str, optional): Target column for correlation analysis
    """
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Plot distributions
    for col in numerical_cols:
        plt.figure(figsize=(10, 6))
        sns.histplot(data=df, x=col, kde=True)
        plt.title(f'Distribution of {col}')
        plt.show()
    
    # Plot correlation matrix if target column is provided
    if target_col and target_col in df.columns:
        plt.figure(figsize=(12, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Correlation Matrix')
        plt.show()

def main():
    # Load data
    train_features, train_labels, test_features = load_and_prepare_data()
    
    # Basic analysis
    basic_data_analysis(train_features, "Training Features")
    basic_data_analysis(train_labels, "Training Labels")
    basic_data_analysis(test_features, "Test Features")
    
    # Plot distributions
    plot_feature_distributions(train_features)
    
    # Clean data
    cleaner = DataCleaner(train_features)
    cleaner.handle_missing_values(strategy='mean')
    
    # Save cleaned data
    cleaned_data_path = Path("data/processed/cleaned_train_features.csv")
    cleaned_data_path.parent.mkdir(parents=True, exist_ok=True)
    cleaner.save_data(str(cleaned_data_path))

if __name__ == "__main__":
    main()
