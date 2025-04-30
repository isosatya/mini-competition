
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
import sys

# Debugging setup
DEBUG = True

def debug_print(message):
    if DEBUG:
        print(f"[DEBUG] {message}")

# Add parent directory to Python path to make lib module importable
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
debug_print(f"Added {project_root} to Python path")

# Configure matplotlib
plt.ion()  # Turn on interactive mode
plt.style.use('seaborn-v0_8')  # Use a nice style
debug_print("Matplotlib configured")

from libs.core_classes import DataLoader, DataCleaner
debug_print("Imported core classes")

def print_section(title):
    """Print a formatted section title."""
    print("\n" + "="*80)
    print(f" {title} ".center(80, "="))
    print("="*80 + "\n")

def load_and_prepare_data():
    """
    Load and prepare the training and test data.
    Returns:
        tuple: (train_features, train_labels, test_features)
    """
    debug_print("Starting data loading")
    
    # Initialize data paths
    data_dir = Path("data/raw")
    debug_print(f"Data directory: {data_dir}")
    debug_print(f"Directory exists: {data_dir.exists()}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")
    
    train_features_path = data_dir / "Training_Data_Features.csv"
    train_labels_path = data_dir / "Training_Data_Labels.csv"
    test_features_path = data_dir / "Test_Data_Features.csv"
    
    debug_print(f"Training features path: {train_features_path}")
    debug_print(f"Training labels path: {train_labels_path}")
    debug_print(f"Test features path: {test_features_path}")
    
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

def analyze_temporal_features(df):
    """
    Analyze temporal features of the dataset.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    print_section("Analyzing Temporal Features")
    
    if 'week_start_date' in df.columns:
        # Convert to datetime
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        
        # Extract temporal features
        df['year'] = df['week_start_date'].dt.year
        df['month'] = df['week_start_date'].dt.month
        df['week'] = df['week_start_date'].dt.isocalendar().week
        
        # Print temporal statistics
        print("\nTemporal Coverage:")
        print(f"Start Date: {df['week_start_date'].min()}")
        print(f"End Date: {df['week_start_date'].max()}")
        print(f"Total Weeks: {len(df['week_start_date'].unique())}")
        print(f"Years Covered: {df['year'].unique()}")
        
        # Plot temporal distribution
        plt.figure(figsize=(15, 5))
        df.groupby('year')['week'].count().plot(kind='bar')
        plt.title('Data Distribution by Year')
        plt.xlabel('Year')
        plt.ylabel('Number of Weeks')
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)  # Ensure plot is displayed

def analyze_weather_features(df):
    """
    Analyze weather-related features.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    print_section("Analyzing Weather Features")
    
    # Group weather features
    temp_features = [col for col in df.columns if 'temp' in col.lower()]
    precip_features = [col for col in df.columns if 'precip' in col.lower()]
    humidity_features = [col for col in df.columns if 'humidity' in col.lower()]
    
    # Print feature counts
    print(f"\nTemperature Features: {len(temp_features)}")
    print(f"Precipitation Features: {len(precip_features)}")
    print(f"Humidity Features: {len(humidity_features)}")
    
    # Plot temperature features
    if temp_features:
        plt.figure(figsize=(15, 5))
        for feature in temp_features:
            plt.plot(df[feature], label=feature)
        plt.title('Temperature Features')
        plt.xlabel('Time')
        plt.ylabel('Temperature')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)
    
    # Plot precipitation features
    if precip_features:
        plt.figure(figsize=(15, 5))
        for feature in precip_features:
            plt.plot(df[feature], label=feature)
        plt.title('Precipitation Features')
        plt.xlabel('Time')
        plt.ylabel('Precipitation')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

def analyze_ndvi_features(df):
    """
    Analyze NDVI (vegetation) features.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    print_section("Analyzing NDVI Features")
    
    ndvi_features = [col for col in df.columns if 'ndvi' in col.lower()]
    
    print(f"\nNumber of NDVI Features: {len(ndvi_features)}")
    print("NDVI Features:", ndvi_features)
    
    if ndvi_features:
        plt.figure(figsize=(15, 5))
        for feature in ndvi_features:
            plt.plot(df[feature], label=feature)
        plt.title('NDVI Features')
        plt.xlabel('Time')
        plt.ylabel('NDVI')
        plt.legend()
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

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
    
    # Suggest handling strategies based on feature type
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

def handle_missing_values(df, strategy='auto'):
    """
    Handle missing values based on the suggested strategy.
    
    Args:
        df (pd.DataFrame): DataFrame to process
        strategy (str): 'auto' for automatic strategy selection, or specific strategy
        
    Returns:
        pd.DataFrame: Processed DataFrame
    """
    # Create a copy to avoid modifying the original DataFrame
    df = df.copy()
    
    if strategy == 'auto':
        # Extract month from week_start_date if it exists
        if 'week_start_date' in df.columns:
            df['month'] = pd.to_datetime(df['week_start_date']).dt.month
        
        for col in df.columns:
            if df[col].isnull().sum() > 0:
                percentage = (df[col].isnull().sum() / len(df)) * 100
                
                if percentage > 50:
                    # Drop columns with more than 50% missing values
                    df = df.drop(columns=[col])
                else:
                    # Apply feature-specific imputation
                    if 'temp' in col.lower():
                        # Linear interpolation for temperature data
                        df[col] = df[col].interpolate(method='linear')
                    elif 'precip' in col.lower():
                        # Zero imputation for precipitation data
                        df[col] = df[col].fillna(0)
                    elif 'ndvi' in col.lower():
                        # Seasonal mean imputation for NDVI data
                        if 'month' in df.columns:
                            df[col] = df.groupby('month')[col].transform(lambda x: x.fillna(x.mean()))
                        else:
                            # Fallback to simple mean if no date information is available
                            df[col] = df[col].fillna(df[col].mean())
                    else:
                        # Mean imputation for other features
                        df[col] = df[col].fillna(df[col].mean())
        
        # Remove temporary month column if it was created
        if 'month' in df.columns and 'month' not in df.columns.tolist():
            df = df.drop(columns=['month'])
            
    else:
        # Use the specified strategy for all columns
        if strategy == 'mean':
            df = df.fillna(df.mean())
        elif strategy == 'median':
            df = df.fillna(df.median())
        elif strategy == 'mode':
            df = df.fillna(df.mode().iloc[0])
        elif strategy == 'drop':
            df = df.dropna()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")
    
    return df

def basic_data_analysis(df, name="Dataset"):
    """
    Perform basic data analysis on a DataFrame.
    """
    debug_print(f"Starting basic analysis for {name}")
    
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
    
    debug_print(f"Completed basic analysis for {name}")

def analyze_correlations(df, target_col=None):
    """
    Analyze correlations between features and optionally with target variable.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        target_col (str, optional): Target column for correlation analysis
    """
    print_section("Correlation Analysis")
    
    # Select numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns
    
    # Calculate correlations
    corr_matrix = df[numerical_cols].corr()
    
    # Plot correlation matrix
    plt.figure(figsize=(15, 12))
    plt.matshow(corr_matrix, fignum=1)
    plt.colorbar()
    plt.xticks(range(len(numerical_cols)), numerical_cols, rotation=90)
    plt.yticks(range(len(numerical_cols)), numerical_cols)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    plt.pause(0.1)
    
    # If target column is provided, show top correlations with target
    if target_col and target_col in df.columns:
        target_correlations = corr_matrix[target_col].sort_values(ascending=False)
        print("\nTop correlations with target variable:")
        print(target_correlations.head(10))
        
        # Plot top correlations
        plt.figure(figsize=(10, 6))
        target_correlations.drop(target_col).head(10).plot(kind='bar')
        plt.title(f'Top 10 Correlations with {target_col}')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

def analyze_time_series(df, target_col=None):
    """
    Analyze time series patterns in the data.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
        target_col (str, optional): Target column to plot
    """
    print_section("Time Series Analysis")
    
    # Convert week_start_date to datetime if it exists
    if 'week_start_date' in df.columns:
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        
        # Plot time series of target variable if provided
        if target_col and target_col in df.columns:
            plt.figure(figsize=(15, 6))
            df.set_index('week_start_date')[target_col].plot()
            plt.title(f'Time Series of {target_col}')
            plt.xlabel('Date')
            plt.ylabel(target_col)
            plt.tight_layout()
            plt.show()
            plt.pause(0.1)
        
        # Plot monthly averages
        df['month'] = df['week_start_date'].dt.month
        monthly_avg = df.groupby('month')[target_col].mean()
        
        plt.figure(figsize=(10, 6))
        monthly_avg.plot(kind='bar')
        plt.title(f'Monthly Average of {target_col}')
        plt.xlabel('Month')
        plt.ylabel(f'Average {target_col}')
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

def analyze_city_differences(df):
    """
    Analyze differences between cities.
    
    Args:
        df (pd.DataFrame): DataFrame to analyze
    """
    print_section("City Analysis")
    
    if 'city' in df.columns:
        # Group by city and calculate statistics
        city_stats = df.groupby('city').agg({
            'total_cases': ['mean', 'std', 'min', 'max', 'sum']
        })
        print("\nStatistics by City:")
        print(city_stats)
        
        # Plot total cases by city
        plt.figure(figsize=(10, 6))
        df.groupby('city')['total_cases'].sum().plot(kind='bar')
        plt.title('Total Cases by City')
        plt.xlabel('City')
        plt.ylabel('Total Cases')
        plt.tight_layout()
        plt.show()
        plt.pause(0.1)

def main():
    try:
        debug_print("Starting main function")
        
        # Load data
        debug_print("Loading data...")
        train_features, train_labels, test_features = load_and_prepare_data()
        
        # Merge features and labels for complete analysis
        train_data = pd.merge(train_features, train_labels, on=['city', 'year', 'weekofyear'])
        
        # Basic analysis
        debug_print("Performing basic analysis...")
        basic_data_analysis(train_features, "Training Features")
        basic_data_analysis(train_labels, "Training Labels")
        basic_data_analysis(test_features, "Test Features")
        
        # Correlation analysis
        analyze_correlations(train_data, target_col='total_cases')
        
        # Time series analysis
        analyze_time_series(train_data, target_col='total_cases')
        
        # City analysis
        analyze_city_differences(train_data)
        
        # Handle missing values using automatic strategy
        print_section("Handling Missing Values")
        train_features_clean = handle_missing_values(train_features, strategy='auto')
        train_labels_clean = handle_missing_values(train_labels, strategy='auto')
        test_features_clean = handle_missing_values(test_features, strategy='auto')
        
        # Save cleaned data
        print_section("Saving Cleaned Data")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        train_features_clean.to_csv(processed_dir / "cleaned_train_features.csv", index=False)
        train_labels_clean.to_csv(processed_dir / "cleaned_train_labels.csv", index=False)
        test_features_clean.to_csv(processed_dir / "cleaned_test_features.csv", index=False)
        
        print("\nCleaned data has been saved to the 'data/processed' directory.")
        
        # Keep plots open
        plt.show(block=True)
        
        debug_print("Analysis completed successfully")
        
    except Exception as e:
        debug_print(f"Error occurred: {str(e)}")
        import traceback
        debug_print("Traceback:")
        debug_print(traceback.format_exc())
        raise

if __name__ == "__main__":
    debug_print("Script started")
    main()
    debug_print("Script completed")
