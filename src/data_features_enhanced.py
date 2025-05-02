"""
Enhanced Feature Engineering Module

This module handles advanced feature engineering for the Dengue Fever prediction model.
It creates more sophisticated features from the cleaned data to improve model performance,
with special attention to temporal dynamics, mosquito lifecycle, and disease patterns.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import os
import sys
from sklearn.preprocessing import PowerTransformer

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

def add_enhanced_temporal_features(df):
    """
    Add enhanced temporal features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with date column
    
    Returns:
        pd.DataFrame: DataFrame with added temporal features
    """
    print_section("Adding Enhanced Temporal Features")
    
    # Create a copy to avoid modifying the original
    df_temp = df.copy()
    
    # Convert date to datetime
    df_temp['week_start_date'] = pd.to_datetime(df_temp['week_start_date'])
    
    # Basic temporal features
    df_temp['year'] = df_temp['week_start_date'].dt.year
    df_temp['month'] = df_temp['week_start_date'].dt.month
    df_temp['weekofyear'] = df_temp['week_start_date'].dt.isocalendar().week
    df_temp['dayofyear'] = df_temp['week_start_date'].dt.dayofyear
    
    # Better cyclic features for week of year using sine/cosine transformations
    # The period for week of year is 52 weeks
    df_temp['weekofyear_sin'] = np.sin(2 * np.pi * df_temp['weekofyear'] / 52)
    df_temp['weekofyear_cos'] = np.cos(2 * np.pi * df_temp['weekofyear'] / 52)
    
    # Add month cyclical features
    df_temp['month_sin'] = np.sin(2 * np.pi * df_temp['month'] / 12)
    df_temp['month_cos'] = np.cos(2 * np.pi * df_temp['month'] / 12)
    
    # Add day of year cyclical features (365 days)
    df_temp['dayofyear_sin'] = np.sin(2 * np.pi * df_temp['dayofyear'] / 365)
    df_temp['dayofyear_cos'] = np.cos(2 * np.pi * df_temp['dayofyear'] / 365)
    
    # Add quarter feature
    df_temp['quarter'] = df_temp['week_start_date'].dt.quarter
    df_temp['quarter_sin'] = np.sin(2 * np.pi * df_temp['quarter'] / 4)
    df_temp['quarter_cos'] = np.cos(2 * np.pi * df_temp['quarter'] / 4)
    
    # Add custom season indicators (specific to tropical dengue regions)
    # For Northern Hemisphere locations like San Juan
    # Wet season: May to November, Dry season: December to April
    df_temp['is_wet_season_north'] = ((df_temp['month'] >= 5) & (df_temp['month'] <= 11)).astype(int)
    
    # For Southern Hemisphere locations like Iquitos
    # Wet season: November to April, Dry season: May to October
    df_temp['is_wet_season_south'] = (((df_temp['month'] >= 11) & (df_temp['month'] <= 12)) | 
                                     ((df_temp['month'] >= 1) & (df_temp['month'] <= 4))).astype(int)
    
    # Different wet seasons by city (applying the correct season indicator)
    df_temp['is_wet_season'] = np.where(
        df_temp['city'] == 'sj', 
        df_temp['is_wet_season_north'],
        df_temp['is_wet_season_south']
    )
    
    return df_temp

def add_enhanced_weather_features(df):
    """
    Add enhanced weather features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with weather columns
    
    Returns:
        pd.DataFrame: DataFrame with added weather features
    """
    print_section("Adding Enhanced Weather Features")
    
    # Create a copy to avoid modifying the original
    df_weather = df.copy()
    
    # Initialize all required weather features with 0
    required_features = [
        'temp_avg',
        'temp_min',
        'temp_max',
        'temp_range',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg',
        'dew_point'
    ]
    
    for feature in required_features:
        if feature not in df_weather.columns:
            df_weather[feature] = 0
    
    # Temperature features
    temp_cols = [col for col in df.columns if 'temp' in col.lower() and 'lag' not in col.lower()]
    if temp_cols:
        df_weather['temp_avg'] = df[temp_cols].mean(axis=1)
        df_weather['temp_min'] = df[temp_cols].min(axis=1)
        df_weather['temp_max'] = df[temp_cols].max(axis=1)
        df_weather['temp_range'] = df_weather['temp_max'] - df_weather['temp_min']
    
    # Humidity features
    humidity_cols = [col for col in df.columns if 'humidity' in col.lower() and 'lag' not in col.lower()]
    if humidity_cols:
        df_weather['humidity_avg'] = df[humidity_cols].mean(axis=1)
    
    # Precipitation features
    precip_cols = [col for col in df.columns if 'precip' in col.lower() and 'lag' not in col.lower()]
    if precip_cols:
        df_weather['precip_total'] = df[precip_cols].sum(axis=1)
        df_weather['precip_days'] = (df[precip_cols] > 0).sum(axis=1)
        
        # Add precipitation intensity
        precip_days = df_weather['precip_days'].replace(0, 1)  # Avoid division by zero
        df_weather['precip_intensity'] = df_weather['precip_total'] / precip_days
        df_weather.loc[df_weather['precip_total'] == 0, 'precip_intensity'] = 0
    
    # Pressure features
    pressure_cols = [col for col in df.columns if 'pressure' in col.lower() and 'lag' not in col.lower()]
    if pressure_cols:
        df_weather['pressure_avg'] = df[pressure_cols].mean(axis=1)
    
    # Calculate dew point if temperature and humidity are available
    if 'temp_avg' in df_weather.columns and 'humidity_avg' in df_weather.columns:
        # Approximate dew point using the Magnus formula
        a = 17.27
        b = 237.7
        alpha = ((a * df_weather['temp_avg']) / (b + df_weather['temp_avg'])) + np.log(df_weather['humidity_avg'] / 100.0)
        df_weather['dew_point'] = (b * alpha) / (a - alpha)
    
    # Add interaction terms between temperature and humidity
    # These are important for mosquito breeding and survival
    df_weather['temp_humidity_interaction'] = df_weather['temp_avg'] * df_weather['humidity_avg']
    
    # Create "mosquito comfort index" - combining temperature and humidity in ideal mosquito conditions
    # Aedes aegypti mosquitoes thrive in temperatures 25-30°C with high humidity
    # This is a simplified index - could be refined with expert knowledge
    temp_optimal = 27.5  # Midpoint of ideal temperature range
    humidity_optimal = 80  # Approximate ideal humidity
    
    df_weather['mosquito_temp_factor'] = 1 - abs(df_weather['temp_avg'] - temp_optimal) / 10
    df_weather['mosquito_temp_factor'] = df_weather['mosquito_temp_factor'].clip(0, 1)
    
    df_weather['mosquito_humidity_factor'] = 1 - abs(df_weather['humidity_avg'] - humidity_optimal) / 50
    df_weather['mosquito_humidity_factor'] = df_weather['mosquito_humidity_factor'].clip(0, 1)
    
    df_weather['mosquito_comfort_index'] = df_weather['mosquito_temp_factor'] * df_weather['mosquito_humidity_factor']
    
    return df_weather

def add_extended_weather_lag_features(df):
    """
    Add extended lag features for weather data to capture mosquito lifecycle and virus incubation.
    
    Args:
        df (pd.DataFrame): DataFrame with weather features
    
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    print_section("Adding Extended Weather Lag Features")
    
    # Define weather features for lagging
    weather_features = [
        'temp_avg',
        'temp_min',
        'temp_max',
        'temp_range',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg',
        'dew_point',
        'mosquito_comfort_index'
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
        
        # Create extended lag features (8-12 weeks) to capture full mosquito lifecycle
        # and virus incubation period
        for feature in weather_features:
            # Standard lags (1-4 weeks)
            for lag in range(1, 5):
                lag_col = f'{feature}_lag_{lag}'
                df_lagged.loc[city_mask, lag_col] = df_lagged.loc[city_mask, feature].shift(lag)
            
            # Extended lags (8, 10, 12 weeks) to capture longer-term effects
            for lag in [8, 10, 12]:
                lag_col = f'{feature}_lag_{lag}'
                df_lagged.loc[city_mask, lag_col] = df_lagged.loc[city_mask, feature].shift(lag)
    
    # Fill NaN values with 0
    df_lagged = df_lagged.fillna(0)
    
    return df_lagged

def add_cumulative_precipitation_features(df):
    """
    Add cumulative precipitation features to represent standing water over multiple weeks.
    
    Args:
        df (pd.DataFrame): DataFrame with precipitation features
    
    Returns:
        pd.DataFrame: DataFrame with cumulative precipitation features
    """
    print_section("Adding Cumulative Precipitation Features")
    
    # Create a copy to avoid modifying the original
    df_precip = df.copy()
    
    # Ensure precipitation features exist
    if 'precip_total' not in df_precip.columns:
        print("Warning: 'precip_total' column not found. Skipping cumulative precipitation features.")
        return df_precip
    
    # Create cumulative precipitation over different time windows by city
    for city in df['city'].unique():
        city_mask = df['city'] == city
        
        # Calculate rolling sums of precipitation over 2, 3, and 4 weeks
        # This represents standing water accumulation
        for window in [2, 3, 4]:
            col_name = f'precip_total_cum_{window}w'
            df_precip.loc[city_mask, col_name] = df_precip.loc[city_mask, 'precip_total'].rolling(
                window=window, min_periods=1).sum()
        
        # Calculate exponentially weighted precipitation
        # This gives more weight to recent rain but still accounts for standing water
        # Decay factors: 0.8 (slower decay), 0.6 (medium decay), 0.4 (faster decay)
        for alpha in [0.8, 0.6, 0.4]:
            col_name = f'precip_total_ewm_a{int(alpha*10)}'
            df_precip.loc[city_mask, col_name] = df_precip.loc[city_mask, 'precip_total'].ewm(
                alpha=alpha, adjust=False).mean()
    
    # Fill NaN values with 0
    df_precip = df_precip.fillna(0)
    
    return df_precip

def add_enhanced_ndvi_features(df):
    """
    Add enhanced NDVI-related features to the DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame with NDVI columns
    
    Returns:
        pd.DataFrame: DataFrame with added NDVI features
    """
    print_section("Adding Enhanced NDVI Features")
    
    # Create a copy to avoid modifying the original
    df_ndvi = df.copy()
    
    # Calculate NDVI statistics
    ndvi_cols = [col for col in df.columns if 'ndvi' in col.lower()]
    
    if ndvi_cols:
        # Calculate statistics for each city separately
        for city in df['city'].unique():
            city_mask = df['city'] == city
            
            # More sophisticated rolling statistics for NDVI
            for col in ndvi_cols:
                # Rolling mean with different windows
                for window in [2, 4, 8]:
                    df_ndvi.loc[city_mask, f'{col}_mean_{window}w'] = (
                        df.loc[city_mask, col].rolling(window=window, min_periods=1).mean()
                    )
                
                # Rolling standard deviation with different windows
                for window in [4, 8]:
                    df_ndvi.loc[city_mask, f'{col}_std_{window}w'] = (
                        df.loc[city_mask, col].rolling(window=window, min_periods=2).std()
                    )
                
                # Exponentially weighted mean for NDVI
                df_ndvi.loc[city_mask, f'{col}_ewm'] = (
                    df.loc[city_mask, col].ewm(span=4, min_periods=1).mean()
                )
                
                # Rate of change in NDVI (vegetation growth/decline)
                df_ndvi.loc[city_mask, f'{col}_diff'] = df.loc[city_mask, col].diff()
                
                # NDVI anomaly from seasonal average
                # Group by week of year and calculate the mean NDVI for each week
                weekly_ndvi_mean = df.loc[city_mask].groupby('weekofyear')[col].transform('mean')
                df_ndvi.loc[city_mask, f'{col}_seasonality'] = df.loc[city_mask, col] - weekly_ndvi_mean
    
    # Fill NaN values with 0
    df_ndvi = df_ndvi.fillna(0)
    
    return df_ndvi

def add_autoregressive_features(df):
    """
    Add autoregressive features to capture outbreak dynamics.
    
    Args:
        df (pd.DataFrame): DataFrame with total_cases
    
    Returns:
        pd.DataFrame: DataFrame with autoregressive features
    """
    print_section("Adding Autoregressive Features")
    
    # Create a copy to avoid modifying the original
    df_ar = df.copy()
    
    # Check if total_cases is in the dataframe (should be in training data but not test)
    if 'total_cases' not in df_ar.columns:
        print("Warning: 'total_cases' column not found. Skipping autoregressive features.")
        return df_ar
    
    # Process by city
    for city in df['city'].unique():
        city_mask = df['city'] == city
        
        # Lag features of total_cases (previous case counts)
        for lag in [1, 2, 3, 4, 8, 12]:
            lag_col = f'cases_lag_{lag}'
            df_ar.loc[city_mask, lag_col] = df_ar.loc[city_mask, 'total_cases'].shift(lag)
        
        # Moving averages of cases
        for window in [2, 4, 8]:
            ma_col = f'cases_ma_{window}'
            df_ar.loc[city_mask, ma_col] = df_ar.loc[city_mask, 'total_cases'].rolling(
                window=window, min_periods=1).mean()
        
        # Exponentially weighted moving average of cases
        for alpha in [0.8, 0.6, 0.4]:
            ewma_col = f'cases_ewma_a{int(alpha*10)}'
            df_ar.loc[city_mask, ewma_col] = df_ar.loc[city_mask, 'total_cases'].ewm(
                alpha=alpha, adjust=False).mean()
        
        # Case momentum (percent change)
        mom_col = 'cases_momentum'
        df_ar.loc[city_mask, mom_col] = df_ar.loc[city_mask, 'total_cases'].pct_change(
            periods=4, fill_method='ffill')
        
        # Outbreak indicator (1 if cases are increasing for 2 consecutive weeks)
        df_ar.loc[city_mask, 'outbreak_indicator'] = (
            (df_ar.loc[city_mask, 'total_cases'].diff() > 0) & 
            (df_ar.loc[city_mask, 'total_cases'].diff().shift() > 0)
        ).astype(int)
        
        # Calculate historical seasonal patterns
        df_ar.loc[city_mask, 'historical_seasonal_pattern'] = df_ar.loc[city_mask].groupby('weekofyear')['total_cases'].transform('mean')
        
        # Deviation from historical pattern
        df_ar.loc[city_mask, 'seasonal_deviation'] = df_ar.loc[city_mask, 'total_cases'] - df_ar.loc[city_mask, 'historical_seasonal_pattern']
    
    # Fill NaN values appropriately
    df_ar = df_ar.fillna(0)
    
    return df_ar

def add_interaction_features(df):
    """
    Add interaction features between key variables.
    
    Args:
        df (pd.DataFrame): DataFrame with weather features
    
    Returns:
        pd.DataFrame: DataFrame with interaction features
    """
    print_section("Adding Interaction Features")
    
    # Create a copy to avoid modifying the original
    df_interact = df.copy()
    
    # Interactions between temperature and humidity
    # Both current values and lagged values
    temp_humidity_pairs = [
        # Current interactions
        ('temp_avg', 'humidity_avg'),
        ('temp_min', 'humidity_avg'),
        ('temp_max', 'humidity_avg'),
        # Lagged interactions (important for mosquito breeding cycles)
        ('temp_avg_lag_2', 'humidity_avg_lag_2'),
        ('temp_avg_lag_4', 'humidity_avg_lag_4'),
        ('temp_avg_lag_8', 'humidity_avg_lag_8')
    ]
    
    for temp_col, humidity_col in temp_humidity_pairs:
        if temp_col in df_interact.columns and humidity_col in df_interact.columns:
            interaction_col = f"{temp_col}_{humidity_col}_interact"
            df_interact[interaction_col] = df_interact[temp_col] * df_interact[humidity_col]
    
    # Interactions between precipitation and temperature
    precip_temp_pairs = [
        ('precip_total', 'temp_avg'),
        ('precip_total_cum_4w', 'temp_avg'),
        ('precip_total_lag_2', 'temp_avg_lag_2')
    ]
    
    for precip_col, temp_col in precip_temp_pairs:
        if precip_col in df_interact.columns and temp_col in df_interact.columns:
            interaction_col = f"{precip_col}_{temp_col}_interact"
            df_interact[interaction_col] = df_interact[precip_col] * df_interact[temp_col]
    
    # Interactions between NDVI and precipitation 
    # (vegetation response to rainfall, which affects mosquito habitats)
    ndvi_cols = [col for col in df_interact.columns if 'ndvi' in col.lower() and not 'mean' in col.lower() and not 'std' in col.lower()]
    
    if ndvi_cols and 'precip_total' in df_interact.columns:
        for ndvi_col in ndvi_cols[:1]:  # Limit to one NDVI column to avoid too many features
            interaction_col = f"{ndvi_col}_precip_interact"
            df_interact[interaction_col] = df_interact[ndvi_col] * df_interact['precip_total']
    
    return df_interact

def prepare_enhanced_features(df):
    """
    Prepare all enhanced features for the model.
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with all enhanced features
    """
    print_section("Preparing All Enhanced Features")
    
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # Add enhanced temporal features
    df_features = add_enhanced_temporal_features(df_features)
    
    # Add enhanced weather features
    df_features = add_enhanced_weather_features(df_features)
    
    # Add extended weather lag features
    df_features = add_extended_weather_lag_features(df_features)
    
    # Add cumulative precipitation features
    df_features = add_cumulative_precipitation_features(df_features)
    
    # Add enhanced NDVI features
    df_features = add_enhanced_ndvi_features(df_features)
    
    # Add interaction features
    df_features = add_interaction_features(df_features)
    
    # Add autoregressive features (only for training data)
    if 'total_cases' in df_features.columns:
        df_features = add_autoregressive_features(df_features)
    
    return df_features

def save_featured_data(train_data, test_features):
    """
    Save the featured datasets to the processed directory.
    
    Args:
        train_data (pd.DataFrame): Featured training data
        test_features (pd.DataFrame): Featured test features
    """
    print_section("Saving Enhanced Featured Data")
    
    # Create processed directory if it doesn't exist
    processed_dir = Path("data/processed")
    processed_dir.mkdir(exist_ok=True)
    
    # Save featured data
    train_path = processed_dir / "enhanced_featured_train_data.csv"
    test_path = processed_dir / "enhanced_featured_test_features.csv"
    
    train_data.to_csv(train_path, index=False)
    test_features.to_csv(test_path, index=False)
    
    print(f"Enhanced featured training data saved to {train_path}")
    print(f"Enhanced featured test features saved to {test_path}")
    
    # Save a feature list for documentation
    feature_cols = [col for col in train_data.columns if col not in ['city', 'week_start_date', 'total_cases']]
    feature_list_path = processed_dir / "enhanced_feature_list.txt"
    
    with open(feature_list_path, 'w') as f:
        f.write(f"Total number of features: {len(feature_cols)}\n\n")
        for i, feature in enumerate(feature_cols, 1):
            f.write(f"{i}. {feature}\n")
    
    print(f"Feature list saved to {feature_list_path}")

def main():
    """Main function to run the enhanced feature engineering process."""
    print_section("Starting Enhanced Feature Engineering")
    
    # Load cleaned data
    train_data, test_features = load_cleaned_data()
    
    # Prepare enhanced features for training data
    print("Preparing enhanced features for training data...")
    featured_train_data = prepare_enhanced_features(train_data)
    
    # Prepare enhanced features for test data
    print("Preparing enhanced features for test data...")
    featured_test_features = prepare_enhanced_features(test_features)
    
    # Save featured data
    save_featured_data(featured_train_data, featured_test_features)
    
    print_section("Next Steps")
    print("1. Review the enhanced featured data")
    print("2. Proceed with improved model training")
    print("3. If needed, further adjust feature engineering")

if __name__ == "__main__":
    main()