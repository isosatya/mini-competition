"""
XGBoost Model Training with City-Specific Timeframes

This module trains city-specific XGBoost models for dengue fever prediction
using distinct timeframes for each city:
- San Juan: Training period 1990-2000, Testing period 2000-2007
- Iquitos: Training period 2000-2007, Testing period 2007-2010

The module implements the following features:
- Climate lag features (1, 4, 12 weeks) for temperature, humidity, precipitation
- Cyclical encoding of week-of-year using sine and cosine transformations
- Mosquito breeding condition features
- Evaluation using standard and outbreak detection metrics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import logging
import sys
import time
import warnings
from datetime import datetime

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f"logs/xgboost_timeframes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Suppress warnings
warnings.filterwarnings('ignore')

def print_section(title):
    """Print a formatted section title."""
    section = "\n" + "="*80 + "\n" + f" {title} ".center(80, "=") + "\n" + "="*80 + "\n"
    logger.info(section)

def load_data():
    """
    Load dengue fever dataset and handle missing values.
    
    Returns:
        pd.DataFrame: Combined dataset with both cities
    """
    print_section("Loading Data")
    
    try:
        # Load the merged dataset
        logger.info("Loading cleaned data...")
        df = pd.read_csv('data/processed/cleaned_data_merged.csv')
        
        # Ensure date column is datetime
        df['week_start_date'] = pd.to_datetime(df['week_start_date'])
        
        # Check for and handle missing values in important columns
        missing_count = df.isnull().sum()
        if missing_count.sum() > 0:
            logger.info(f"Found missing values: \n{missing_count[missing_count > 0]}")
            
            # Fill missing values by city
            for city in df['city'].unique():
                city_mask = df['city'] == city
                city_df = df[city_mask]
                
                for col in df.columns:
                    if df[col].isnull().sum() > 0:
                        # For numeric columns, use city mean
                        if np.issubdtype(df[col].dtype, np.number):
                            fill_value = city_df[col].mean()
                            df.loc[city_mask, col] = df.loc[city_mask, col].fillna(fill_value)
                        else:
                            # For non-numeric, forward fill then backward fill
                            df.loc[city_mask, col] = df.loc[city_mask, col].fillna(method='ffill').fillna(method='bfill')
        
        # Log basic information
        logger.info(f"Total rows: {len(df)}")
        logger.info(f"Cities: {', '.join(df['city'].unique())}")
        
        # Get date ranges for each city
        for city in df['city'].unique():
            city_data = df[df['city'] == city]
            logger.info(f"{city} date range: {city_data['week_start_date'].min()} to {city_data['week_start_date'].max()}")
            logger.info(f"{city} total rows: {len(city_data)}")
        
        # Check if 'is_test' column exists
        if 'is_test' in df.columns:
            test_count = df['is_test'].sum()
            train_count = len(df) - test_count
            logger.info(f"Training rows: {train_count}, Test rows: {test_count}")
        
        return df
    except Exception as e:
        logger.error(f"Error loading data: {str(e)}")
        raise

def split_data_by_timeframes(df):
    """
    Split data by city and timeframes:
    - San Juan: Training period 1990-2000, Testing period 2000-2007
    - Iquitos: Training period 2000-2007, Testing period 2007-2010
    
    Args:
        df (pd.DataFrame): Combined dataset
    
    Returns:
        dict: Dictionary with train/test splits for each city
    """
    print_section("Splitting Data by Timeframes")
    
    # Initialize results dictionary
    splits = {}
    
    # Define exact timeframes for each city based on requirements
    timeframes = {
        'sj': {
            'train_start': '1990-01-01',
            'train_end': '2000-01-01',
            'test_start': '2000-01-01',
            'test_end': '2008-01-01'
        },
        'iq': {
            'train_start': '2000-01-01',
            'train_end': '2007-01-01',
            'test_start': '2007-01-01',
            'test_end': '2011-01-01'
        }
    }
    
    # Check if we have data for each city in the required timeframes
    for city, periods in timeframes.items():
        city_data = df[df['city'] == city]
        actual_range = (city_data['week_start_date'].min(), city_data['week_start_date'].max())
        logger.info(f"{city} actual date range: {actual_range[0]} to {actual_range[1]}")
        
        # Convert period strings to timestamps for comparison
        train_start = pd.to_datetime(periods['train_start'])
        test_end = pd.to_datetime(periods['test_end'])
        
        # Check if we have sufficient data for the required timeframes
        if actual_range[0] > train_start:
            logger.warning(f"{city}: Actual start date ({actual_range[0]}) is later than required train start ({train_start})")
            # Update the start date to match available data
            timeframes[city]['train_start'] = actual_range[0].strftime('%Y-%m-%d')
        
        if actual_range[1] < test_end:
            logger.warning(f"{city}: Actual end date ({actual_range[1]}) is earlier than required test end ({test_end})")
            # Update the end date to match available data
            timeframes[city]['test_end'] = actual_range[1].strftime('%Y-%m-%d')
    
    # Split data for each city
    for city, periods in timeframes.items():
        logger.info(f"Processing {city} timeframes...")
        logger.info(f"Train period: {periods['train_start']} to {periods['train_end']}")
        logger.info(f"Test period: {periods['test_start']} to {periods['test_end']}")
        
        # Filter data for current city
        city_data = df[df['city'] == city].copy()
        
        # Convert all date strings to datetime for comparison
        train_start = pd.to_datetime(periods['train_start'])
        train_end = pd.to_datetime(periods['train_end'])
        test_start = pd.to_datetime(periods['test_start'])
        test_end = pd.to_datetime(periods['test_end'])
        
        # Split based on timeframes
        train_mask = (
            (city_data['week_start_date'] >= train_start) & 
            (city_data['week_start_date'] < train_end)
        )
        test_mask = (
            (city_data['week_start_date'] >= test_start) & 
            (city_data['week_start_date'] < test_end)
        )
        
        train_data = city_data[train_mask].copy()
        test_data = city_data[test_mask].copy()
        
        # Ensure we have the total_cases column for both training and testing
        if 'total_cases' not in train_data.columns or train_data['total_cases'].isnull().all():
            logger.error(f"Missing 'total_cases' in training data for {city}")
            raise ValueError(f"Missing 'total_cases' in training data for {city}")
        
        # For testing data, if it's part of the test set without labels, 
        # we'll need to handle it differently during evaluation
        test_has_labels = 'total_cases' in test_data.columns and not test_data['total_cases'].isnull().all()
        if not test_has_labels:
            logger.warning(f"Testing data for {city} does not have labels")
        
        # Log split information
        logger.info(f"{city} - Training period: {train_data['week_start_date'].min()} to {train_data['week_start_date'].max()}")
        logger.info(f"{city} - Testing period: {test_data['week_start_date'].min()} to {test_data['week_start_date'].max()}")
        logger.info(f"{city} - Training samples: {len(train_data)}, Testing samples: {len(test_data)}")
        
        # Store splits
        splits[city] = {
            'train': train_data,
            'test': test_data,
            'test_has_labels': test_has_labels
        }
    
    return splits

def engineer_features(df):
    """
    Engineer features for the model including:
    - Climate lag features (1, 4, 12 weeks)
    - Cyclical encoding of week-of-year
    - Mosquito breeding condition features
    
    Args:
        df (pd.DataFrame): Input DataFrame
    
    Returns:
        pd.DataFrame: DataFrame with engineered features
    """
    print_section("Engineering Features")
    
    # Create a copy to avoid modifying the original
    df_features = df.copy()
    
    # 1. Climate lag features
    logger.info("Creating climate lag features...")
    
    # Define climate features to create lags for
    climate_features = [
        'reanalysis_air_temp_k',
        'reanalysis_relative_humidity_percent', 
        'reanalysis_precip_amt_kg_per_m2',
        'reanalysis_dew_point_temp_k',
        'reanalysis_specific_humidity_g_per_kg',
        'station_avg_temp_c',
        'station_precip_mm'
    ]
    
    # Filter only columns that exist in the dataframe
    available_climate_features = [col for col in climate_features if col in df_features.columns]
    if len(available_climate_features) < len(climate_features):
        logger.warning(f"Some climate features are missing from dataset. Available: {available_climate_features}")
    
    # Create lag features for each city separately
    for city in df_features['city'].unique():
        city_mask = df_features['city'] == city
        city_df = df_features[city_mask].sort_values('week_start_date')
        
        for feature in available_climate_features:
            # Create specific lags at 1, 4, and 12 weeks as required
            for lag in [1, 4, 12]:
                lag_col = f'{feature}_lag_{lag}'
                
                # Shift values within each city's data
                df_features.loc[city_mask, lag_col] = city_df[feature].shift(lag).values
    
    # 2. Cyclical encoding of week-of-year
    logger.info("Creating cyclical time features...")
    
    # Sine and cosine transformations for week of year (period = 52 weeks)
    df_features['weekofyear_sin'] = np.sin(2 * np.pi * df_features['weekofyear'] / 52)
    df_features['weekofyear_cos'] = np.cos(2 * np.pi * df_features['weekofyear'] / 52)
    
    # Add month cyclical features
    if 'month' in df_features.columns:
        df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
        df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # 3. Create mosquito breeding condition features
    logger.info("Creating mosquito breeding condition features...")
    
    # Optimal temperature for Aedes aegypti mosquitoes: 25-30°C
    # Optimal humidity: 70-90%
    
    # Convert temperature to Celsius for easier interpretation of breeding conditions
    if 'reanalysis_air_temp_k' in df_features.columns:
        df_features['temp_c'] = df_features['reanalysis_air_temp_k'] - 273.15
    elif 'station_avg_temp_c' in df_features.columns:
        df_features['temp_c'] = df_features['station_avg_temp_c']
    
    # Temperature suitability for mosquito breeding (optimal around 28°C)
    if 'temp_c' in df_features.columns:
        df_features['temp_suitability'] = 1 - abs(df_features['temp_c'] - 28) / 15
        df_features['temp_suitability'] = df_features['temp_suitability'].clip(0, 1)
    
    # Humidity suitability for mosquito breeding (optimal around 80%)
    if 'reanalysis_relative_humidity_percent' in df_features.columns:
        humidity_col = 'reanalysis_relative_humidity_percent'
        df_features['humidity_suitability'] = 1 - abs(df_features[humidity_col] - 80) / 50
        df_features['humidity_suitability'] = df_features['humidity_suitability'].clip(0, 1)
    
    # Mosquito breeding index: combining temperature and humidity
    if 'temp_suitability' in df_features.columns and 'humidity_suitability' in df_features.columns:
        df_features['mosquito_breeding_index'] = df_features['temp_suitability'] * df_features['humidity_suitability']
    
    # Standing water index (important for mosquito breeding)
    for city in df_features['city'].unique():
        city_mask = df_features['city'] == city
        
        # Use available precipitation column
        precip_col = None
        for col in ['reanalysis_precip_amt_kg_per_m2', 'precipitation_amt_mm', 'station_precip_mm']:
            if col in df_features.columns:
                precip_col = col
                break
        
        if precip_col is not None:
            # Calculate rolling precipitation metrics with different windows
            city_df = df_features[city_mask].sort_values('week_start_date')
            
            # Recent precipitation (1-2 weeks) - short-term breeding sites
            df_features.loc[city_mask, 'precip_recent'] = city_df[precip_col].rolling(window=2, min_periods=1).sum().values
            
            # Medium-term accumulation (3-4 weeks) - sustained breeding sites
            df_features.loc[city_mask, 'precip_medium'] = city_df[precip_col].rolling(window=4, min_periods=1).sum().values
            
            # Longer-term accumulation (8-12 weeks) - environmental saturation
            df_features.loc[city_mask, 'precip_long'] = city_df[precip_col].rolling(window=12, min_periods=1).sum().values
            
            # Exponential decay for precipitation (recent rain matters more)
            # This simulates standing water that evaporates over time
            alpha = 0.7  # Decay factor
            df_features.loc[city_mask, 'precip_exp_decay'] = city_df[precip_col].ewm(alpha=alpha, adjust=False).mean().values
    
    # Create interaction terms between temperature and precipitation
    if 'temp_c' in df_features.columns and 'precip_recent' in df_features.columns:
        # Warm and wet conditions are ideal for mosquito breeding
        df_features['warm_wet_interaction'] = df_features['temp_c'] * df_features['precip_recent'] / 100
    
    # Diurnal temperature range effect (DTR affects mosquito survival)
    if 'reanalysis_tdtr_k' in df_features.columns:
        # Lower DTR (stable temperatures) is better for mosquitoes
        df_features['temp_stability'] = 1 / (1 + df_features['reanalysis_tdtr_k'])
    elif 'station_diur_temp_rng_c' in df_features.columns:
        df_features['temp_stability'] = 1 / (1 + df_features['station_diur_temp_rng_c'])
    
    # Create vegetation index features if NDVI columns are available
    ndvi_cols = [col for col in df_features.columns if 'ndvi' in col.lower()]
    if ndvi_cols:
        # Average NDVI (vegetation indicator, related to humidity and habitat)
        df_features['ndvi_avg'] = df_features[ndvi_cols].mean(axis=1)
        
        # For each city, create rolling NDVI features
        for city in df_features['city'].unique():
            city_mask = df_features['city'] == city
            city_df = df_features[city_mask].sort_values('week_start_date')
            
            # NDVI change (vegetation growth or decay)
            df_features.loc[city_mask, 'ndvi_change'] = city_df['ndvi_avg'].diff().values
    
    # Fill missing values created by lag and rolling operations, using city-specific means
    for col in df_features.columns:
        if df_features[col].isnull().any():
            for city in df_features['city'].unique():
                city_mask = df_features['city'] == city
                city_mean = df_features.loc[city_mask, col].mean()
                
                # If city mean is also NaN, use 0
                if pd.isna(city_mean):
                    city_mean = 0
                
                # Fill missing values
                df_features.loc[city_mask & df_features[col].isnull(), col] = city_mean
    
    # Final check for any remaining missing values
    missing_values = df_features.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Found {missing_values} missing values after feature engineering. Filling with zeros.")
        df_features = df_features.fillna(0)
    
    # Log created features
    feature_cols = [col for col in df_features.columns if col not in ['city', 'year', 'week_start_date', 'weekofyear', 'total_cases']]
    logger.info(f"Created {len(feature_cols)} features.")
    logger.info(f"Engineered features shape: {df_features.shape}")
    
    # Log a few examples of the new features
    new_feature_examples = [col for col in feature_cols if col not in df.columns][:10]
    if new_feature_examples:
        logger.info(f"Examples of new features: {new_feature_examples}")
    
    return df_features

def prepare_features(df, is_training=True, scaler=None):
    """
    Prepare features for model training or prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        is_training (bool): Whether preparing for training or prediction
        scaler (object, optional): Fitted scaler for feature normalization
    
    Returns:
        tuple: (X, y, scaler) where y and scaler are None if is_training is False
    """
    from sklearn.preprocessing import StandardScaler
    
    # Get target variable first (before dropping columns)
    y = df['total_cases'] if is_training else None
    
    # Drop non-feature columns
    non_feature_cols = ['city', 'year', 'week_start_date', 'weekofyear', 'total_cases']
    
    X = df.drop(columns=[col for col in non_feature_cols if col in df.columns])
    
    # Ensure no missing values in features
    X = X.fillna(0)
    
    # Normalize numerical features
    if is_training:
        # Create and fit a new scaler
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(
            scaler.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled, y, scaler
    elif scaler is not None:
        # Use the provided scaler
        X_scaled = pd.DataFrame(
            scaler.transform(X),
            columns=X.columns,
            index=X.index
        )
        return X_scaled, y, None
    else:
        # No scaling
        return X, y, None

def train_xgboost_model(X_train, y_train, X_val=None, y_val=None, city=None):
    """
    Train an XGBoost model for dengue fever prediction.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
        X_val (pd.DataFrame, optional): Validation features
        y_val (pd.Series, optional): Validation target
        city (str, optional): City name for logging
    
    Returns:
        tuple: (xgb.Booster model, log_transform_flag)
    """
    print_section(f"Training XGBoost Model for {city}" if city else "Training XGBoost Model")
    
    # Check for training data
    if X_train.shape[0] == 0 or y_train.shape[0] == 0:
        raise ValueError("Empty training data")
    
    # Log training data statistics
    logger.info(f"Training data shape: {X_train.shape}")
    logger.info(f"Target statistics - Min: {y_train.min()}, Max: {y_train.max()}, Mean: {y_train.mean():.2f}")
    
    # Handle zero values in target (important for count-based objectives)
    if (y_train == 0).any():
        zero_count = (y_train == 0).sum()
        zero_pct = zero_count / len(y_train) * 100
        logger.info(f"Target contains {zero_count} zeros ({zero_pct:.2f}%)")
    
    # Create internal cross-validation set if validation set is not provided
    if X_val is None or y_val is None:
        logger.info("No validation set provided, creating an internal 20% validation set")
        from sklearn.model_selection import train_test_split
        
        # To maintain temporal order, use the last 20% of the data as validation
        val_size = int(0.2 * X_train.shape[0])
        
        X_tr = X_train.iloc[:-val_size]
        y_tr = y_train.iloc[:-val_size]
        X_v = X_train.iloc[-val_size:]
        y_v = y_train.iloc[-val_size:]
        
        logger.info(f"Split training data: train={X_tr.shape}, validation={X_v.shape}")
    else:
        X_tr, y_tr = X_train, y_train
        X_v, y_v = X_val, y_val
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_tr, label=y_tr)
    dval = xgb.DMatrix(X_v, label=y_v)
    watchlist = [(dtrain, 'train'), (dval, 'val')]
    
    # Define parameters optimized for count data (dengue cases)
    # Using standard regression is more stable for this task
    params = {
        'objective': 'reg:squarederror',  # Standard regression objective
        'tree_method': 'hist',            # Fast histogram-based algorithm
        'eval_metric': ['rmse', 'mae'],   # Standard evaluation metrics
        'max_depth': 5,                   # Reduced to prevent overfitting
        'learning_rate': 0.01,            # Smaller learning rate for better convergence
        'min_child_weight': 3,            # Minimum sum of instance weight needed in a child
        'subsample': 0.7,                 # Prevents overfitting
        'colsample_bytree': 0.7,          # Prevents overfitting
        'gamma': 0.1,                     # Minimum loss reduction for partition
        'alpha': 0.2,                     # L1 regularization (increased)
        'lambda': 2.0,                    # L2 regularization (increased)
        'base_score': y_train.mean(),     # Start with mean prediction
        'seed': 42                        # For reproducibility
    }
    
    # For very skewed datasets with many zeros, log transform the target
    if (y_train == 0).sum() / len(y_train) > 0.2:  # If more than 20% zeros
        logger.info("Target has many zeros, using log(y+1) transformation")
        # Transform targets using log(y+1)
        y_tr = np.log1p(y_tr)
        
        if X_v is not None and y_v is not None:
            y_v = np.log1p(y_v)
            
        # Note this transformation for later inverse transform during prediction
        params['log_transform'] = True
    else:
        params['log_transform'] = False
    
    logger.info(f"Training with parameters: {params}")
    
    # Train model with early stopping
    start_time = time.time()
    # Store log transform setting for later
    log_transform_used = False
    if 'log_transform' in params:
        log_transform_used = params['log_transform']
        # Remove from params to avoid XGBoost errors
        del params['log_transform']
        
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=500,         # Maximum number of boosting rounds
        evals=watchlist,
        early_stopping_rounds=50,    # Stop if no improvement for 50 rounds
        verbose_eval=50              # Print evaluation every 50 rounds
    )
    
    # Store log transform info in the city_results dictionary
    logger.info(f"Log transform used: {log_transform_used}")
    training_time = time.time() - start_time
    
    # Get model info
    best_iteration = model.best_iteration
    best_score = model.best_score
    
    logger.info(f"Model trained in {training_time:.2f} seconds")
    logger.info(f"Best iteration: {best_iteration}")
    logger.info(f"Best validation score: {best_score}")
    
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    top_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("Top 10 features by importance:")
    for feature, score in top_features:
        logger.info(f"  {feature}: {score:.2f}")
    
    return model, log_transform_used

def evaluate_model(model, X_test, y_test, dates=None, city=None, city_results=None):
    """
    Evaluate the model using standard and custom metrics focused on dengue outbreak prediction.
    
    Args:
        model: Trained XGBoost model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target values
        dates (pd.Series, optional): Dates for visualization
        city (str, optional): City name for logging
    
    Returns:
        tuple: (metrics_dict, predictions_array)
    """
    print_section(f"Evaluating Model for {city}" if city else "Evaluating Model")
    
    # Check if we have test data with labels
    if y_test is None or len(y_test) == 0:
        logger.warning("No labeled test data available for evaluation")
        return None, None
    
    logger.info(f"Evaluating on {len(y_test)} test samples")
    
    # Convert test data to DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # If we used log transformation, apply inverse transform
    # This is now passed directly in the city_results dictionary
    log_transform = False
    if 'log_transform_used' in city_results.get(city, {}):
        log_transform = city_results[city]['log_transform_used']
    
    if log_transform:
        logger.info("Applying inverse log transformation to predictions")
        y_pred = np.expm1(y_pred)  # inverse of log1p is expm1
    
    # Ensure predictions are non-negative
    y_pred = np.maximum(0, y_pred)
    
    # Round predictions for count data (for interpretation)
    y_pred_rounded = np.round(y_pred).astype(int)
    
    # Calculate standard regression metrics (using raw predictions)
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
        'mae': mean_absolute_error(y_test, y_pred),  # Fixed extra parameter
        'r2': r2_score(y_test, y_pred)
    }
    
    # Log standard metrics
    logger.info(f"Standard Metrics:")
    logger.info(f"  RMSE: {metrics['rmse']:.4f}")
    logger.info(f"  MAE: {metrics['mae']:.4f}")
    logger.info(f"  R²: {metrics['r2']:.4f}")
    
    # Create metrics for outbreak detection
    # Define outbreaks as periods above 75th percentile (as per requirements)
    outbreak_threshold = np.percentile(y_test, 75)
    logger.info(f"Outbreak threshold (75th percentile): {outbreak_threshold}")
    
    # Calculate actual and predicted outbreaks
    actual_outbreaks = (y_test > outbreak_threshold).astype(int)
    predicted_outbreaks = (y_pred > outbreak_threshold).astype(int)
    
    # Calculate confusion matrix elements for outbreak detection
    tp = np.sum((actual_outbreaks == 1) & (predicted_outbreaks == 1))
    fp = np.sum((actual_outbreaks == 0) & (predicted_outbreaks == 1))
    fn = np.sum((actual_outbreaks == 1) & (predicted_outbreaks == 0))
    tn = np.sum((actual_outbreaks == 0) & (predicted_outbreaks == 0))
    
    # Calculate outbreak detection metrics
    if (tp + fp) > 0:
        precision = tp / (tp + fp)
    else:
        precision = 0
        
    if (tp + fn) > 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
        
    if (precision + recall) > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0
    
    # Add outbreak metrics to results dictionary
    metrics['outbreak_precision'] = precision
    metrics['outbreak_recall'] = recall
    metrics['outbreak_f1'] = f1
    
    # Calculate additional useful metrics
    metrics['outbreak_accuracy'] = (tp + tn) / (tp + tn + fp + fn)
    
    # Log outbreak detection metrics
    logger.info(f"Outbreak Detection Metrics:")
    logger.info(f"  Precision: {precision:.4f}")
    logger.info(f"  Recall: {recall:.4f}")
    logger.info(f"  F1 Score: {f1:.4f}")
    logger.info(f"  Accuracy: {metrics['outbreak_accuracy']:.4f}")
    
    # Calculate outbreak-specific RMSE (errors during outbreaks weighted more heavily)
    outbreak_mask = (actual_outbreaks == 1)
    if outbreak_mask.sum() > 0:  # If we have any outbreaks
        outbreak_rmse = np.sqrt(mean_squared_error(
            y_test[outbreak_mask], y_pred[outbreak_mask]
        ))
        metrics['outbreak_rmse'] = outbreak_rmse
        logger.info(f"  Outbreak RMSE: {outbreak_rmse:.4f}")
    
    # Calculate weighted RMSE that penalizes errors during outbreaks more heavily
    weights = np.ones_like(y_test, dtype=float)
    weights[outbreak_mask] = 2.0  # Weight outbreak periods more heavily
    
    weighted_se = (y_test - y_pred) ** 2 * weights
    weighted_rmse = np.sqrt(weighted_se.mean())
    metrics['weighted_rmse'] = weighted_rmse
    logger.info(f"  Weighted RMSE: {weighted_rmse:.4f}")
    
    # Calculate prediction bias
    bias = np.mean(y_pred) - np.mean(y_test)
    metrics['bias'] = bias
    logger.info(f"  Prediction Bias: {bias:.4f}")
    
    # Calculate correlation between actual and predicted values
    corr = np.corrcoef(y_test, y_pred)[0, 1]
    metrics['correlation'] = corr
    logger.info(f"  Correlation: {corr:.4f}")
    
    # Create confusion matrix for outbreak detection
    cm = {
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn)
    }
    metrics['confusion_matrix'] = cm
    
    logger.info(f"Confusion Matrix for Outbreak Detection:")
    logger.info(f"  True Positives: {tp}, False Positives: {fp}")
    logger.info(f"  True Negatives: {tn}, False Negatives: {fn}")
    
    # Overall evaluation summary
    logger.info(f"Overall Evaluation Summary:")
    if metrics['r2'] > 0.5 and metrics['outbreak_f1'] > 0.6:
        logger.info(f"  Model performance: GOOD")
    elif metrics['r2'] > 0.3 and metrics['outbreak_f1'] > 0.4:
        logger.info(f"  Model performance: ACCEPTABLE")
    else:
        logger.info(f"  Model performance: NEEDS IMPROVEMENT")
    
    return metrics, y_pred

def plot_feature_importance(model, feature_names, city=None):
    """
    Plot feature importance from the model.
    
    Args:
        model: Trained XGBoost model
        feature_names (list): List of feature names
        city (str, optional): City name for plot title
    
    Returns:
        plt.Figure: Feature importance plot
    """
    print_section(f"Feature Importance for {city}" if city else "Feature Importance")
    
    # Get feature importance
    importance = model.get_score(importance_type='gain')
    importance = {k: importance.get(k, 0) for k in feature_names}
    importance = pd.Series(importance).sort_values(ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance{" for " + city if city else ""}')
    importance.head(20).plot(kind='barh')
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(f"results/feature_importance_xgboost_timeframes{('_' + city) if city else ''}.png")
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path)
    
    logger.info(f"Feature importance plot saved to {plot_path}")
    
    return plt.gcf()

def plot_predictions(y_test, y_pred, dates, metrics=None, city=None):
    """
    Plot predictions vs. actual values with enhanced visualization.
    
    Args:
        y_test (pd.Series): Actual values
        y_pred (np.ndarray): Predicted values
        dates (pd.Series): Dates for x-axis
        metrics (dict, optional): Evaluation metrics for annotation
        city (str, optional): City name for plot title
    
    Returns:
        plt.Figure: Predictions plot
    """
    print_section(f"Prediction Plot for {city}" if city else "Prediction Plot")
    
    if y_test is None or len(y_test) == 0 or dates is None:
        logger.warning("Cannot create prediction plot: missing data")
        return None
    
    # Convert dates to datetime if needed
    if not isinstance(dates, pd.DatetimeIndex) and not isinstance(dates.iloc[0], pd.Timestamp):
        try:
            dates = pd.to_datetime(dates)
        except:
            logger.warning("Could not convert dates to datetime format for plotting")
    
    # Create figure and primary axis
    fig, ax1 = plt.subplots(figsize=(14, 8))
    
    # Plot actual and predicted values
    ax1.plot(dates, y_test, 'b-', linewidth=2, label='Actual Cases')
    ax1.plot(dates, y_pred, 'r--', linewidth=2, label='Predicted Cases')
    
    # Highlight outbreak periods (above 75th percentile)
    threshold = np.percentile(y_test, 75)
    outbreak_mask = y_test > threshold
    
    # Highlight outbreak periods with background shading
    if outbreak_mask.sum() > 0:
        # Safety check - convert outbreak_mask to numpy array if it's pandas
        if isinstance(outbreak_mask, pd.Series):
            outbreak_mask = outbreak_mask.values
            
        # Find continuous outbreak periods
        outbreak_indices = np.where(outbreak_mask)[0]
        
        if len(outbreak_indices) > 0:
            outbreak_starts = [outbreak_indices[0]]
            outbreak_ends = []
            
            for i in range(1, len(outbreak_indices)):
                if outbreak_indices[i] > outbreak_indices[i-1] + 1:
                    outbreak_ends.append(outbreak_indices[i-1])
                    outbreak_starts.append(outbreak_indices[i])
            outbreak_ends.append(outbreak_indices[-1])
            
            # Add shaded areas for outbreaks
            for i, (start_idx, end_idx) in enumerate(zip(outbreak_starts, outbreak_ends)):
                # Convert indices to integer to avoid numpy int64 issues
                start_idx = int(start_idx)
                end_idx = int(end_idx)
                
                if 0 <= start_idx < len(dates) and 0 <= end_idx < len(dates):
                    try:
                        ax1.axvspan(dates.iloc[start_idx], dates.iloc[end_idx], 
                                   alpha=0.15, color='red', 
                                   label='Outbreak Period' if i == 0 else "")
                    except:
                        logger.warning(f"Could not highlight outbreak period {start_idx}-{end_idx}")
    
    # Mark outbreak points
    ax1.scatter(
        dates[outbreak_mask], 
        y_test[outbreak_mask], 
        color='red', 
        s=70, 
        marker='o',
        label='Outbreak Points'
    )
    
    # Add error bars for key points
    try:
        error = y_test - y_pred
        # Handle potential pandas vs numpy type mismatches
        if isinstance(error, pd.Series):
            significant_errors = abs(error) > np.percentile(abs(error), 90)  # Top 10% of errors
            if significant_errors.sum() > 0:
                # Plot error bars for significant errors
                for i in range(len(error)):
                    if significant_errors.iloc[i]:
                        # Draw vertical error line
                        ax1.plot([dates.iloc[i], dates.iloc[i]], 
                                [y_test.iloc[i], y_pred[i]], 
                                'k-', alpha=0.5, linewidth=1)
        else:
            # Handle numpy arrays
            significant_errors = abs(error) > np.percentile(abs(error), 90)
            if significant_errors.sum() > 0:
                for i in range(len(error)):
                    if significant_errors[i]:
                        # Draw vertical error line using integer indices
                        ax1.plot([dates.iloc[i], dates.iloc[i]], 
                                [y_test.iloc[i], y_pred[i]], 
                                'k-', alpha=0.5, linewidth=1)
    except Exception as e:
        logger.warning(f"Could not plot error bars: {str(e)}")
    
    # Calculate and plot centered rolling mean for trends (7-week window)
    if len(y_test) > 7:
        window_size = 7  # 7-week rolling window
        y_test_rolmean = pd.Series(y_test).rolling(window=window_size, center=True).mean()
        y_pred_rolmean = pd.Series(y_pred).rolling(window=window_size, center=True).mean()
        
        # Plot rolling means with more transparent lines
        ax1.plot(dates, y_test_rolmean, 'b-', alpha=0.5, linewidth=1.5, 
                label='Actual (7-week avg)')
        ax1.plot(dates, y_pred_rolmean, 'r--', alpha=0.5, linewidth=1.5, 
                label='Predicted (7-week avg)')
    
    # Create a secondary y-axis for prediction error
    ax2 = ax1.twinx()
    ax2.plot(dates, error, 'g-', alpha=0.5, label='Prediction Error')
    ax2.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
    ax2.set_ylabel('Prediction Error', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    
    # Format the primary axis
    title = f'Dengue Fever Cases Prediction for {city}' if city else 'Dengue Fever Cases Prediction'
    ax1.set_title(title, fontsize=16)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Total Cases', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Add metrics annotation if provided
    if metrics is not None:
        metrics_text = (
            f"RMSE: {metrics.get('rmse', 0):.2f}\n"
            f"MAE: {metrics.get('mae', 0):.2f}\n"
            f"R²: {metrics.get('r2', 0):.2f}\n"
            f"Outbreak F1: {metrics.get('outbreak_f1', 0):.2f}"
        )
        
        # Position the text box in figure coords
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax1.text(0.02, 0.97, metrics_text, transform=ax1.transAxes, 
                fontsize=10, verticalalignment='top', bbox=props)
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    # Rotate date labels for better readability
    fig.autofmt_xdate()
    
    # Adjust layout for better spacing
    plt.tight_layout()
    
    # Save plot
    plot_path = Path(f"results/predictions_xgboost_timeframes{('_' + city) if city else ''}.png")
    plot_path.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Predictions plot saved to {plot_path}")
    
    return fig

def save_model(model, feature_names=None, metrics=None, city=None):
    """
    Save the trained model and associated metadata.
    
    Args:
        model: Trained XGBoost model
        feature_names (list, optional): Names of features used in the model
        metrics (dict, optional): Evaluation metrics
        city (str, optional): City name for filename
    
    Returns:
        str: Path to saved model
    """
    print_section(f"Saving Model for {city}" if city else "Saving Model")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    city_suffix = f"_{city}" if city else ""
    
    # Create models directory structure
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    city_dir = models_dir / f"{city}" if city else models_dir
    city_dir.mkdir(exist_ok=True)
    
    # Create a version-specific directory for this model
    model_version_dir = city_dir / f"xgboost_timeframes{city_suffix}_{timestamp}"
    model_version_dir.mkdir(exist_ok=True)
    
    # Save the model
    model_path = model_version_dir / "model.json"
    model.save_model(str(model_path))
    logger.info(f"Model saved to {model_path}")
    
    # Save feature names if provided
    if feature_names is not None:
        feature_path = model_version_dir / "feature_names.txt"
        with open(feature_path, 'w') as f:
            for i, feature in enumerate(feature_names):
                f.write(f"{i+1}. {feature}\n")
        logger.info(f"Feature names saved to {feature_path}")
    
    # Save metrics if provided
    if metrics is not None:
        metrics_path = model_version_dir / "metrics.json"
        
        # Convert numpy values to Python native types for JSON serialization
        metrics_json = {}
        for k, v in metrics.items():
            if isinstance(v, (np.int64, np.int32, np.int16, np.int8)):
                metrics_json[k] = int(v)
            elif isinstance(v, (np.float64, np.float32, np.float16)):
                metrics_json[k] = float(v)
            elif isinstance(v, dict):
                metrics_json[k] = {key: int(val) if isinstance(val, (np.int64, np.int32)) 
                                 else float(val) if isinstance(val, (np.float64, np.float32)) 
                                 else val for key, val in v.items()}
            else:
                metrics_json[k] = v
        
        # Save metrics as JSON
        with open(metrics_path, 'w') as f:
            import json
            json.dump(metrics_json, f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Save model parameters
    params_path = model_version_dir / "params.txt"
    try:
        # For sklearn-style XGBoost models
        if hasattr(model, 'get_params'):
            with open(params_path, 'w') as f:
                f.write(f"Model Parameters:\n")
                for param, value in model.get_params().items():
                    f.write(f"{param}: {value}\n")
            logger.info(f"Model parameters saved to {params_path}")
        else:
            # For native XGBoost Booster objects
            with open(params_path, 'w') as f:
                f.write(f"Model Parameters:\n")
                # Not using attributes since not all XGBoost versions support it
                if 'log_transform_used' in city_results.get(city, {}):
                    f.write(f"log_transform_used: {city_results[city]['log_transform_used']}\n")
                f.write(f"best_iteration: {model.best_iteration}\n")
                f.write(f"best_score: {model.best_score}\n")
                f.write(f"num_features: {len(feature_names) if feature_names is not None else 'unknown'}\n")
            logger.info(f"Model parameters saved to {params_path}")
    except Exception as e:
        logger.warning(f"Could not save model parameters: {str(e)}")
    
    # Save a simple README file
    readme_path = model_version_dir / "README.md"
    with open(readme_path, 'w') as f:
        f.write(f"# XGBoost Time-based Model for {city if city else 'Dengue Prediction'}\n\n")
        f.write(f"Trained on: {timestamp}\n\n")
        
        if metrics is not None:
            f.write("## Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            for metric, value in metrics.items():
                if not isinstance(value, dict):
                    f.write(f"| {metric} | {value:.4f} |\n")
        
        f.write("\n## Model Description\n\n")
        f.write("This model was trained using XGBoost with city-specific timeframes:\n\n")
        f.write("- San Juan: Training period 1990-2000, Testing period 2000-2007\n")
        f.write("- Iquitos: Training period 2000-2007, Testing period 2007-2010\n\n")
        
        f.write("The model uses climate lag features, cyclical encoding of temporal features, ")
        f.write("and mosquito breeding condition indicators to predict dengue fever cases.\n")
    
    logger.info(f"Model documentation saved to {readme_path}")
    
    # Create a symlink to the latest model
    latest_link = city_dir / f"xgboost_timeframes{city_suffix}_latest"
    if latest_link.exists():
        latest_link.unlink()
    
    try:
        latest_link.symlink_to(model_version_dir.name)
        logger.info(f"Created symlink at {latest_link} -> {model_version_dir.name}")
    except Exception as e:
        logger.warning(f"Could not create symlink: {str(e)}")
    
    return str(model_path)

def generate_submission(city_predictions):
    """
    Generate submission file with predictions for competition.
    
    Args:
        city_predictions (dict): Dictionary with predictions for each city
    
    Returns:
        str: Path to submission file
    """
    print_section("Generating Submission")
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create results directory
    results_dir = Path("results")
    submissions_dir = results_dir / "submissions"
    submissions_dir.mkdir(exist_ok=True, parents=True)
    
    # Combine predictions from all cities
    all_submissions = []
    city_info = {}
    
    for city, data in city_predictions.items():
        logger.info(f"Processing submission for {city}...")
        
        # Ensure we have test data
        if 'test_df' not in data or 'predictions' not in data:
            logger.warning(f"Missing test data or predictions for {city}")
            continue
        
        # Format predictions for submission
        test_df = data['test_df']
        predictions = data['predictions']
        
        # Round predictions for count data
        predictions_rounded = np.round(predictions).astype(int)
        
        # Ensure no negative predictions
        predictions_rounded = np.maximum(0, predictions_rounded)
        
        # Format city predictions
        city_submission = pd.DataFrame({
            'city': city,
            'year': test_df['year'],
            'weekofyear': test_df['weekofyear'],
            'total_cases': predictions_rounded
        })
        
        # Log information about this city's predictions
        city_info[city] = {
            'min_pred': int(predictions_rounded.min()),
            'max_pred': int(predictions_rounded.max()),
            'mean_pred': float(predictions_rounded.mean()),
            'samples': len(predictions_rounded)
        }
        
        # Add to combined submissions
        all_submissions.append(city_submission)
    
    # Check if we have any submissions
    if not all_submissions:
        logger.error("No valid predictions found for any city")
        return None
    
    # Combine submissions
    submission = pd.concat(all_submissions, ignore_index=True)
    
    # Verify submission format
    if 'city' not in submission.columns or 'year' not in submission.columns or 'weekofyear' not in submission.columns:
        logger.error("Submission missing required columns: city, year, weekofyear")
        return None
    
    # Create a custom order for cities to ensure San Juan (sj) comes first, followed by Iquitos (iq)
    city_order = {'sj': 0, 'iq': 1}  # Define sorting order: sj first, then iq
    
    # Create temporary column for sorting by our custom city order
    submission['city_order'] = submission['city'].map(city_order)
    
    # Sort submission by custom city order, year, and week
    submission = submission.sort_values(['city_order', 'year', 'weekofyear'])
    
    # Remove the temporary sorting column
    submission = submission.drop(columns=['city_order'])
    
    # Create submission path with timestamp
    submission_path = submissions_dir / f"submission_xgboost_timeframes_{timestamp}.csv"
    submission.to_csv(submission_path, index=False)
    
    # Also save a "latest" version
    latest_path = results_dir / "submission_xgboost_timeframes_latest.csv"
    submission.to_csv(latest_path, index=False)
    
    # Create a summary file with information about the submission
    summary_path = submissions_dir / f"summary_xgboost_timeframes_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'total_predictions': len(submission),
        'cities': city_info,
        'submission_path': str(submission_path)
    }
    
    # Save summary as JSON
    with open(summary_path, 'w') as f:
        import json
        json.dump(summary, f, indent=2)
    
    logger.info(f"Submission saved to {submission_path}")
    logger.info(f"Summary saved to {summary_path}")
    logger.info(f"Latest submission also saved to {latest_path}")
    
    # Log summary information
    logger.info(f"Submission Summary:")
    logger.info(f"  Total predictions: {len(submission)}")
    
    for city, info in city_info.items():
        logger.info(f"  {city}: {info['samples']} samples, range: {info['min_pred']}-{info['max_pred']}, mean: {info['mean_pred']:.2f}")
    
    # Create a basic visualization of the submission
    try:
        plt.figure(figsize=(12, 6))
        
        for city in submission['city'].unique():
            city_data = submission[submission['city'] == city]
            plt.plot(range(len(city_data)), city_data['total_cases'], label=city)
            
        plt.title('Submission Predictions by City')
        plt.xlabel('Sample Index')
        plt.ylabel('Predicted Cases')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        viz_path = submissions_dir / f"visualization_xgboost_timeframes_{timestamp}.png"
        plt.savefig(viz_path)
        logger.info(f"Visualization saved to {viz_path}")
    except Exception as e:
        logger.warning(f"Could not create submission visualization: {str(e)}")
    
    return str(submission_path)

def main():
    """
    Main function to run the XGBoost timeframes training process.
    
    This function orchestrates the entire training pipeline:
    1. Load and prepare the dengue fever dataset
    2. Split data by city-specific timeframes
    3. Engineer features for each city's data
    4. Train XGBoost models for each city
    5. Evaluate models and visualize results
    6. Save models and generate submission file
    
    The script uses distinct timeframes for each city:
    - San Juan: Training period 1990-2000, Testing period 2000-2007
    - Iquitos: Training period 2000-2007, Testing period 2007-2010
    """
    print_section("Starting XGBoost Timeframes Training")
    
    # Set up error tracking
    errors = []
    success = False
    
    try:
        # Create results directories
        Path("logs").mkdir(exist_ok=True)
        Path("models").mkdir(exist_ok=True)
        Path("results").mkdir(exist_ok=True)
        
        logger.info("Starting dengue fever prediction with time-specific XGBoost models")
        
        # Step 1: Load data
        try:
            logger.info("Step 1: Loading dengue fever dataset")
            df = load_data()
            logger.info("Data loaded successfully")
        except Exception as e:
            error_msg = f"Failed to load data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            raise
        
        # Step 2: Split data by timeframes
        try:
            logger.info("Step 2: Splitting data by city-specific timeframes")
            splits = split_data_by_timeframes(df)
            logger.info(f"Data split into {len(splits)} city-specific datasets")
            
            # Basic validation
            for city, data in splits.items():
                train_size = len(data['train'])
                test_size = len(data['test'])
                logger.info(f"  {city}: {train_size} training samples, {test_size} testing samples")
                
                # Check for empty datasets
                if train_size == 0:
                    error_msg = f"Empty training dataset for {city}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                if test_size == 0:
                    error_msg = f"Empty test dataset for {city}"
                    logger.error(error_msg)
                    errors.append(error_msg)
        except Exception as e:
            error_msg = f"Failed to split data: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            raise
        
        # Step 3: Engineer features
        try:
            logger.info("Step 3: Engineering features for each city")
            for city in splits.keys():
                logger.info(f"Engineering features for {city}")
                splits[city]['train'] = engineer_features(splits[city]['train'])
                splits[city]['test'] = engineer_features(splits[city]['test'])
                
                # Log feature counts
                feature_count = splits[city]['train'].shape[1]
                logger.info(f"  {city}: {feature_count} features created")
        except Exception as e:
            error_msg = f"Failed to engineer features: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
            raise
        
        # Step 4-6: Train, evaluate, and save models for each city
        city_results = {}
        
        for city, data in splits.items():
            logger.info(f"\nProcessing city: {city}")
            city_success = False
            
            try:
                # Prepare features
                logger.info(f"Preparing features for {city}")
                X_train, y_train, scaler = prepare_features(data['train'], is_training=True)
                X_test, y_test, _ = prepare_features(data['test'], is_training=True, scaler=scaler)
                
                # Validate prepared data
                if X_train.shape[0] == 0 or y_train.shape[0] == 0:
                    error_msg = f"Empty training data for {city} after feature preparation"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Train model
                logger.info(f"Training XGBoost model for {city}")
                model, log_transform_used = train_xgboost_model(X_train, y_train, city=city)
                
                # Initialize city results dict
                city_results[city] = city_results.get(city, {})
                
                # Store the scaler and log transform info for future predictions
                city_results[city]['scaler'] = scaler
                city_results[city]['log_transform_used'] = log_transform_used
                
                # Check if model was trained successfully
                if model is None:
                    error_msg = f"Model training failed for {city}"
                    logger.error(error_msg)
                    errors.append(error_msg)
                    continue
                
                # Evaluate model
                logger.info(f"Evaluating model for {city}")
                metrics, predictions = evaluate_model(model, X_test, y_test, data['test']['week_start_date'], city=city, city_results=city_results)
                
                # Plot feature importance
                logger.info(f"Generating feature importance visualization for {city}")
                plot_feature_importance(model, X_train.columns, city=city)
                
                # Plot predictions
                logger.info(f"Generating prediction visualization for {city}")
                plot_predictions(y_test, predictions, data['test']['week_start_date'], metrics, city=city)
                
                # Save model
                logger.info(f"Saving model for {city}")
                model_path = save_model(model, X_train.columns, metrics, city=city)
                
                # Update results
                city_results[city].update({
                    'model': model,
                    'model_path': model_path,
                    'metrics': metrics,
                    'predictions': predictions,
                    'test_df': data['test'],
                    'feature_names': list(X_train.columns)
                })
                
                city_success = True
                logger.info(f"Successfully completed processing for {city}")
                
            except Exception as e:
                error_msg = f"Error processing {city}: {str(e)}"
                logger.error(error_msg, exc_info=True)
                errors.append(error_msg)
                # Continue with other cities
        
        # Check if we have any successful models
        if not city_results:
            error_msg = "No models were successfully trained"
            logger.error(error_msg)
            errors.append(error_msg)
            # Continue to generate summary rather than stopping completely
            success = False
        else:
            # We have at least one successful model
            success = True
        
        # Step 7: Generate submission file
        try:
            logger.info("Step 7: Generating submission file")
            submission_path = generate_submission(city_results)
            
            if submission_path:
                logger.info(f"Submission file generated: {submission_path}")
            else:
                error_msg = "Failed to generate submission file"
                logger.error(error_msg)
                errors.append(error_msg)
        except Exception as e:
            error_msg = f"Error generating submission: {str(e)}"
            logger.error(error_msg, exc_info=True)
            errors.append(error_msg)
        
        # Print final results summary
        print_section("XGBoost Timeframes Training Results")
        
        for city, results in city_results.items():
            if 'metrics' in results:
                logger.info(f"\n{city} Results Summary:")
                logger.info(f"  Model saved to: {results['model_path']}")
                
                if isinstance(results['metrics'], dict):
                    # Standard metrics
                    std_metrics = ['rmse', 'mae', 'r2']
                    logger.info("  Standard Metrics:")
                    for metric in std_metrics:
                        if metric in results['metrics']:
                            logger.info(f"    {metric.upper()}: {results['metrics'][metric]:.4f}")
                    
                    # Outbreak metrics
                    outbreak_metrics = ['outbreak_precision', 'outbreak_recall', 'outbreak_f1']
                    logger.info("  Outbreak Detection Metrics:")
                    for metric in outbreak_metrics:
                        if metric in results['metrics']:
                            logger.info(f"    {metric}: {results['metrics'][metric]:.4f}")
        
        if submission_path:
            logger.info(f"\nFinal submission file: {submission_path}")
        
        # Overall success if no critical errors
        if not errors:
            success = True
            logger.info("\nTraining process completed successfully with no errors.")
        else:
            logger.warning(f"\nTraining process completed with {len(errors)} errors.")
            logger.warning("Please check the log file for details.")
        
    except Exception as e:
        error_msg = f"Critical error in main execution: {str(e)}"
        logger.error(error_msg, exc_info=True)
        errors.append(error_msg)
    
    finally:
        # Print final status
        print_section("XGBoost Timeframes Training Complete")
        
        if success:
            logger.info("✅ Training completed successfully!")
        else:
            logger.error("❌ Training completed with errors!")
            
            if errors:
                logger.error("\nError Summary:")
                for i, error in enumerate(errors, 1):
                    logger.error(f"{i}. {error}")
        
        # Return execution status
        return success

if __name__ == "__main__":
    main()