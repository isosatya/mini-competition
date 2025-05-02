"""
Time Series Model Training Module

This module implements time series-specific methods for the Dengue Fever prediction model,
including Prophet, SARIMAX components, and customized time series validation.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import os
import sys
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import PoissonRegressor, TweedieRegressor
from sklearn.pipeline import Pipeline

# Suppress warnings
warnings.filterwarnings('ignore')

# Try to import Prophet - a time series forecasting library
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    print("Warning: Prophet not installed. Prophet model will not be available.")

# Try to import statsmodels for SARIMAX
try:
    import statsmodels.api as sm
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. SARIMAX model will not be available.")

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

def load_enhanced_featured_data():
    """
    Load the enhanced featured training and test data.
    
    Returns:
        tuple: (train_data, test_features)
    
    Raises:
        FileNotFoundError: If featured data files are missing
    """
    debug_print("Starting enhanced featured data loading")
    
    # Initialize data paths
    data_dir = Path("data/processed")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    # Define file paths - using enhanced features
    train_path = data_dir / "enhanced_featured_train_data.csv"
    test_path = data_dir / "enhanced_featured_test_features.csv"
    
    # Fall back to regular features if enhanced ones are not available
    if not train_path.exists() or not test_path.exists():
        print("Enhanced featured data not found. Falling back to regular featured data.")
        train_path = data_dir / "featured_train_data.csv"
        test_path = data_dir / "featured_test_features.csv"
    
    # Check if files exist
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    debug_print("Loading featured training data...")
    train_data = pd.read_csv(train_path)
    # Ensure datetime format for week_start_date
    if 'week_start_date' in train_data.columns:
        train_data['week_start_date'] = pd.to_datetime(train_data['week_start_date'])
    debug_print(f"Training data loaded. Shape: {train_data.shape}")
    
    debug_print("Loading featured test features...")
    test_features = pd.read_csv(test_path)
    # Ensure datetime format for week_start_date
    if 'week_start_date' in test_features.columns:
        test_features['week_start_date'] = pd.to_datetime(test_features['week_start_date'])
    debug_print(f"Test features loaded. Shape: {test_features.shape}")
    
    return train_data, test_features

def prepare_data_for_prophet(df, city=None):
    """
    Prepare data for Prophet model.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str, optional): City to filter for
    
    Returns:
        tuple: (prophet_df, exog_features) for Prophet model
    """
    # Filter data for specific city if provided
    if city:
        city_data = df[df['city'] == city].copy()
    else:
        city_data = df.copy()
    
    # Prophet requires 'ds' (date) and 'y' (target) columns
    prophet_df = pd.DataFrame({
        'ds': city_data['week_start_date'],
        'y': city_data['total_cases']
    })
    
    # Select exogenous features (excluding date, city, and target)
    exog_cols = [col for col in city_data.columns if col not in ['week_start_date', 'city', 'total_cases']]
    exog_features = city_data[exog_cols].copy()
    
    return prophet_df, exog_features

def train_prophet_model(df, city):
    """
    Train a Prophet model for a specific city.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str): City to train model for
    
    Returns:
        dict: Model and metrics
    """
    if not PROPHET_AVAILABLE:
        print(f"Prophet is not available. Skipping Prophet model for {city}.")
        return None
    
    print_section(f"Training Prophet Model for {city}")
    
    # Prepare data for Prophet
    prophet_df, exog_features = prepare_data_for_prophet(df, city)
    
    # Initialize Prophet model with yearly seasonality and additional regressors
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='multiplicative'  # Better for count data that has increasing variance with level
    )
    
    # Add selected exogenous features as regressors
    # Limit to most important features to avoid overfitting
    important_features = [
        'temp_avg', 'humidity_avg', 'precip_total',
        'temp_avg_lag_4', 'humidity_avg_lag_4', 'precip_total_lag_4',
        'mosquito_comfort_index'
    ]
    
    for feature in important_features:
        if feature in exog_features.columns:
            model.add_regressor(feature)
    
    # Fit the model
    model.fit(prophet_df.join(exog_features))
    
    # Make in-sample predictions
    forecast = model.predict(prophet_df.join(exog_features))
    
    # Calculate metrics
    y_true = prophet_df['y'].values
    y_pred = forecast['yhat'].values
    
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    # Print metrics
    print(f"Prophet Model Metrics for {city}:")
    print(f"RMSE: {metrics['rmse']:.2f}")
    print(f"MAE: {metrics['mae']:.2f}")
    print(f"R²: {metrics['r2']:.2f}")
    
    # Visualize forecast
    fig = model.plot(forecast)
    plt.title(f'Prophet Forecast for {city}')
    plt.savefig(f'models/prophet_forecast_{city}.png')
    plt.close()
    
    # Visualize components
    fig = model.plot_components(forecast)
    plt.savefig(f'models/prophet_components_{city}.png')
    plt.close()
    
    return {
        'model': model,
        'metrics': metrics,
        'exog_features': important_features
    }

def prepare_data_for_sarimax(df, city=None):
    """
    Prepare data for SARIMAX model.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str, optional): City to filter for
    
    Returns:
        tuple: (endog, exog) for SARIMAX model
    """
    # Filter data for specific city if provided
    if city:
        city_data = df[df['city'] == city].copy()
    else:
        city_data = df.copy()
    
    # Endogenous variable (target)
    endog = city_data['total_cases']
    
    # Select exogenous features (excluding date, city, and target)
    exog_cols = [
        'temp_avg', 'humidity_avg', 'precip_total',
        'temp_avg_lag_4', 'humidity_avg_lag_4', 'precip_total_lag_4',
        'mosquito_comfort_index',
        'weekofyear_sin', 'weekofyear_cos'
    ]
    
    exog = city_data[exog_cols].copy()
    
    # Add constant to exog
    exog = sm.add_constant(exog)
    
    return endog, exog

def train_sarimax_model(df, city):
    """
    Train a SARIMAX model for a specific city.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str): City to train model for
    
    Returns:
        dict: Model and metrics
    """
    if not STATSMODELS_AVAILABLE:
        print(f"statsmodels is not available. Skipping SARIMAX model for {city}.")
        return None
    
    print_section(f"Training SARIMAX Model for {city}")
    
    # Prepare data for SARIMAX
    endog, exog = prepare_data_for_sarimax(df, city)
    
    # Define SARIMAX order based on city
    # These parameters should be tuned using auto_arima or grid search in practice
    if city == 'sj':
        # San Juan - larger dataset, more complex seasonality
        order = (2, 1, 2)
        seasonal_order = (1, 0, 1, 52)  # 52-week seasonality
    else:
        # Iquitos - smaller dataset, simpler model
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 52)  # 52-week seasonality
    
    # Initialize and fit SARIMAX model
    try:
        model = sm.tsa.SARIMAX(
            endog=endog,
            exog=exog,
            order=order,
            seasonal_order=seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        )
        
        results = model.fit(disp=False)
        
        # Make in-sample predictions
        y_pred = results.predict()
        
        # Calculate metrics
        y_true = endog.values
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }
        
        # Print metrics
        print(f"SARIMAX Model Metrics for {city}:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R²: {metrics['r2']:.2f}")
        
        # Visualize model diagnostics
        fig = results.plot_diagnostics(figsize=(15, 12))
        plt.savefig(f'models/sarimax_diagnostics_{city}.png')
        plt.close()
        
        return {
            'model': results,
            'metrics': metrics,
            'exog_columns': exog.columns.tolist()
        }
    
    except Exception as e:
        print(f"Error training SARIMAX model for {city}: {e}")
        return None

def prepare_data_for_poisson(df, city=None):
    """
    Prepare data for Poisson/Negative Binomial regression.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str, optional): City to filter for
    
    Returns:
        tuple: (X, y) for regression model
    """
    # Filter data for specific city if provided
    if city:
        city_data = df[df['city'] == city].copy()
    else:
        city_data = df.copy()
    
    # Target variable
    y = city_data['total_cases']
    
    # Select features (excluding non-feature columns)
    non_feature_cols = ['week_start_date', 'city', 'total_cases']
    feature_cols = [col for col in city_data.columns if col not in non_feature_cols]
    
    # For Poisson/NB models, limit to most important features to avoid overfitting
    X = city_data[feature_cols].copy()
    
    return X, y

def train_negative_binomial_model(df, city):
    """
    Train a Negative Binomial regression model (via Tweedie) for a specific city.
    Negative Binomial is appropriate for overdispersed count data.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str): City to train model for
    
    Returns:
        dict: Model and metrics
    """
    print_section(f"Training Negative Binomial Model for {city}")
    
    # Prepare data
    X, y = prepare_data_for_poisson(df, city)
    
    # Create time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize metrics dictionary
    metrics = {'rmse': [], 'mae': [], 'r2': []}
    
    # Train model with time series cross-validation
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create pipeline with scaling and Tweedie Regressor (power=2 approximates negative binomial)
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', TweedieRegressor(power=2, alpha=0.5, link='log', max_iter=1000))
        ])
        
        # Fit model
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        
        # Calculate metrics
        metrics['rmse'].append(np.sqrt(mean_squared_error(y_test, y_pred)))
        metrics['mae'].append(mean_absolute_error(y_test, y_pred))
        metrics['r2'].append(r2_score(y_test, y_pred))
    
    # Average metrics across folds
    avg_metrics = {
        'rmse': np.mean(metrics['rmse']),
        'mae': np.mean(metrics['mae']),
        'r2': np.mean(metrics['r2'])
    }
    
    # Print metrics
    print(f"Negative Binomial Model Cross-Validation Metrics for {city}:")
    print(f"RMSE: {avg_metrics['rmse']:.2f}")
    print(f"MAE: {avg_metrics['mae']:.2f}")
    print(f"R²: {avg_metrics['r2']:.2f}")
    
    # Train final model on all data
    final_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('regressor', TweedieRegressor(power=2, alpha=0.5, link='log', max_iter=1000))
    ])
    
    final_pipeline.fit(X, y)
    
    # Get feature importance from coefficients
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': np.abs(final_pipeline.named_steps['regressor'].coef_)
    })
    
    feature_importance = feature_importance.sort_values('importance', ascending=False)
    
    # Plot feature importance
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance['feature'].head(20), feature_importance['importance'].head(20))
    plt.title(f'Top 20 Feature Importance for {city} (Negative Binomial Model)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    plt.savefig(f'models/neg_binomial_importance_{city}.png')
    plt.close()
    
    return {
        'model': final_pipeline,
        'metrics': avg_metrics,
        'feature_importance': feature_importance
    }

def make_predictions_for_city(models, test_features, city):
    """
    Make predictions for a specific city using trained models.
    
    Args:
        models (dict): Dictionary of trained models
        test_features (pd.DataFrame): Test features
        city (str): City to make predictions for
    
    Returns:
        pd.DataFrame: Predictions for the city
    """
    print_section(f"Making Predictions for {city}")
    
    # Filter test data for the specific city
    city_test = test_features[test_features['city'] == city].copy()
    
    # Initialize a DataFrame to store predictions from different models
    all_predictions = pd.DataFrame()
    all_predictions['city'] = city_test['city']
    all_predictions['year'] = city_test['year']
    all_predictions['weekofyear'] = city_test['weekofyear']
    
    # Make predictions using Prophet model
    if 'prophet' in models[city] and models[city]['prophet'] is not None:
        # Prepare data for Prophet
        prophet_df, exog_features = prepare_data_for_prophet(city_test, city)
        prophet_df = prophet_df.drop('y', axis=1)  # Remove target column which might not exist in test data
        
        # Select only the exogenous features used in training
        used_features = models[city]['prophet']['exog_features']
        available_features = [f for f in used_features if f in city_test.columns]
        
        prophet_exog = city_test[available_features].copy()
        
        # Make predictions
        forecast = models[city]['prophet']['model'].predict(prophet_df.join(prophet_exog))
        all_predictions['prophet_pred'] = forecast['yhat'].values
    
    # Make predictions using SARIMAX model
    if 'sarimax' in models[city] and models[city]['sarimax'] is not None:
        try:
            # Prepare data for SARIMAX
            _, exog = prepare_data_for_sarimax(city_test, city)
            
            # Make predictions
            sarimax_pred = models[city]['sarimax']['model'].get_forecast(
                steps=len(city_test),
                exog=exog
            ).predicted_mean
            
            all_predictions['sarimax_pred'] = sarimax_pred.values
        except Exception as e:
            print(f"Error making SARIMAX predictions for {city}: {e}")
    
    # Make predictions using Negative Binomial model
    if 'neg_binomial' in models[city] and models[city]['neg_binomial'] is not None:
        # Prepare data for Negative Binomial
        X, _ = prepare_data_for_poisson(city_test, city)
        
        # Make predictions
        nb_pred = models[city]['neg_binomial']['model'].predict(X)
        all_predictions['neg_binomial_pred'] = nb_pred
    
    # Combine predictions (ensemble) - weighted average based on model performance
    # Use inverse RMSE as weights to give more weight to better-performing models
    weights = {}
    available_models = []
    
    for model_type in ['prophet', 'sarimax', 'neg_binomial']:
        pred_col = f'{model_type}_pred'
        if pred_col in all_predictions.columns and model_type in models[city]:
            # Use inverse RMSE as weight
            weights[model_type] = 1 / models[city][model_type]['metrics']['rmse']
            available_models.append(model_type)
    
    # Normalize weights
    if available_models:
        weight_sum = sum(weights.values())
        for model_type in weights:
            weights[model_type] /= weight_sum
    
    # Calculate weighted average prediction
    all_predictions['ensemble_pred'] = 0
    for model_type in available_models:
        pred_col = f'{model_type}_pred'
        all_predictions['ensemble_pred'] += all_predictions[pred_col] * weights[model_type]
    
    # Round predictions to integers (dengue cases are counts)
    all_predictions['total_cases'] = np.round(all_predictions['ensemble_pred']).astype(int)
    
    # Ensure no negative predictions
    all_predictions['total_cases'] = all_predictions['total_cases'].clip(lower=0)
    
    # Return prediction DataFrame with required columns
    return all_predictions[['city', 'year', 'weekofyear', 'total_cases']]

def train_models_for_city(train_data, city):
    """
    Train all models for a specific city.
    
    Args:
        train_data (pd.DataFrame): Training data
        city (str): City to train models for
    
    Returns:
        dict: Dictionary of trained models
    """
    print_section(f"Training Models for {city}")
    
    # Filter data for specific city
    city_data = train_data[train_data['city'] == city].copy()
    
    # Initialize models dictionary
    city_models = {}
    
    # Train Prophet model
    print(f"Training Prophet model for {city}...")
    prophet_model = train_prophet_model(city_data, city)
    city_models['prophet'] = prophet_model
    
    # Train SARIMAX model
    print(f"Training SARIMAX model for {city}...")
    sarimax_model = train_sarimax_model(city_data, city)
    city_models['sarimax'] = sarimax_model
    
    # Train Negative Binomial model
    print(f"Training Negative Binomial model for {city}...")
    nb_model = train_negative_binomial_model(city_data, city)
    city_models['neg_binomial'] = nb_model
    
    return city_models

def train_all_models(train_data, test_features):
    """
    Train models for all cities.
    
    Args:
        train_data (pd.DataFrame): Training data
        test_features (pd.DataFrame): Test features
    
    Returns:
        dict: Dictionary of trained models for each city
    """
    print_section("Training Models for All Cities")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Get unique cities
    cities = train_data['city'].unique()
    
    # Initialize models dictionary
    all_models = {}
    
    # Train models for each city
    for city in cities:
        print(f"Training models for {city}...")
        all_models[city] = train_models_for_city(train_data, city)
    
    return all_models

def save_models(models):
    """
    Save trained models.
    
    Args:
        models (dict): Dictionary of trained models
    """
    print_section("Saving Models")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    model_path = models_dir / "time_series_models.joblib"
    joblib.dump(models, model_path)
    print(f"Models saved to {model_path}")

def generate_submission(models, test_features):
    """
    Generate submission file.
    
    Args:
        models (dict): Dictionary of trained models
        test_features (pd.DataFrame): Test features
        
    Returns:
        pd.DataFrame: Submission dataframe
    """
    print_section("Generating Submission")
    
    # Get unique cities
    cities = test_features['city'].unique()
    
    # Generate predictions for each city
    all_predictions = []
    for city in cities:
        print(f"Generating predictions for {city}...")
        city_predictions = make_predictions_for_city(models, test_features, city)
        all_predictions.append(city_predictions)
    
    # Combine predictions
    submission = pd.concat(all_predictions, ignore_index=True)
    
    # Save submission
    submission_path = Path("data/processed/submission_time_series.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    return submission

def main():
    """Main function to run the time series model training process."""
    print_section("Starting Time Series Model Training")
    
    # Load enhanced featured data
    train_data, test_features = load_enhanced_featured_data()
    
    # Train models for all cities
    models = train_all_models(train_data, test_features)
    
    # Save models
    save_models(models)
    
    # Generate submission
    submission = generate_submission(models, test_features)
    
    print_section("Training Complete")
    print("Time series models have been trained and predictions generated.")
    print("Next steps:")
    print("1. Evaluate the models")
    print("2. Implement ensembling with the other models")
    print("3. Optimize model parameters")

if __name__ == "__main__":
    main()