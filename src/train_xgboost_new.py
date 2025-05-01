"""
XGBoost Model Training Module

This module trains an XGBoost model on the featured data.
It includes hyperparameter tuning and model evaluation.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
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

def load_featured_data():
    """
    Load the featured training and test data.
    
    Returns:
        tuple: (train_data, test_features)
    
    Raises:
        FileNotFoundError: If featured data files are missing
    """
    debug_print("Starting featured data loading")
    
    # Initialize data paths
    data_dir = Path("data/processed")
    debug_print(f"Data directory: {data_dir}")
    
    if not data_dir.exists():
        raise FileNotFoundError(f"Processed data directory not found: {data_dir}")
    
    # Define file paths
    train_path = data_dir / "featured_train_data.csv"
    test_path = data_dir / "featured_test_features.csv"
    
    # Check if files exist
    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
    
    # Load data
    debug_print("Loading featured training data...")
    train_data = pd.read_csv(train_path)
    debug_print(f"Training data loaded. Shape: {train_data.shape}")
    
    debug_print("Loading featured test features...")
    test_features = pd.read_csv(test_path)
    debug_print(f"Test features loaded. Shape: {test_features.shape}")
    
    return train_data, test_features

def prepare_features(df, is_training=True):
    """
    Prepare features for model training or prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        is_training (bool): Whether preparing for training or prediction
    
    Returns:
        pd.DataFrame: DataFrame with prepared features
    """
    print_section("Preparing Features")
    
    # Drop non-feature columns
    non_feature_cols = ['city', 'week_start_date']
    if is_training:
        non_feature_cols.append('total_cases')
    
    features = df.drop(columns=non_feature_cols)
    
    # Ensure all lag features are present
    weather_features = [
        'temp_avg',
        'humidity_avg',
        'precip_total',
        'precip_days',
        'pressure_avg'
    ]
    
    for feature in weather_features:
        for lag in [1, 2, 3, 4]:
            lag_col = f'{feature}_lag_{lag}'
            if lag_col not in features.columns:
                features[lag_col] = 0
    
    return features

def train_model(X_train, y_train):
    """
    Train an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        xgb.XGBRegressor: Trained model
    """
    print_section("Training Model")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 6, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Initialize model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Initialize grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    # Fit grid search
    print("Performing grid search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Print best parameters
    print("\nBest Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"{param}: {value}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    
    Returns:
        dict: Evaluation metrics
    """
    print_section("Evaluating Model")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # Print metrics
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    
    return {
        'rmse': rmse,
        'r2': r2
    }

def save_model(model, metrics):
    """
    Save the trained model and metrics.
    
    Args:
        model: Trained model
        metrics (dict): Model evaluation metrics
    """
    print_section("Saving Model")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "xgboost_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "xgboost_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"R² Score: {metrics['r2']:.2f}\n")
    print(f"Metrics saved to {metrics_path}")

def make_predictions(model, test_features):
    """
    Make predictions on test data and create submission file.
    
    Args:
        model: Trained model
        test_features (pd.DataFrame): Test features
    """
    print_section("Making Predictions")
    
    # Prepare test features
    X_test = prepare_features(test_features, is_training=False)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission DataFrame
    submission = pd.DataFrame({
        'city': test_features['city'],
        'week_start_date': test_features['week_start_date'],
        'total_cases': predictions.round().astype(int)
    })
    
    # Save submission
    submission_path = Path("data/processed/submission.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")

def main():
    """Main function to run the model training process."""
    print_section("Starting Model Training")
    
    # Load data
    train_data, test_features = load_featured_data()
    
    # Prepare features
    X = prepare_features(train_data)
    y = train_data['total_cases']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model and metrics
    save_model(model, metrics)
    
    # Make predictions
    make_predictions(model, test_features)
    
    print_section("Next Steps")
    print("1. Review the model metrics")
    print("2. Check the submission file")
    print("3. If needed, adjust the model parameters")

if __name__ == "__main__":
    main() 