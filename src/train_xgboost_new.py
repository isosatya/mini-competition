"""
XGBoost Training Module (New)

This module trains an XGBoost model using the cleaned and featured data.
It includes hyperparameter tuning, model evaluation, and submission file creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import joblib
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

def prepare_features(df, target_col='total_cases', is_test=False):
    """
    Prepare features for training or prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        target_col (str): Name of the target column
        is_test (bool): Whether the data is for testing
    
    Returns:
        tuple: (X, y) for training or (X, None) for testing
    """
    # Drop non-feature columns
    non_feature_cols = ['city', 'week_start_date', 'year', 'month', 'weekofyear']
    
    # Only remove target column from training data
    if not is_test and target_col in df.columns:
        non_feature_cols.append(target_col)
    
    # Keep the cyclic transformations of weekofyear
    if 'weekofyear_sin' in df.columns and 'weekofyear_cos' in df.columns:
        non_feature_cols.remove('weekofyear')
    
    # For test data, we need to handle lag features
    if is_test:
        # Create empty lag features
        for lag in [1, 2, 3, 4]:
            df[f'total_cases_lag_{lag}'] = 0
    
    X = df.drop(columns=non_feature_cols)
    y = df[target_col] if not is_test else None
    
    return X, y

def train_model(X_train, y_train):
    """
    Train an XGBoost model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        xgb.XGBRegressor: Trained model
    """
    print_section("Training XGBoost Model")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    }
    
    # Create base model
    xgb_model = xgb.XGBRegressor(
        objective='reg:squarederror',
        random_state=42,
        n_jobs=-1
    )
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        verbose=2
    )
    
    # Fit model
    print("Starting grid search...")
    grid_search.fit(X_train, y_train)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    print("\nBest parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"- {param}: {value}")
    
    return best_model

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model on test data.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
    
    Returns:
        dict: Evaluation metrics
    """
    print_section("Model Evaluation")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'rmse': rmse,
        'r2': r2
    }

def save_model(model, metrics):
    """
    Save the trained model and its metrics.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        metrics (dict): Model evaluation metrics
    """
    print_section("Saving Model")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "xgboost_new.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "xgboost_new_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
    print(f"Metrics saved to {metrics_path}")

def make_predictions(model, test_features):
    """
    Make predictions on test data and create submission file.
    
    Args:
        model (xgb.XGBRegressor): Trained model
        test_features (pd.DataFrame): Test features
    
    Returns:
        pd.DataFrame: Submission data
    """
    print_section("Making Predictions")
    
    # Prepare test features
    X_test, _ = prepare_features(test_features, is_test=True)
    
    # Make predictions
    predictions = model.predict(X_test)
    
    # Create submission DataFrame with correct column order
    submission = pd.DataFrame({
        'year': test_features['year'],
        'weekofyear': test_features['weekofyear'],  # Use original weekofyear for submission
        'city': test_features['city'],
        'total_cases': predictions.round().astype(int)
    })
    
    # Ensure correct column order
    submission = submission[['city', 'year', 'weekofyear', 'total_cases']]
    
    # Save submission file
    submission_path = Path("data/processed/submission_xgboost.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    
    return submission

def main():
    """Main function to run the training process."""
    print_section("Starting XGBoost Training")
    
    # Load data
    train_data, test_features = load_featured_data()
    
    # Prepare features
    X, y = prepare_features(train_data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Save model
    save_model(model, metrics)
    
    # Make predictions and create submission
    submission = make_predictions(model, test_features)
    
    print_section("Next Steps")
    print("1. Review model performance")
    print("2. Check submission file in data/processed/submission.csv")
    print("3. If not satisfied, adjust hyperparameters or feature engineering")

if __name__ == "__main__":
    main() 