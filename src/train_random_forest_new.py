"""
Random Forest Training Module (New)

This module trains a Random Forest model using the cleaned and featured data.
It includes hyperparameter tuning, model evaluation, and submission file creation.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
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

def prepare_features(df, target_col='total_cases', is_test=False, train_data=None):
    """
    Prepare features for training or prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        target_col (str): Name of the target column
        is_test (bool): Whether the data is for testing
        train_data (pd.DataFrame): Training data for lag features
    
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
    if is_test and train_data is not None:
        # Get the last known values for each city
        last_values = {}
        for city in df['city'].unique():
            city_data = train_data[train_data['city'] == city].sort_values('week_start_date')
            if len(city_data) > 0:
                last_values[city] = city_data[target_col].iloc[-4:].values
        
        # Create lag features using the last known values
        for city in df['city'].unique():
            city_mask = df['city'] == city
            if city in last_values and len(last_values[city]) == 4:
                for lag in range(1, 5):
                    df.loc[city_mask, f'total_cases_lag_{lag}'] = last_values[city][-lag]
            else:
                # If no historical data, use 0
                for lag in range(1, 5):
                    df.loc[city_mask, f'total_cases_lag_{lag}'] = 0
    
    # For training data, we need to create lag features without using future values
    elif not is_test:
        # Sort data by city and date
        df = df.sort_values(['city', 'week_start_date'])
        
        # Create lag features for each city
        for city in df['city'].unique():
            city_mask = df['city'] == city
            city_data = df[city_mask].copy()
            
            # Create lag features
            for lag in range(1, 5):
                df.loc[city_mask, f'total_cases_lag_{lag}'] = city_data[target_col].shift(lag)
        
        # Fill NaN values with 0
        for lag in range(1, 5):
            df[f'total_cases_lag_{lag}'] = df[f'total_cases_lag_{lag}'].fillna(0)
    
    X = df.drop(columns=non_feature_cols)
    y = df[target_col] if not is_test else None
    
    return X, y

def train_model(X_train, y_train):
    """
    Train a Random Forest model with hyperparameter tuning.
    
    Args:
        X_train (pd.DataFrame): Training features
        y_train (pd.Series): Training target
    
    Returns:
        RandomForestRegressor: Trained model
    """
    print_section("Training Random Forest Model")
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Create base model
    rf = RandomForestRegressor(random_state=42)
    
    # Grid search
    grid_search = GridSearchCV(
        estimator=rf,
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
        model (RandomForestRegressor): Trained model
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
        model (RandomForestRegressor): Trained model
        metrics (dict): Model evaluation metrics
    """
    print_section("Saving Model")
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save model
    model_path = models_dir / "random_forest_new.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "random_forest_new_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"R²: {metrics['r2']:.4f}\n")
    print(f"Metrics saved to {metrics_path}")

def make_predictions(model, test_features, train_data):
    """
    Make predictions on test data and create submission file.
    
    Args:
        model (RandomForestRegressor): Trained model
        test_features (pd.DataFrame): Test features
        train_data (pd.DataFrame): Training data for lag features
    
    Returns:
        pd.DataFrame: Submission data
    """
    print_section("Making Predictions")
    
    # Prepare test features
    X_test, _ = prepare_features(test_features, is_test=True, train_data=train_data)
    
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
    submission_path = Path("data/processed/submission_random_forest.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission file saved to {submission_path}")
    
    return submission

def main():
    """Main function to run the training process."""
    print_section("Starting Random Forest Training")
    
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
    submission = make_predictions(model, test_features, train_data)
    
    print_section("Next Steps")
    print("1. Review model performance")
    print("2. Check submission file in data/processed/submission.csv")
    print("3. If not satisfied, adjust hyperparameters or feature engineering")

if __name__ == "__main__":
    main() 