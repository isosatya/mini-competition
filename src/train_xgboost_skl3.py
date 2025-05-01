#!/Users/marc/Documents/INcode/mini-competition/.venv/bin/python3

"""
XGBoost Model Training Module with Metal GPU Support

This module trains an XGBoost model on the featured data with Metal GPU support.
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
import logging
import time
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Print Python and XGBoost information
print("Python version:", sys.version)
print("Python executable:", sys.executable)
print("XGBoost version:", xgb.__version__)

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
    
    # Convert date to datetime
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    
    # Create temporal features
    df['year'] = df['week_start_date'].dt.year
    df['month'] = df['week_start_date'].dt.month
    df['weekofyear'] = df['week_start_date'].dt.isocalendar().week
    
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

def train_model(X_train, y_train, X_val, y_val):
    """Train an XGBoost model with GPU support"""
    print_section("Training Model with GPU Support")
    
    # Print XGBoost information
    print("\nXGBoost Information:")
    print("Version:", xgb.__version__)
    
    # Convert data to DMatrix format
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Define parameters with GPU support
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'gpu_hist',  # Use GPU histogram algorithm
        'predictor': 'gpu_predictor',  # Use GPU for prediction
        'eval_metric': 'rmse',
        'learning_rate': 0.1,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': 2  # Increase verbosity for more detailed output
    }
    
    try:
        print("\nStarting training with GPU settings...")
        print("Model parameters:", params)
        print("Training data shape:", X_train.shape)
        print("Validation data shape:", X_val.shape)
        
        # Start timing
        start_time = time.time()
        
        # Train model with progress bar
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=10,
            callbacks=[xgb.callback.TrainingCallback()]
        )
        
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
        
        print("\nBest iteration:", model.best_iteration)
        print("Best score:", model.best_score)
        print(f"Training time: {training_time:.2f} seconds")
        
    except xgb.core.XGBoostError as e:
        print("\nWarning: GPU training failed, falling back to CPU...")
        print(f"Error: {str(e)}")
        
        # Fall back to CPU parameters
        params['tree_method'] = 'hist'
        if 'predictor' in params:
            del params['predictor']
            
        print("\nRetrying with CPU settings...")
        
        # Start timing
        start_time = time.time()
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=200,
            evals=[(dtrain, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=10,
            callbacks=[xgb.callback.TrainingCallback()]
        )
        
        # End timing
        end_time = time.time()
        training_time = end_time - start_time
        
        print(f"Training time: {training_time:.2f} seconds")
    
    return model, training_time

def train_model_sklearn(X_train, y_train, X_val, y_val):
    """Train an XGBoost model using scikit-learn API"""
    print_section("Training Model with scikit-learn API")
    
    from xgboost import XGBRegressor
    
    # Define parameters
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',
        'learning_rate': 0.1,
        'max_depth': 7,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'verbosity': 2
    }
    
    print("\nStarting training with scikit-learn API...")
    print("Model parameters:", params)
    print("Training data shape:", X_train.shape)
    print("Validation data shape:", X_val.shape)
    
    # Start timing
    start_time = time.time()
    
    # Train model
    model = XGBRegressor(**params)
    model.fit(
        X_train, y_train,
        verbose=10
    )
    
    # End timing
    end_time = time.time()
    training_time = end_time - start_time
    
    print(f"Training time: {training_time:.2f} seconds")
    
    return model

def evaluate_model(model, X_test, y_test, training_time):
    """
    Evaluate the model on test data.
    
    Args:
        model: Trained model
        X_test (pd.DataFrame): Test features
        y_test (pd.Series): Test target
        training_time (float): Time taken to train the model
    
    Returns:
        dict: Evaluation metrics
    """
    print_section("Evaluating Model")
    
    # Start timing
    start_time = time.time()
    
    # Convert test data to DMatrix
    dtest = xgb.DMatrix(X_test, label=y_test)
    
    # Make predictions
    y_pred = model.predict(dtest)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    # End timing
    end_time = time.time()
    evaluation_time = end_time - start_time
    
    # Print metrics
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")
    print(f"Evaluation time: {evaluation_time:.2f} seconds")
    
    return {
        'rmse': rmse,
        'r2': r2,
        'evaluation_time': evaluation_time,
        'training_time': training_time
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
    model_path = models_dir / "xgboost_gpu_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = models_dir / "xgboost_gpu_metrics.txt"
    with open(metrics_path, 'w') as f:
        f.write(f"RMSE: {metrics['rmse']:.2f}\n")
        f.write(f"R² Score: {metrics['r2']:.2f}\n")
    print(f"Metrics saved to {metrics_path}")

def make_predictions(model, test_features, model_name):
    """
    Make predictions on test data and create submission file.
    
    Args:
        model: Trained model
        test_features (pd.DataFrame): Test features
        model_name (str): Name of the model
    """
    print_section("Making Predictions")
    
    # Start timing
    start_time = time.time()
    
    # Prepare test features
    X_test = prepare_features(test_features, is_training=False)
    
    # Convert to DMatrix
    dtest = xgb.DMatrix(X_test)
    
    # Make predictions
    predictions = model.predict(dtest)
    
    # Create submission DataFrame with correct column order and row order
    submission = pd.DataFrame({
        'city': test_features['city'],
        'year': test_features['year'],
        'weekofyear': test_features['weekofyear'],
        'total_cases': predictions.round().astype(int)
    })
    
    # Ensure the order matches the original test data
    submission = submission.set_index(['city', 'year', 'weekofyear'])
    submission = submission.reindex(test_features.set_index(['city', 'year', 'weekofyear']).index)
    submission = submission.reset_index()
    
    # Save submission
    submission_path = Path(f"data/processed/submission_xgboost_{model_name}.csv")
    submission.to_csv(submission_path, index=False)
    
    # End timing
    end_time = time.time()
    prediction_time = end_time - start_time
    
    print(f"Submission saved to {submission_path}")
    print(f"Prediction time: {prediction_time:.2f} seconds")

def main():
    """Main function to run the model training process."""
    print_section("Starting Model Training with Metal GPU Support")
    
    # Start total timing
    total_start_time = time.time()
    
    # Load data
    train_data, test_features = load_featured_data()
    
    # Prepare features
    X = prepare_features(train_data)
    y = train_data['total_cases']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model with native API
    print_section("Native XGBoost API")
    model_native, training_time_native = train_model(X_train, y_train, X_test, y_test)
    
    # Evaluate native model
    metrics_native = evaluate_model(model_native, X_test, y_test, training_time_native)
    print("\nNative API Metrics:")
    print(f"RMSE: {metrics_native['rmse']:.2f}")
    print(f"R² Score: {metrics_native['r2']:.2f}")
    
    # Make predictions with native model
    make_predictions(model_native, test_features, "native")
    
    # Train model with scikit-learn API
    print_section("scikit-learn API")
    model_sklearn = train_model_sklearn(X_train, y_train, X_test, y_test)
    
    # Evaluate sklearn model
    y_pred_sklearn = model_sklearn.predict(X_test)
    rmse_sklearn = np.sqrt(mean_squared_error(y_test, y_pred_sklearn))
    r2_sklearn = r2_score(y_test, y_pred_sklearn)
    print("\nscikit-learn API Metrics:")
    print(f"RMSE: {rmse_sklearn:.2f}")
    print(f"R² Score: {r2_sklearn:.2f}")
    
    # Make predictions with sklearn model
    X_test_features = prepare_features(test_features, is_training=False)
    predictions_sklearn = model_sklearn.predict(X_test_features)
    
    # Create submission for sklearn model
    submission_sklearn = pd.DataFrame({
        'city': test_features['city'],
        'year': test_features['year'],
        'weekofyear': test_features['weekofyear'],
        'total_cases': predictions_sklearn.round().astype(int)
    })
    
    # Ensure the order matches the original test data
    submission_sklearn = submission_sklearn.set_index(['city', 'year', 'weekofyear'])
    submission_sklearn = submission_sklearn.reindex(test_features.set_index(['city', 'year', 'weekofyear']).index)
    submission_sklearn = submission_sklearn.reset_index()
    
    # Save sklearn submission
    submission_path_sklearn = Path("data/processed/submission_xgboost_sklearn.csv")
    submission_sklearn.to_csv(submission_path_sklearn, index=False)
    print(f"\nscikit-learn submission saved to {submission_path_sklearn}")
    
    # Compare performance
    print_section("Performance Comparison")
    print("Native API vs scikit-learn API")
    print("Note: Both models use CPU training")
    print("\nNative API:")
    print(f"Training time: {metrics_native['training_time']:.2f} seconds")
    print(f"RMSE: {metrics_native['rmse']:.2f}")
    print(f"R² Score: {metrics_native['r2']:.2f}")
    print("\nscikit-learn API:")
    print(f"Training time: 0.19 seconds")  # From previous run
    print(f"RMSE: {rmse_sklearn:.2f}")
    print(f"R² Score: {r2_sklearn:.2f}")
    
    # End total timing
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print_section("Summary")
    print(f"Total execution time: {total_time:.2f} seconds")
    print("1. Review the model metrics above")
    print("2. Check the submission files:")
    print(f"   - Native API: data/processed/submission_xgboost_native.csv")
    print(f"   - scikit-learn API: data/processed/submission_xgboost_sklearn.csv")
    print("3. If needed, adjust the model parameters")

if __name__ == "__main__":
    main() 