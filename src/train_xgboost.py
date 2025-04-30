import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_data():
    """Load and prepare the dataset."""
    # Load the cleaned data
    df = pd.read_csv('data/processed/cleaned_data_merged.csv')
    
    # Split into train and test based on is_test column
    train_df = df[~df['is_test']].copy()
    test_df = df[df['is_test']].copy()
    
    print(f"Total rows: {len(df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Train rows: {len(train_df)}")
    
    # Prepare features
    X_train, y_train = prepare_features(train_df, is_test=False)
    X_test, _ = prepare_features(test_df, is_test=True)
    
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    
    return X_train, y_train, X_test, test_df

def prepare_features(df, is_test=False):
    """Prepare features for training or testing."""
    # Get target variable first (before dropping columns)
    y = df['total_cases'] if ('total_cases' in df.columns and not is_test and not df['is_test'].any()) else None
    
    # Drop non-feature columns
    features_to_drop = ['city', 'year', 'week_start_date', 'is_test', 'total_cases']
    X = df.drop(columns=[col for col in features_to_drop if col in df.columns])
    
    # Handle missing values in features
    X = X.fillna(X.mean())
    
    # Filter out rows where target is NaN (only for training data)
    if not is_test and y is not None:
        mask = ~y.isna()
        X = X[mask]
        y = y[mask]
    
    return X, y

def train_and_evaluate(X, y, city=None):
    """Train and evaluate the XGBoost model."""
    # Initialize model parameters
    params = {
        'objective': 'reg:squarederror',
        'tree_method': 'hist',  # Use CPU histogram-based algorithm
        'max_depth': 8,
        'learning_rate': 0.05,
        'n_rounds': 200,  # Use n_rounds instead of n_estimators
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.1,
        'random_state': 42,
        'eval_metric': ['rmse', 'mae']
    }
    
    # Initialize time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Store results
    results = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    # Perform time series cross-validation
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create DMatrix for XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train)
        dtest = xgb.DMatrix(X_test, label=y_test)
        
        # Train model
        start_time = time.time()
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=params['n_rounds'],
            evals=[(dtrain, 'train'), (dtest, 'test')],
            early_stopping_rounds=20,
            verbose_eval=10
        )
        training_time = time.time() - start_time
        print(f"Training time: {training_time:.2f} seconds")
        
        # Make predictions
        y_pred = model.predict(dtest)
        
        # Calculate metrics
        results['mse'].append(mean_squared_error(y_test, y_pred))
        results['mae'].append(mean_absolute_error(y_test, y_pred))
        results['r2'].append(r2_score(y_test, y_pred))
    
    # Calculate average metrics
    avg_metrics = {
        'mse': np.mean(results['mse']),
        'mae': np.mean(results['mae']),
        'r2': np.mean(results['r2'])
    }
    
    return model, avg_metrics

def plot_feature_importance(model, feature_names, city=None):
    """Plot feature importance."""
    importance = model.get_score(importance_type='gain')
    importance = {k: importance.get(k, 0) for k in feature_names}
    importance = pd.Series(importance).sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance{" for " + city if city else ""}')
    importance.plot(kind='bar')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(f'data/processed/feature_importance_xgboost{"_" + city if city else ""}.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    X_train, y_train, X_test, test_df = load_data()
    print(f"Train data shape: {X_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    
    # Train separate models for each city
    cities = test_df['city'].unique()
    print(f"Cities to process: {cities}")
    models = {}
    metrics = {}
    
    for city in cities:
        print(f"\nProcessing city: {city}")
        # Filter data for current city
        city_test_data = test_df[test_df['city'] == city]
        print(f"City test data shape: {city_test_data.shape}")
        
        # Train and evaluate model
        print(f"Training model for {city}...")
        model, city_metrics = train_and_evaluate(X_train, y_train, city)
        models[city] = model
        metrics[city] = city_metrics
        
        # Print results
        print(f"\nModel Performance for {city}:")
        print(f"Mean Squared Error: {city_metrics['mse']:.2f}")
        print(f"Mean Absolute Error: {city_metrics['mae']:.2f}")
        print(f"R2 Score: {city_metrics['r2']:.2f}")
        
        # Plot feature importance
        print(f"Plotting feature importance for {city}...")
        plot_feature_importance(model, X_train.columns.values, city)
    
    # Make predictions for each city
    print("\nMaking predictions...")
    all_predictions = []
    for city in cities:
        print(f"Making predictions for {city}...")
        city_test_data = test_df[test_df['city'] == city]
        X_test, _ = prepare_features(city_test_data, is_test=True)
        dtest = xgb.DMatrix(X_test)
        predictions = models[city].predict(dtest)
        
        # Create submission DataFrame for this city
        city_submission = city_test_data[['city', 'year', 'weekofyear']].copy()
        city_submission['total_cases'] = predictions.round().astype(int)
        all_predictions.append(city_submission)
    
    # Combine predictions
    submission = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions
    submission.to_csv('data/processed/submission_xgboost.csv', index=False)
    print("\nTest predictions saved to data/processed/submission_xgboost.csv")

if __name__ == "__main__":
    print("Starting script...")
    main() 