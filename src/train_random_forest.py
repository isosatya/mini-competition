import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys

# Add parent directory to Python path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

def load_data():
    """Load the merged dataset and separate training and test data."""
    processed_dir = Path("data/processed")
    df = pd.read_csv(processed_dir / "cleaned_data_merged.csv")
    
    print(f"\nTotal rows in dataset: {len(df)}")
    print(f"Number of test rows: {len(df[df['is_test'] == True])}")
    print(f"Number of training rows: {len(df[df['is_test'] == False])}")
    
    # Separate training and test data
    train_data = df[df['is_test'] == False]
    test_data = df[df['is_test'] == True]
    
    return train_data, test_data

def create_lag_features(df, target_col, lags=[1, 2, 3, 4]):
    """Create lag features for the target variable."""
    df = df.copy()
    for lag in lags:
        df[f'{target_col}_lag_{lag}'] = df.groupby('city')[target_col].shift(lag)
    return df

def create_rolling_features(df, window_sizes=[4, 8, 12]):
    """Create rolling average features for weather variables."""
    weather_cols = [col for col in df.columns if any(term in col.lower() for term in ['temp', 'precip', 'humidity'])]
    df = df.copy()
    
    for window in window_sizes:
        for col in weather_cols:
            df[f'{col}_rolling_mean_{window}'] = df.groupby('city')[col].transform(lambda x: x.rolling(window=window, min_periods=1).mean())
            df[f'{col}_rolling_std_{window}'] = df.groupby('city')[col].transform(lambda x: x.rolling(window=window, min_periods=1).std())
    
    return df

def create_seasonal_features(df):
    """Create seasonal features from week_start_date."""
    df = df.copy()
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    df['month'] = df['week_start_date'].dt.month
    df['season'] = df['month'] % 12 // 3 + 1  # 1: Winter, 2: Spring, 3: Summer, 4: Fall
    return df

def prepare_features(df, is_test=False, train_data=None):
    """Prepare features for training or testing."""
    # Get target variable first (before dropping columns)
    y = df['total_cases'] if ('total_cases' in df.columns and not is_test) else None
    
    # Create engineered features
    if is_test:
        if train_data is None:
            raise ValueError("train_data must be provided when preparing test features")
        
        # Create lag features using only training data
        train_data = create_lag_features(train_data, 'total_cases')
        last_values = train_data.groupby('city')['total_cases'].last()
        
        # Create lag features for test data using last known values from training
        df = df.copy()
        for lag in [1, 2, 3, 4]:
            col = f'total_cases_lag_{lag}'
            df[col] = df['city'].map(last_values)
    else:
        df = create_lag_features(df, 'total_cases')
    
    # Create rolling features using only the current data (no leakage)
    df = create_rolling_features(df)
    df = create_seasonal_features(df)
    
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
    """Train and evaluate the Random Forest model with hyperparameter tuning."""
    # Define parameter grid for GridSearchCV
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    
    # Initialize base model
    rf = RandomForestRegressor(random_state=42)
    
    # Initialize time series split
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize GridSearchCV
    grid_search = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )
    
    # Fit GridSearchCV
    print(f"\nTraining model{' for ' + city if city else ''}...")
    grid_search.fit(X, y)
    
    # Get best model
    best_model = grid_search.best_estimator_
    
    # Store results
    results = {
        'mse': [],
        'mae': [],
        'r2': []
    }
    
    # Perform time series cross-validation with best model
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Train model
        best_model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = best_model.predict(X_test)
        
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
    
    return best_model, avg_metrics

def main():
    # Load data
    print("Loading data...")
    train_data, test_data = load_data()
    
    # Train separate models for each city
    cities = train_data['city'].unique()
    models = {}
    metrics = {}
    prepared_test_data = {}  # Store prepared test data for each city
    
    for city in cities:
        # Filter data for current city
        city_train_data = train_data[train_data['city'] == city]
        city_test_data = test_data[test_data['city'] == city]
        
        # Prepare features
        print(f"\nPreparing features for {city}...")
        X_train, y_train = prepare_features(city_train_data, is_test=False)
        X_test, _ = prepare_features(city_test_data, is_test=True, train_data=city_train_data)
        
        # Verify feature consistency
        train_features = set(X_train.columns)
        test_features = set(X_test.columns)
        
        if train_features != test_features:
            missing_in_test = train_features - test_features
            missing_in_train = test_features - train_features
            print(f"\nWarning: Feature mismatch for {city}:")
            if missing_in_test:
                print(f"Features missing in test data: {missing_in_test}")
            if missing_in_train:
                print(f"Features missing in training data: {missing_in_train}")
            raise ValueError("Feature mismatch between training and test data")
        
        prepared_test_data[city] = X_test  # Store prepared test data
        
        # Train and evaluate model
        model, city_metrics = train_and_evaluate(X_train, y_train, city)
        models[city] = model
        metrics[city] = city_metrics
        
        # Print results
        print(f"\nModel Performance for {city}:")
        print(f"Mean Squared Error: {city_metrics['mse']:.2f}")
        print(f"Mean Absolute Error: {city_metrics['mae']:.2f}")
        print(f"R2 Score: {city_metrics['r2']:.2f}")
        
        # Plot feature importance
        print(f"\nPlotting feature importance for {city}...")
        plot_feature_importance(model, X_train.columns.values, city)
    
    # Make predictions for each city using stored prepared test data
    all_predictions = []
    for city in cities:
        city_test_data = test_data[test_data['city'] == city]
        X_test = prepared_test_data[city]  # Use stored prepared test data
        predictions = models[city].predict(X_test)
        
        # Create submission DataFrame for this city
        city_submission = city_test_data[['city', 'year', 'weekofyear']].copy()
        city_submission['total_cases'] = predictions.round().astype(int)
        all_predictions.append(city_submission)
    
    # Combine predictions
    submission = pd.concat(all_predictions, ignore_index=True)
    
    # Save predictions
    submission.to_csv('data/processed/submission.csv', index=False)
    print("\nTest predictions saved to data/processed/submission.csv")

def plot_feature_importance(model, feature_names, city=None):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importance{" for " + city if city else ""}')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig(f'data/processed/feature_importance{"_" + city if city else ""}.png')
    plt.close()

if __name__ == "__main__":
    main() 