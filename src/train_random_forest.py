import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
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

def prepare_features(df, is_test=False):
    """Prepare features for training or testing."""
    # Get target variable first (before dropping columns)
    y = df['total_cases'] if ('total_cases' in df.columns and not is_test) else None
    
    # Drop non-feature columns
    #features_to_drop = ['city', 'year', 'weekofyear', 'week_start_date', 'is_test', 'total_cases']
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

def train_and_evaluate(X, y):
    """Train and evaluate the Random Forest model."""
    # Initialize model
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42
    )
    
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
        
        # Train model
        rf.fit(X_train, y_train)
        
        # Make predictions
        y_pred = rf.predict(X_test)
        
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
    
    return rf, avg_metrics

def plot_feature_importance(model, feature_names):
    """Plot feature importance."""
    importance = model.feature_importances_
    indices = np.argsort(importance)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title('Feature Importance')
    plt.bar(range(len(importance)), importance[indices])
    plt.xticks(range(len(importance)), feature_names[indices], rotation=90)
    plt.tight_layout()
    plt.savefig('data/processed/feature_importance.png')
    plt.close()

def main():
    # Load data
    print("Loading data...")
    train_data, test_data = load_data()
    
    # Prepare training features
    print("\nPreparing features...")
    X_train, y_train = prepare_features(train_data, is_test=False)
    print(f"Training features shape: {X_train.shape}")
    print(f"Training target shape: {y_train.shape}")
    
    # Train and evaluate model
    print("\nTraining and evaluating model...")
    model, metrics = train_and_evaluate(X_train, y_train)
    
    # Print results
    print("\nModel Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.2f}")
    print(f"Mean Absolute Error: {metrics['mae']:.2f}")
    print(f"R2 Score: {metrics['r2']:.2f}")
    
    # Plot feature importance
    print("\nPlotting feature importance...")
    plot_feature_importance(model, X_train.columns.values)
    print("Feature importance plot saved to data/processed/feature_importance.png")
    
    # Prepare test features for prediction
    X_test, _ = prepare_features(test_data, is_test=True)
    print(f"\nTest features shape: {X_test.shape}")
    
    # Make predictions on test data
    test_predictions = model.predict(X_test)
    
    # Create submission DataFrame
    submission = test_data[['city', 'year', 'weekofyear']].copy()
    submission['total_cases'] = test_predictions.round().astype(int)  # Round predictions to integers
    
    # Save predictions
    submission.to_csv('data/processed/submission.csv', index=False)
    print("\nTest predictions saved to data/processed/submission.csv")

if __name__ == "__main__":
    main() 