"""
Negative Binomial Regression Model Training Module

This module implements a negative binomial regression model specifically designed
for overdispersed count data like dengue case counts.
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
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.feature_selection import SelectFromModel
from sklearn.pipeline import Pipeline
import time

# Try to import statsmodels for negative binomial regression
try:
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Using sklearn's TweedieRegressor as fallback.")

# If statsmodels is not available, use sklearn's TweedieRegressor as fallback
if not STATSMODELS_AVAILABLE:
    from sklearn.linear_model import TweedieRegressor

# Suppress warnings
warnings.filterwarnings('ignore')

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

def prepare_data_for_negative_binomial(df, city=None):
    """
    Prepare data for negative binomial regression.
    
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
    if 'total_cases' in city_data.columns:
        y = city_data['total_cases']
    else:
        y = None
    
    # Remove non-feature columns
    non_feature_cols = ['city', 'week_start_date', 'total_cases']
    feature_cols = [col for col in city_data.columns if col not in non_feature_cols]
    
    # Select features
    X = city_data[feature_cols].copy()
    
    return X, y

def select_important_features(X, y, threshold=0.01):
    """
    Select important features using a simple model.
    
    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target variable
        threshold (float): Feature importance threshold
    
    Returns:
        list: Selected feature names
    """
    # Use TweedieRegressor with power=2 (gamma) for feature selection
    selector = SelectFromModel(
        TweedieRegressor(power=2, alpha=0.1, link='log'),
        threshold=threshold
    )
    
    # Fit the selector
    selector.fit(X, y)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    
    # Get selected feature names
    selected_features = X.columns[selected_indices].tolist()
    
    return selected_features

def train_statsmodels_nb(X, y, city):
    """
    Train a negative binomial model using statsmodels.
    
    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target variable
        city (str): City name
    
    Returns:
        dict: Model and metrics
    """
    print(f"Training statsmodels negative binomial model for {city}...")
    
    # Combine features and target into a single DataFrame
    data = X.copy()
    data['total_cases'] = y
    
    # Select important features to avoid convergence issues
    important_features = select_important_features(X, y)
    print(f"Selected {len(important_features)} important features")
    
    # Create formula (exclude very high cardinality features)
    formula = "total_cases ~ " + " + ".join(important_features)
    
    # Train model
    try:
        nb_model = smf.glm(
            formula=formula,
            data=data,
            family=sm.families.NegativeBinomial()
        ).fit()
        
        # Make in-sample predictions
        y_pred = nb_model.predict(data[important_features])
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Print metrics
        print(f"Model Metrics:")
        print(f"RMSE: {metrics['rmse']:.2f}")
        print(f"MAE: {metrics['mae']:.2f}")
        print(f"R²: {metrics['r2']:.2f}")
        
        # Extract feature importance from model summary
        feature_importance = pd.DataFrame({
            'feature': important_features,
            'importance': np.abs(nb_model.params[1:])  # Skip intercept
        }).sort_values('importance', ascending=False)
        
        return {
            'model': nb_model,
            'metrics': metrics,
            'feature_importance': feature_importance,
            'important_features': important_features
        }
        
    except Exception as e:
        print(f"Error training statsmodels negative binomial model: {e}")
        return None

def train_sklearn_nb(X, y, city):
    """
    Train a negative binomial-like model using sklearn's TweedieRegressor.
    
    Args:
        X (pd.DataFrame): Feature DataFrame
        y (pd.Series): Target variable
        city (str): City name
    
    Returns:
        dict: Model and metrics
    """
    print(f"Training sklearn Tweedie model for {city}...")
    
    # Create time series split for validation
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Initialize metrics dictionary
    metrics = {'rmse': [], 'mae': [], 'r2': []}
    
    # Select important features
    important_features = select_important_features(X, y)
    print(f"Selected {len(important_features)} important features")
    
    # Use only important features
    X_selected = X[important_features]
    
    # Train model with time series cross-validation
    for train_idx, test_idx in tscv.split(X_selected):
        X_train, X_test = X_selected.iloc[train_idx], X_selected.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Create pipeline with scaling and Tweedie Regressor (power=2 approximates negative binomial)
        pipeline = Pipeline([
            ('scaler', RobustScaler()),
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
    print(f"Model Cross-Validation Metrics:")
    print(f"RMSE: {avg_metrics['rmse']:.2f}")
    print(f"MAE: {avg_metrics['mae']:.2f}")
    print(f"R²: {avg_metrics['r2']:.2f}")
    
    # Train final model on all data
    final_pipeline = Pipeline([
        ('scaler', RobustScaler()),
        ('regressor', TweedieRegressor(power=2, alpha=0.5, link='log', max_iter=1000))
    ])
    
    final_pipeline.fit(X_selected, y)
    
    # Get feature importance from coefficients
    feature_importance = pd.DataFrame({
        'feature': important_features,
        'importance': np.abs(final_pipeline.named_steps['regressor'].coef_)
    }).sort_values('importance', ascending=False)
    
    return {
        'model': final_pipeline,
        'metrics': avg_metrics,
        'feature_importance': feature_importance,
        'important_features': important_features
    }

def train_negative_binomial_model(train_data, city):
    """
    Train a negative binomial model for a specific city.
    
    Args:
        train_data (pd.DataFrame): Training data
        city (str): City to train model for
    
    Returns:
        dict: Model details
    """
    print_section(f"Training Negative Binomial Model for {city}")
    
    # Prepare data
    X, y = prepare_data_for_negative_binomial(train_data, city)
    
    # Train model using statsmodels if available, otherwise use sklearn
    if STATSMODELS_AVAILABLE:
        model_details = train_statsmodels_nb(X, y, city)
        if model_details is None:
            print("Falling back to sklearn Tweedie model...")
            model_details = train_sklearn_nb(X, y, city)
    else:
        model_details = train_sklearn_nb(X, y, city)
    
    # Plot feature importance
    if model_details and 'feature_importance' in model_details:
        feature_importance = model_details['feature_importance']
        
        plt.figure(figsize=(10, 8))
        plt.barh(feature_importance['feature'].head(20), feature_importance['importance'].head(20))
        plt.title(f'Top 20 Feature Importance for {city}')
        plt.xlabel('Absolute Coefficient Value')
        plt.tight_layout()
        
        # Create models directory if it doesn't exist
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        plt.savefig(f'models/nb_feature_importance_{city}.png')
        plt.close()
    
    return model_details

def generate_predictions(model_details, test_features, city):
    """
    Generate predictions for a specific city.
    
    Args:
        model_details (dict): Model details
        test_features (pd.DataFrame): Test features
        city (str): City to make predictions for
    
    Returns:
        pd.DataFrame: Predictions for the city
    """
    print_section(f"Generating Predictions for {city}")
    
    # Filter test features for the specific city
    city_test = test_features[test_features['city'] == city].copy()
    
    # Prepare data
    X_test, _ = prepare_data_for_negative_binomial(city_test, city)
    
    # Use only the important features used in training
    important_features = model_details['important_features']
    X_test_selected = X_test[important_features]
    
    # Make predictions
    if STATSMODELS_AVAILABLE and isinstance(model_details['model'], sm.GLMResults):
        # Add constant for statsmodels prediction if needed
        if 'const' in model_details['model'].params:
            X_test_selected = sm.add_constant(X_test_selected)
        
        # Predict using statsmodels
        predictions = model_details['model'].predict(X_test_selected)
    else:
        # Predict using sklearn pipeline
        predictions = model_details['model'].predict(X_test_selected)
    
    # Create predictions DataFrame
    predictions_df = pd.DataFrame({
        'city': city_test['city'],
        'year': city_test['year'],
        'weekofyear': city_test['weekofyear'],
        'total_cases': np.round(predictions).clip(0).astype(int)
    })
    
    return predictions_df

def main():
    """Main function to run the negative binomial model training process."""
    print_section("Starting Negative Binomial Model Training")
    
    # Load enhanced featured data
    train_data, test_features = load_enhanced_featured_data()
    
    # Train models and generate predictions for each city
    cities = train_data['city'].unique()
    
    # Dictionary to store models
    models = {}
    
    # List to store predictions
    all_predictions = []
    
    for city in cities:
        print(f"Processing {city}...")
        
        # Train model
        start_time = time.time()
        model_details = train_negative_binomial_model(train_data, city)
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Store model
        models[city] = model_details
        
        # Generate predictions
        predictions = generate_predictions(model_details, test_features, city)
        all_predictions.append(predictions)
    
    # Combine predictions
    submission = pd.concat(all_predictions, ignore_index=True)
    
    # Save submission
    submission_path = Path("data/processed/submission_negative_binomial.csv")
    submission.to_csv(submission_path, index=False)
    print(f"Submission saved to {submission_path}")
    
    # Save models
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / "negative_binomial_models.joblib"
    joblib.dump(models, model_path)
    print(f"Models saved to {model_path}")
    
    print_section("Training Complete")
    print("Negative binomial models have been trained and predictions generated.")
    print("Next steps:")
    print("1. Evaluate the models")
    print("2. Implement ensembling with other models")
    print("3. Optimize model parameters")

if __name__ == "__main__":
    main()