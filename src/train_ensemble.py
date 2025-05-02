"""
Ensemble Model Training Module

This module implements a stacking ensemble approach that combines multiple models
to improve prediction performance for the Dengue Fever prediction task.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
import os
import sys
import warnings
import time
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
import xgboost as xgb
import lightgbm as lgb

# Import custom modules
sys.path.append(str(Path(__file__).parent.parent))

try:
    from src.time_series_validation import TimeSeriesValidator, save_validation_results, OutbreakDetectionMetrics
except ImportError:
    print("Warning: time_series_validation module not found. Validation will be limited.")

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

def prepare_data_for_modeling(df, city=None, is_training=True):
    """
    Prepare data for model training or prediction.
    
    Args:
        df (pd.DataFrame): DataFrame with features
        city (str, optional): City to filter for
        is_training (bool): Whether preparing for training or prediction
    
    Returns:
        tuple: (X, y) for training, or (X, None) for prediction
    """
    # Filter data for specific city if provided
    if city:
        city_data = df[df['city'] == city].copy()
    else:
        city_data = df.copy()
    
    # Target variable (only for training data)
    if is_training and 'total_cases' in city_data.columns:
        y = city_data['total_cases']
    else:
        y = None
    
    # Remove non-feature columns
    non_feature_cols = ['city', 'week_start_date', 'total_cases']
    feature_cols = [col for col in city_data.columns if col not in non_feature_cols]
    
    # Select features
    X = city_data[feature_cols].copy()
    
    # Handle missing values
    X = X.fillna(0)
    
    # Handle infinite values by replacing them with large but finite values
    # This is crucial for XGBoost which can't handle inf values
    X = X.replace([np.inf, -np.inf], [1e9, -1e9])
    
    # Remove columns with extremely large values or std=0
    # These can cause numerical issues in model training
    for col in X.columns:
        if X[col].std() == 0:  # Constant columns
            print(f"Dropping constant column: {col}")
            X = X.drop(columns=[col])
        elif X[col].abs().max() > 1e10:  # Extremely large values
            print(f"Dropping column with extreme values: {col}")
            X = X.drop(columns=[col])
    
    # Check for NaN values that might have been introduced during computation
    X = X.fillna(0)
    
    return X, y

class BaseModel:
    """Base class for all models in the ensemble."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
    
    def fit(self, X, y):
        raise NotImplementedError("Subclasses must implement fit method")
    
    def predict(self, X):
        raise NotImplementedError("Subclasses must implement predict method")
    
    def get_feature_importance(self):
        raise NotImplementedError("Subclasses must implement get_feature_importance method")

class XGBoostModel(BaseModel):
    """XGBoost model wrapper."""
    
    def __init__(self, params=None):
        super().__init__("XGBoost")
        self.params = params or {
            'objective': 'reg:squarederror',
            'tree_method': 'hist',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 0.1,
            'random_state': 42,
            'missing': 0.0,  # Explicitly handle missing values
            'reg_alpha': 0.1,  # L1 regularization to prevent overfitting
            'reg_lambda': 1.0  # L2 regularization to prevent overfitting
        }
        self.model = xgb.XGBRegressor(**self.params)
    
    def fit(self, X, y):
        try:
            # Make sure data doesn't have inf values that XGBoost can't handle
            X_clean = X.replace([np.inf, -np.inf], [1e9, -1e9])
            X_clean = X_clean.fillna(0)
            
            self.model.fit(X_clean, y)
            return self
        except Exception as e:
            print(f"Error fitting XGBoost model: {str(e)}")
            # Fall back to a simple model if XGBoost fails
            from sklearn.ensemble import RandomForestRegressor
            print("Falling back to RandomForest model")
            self.model = RandomForestRegressor(n_estimators=100, random_state=42)
            self.model.fit(X.fillna(0), y)
            return self
    
    def predict(self, X):
        try:
            # Clean data for prediction too
            X_clean = X.replace([np.inf, -np.inf], [1e9, -1e9])
            X_clean = X_clean.fillna(0)
            return self.model.predict(X_clean)
        except Exception as e:
            print(f"Error in XGBoost prediction: {str(e)}")
            # Return a simple prediction if model fails
            return np.ones(len(X)) * y.mean() if hasattr(self, 'y_mean') else np.zeros(len(X))
    
    def get_feature_importance(self):
        try:
            importance = self.model.feature_importances_
            return pd.DataFrame({
                'feature': self.model.feature_names_in_,
                'importance': importance
            }).sort_values('importance', ascending=False)
        except:
            return pd.DataFrame({'feature': [], 'importance': []})

class LightGBMModel(BaseModel):
    """LightGBM model wrapper."""
    
    def __init__(self, params=None):
        super().__init__("LightGBM")
        self.params = params or {
            'objective': 'regression',
            'boosting_type': 'gbdt',
            'max_depth': 6,
            'learning_rate': 0.05,
            'n_estimators': 300,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        self.model = lgb.LGBMRegressor(**self.params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

class RandomForestModel(BaseModel):
    """Random Forest model wrapper."""
    
    def __init__(self, params=None):
        super().__init__("RandomForest")
        self.params = params or {
            'n_estimators': 300,
            'max_depth': 10,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1
        }
        self.model = RandomForestRegressor(**self.params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

class GradientBoostingModel(BaseModel):
    """Gradient Boosting model wrapper."""
    
    def __init__(self, params=None):
        super().__init__("GradientBoosting")
        self.params = params or {
            'n_estimators': 200,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'random_state': 42
        }
        self.model = GradientBoostingRegressor(**self.params)
    
    def fit(self, X, y):
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        return self.model.predict(X)
    
    def get_feature_importance(self):
        importance = self.model.feature_importances_
        return pd.DataFrame({
            'feature': X.columns,
            'importance': importance
        }).sort_values('importance', ascending=False)

class StackingEnsemble:
    """Stacking ensemble that combines multiple models."""
    
    def __init__(self, base_models, meta_model=None):
        """
        Initialize the stacking ensemble.
        
        Args:
            base_models (list): List of base models
            meta_model: Meta model for combining base models
        """
        self.base_models = base_models
        self.meta_model = meta_model or RidgeCV(alphas=[0.1, 1.0, 10.0])
        self.base_predictions = None
        self.successful_models = []
    
    def fit(self, X, y):
        """
        Fit the stacking ensemble.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
        
        Returns:
            self: Fitted ensemble
        """
        print("Fitting base models...")
        
        # Store the mean target for fallback predictions
        self.y_mean = y.mean()
        
        # Create time series cross-validator
        cv = TimeSeriesSplit(n_splits=5)
        
        # Track which models successfully trained
        successful_models = []
        successful_model_predictions = []
        
        # Train base models with cross-validation
        for i, model in enumerate(self.base_models):
            try:
                print(f"Training {model.name}...")
                
                # Initialize out-of-fold predictions for this model
                oof_predictions = np.zeros(X.shape[0])
                
                # Train and predict with cross-validation
                success = True
                for train_idx, test_idx in cv.split(X):
                    try:
                        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                        
                        # Fit model on training data
                        model.fit(X_train, y_train)
                        
                        # Make predictions on test data
                        oof_predictions[test_idx] = model.predict(X_test)
                    except Exception as e:
                        print(f"Error in CV fold for {model.name}: {str(e)}")
                        success = False
                        break
                
                if success:
                    # Store out-of-fold predictions
                    successful_model_predictions.append(oof_predictions)
                    
                    # Fit model on all data
                    model.fit(X, y)
                    successful_models.append(model)
                    print(f"Successfully trained {model.name}")
                else:
                    print(f"Failed to train {model.name} during cross-validation")
            except Exception as e:
                print(f"Error training {model.name}: {str(e)}")
        
        # Store the successful models
        self.successful_models = successful_models
        
        # Only proceed with meta-learning if we have at least one successful model
        if len(successful_models) > 0:
            # Create base predictions matrix from successful models
            self.base_predictions = np.column_stack(successful_model_predictions)
            
            # Train meta model on out-of-fold predictions
            print(f"Training meta model with {len(successful_models)} base models...")
            try:
                self.meta_model.fit(self.base_predictions, y)
                print("Meta model trained successfully")
            except Exception as e:
                print(f"Error training meta model: {str(e)}")
                # If meta model fails, we'll use simple averaging in predict()
                print("Will use simple averaging for prediction")
        else:
            print("No models trained successfully. Ensemble will use fallback prediction.")
        
        return self
    
    def predict(self, X):
        """
        Make predictions using the stacking ensemble.
        
        Args:
            X (pd.DataFrame): Features
        
        Returns:
            np.array: Predictions
        """
        # If no models trained successfully, return mean target
        if not self.successful_models:
            print("No successful models. Using fallback prediction.")
            return np.ones(len(X)) * self.y_mean
        
        try:
            # Make predictions with base models
            base_predictions = []
            for model in self.successful_models:
                try:
                    model_preds = model.predict(X)
                    base_predictions.append(model_preds)
                except Exception as e:
                    print(f"Error in {model.name} prediction: {str(e)}")
                    # Use mean prediction as fallback
                    base_predictions.append(np.ones(len(X)) * self.y_mean)
            
            # Stack predictions into a matrix
            base_predictions = np.column_stack(base_predictions)
            
            # Make predictions with meta model if it was trained successfully
            if hasattr(self, 'meta_model') and self.meta_model is not None:
                try:
                    meta_predictions = self.meta_model.predict(base_predictions)
                    return meta_predictions
                except Exception as e:
                    print(f"Error in meta model prediction: {str(e)}")
                    # Fall back to averaging base models
                    return np.mean(base_predictions, axis=1)
            else:
                # Simple averaging if meta model wasn't trained
                return np.mean(base_predictions, axis=1)
            
        except Exception as e:
            print(f"Error in ensemble prediction: {str(e)}")
            return np.ones(len(X)) * self.y_mean
    
    def evaluate(self, X, y):
        """
        Evaluate the ensemble.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
        
        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = self.predict(X)
        
        # Calculate metrics
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Calculate outbreak metrics if available
        try:
            from sklearn.metrics import f1_score
            # Define outbreaks based on percentile
            outbreak_threshold = np.percentile(y, 75)
            outbreak_true = (y > outbreak_threshold).astype(int)
            outbreak_pred = (y_pred > outbreak_threshold).astype(int)
            
            # Calculate F1 score for outbreak detection
            metrics['outbreak_f1'] = f1_score(outbreak_true, outbreak_pred)
            
            # Try to use the OutbreakDetectionMetrics if available
            try:
                outbreak_metrics = OutbreakDetectionMetrics.outbreak_detection_summary(y, y_pred)
                metrics.update(outbreak_metrics)
            except:
                pass
            
        except Exception as e:
            print(f"Error calculating outbreak metrics: {str(e)}")
        
        return metrics

def train_ensemble_for_city(train_data, city):
    """
    Train an ensemble model for a specific city.
    
    Args:
        train_data (pd.DataFrame): Training data
        city (str): City to train model for
    
    Returns:
        dict: Ensemble model and metrics
    """
    print_section(f"Training Ensemble Model for {city}")
    
    # Prepare data
    X, y = prepare_data_for_modeling(train_data, city)
    
    # Initialize base models
    base_models = [
        XGBoostModel(),
        LightGBMModel(),
        RandomForestModel(),
        GradientBoostingModel()
    ]
    
    # Initialize ensemble
    ensemble = StackingEnsemble(base_models)
    
    # Train ensemble
    start_time = time.time()
    ensemble.fit(X, y)
    training_time = time.time() - start_time
    
    # Evaluate ensemble
    metrics = ensemble.evaluate(X, y)
    
    # Print metrics
    print(f"Ensemble Model Metrics for {city}:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print(f"Training time: {training_time:.2f} seconds")
    
    return {
        'ensemble': ensemble,
        'metrics': metrics,
        'training_time': training_time
    }

def generate_ensemble_predictions(ensemble_models, test_features):
    """
    Generate predictions using ensemble models.
    
    Args:
        ensemble_models (dict): Dictionary of ensemble models by city
        test_features (pd.DataFrame): Test features
    
    Returns:
        pd.DataFrame: Predictions
    """
    print_section("Generating Ensemble Predictions")
    
    # Initialize list to store predictions for each city
    all_predictions = []
    
    # Generate predictions for each city
    for city, model_dict in ensemble_models.items():
        print(f"Generating predictions for {city}...")
        
        # Prepare test data for the specific city
        city_test = test_features[test_features['city'] == city]
        X_test, _ = prepare_data_for_modeling(city_test, city, is_training=False)
        
        # Get ensemble
        ensemble = model_dict['ensemble']
        
        # Make predictions
        predictions = ensemble.predict(X_test)
        
        # Round predictions to integers (dengue cases are counts)
        predictions = np.round(predictions).clip(0).astype(int)
        
        # Create predictions DataFrame
        city_predictions = pd.DataFrame({
            'city': city_test['city'],
            'year': city_test['year'],
            'weekofyear': city_test['weekofyear'],
            'total_cases': predictions
        })
        
        all_predictions.append(city_predictions)
    
    # Combine predictions
    combined_predictions = pd.concat(all_predictions, ignore_index=True)
    
    return combined_predictions

def save_ensemble_models(ensemble_models):
    """
    Save ensemble models.
    
    Args:
        ensemble_models (dict): Dictionary of ensemble models by city
    """
    print_section("Saving Ensemble Models")
    
    # Create models directory if it doesn't exist
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Save models
    model_path = models_dir / "ensemble_models.joblib"
    joblib.dump(ensemble_models, model_path)
    print(f"Models saved to {model_path}")

def main():
    """Main function to run the ensemble model training process."""
    print_section("Starting Ensemble Model Training")
    
    # Load enhanced featured data
    train_data, test_features = load_enhanced_featured_data()
    
    # Train ensemble models for each city
    cities = train_data['city'].unique()
    ensemble_models = {}
    
    for city in cities:
        ensemble_models[city] = train_ensemble_for_city(train_data, city)
    
    # Generate predictions
    predictions = generate_ensemble_predictions(ensemble_models, test_features)
    
    # Save predictions
    predictions_path = Path("data/processed/submission_ensemble.csv")
    predictions.to_csv(predictions_path, index=False)
    print(f"Predictions saved to {predictions_path}")
    
    # Save models
    save_ensemble_models(ensemble_models)
    
    print_section("Ensemble Training Complete")
    print("Final steps:")
    print("1. Review ensemble model performance")
    print("2. Compare with individual models")
    print("3. Submit predictions for evaluation")

if __name__ == "__main__":
    main()