# Dengue Fever Prediction Model

This project implements machine learning models to predict dengue fever cases based on weather and environmental data for San Juan, Puerto Rico and Iquitos, Peru.

## Enhanced Pipeline

A new comprehensive prediction pipeline has been developed that orchestrates all steps from data preprocessing to model evaluation:

```bash
# Run the complete pipeline with default configuration
python src/dengai_pipeline.py

# Run with enhanced implementation
python src/dengai_pipeline.py --mode enhanced

# Run specific stages only
python src/dengai_pipeline.py --stages preprocess feature_engineering model_training
```

See `README_pipeline.md` for detailed information about the enhanced pipeline.

## Models

The project includes multiple model implementations:

### Basic Models
1. **Random Forest**
   - Implemented in `train_random_forest_new.py`
   - Uses scikit-learn's RandomForestRegressor
   - Includes hyperparameter tuning and model evaluation

2. **XGBoost**
   - Implemented in `train_xgboost_new.py`
   - Uses XGBoost's XGBRegressor
   - Includes hyperparameter tuning and model evaluation

3. **LightGBM**
   - Implemented in `train_lightgbm.py`
   - Uses LightGBM's LGBMRegressor
   - Includes hyperparameter tuning and model evaluation

### Enhanced Models
4. **Time Series Models**
   - Implemented in `train_time_series.py`
   - Includes Prophet and SARIMAX models
   - Captures seasonal patterns and trends

5. **Negative Binomial Regression**
   - Implemented in `train_negative_binomial.py`
   - Specifically designed for overdispersed count data
   - Handles the skewed distribution of dengue cases

6. **Stacking Ensemble**
   - Implemented in `train_ensemble.py`
   - Combines multiple models for improved performance
   - Weights models based on validation performance

## Data Processing and Feature Engineering

The data processing pipeline includes:

1. **Data Preprocessing**
   - Basic: Standard cleaning and imputation
   - Enhanced: Seasonal imputation, outlier detection, and stationarity transformations

2. **Feature Engineering**
   - Basic: Temporal features, weather averages, and simple lag features
   - Enhanced: Extended lag features (8-12 weeks), cumulative precipitation, interaction terms, enhanced cyclic features, and outbreak dynamics features

3. **Model Training**
   - Time series cross-validation to prevent data leakage
   - City-specific modeling to capture local patterns
   - Specialized evaluation for outbreak detection

See `docs/README_processing.md` for detailed information about the basic data processing steps and `README_ENHANCED_MODEL.md` for the enhanced approach.

## Setup

### Using Conda
```bash
# Create and activate the conda environment
conda env create -f environment.yml
conda activate mini-competition
```

### Using Pip
```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Basic Workflow
1. Run the data processing pipeline:
   ```bash
   python src/data_investigation.py
   python src/data_cleaner.py
   python src/data_cleaner_combinations.py
   python src/data_features.py
   ```

2. Train individual models:
   ```bash
   # For Random Forest
   python src/train_random_forest_new.py
   
   # For XGBoost
   python src/train_xgboost_new.py
   
   # For LightGBM
   python src/train_lightgbm.py
   ```

### Enhanced Workflow
1. Run the complete pipeline:
   ```bash
   python src/dengai_pipeline.py --mode enhanced
   ```

2. Or run specific enhanced components:
   ```bash
   # Enhanced preprocessing
   python src/enhanced_preprocessing.py
   
   # Enhanced feature engineering
   python src/data_features_enhanced.py
   
   # Time series models
   python src/train_time_series.py
   
   # Negative binomial model
   python src/train_negative_binomial.py
   
   # Ensemble model
   python src/train_ensemble.py
   
   # Comprehensive evaluation
   python src/evaluation_framework.py
   ```

## Project Structure

```
.
├── config/             # Configuration files
│   └── pipeline_config.ini  # Pipeline configuration
├── data/
│   ├── raw/            # Raw data files
│   └── processed/      # Processed data files
├── docs/               # Documentation
├── logs/               # Pipeline execution logs
├── models/             # Trained models and metrics
├── results/            # Pipeline run results with timestamps
├── src/                # Source code
│   ├── dengai_pipeline.py              # Main pipeline script
│   ├── data_cleaner.py                 # Basic data preprocessing
│   ├── enhanced_preprocessing.py       # Enhanced preprocessing
│   ├── data_features.py                # Basic feature engineering
│   ├── data_features_enhanced.py       # Enhanced feature engineering
│   ├── train_*.py                      # Model training scripts
│   ├── time_series_validation.py       # Time series validation utilities
│   └── evaluation_framework.py         # Comprehensive evaluation framework
├── environment.yml     # Conda environment specification
└── requirements.txt    # Pip requirements file
```

## Dependencies

- Python 3.9
- Core data science: pandas, numpy, scikit-learn, scipy
- Machine learning: xgboost, lightgbm, statsmodels
- Time series: prophet (optional), pmdarima
- Visualization: matplotlib, seaborn, plotly
- Utilities: joblib, psutil, tqdm

See `environment.yml` or `requirements.txt` for the complete list of dependencies.