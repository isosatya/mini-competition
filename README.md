# Dengue Fever Prediction Model

This project implements machine learning models to predict dengue fever cases based on weather and environmental data.

## Models

The project includes three different model implementations:

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
   - Currently running on CPU (GPU support removed due to compatibility issues)

## Data Processing

The data processing pipeline includes:

1. Data investigation and cleaning
2. Feature engineering
3. Feature combination
4. Model training and evaluation

See `docs/README_processing.md` for detailed information about the data processing steps.

## Setup

1. Create and activate the conda environment:
   ```bash
   conda env create -f environment.yaml
   conda activate mini-competition
   ```

2. Install the package in development mode:
   ```bash
   pip install -e .
   ```

## Usage

1. Run the data processing pipeline:
   ```bash
   python src/data_investigation.py
   python src/data_cleaner.py
   python src/data_cleaner_combinations.py
   python src/data_features.py
   ```

2. Train a model:
   ```bash
   # For Random Forest
   python src/train_random_forest_new.py
   
   # For XGBoost
   python src/train_xgboost_new.py
   
   # For LightGBM
   python src/train_lightgbm.py
   ```

## Project Structure

```
.
├── data/
│   ├── raw/           # Raw data files
│   └── processed/     # Processed data files
├── docs/              # Documentation
├── models/            # Trained models and metrics
├── src/               # Source code
└── environment.yaml   # Conda environment specification
```

## Dependencies

- Python 3.9
- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- joblib

See `environment.yaml` for the complete list of dependencies.