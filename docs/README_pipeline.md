# DengAI Prediction Pipeline

A comprehensive, modular pipeline for predicting dengue fever cases in San Juan, Puerto Rico and Iquitos, Peru based on climate variables and historical data.

## Overview and Motivation

The DengAI Prediction Pipeline orchestrates the entire process from raw data to model evaluation for predicting dengue fever outbreaks. It implements both basic approaches and an enhanced implementation that incorporates domain knowledge about dengue transmission, mosquito lifecycle, and virus incubation periods.

Key features of the pipeline:
- Modular architecture allowing selective execution of pipeline stages
- Support for both basic and enhanced implementation strategies
- Comprehensive logging and error handling
- Reproducible runs with configuration tracking
- Multiple modeling approaches with ensemble capabilities
- Specialized evaluation for outbreak detection
- Detailed visualization and model comparison

This pipeline was designed to improve prediction accuracy for dengue fever cases, with particular emphasis on capturing outbreak dynamics, which is critical for public health planning and response.

## Installation Requirements

### Basic Requirements

```bash
# Create a conda environment
conda create -n dengai python=3.8
conda activate dengai

# Install basic dependencies
pip install -r requirements.txt
```

### Requirements File (requirements.txt)
```
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
xgboost>=1.4.0
lightgbm>=3.2.0
statsmodels>=0.13.0
shap>=0.40.0
joblib>=1.0.0
psutil>=5.8.0
```

### Additional Dependencies for Enhanced Implementation

For the enhanced implementation with specialized time series models, additional dependencies are required:

```bash
# Install Prophet (may require Rust compiler)
pip install prophet

# Install other specialized packages
pip install pmdarima  # For auto ARIMA model selection
pip install darts     # Time series library
```

Note: Prophet can be challenging to install on some systems due to its Stan dependency. If you encounter issues, see the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html) for detailed instructions.

## Directory Structure

The pipeline expects/creates the following directory structure:

```
mini-competition/
├── config/                  # Configuration files
│   └── pipeline_config.ini  # Main pipeline configuration
├── data/
│   ├── raw/                 # Raw input data
│   │   ├── dengue_features_train.csv
│   │   ├── dengue_labels_train.csv
│   │   └── dengue_features_test.csv
│   └── processed/           # Processed data and intermediary results
├── logs/                    # Pipeline execution logs
├── models/                  # Trained models
├── results/                 # Pipeline run results
│   └── YYYYMMDD_HHMMSS/     # Run-specific directory with timestamp
│       ├── run_config.ini   # Configuration used for this run
│       ├── run_summary.json # Run summary
│       ├── processed_data/  # Copy of processed data for this run
│       ├── models/          # Copy of models for this run
│       └── evaluation/      # Evaluation results
├── src/                     # Source code
│   ├── dengai_pipeline.py   # Main pipeline script
│   ├── data_cleaner.py      # Basic data preprocessing
│   ├── enhanced_preprocessing.py # Enhanced preprocessing
│   ├── data_features.py     # Basic feature engineering
│   ├── data_features_enhanced.py # Enhanced feature engineering
│   ├── train_*.py           # Model training modules
│   ├── time_series_validation.py # Time series validation utilities
│   └── evaluation_framework.py # Comprehensive evaluation framework
└── README_pipeline.md       # This documentation
```

## Pipeline Components Explanation

The pipeline consists of several main components:

### 1. Data Preprocessing
- **Basic Implementation**: Standard cleaning, imputation, and normalization
- **Enhanced Implementation**: Seasonal imputation, outlier detection, stationarity transformations

### 2. Feature Engineering
- **Basic Implementation**: Basic temporal features, weather averages, and simple lag features
- **Enhanced Implementation**: 
  - Extended lag features (8-12 weeks) capturing mosquito lifecycle and virus incubation
  - Cumulative precipitation features representing standing water
  - Temperature-humidity interaction terms
  - Enhanced cyclic features for seasonality
  - Autoregressive features capturing outbreak dynamics

### 3. Model Training
- **Random Forest**: Tree-based ensemble model
- **XGBoost/LightGBM**: Gradient boosting frameworks
- **Time Series Models**: Prophet and SARIMAX for time series forecasting
- **Negative Binomial Regression**: Specialized for overdispersed count data
- **Ensemble Approach**: Stacking multiple models for improved performance

### 4. Model Evaluation
- **Standard Metrics**: RMSE, MAE, R²
- **Outbreak-Specific Metrics**: Specialized for dengue outbreak detection
- **Visualization**: Comprehensive visualization of model performance
- **Feature Importance**: Understanding key predictors

## Usage Instructions

### Basic Usage

```bash
# Run the entire pipeline with default configuration
python src/dengai_pipeline.py

# Run specific stages in basic mode
python src/dengai_pipeline.py --mode basic --stages preprocess feature_engineering model_training

# Use a custom configuration file
python src/dengai_pipeline.py --config custom_config.ini
```

### Advanced Usage

```bash
# Run the enhanced implementation
python src/dengai_pipeline.py --mode enhanced

# Run only model training and evaluation stages
python src/dengai_pipeline.py --stages model_training model_evaluation

# Run with custom configuration and specific stages
python src/dengai_pipeline.py --config custom_config.ini --mode enhanced --stages feature_engineering model_training
```

### Configuration Example

The pipeline is highly configurable via the `config/pipeline_config.ini` file. Here's an example configuration for an enhanced run:

```ini
[GENERAL]
mode = enhanced
run_stages = preprocess,feature_engineering,model_training,model_evaluation

[PATHS]
raw_data_dir = data/raw
processed_data_dir = data/processed
models_dir = models
results_dir = results

[PREPROCESSING]
imputation_method = seasonal
handle_outliers = true
outlier_method = isolation_forest
apply_transformations = true

[FEATURE_ENGINEERING]
extended_lag_features = true
max_lag_weeks = 12
create_interaction_terms = true
create_cumulative_precip = true
create_autoregressive_features = true
create_enhanced_cyclic = true

[MODEL_TRAINING]
models = random_forest,xgboost,time_series,negative_binomial,ensemble
city_specific_models = true
random_seed = 42
validation_method = time_series_split
n_splits = 5
save_models = true

[EVALUATION]
outbreak_percentile = 75
outbreak_penalty_weight = 2.0
generate_visualizations = true
save_feature_importance = true
use_outbreak_metrics = true
city_specific_metrics = true
```

## Configuration Options

### General Settings
- `mode`: `basic` or `enhanced` mode for implementation complexity
- `run_stages`: Comma-separated list of stages to run (`preprocess`, `feature_engineering`, `model_training`, `model_evaluation`)

### Path Settings
- `raw_data_dir`: Directory containing raw data files
- `processed_data_dir`: Directory for processed data and intermediary results
- `models_dir`: Directory for trained models
- `results_dir`: Directory for run results

### Preprocessing Settings
- `imputation_method`: Method for missing value imputation (`mean`, `median`, `seasonal`)
- `handle_outliers`: Whether to detect and handle outliers
- `outlier_method`: Method for outlier detection (`isolation_forest`, `zscore`, `iqr`)
- `apply_transformations`: Whether to apply transformations for stationarity

### Feature Engineering Settings
- `extended_lag_features`: Whether to create extended lag features
- `max_lag_weeks`: Maximum number of weeks for lag features
- `create_interaction_terms`: Whether to create interaction terms
- `create_cumulative_precip`: Whether to create cumulative precipitation features
- `create_autoregressive_features`: Whether to create autoregressive features
- `create_enhanced_cyclic`: Whether to create enhanced cyclic features

### Model Training Settings
- `models`: Comma-separated list of models to train
- `city_specific_models`: Whether to train city-specific models
- `random_seed`: Random seed for reproducibility
- `validation_method`: Method for time series validation
- `n_splits`: Number of splits for cross-validation
- `save_models`: Whether to save trained models

### Evaluation Settings
- `outbreak_percentile`: Percentile threshold for defining outbreaks
- `outbreak_penalty_weight`: Weight for outbreak periods in weighted metrics
- `generate_visualizations`: Whether to generate visualizations
- `save_feature_importance`: Whether to save feature importance analysis
- `use_outbreak_metrics`: Whether to use specialized outbreak detection metrics
- `city_specific_metrics`: Whether to generate city-specific metrics

## Expected Outputs

After a successful pipeline run, you'll find the following outputs:

### 1. Run-Specific Directory
A timestamped directory in the `results/` folder containing all outputs from the run.

### 2. Processed Data
Cleaned and feature-engineered datasets ready for modeling.

### 3. Trained Models
Serialized model files saved in the `models/` directory.

### 4. Evaluation Results
- Model comparison metrics in CSV format
- Detailed evaluation metrics for each model
- Visualizations of model performance
- Feature importance analysis

### 5. Logs
Detailed logs of the pipeline execution in the `logs/` directory.

### 6. Run Summary
A JSON file summarizing the run, including:
- Configuration used
- Execution time
- Success/failure status of each stage
- System information for reproducibility

## Performance Metrics Explanation

The pipeline uses several metrics to evaluate model performance:

### Standard Metrics
- **RMSE (Root Mean Squared Error)**: Measures the square root of the average squared differences between predicted and actual values
- **MAE (Mean Absolute Error)**: Measures the average absolute differences between predicted and actual values
- **R² (R-squared)**: Proportion of the variance in the dependent variable that is predictable from the independent variables

### Specialized Outbreak Metrics
- **Outbreak F1 Score**: F1 score for detecting outbreak periods (balancing precision and recall)
- **Weighted RMSE**: RMSE with higher weights for outbreak periods
- **Outbreak RMSE**: RMSE calculated only during outbreak periods
- **Dengue Competition Metric**: Mean absolute error divided by the true case count (MAE relative to magnitude)

## Troubleshooting Guide

### Common Issues and Solutions

#### Missing Dependencies
```
Error: Missing required dependencies: prophet, statsmodels
```
**Solution**: Install the missing dependencies using pip.
```bash
pip install prophet statsmodels
```

#### File Not Found Errors
```
FileNotFoundError: Missing required data files: dengue_features_train.csv
```
**Solution**: Ensure the required data files are in the `data/raw/` directory.

#### Memory Issues
```
MemoryError: Unable to allocate array with shape (1000000, 500)
```
**Solution**: Reduce model complexity or use a machine with more memory.

#### Prophet Installation Issues
```
Error: Failed building wheel for prophet
```
**Solution**: Prophet requires a C++ compiler. See the [Prophet installation guide](https://facebook.github.io/prophet/docs/installation.html).

#### Pipeline Stage Failures
```
Error in preprocessing stage: ...
```
**Solution**: Check the logs for detailed error messages. Fix the issue and run the pipeline again with only the failed stages.

### Debugging Tips

1. Check the logs in the `logs/` directory for detailed error messages
2. Run individual stages to isolate the problem
3. Use the `--mode basic` flag to test with simpler implementations
4. Verify configuration file settings

## References and Citations

### Data Sources
- **DengAI Competition Data**: Provided by DrivenData, NOAA, and IQSS at the Harvard Datverse

### Methodological References
1. Johansson, M. A., Dominici, F., & Glass, G. E. (2009). Local and global effects of climate on dengue transmission in Puerto Rico. PLoS neglected tropical diseases, 3(2), e382.

2. Hii, Y. L., Zhu, H., Ng, N., Ng, L. C., & Rocklöv, J. (2012). Forecast of dengue incidence using temperature and rainfall. PLoS neglected tropical diseases, 6(11), e1908.

3. Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45. (Prophet model)

4. Stolerman, L. M., Coombs, D., & Boatto, S. (2019). SIR-Network Model and Its Application to Dengue Fever. SIAM Journal on Applied Mathematics, 79(2), 625-648.

5. Codeço, C. T., Villela, D. A., & Coelho, F. C. (2018). Estimating the effective reproduction number of dengue considering temperature-dependent generation intervals. Epidemics, 25, 101-111.

### Software Libraries
- scikit-learn: Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. JMLR 12, pp. 2825-2830.
- Prophet: Taylor, S. J., & Letham, B. (2018). Forecasting at scale. The American Statistician, 72(1), 37-45.
- XGBoost: Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.
- LightGBM: Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NIPS.
- statsmodels: Seabold, S., & Perktold, J. (2010). Statsmodels: Econometric and statistical modeling with python. Proceedings of the 9th Python in Science Conference.

---

*This project is part of the DengAI challenge to predict dengue fever cases based on climate variables.*