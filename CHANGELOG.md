# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Comprehensive DengAI prediction pipeline (`dengai_pipeline.py`) that orchestrates all stages
- Enhanced feature engineering with domain knowledge integration (`data_features_enhanced.py`)
- Extended lag features (8-12 weeks) to capture mosquito lifecycle and virus incubation periods
- Cumulative precipitation features to represent standing water over multiple weeks
- Interaction terms between temperature and humidity variables
- Advanced cyclic features for better seasonality representation
- Autoregressive features to capture outbreak dynamics
- Time series models including Prophet and SARIMAX (`train_time_series.py`)
- Negative Binomial Regression for overdispersed count data (`train_negative_binomial.py`)
- Stacking ensemble that combines multiple models (`train_ensemble.py`)
- Enhanced data preprocessing with seasonal imputation and outlier handling (`enhanced_preprocessing.py`)
- Specialized time series validation utilities (`time_series_validation.py`)
- Comprehensive evaluation framework focusing on outbreak detection (`evaluation_framework.py`)
- Configurable pipeline with basic and enhanced modes via `config/pipeline_config.ini`
- Detailed pipeline documentation in `README_pipeline.md`
- Enhanced model documentation in `README_ENHANCED_MODEL.md`
- Support for both conda environment and pip requirements

### Changed
- Updated README.md to include enhanced pipeline information
- Updated environment.yml with new dependencies
- Updated requirements.txt with all necessary packages
- Reorganized project structure to support pipeline architecture
- Changed file name environment.yaml to environment.yml for consistency
- Improved model training code with better error handling

### Fixed
- Bug in order of submission.csv generation in data_features.py
- Fixed broken dependency imports in training scripts
- Improved error handling in data preprocessing steps

## [0.1.0] - 2024-03-20

### Added
- Initial project setup
- Data processing pipeline
- Random Forest and XGBoost model implementations
- Documentation and README files

### Changed
- Removed GPU support from LightGBM model due to compatibility issues
- Updated model training scripts to handle date features correctly

### Fixed
- Fixed date conversion issues in feature preparation
- Fixed feature handling in model training scripts