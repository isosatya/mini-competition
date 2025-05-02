# Dengue Fever Prediction Model - Enhanced Implementation

This document outlines the enhanced approach to predicting dengue fever cases in San Juan, Puerto Rico and Iquitos, Peru. The implementation incorporates domain knowledge about dengue transmission, mosquito lifecycle, and climatic factors to significantly improve prediction accuracy.

## Table of Contents
1. [Key Enhancements](#key-enhancements)
2. [Feature Engineering](#feature-engineering)
3. [Model Architecture](#model-architecture)
4. [Data Preprocessing](#data-preprocessing)
5. [Evaluation Framework](#evaluation-framework)
6. [Usage Guide](#usage-guide)
7. [Results and Performance](#results-and-performance)

## Key Enhancements

The enhanced model addresses several limitations of the original implementation:

1. **Biological Relevance:** Incorporates domain knowledge about dengue transmission, mosquito lifecycle (8-10 days), and virus incubation (4-10 days)
2. **Outbreak Dynamics:** Special focus on capturing outbreak patterns and peaks
3. **Time Series Approach:** Proper time series validation and specialized models
4. **Advanced Features:** Interaction terms, custom indices, and longer time lags
5. **Overdispersion Handling:** Models tailored for overdispersed count data (variance >> mean)
6. **Ensembling:** Combines multiple models to improve prediction robustness

## Feature Engineering

Enhanced feature engineering is implemented in `data_features_enhanced.py`, with key improvements including:

### Extended Temporal Features
- Better cyclic features for seasonality (sine/cosine transformations)
- Multiple temporal scales (day, week, month, quarter, year)
- City-specific wet/dry season indicators based on climate zones

### Extended Weather Features
- Temperature-humidity interaction terms (critical for mosquito breeding)
- "Mosquito comfort index" combining optimal temperature and humidity
- Extended lag features (1-4, 8, 10, 12 weeks) capturing full mosquito lifecycle and virus incubation
- Cumulative precipitation with multiple windows (standing water representation)
- Exponentially weighted moving averages with different decay factors

### Outbreak Dynamics Features
- Autoregressive components from case history
- Moving averages and momentum indicators
- Seasonal pattern detection and deviation
- Outbreak indicators using lagged differences

## Model Architecture

Multiple specialized models are implemented to capture different aspects of dengue dynamics:

### Time Series Models (`train_time_series.py`)
- Prophet for capturing seasonality and trend
- SARIMAX for autoregressive components
- City-specific parameter tuning

### Negative Binomial Models (`train_negative_binomial.py`)
- Specially designed for overdispersed count data
- Capture the skewed distribution of dengue cases
- Log-link function for proper scale

### Advanced Validation (`time_series_validation.py`)
- Time series split with gap to prevent leakage
- Expanding window validation
- Specialized outbreak evaluation metrics

### Stacking Ensemble (`train_ensemble.py`)
- Combines multiple models (XGBoost, LightGBM, RandomForest, GradientBoosting)
- Meta-model learns optimal combination
- Weighted by model performance on validation data

## Data Preprocessing

Enhanced preprocessing in `enhanced_preprocessing.py` includes:

- Seasonal imputation for missing values
- Outlier detection and treatment using Isolation Forest
- Stationarity testing and transformations
- Advanced data cleaning with time series awareness

## Evaluation Framework

A comprehensive evaluation framework in `evaluation_framework.py` focuses on:

- Metrics that penalize missing outbreak peaks
- Weighted RMSE giving higher importance to outbreak periods
- F1 score for outbreak detection
- SHAP values for model explainability
- Visualization of predictions vs. actual cases
- Comparative analysis across models

## Usage Guide

Follow these steps to run the enhanced model:

1. **Data Preprocessing:**
   ```
   python src/enhanced_preprocessing.py
   ```

2. **Feature Engineering:**
   ```
   python src/data_features_enhanced.py
   ```

3. **Model Training (Choose one or run all):**
   ```
   python src/train_time_series.py
   python src/train_negative_binomial.py
   python src/train_ensemble.py
   ```

4. **Evaluation:**
   ```
   python src/evaluation_framework.py
   ```

5. **Check Evaluation Results:**
   - Review metrics and plots in `data/processed/evaluation/`
   - Compare model performances in `data/processed/evaluation/model_comparison.csv`

## Results and Performance

The enhanced model significantly improves prediction accuracy, particularly for outbreak periods, compared to the original implementation:

- **Overall Performance:** Improved RMSE and MAE across both cities
- **Outbreak Detection:** Higher F1 scores for detecting outbreak periods
- **Competition Metric:** Lower mean absolute error relative to case counts
- **Biological Plausibility:** Feature importance aligns with epidemiological understanding

Key factors contributing to performance improvement:
1. Longer lag features capturing mosquito lifecycle and virus incubation
2. Specialized handling of overdispersed count data
3. Interaction terms between temperature and humidity
4. Better seasonal components and weather pattern representation
5. Ensemble approach combining complementary models

The model can be further improved with:
1. Fine-tuning hyperparameters for each city
2. Incorporating additional environmental data if available
3. Testing different ensemble techniques
4. Exploring deep learning approaches for complex pattern detection