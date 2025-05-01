# Data Processing Pipeline

This document describes the data processing pipeline for the Dengue Fever prediction model.

## Overview

The data processing pipeline consists of several steps that are executed sequentially:

1. Data Investigation (`data_investigation.py`)
2. Data Cleaning (`data_cleaner.py`)
3. Feature Combination (`data_cleaner_combinations.py`)
4. Feature Engineering (`data_features.py`)
5. Cleaned Data Analysis (`data_cleaner_investigation.py`)

## Step 1: Data Investigation

**File:** `src/data_investigation.py`

This step analyzes the raw data and identifies:
- Data structure and types
- Missing values
- Outliers
- Feature correlations
- Feature distributions

**Outputs:**
- Missing value statistics
- Correlation matrices
- Distribution plots
- Data cleaning recommendations

## Step 2: Data Cleaning

**File:** `src/data_cleaner.py`

Based on the investigation results, the following cleaning steps are performed:
- Missing value handling
- Outlier removal
- Data type conversion
- Feature normalization
- Creation of `weekofyear` from date

**Outputs:**
- `cleaned_train_data.csv`
- `cleaned_test_features.csv`

## Step 3: Feature Combination

**File:** `src/data_cleaner_combinations.py`

Combines related features to reduce redundancy:
- Temperature features:
  - Reanalysis temperatures (K)
  - Station temperatures (C)
- Date format conversion
- Calculation of means and ranges

**Outputs:**
- `combined_train_data.csv`
- `combined_test_features.csv`

## Step 4: Feature Engineering

**File:** `src/data_features.py`

Creates new features for the model:
- Temporal features:
  - `dayofyear`
  - `quarter`
  - `is_month_start`
  - `is_month_end`
- Weather features:
  - Average temperature
  - Temperature range
  - Precipitation days
  - Humidity
- NDVI features:
  - Mean
  - Standard deviation
  - Range
- Lag features for `total_cases`

**Outputs:**
- `featured_train_data.csv`
- `featured_test_features.csv`

## Step 5: Cleaned Data Analysis

**File:** `src/data_cleaner_investigation.py`

Verifies the quality of the cleaned data:
- Feature distributions
- Correlations
- Missing values
- Outliers
- Data quality

**Outputs:**
- Distribution plots in `data/processed/png/`
- Statistics and metrics

## Execution Order

1. Run `data_investigation.py` first
2. Based on the results, run `data_cleaner.py`
3. Run `data_cleaner_combinations.py`
4. Run `data_features.py`
5. Finally, run `data_cleaner_investigation.py`

## Directory Structure

```
data/
├── raw/                    # Raw data
├── processed/              # Processed data
│   ├── cleaned_*.csv      # Cleaned data
│   ├── combined_*.csv     # Combined features
│   ├── featured_*.csv     # Feature engineering
│   └── png/               # Analysis plots
└── submission/            # Predictions
```

## Next Steps

After completing the data processing:
1. Train the model with the processed data
2. Evaluate model performance
3. Optimize features based on model performance 