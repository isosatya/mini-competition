# Data Cleaning Module

## Overview
The `data_clean.py` module implements the cleaning strategies suggested by the data investigation module. It handles missing values, standardizes temporal features, and prepares the data for feature engineering.

## Key Features
- Creates processed data directory structure
- Handles missing values based on feature type
- Cleans and standardizes temporal features
- Merges training features and labels
- Saves cleaned data for further processing

## Usage
1. Run the module:
```bash
python src/data_clean.py
```

2. The module will:
   - Create necessary directories
   - Load raw data from `data/raw/`
   - Clean temporal features
   - Handle missing values
   - Merge training data
   - Save cleaned data to `data/processed/`

3. Next Steps:
   - Review cleaned data in `data/processed/`
   - Proceed with `data_features.py` to add new features
   - Use cleaned data for model training

## Output Files
- `cleaned_train_data.csv`: Merged and cleaned training data
- `cleaned_test_features.csv`: Cleaned test features

## Cleaning Strategies
- Missing Values:
  - Temperature: Linear interpolation
  - Precipitation: Zero imputation
  - NDVI: Seasonal mean imputation
  - Other: Mean imputation
- Temporal Features:
  - Standardized date formats
  - Extracted year, month, week features

## Dependencies
- pandas
- numpy
- pathlib

## Notes
- Always run `data_investigation.py` first
- Review cleaning strategies before running
- Check output files for completeness
- Ensure all required columns are preserved 