# Feature Engineering Module

## Overview
The `data_features.py` module enhances the cleaned data by adding new features that can improve model performance. It creates derived features from existing data and saves the enhanced datasets for model training.

## Key Features
- Adds temporal features (day of year, quarter, month start/end)
- Creates cyclic week features (sine and cosine transformations)
- Creates weather statistics (mean, std, range)
- Calculates NDVI features
- Generates lag features for target variable
- Saves enhanced datasets

## Usage
1. Run the module:
```bash
python src/data_features.py
```

2. The module will:
   - Load cleaned data from `data/processed/`
   - Add temporal features
   - Create weather statistics
   - Calculate NDVI features
   - Generate lag features
   - Save enhanced data to `data/processed/`

3. Next Steps:
   - Review featured data in `data/processed/`
   - Use enhanced data for model training
   - Consider feature importance in model selection

## Output Files
- `featured_train_data.csv`: Training data with new features
- `featured_test_features.csv`: Test features with new features

## Feature Types
1. Temporal Features:
   - Day of year
   - Quarter
   - Month start/end indicators
   - Cyclic week features (sine and cosine)

2. Weather Features:
   - Temperature statistics
   - Precipitation totals
   - Days with precipitation
   - Humidity means

3. NDVI Features:
   - Mean NDVI
   - NDVI standard deviation
   - NDVI range

4. Lag Features:
   - Previous weeks' case counts (1-4 weeks)

## Dependencies
- pandas
- numpy
- pathlib

## Notes
- Run `data_clean.py` first
- Review feature distributions
- Consider feature importance
- Check for feature correlations
- Ensure test data has all required features
- Cyclic features help capture seasonal patterns 