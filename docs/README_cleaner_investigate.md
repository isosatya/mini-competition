# Data Cleaner Investigation Module

## Overview
The `data_cleaner_investigation.py` module analyzes the cleaned data to verify the success of the cleaning process and identify any remaining issues. It provides detailed insights into the cleaned datasets before proceeding with feature engineering.

## Key Features
- Loads cleaned training and test data
- Analyzes remaining missing values
- Examines temporal feature distributions
- Investigates categorical and numerical feature distributions
- Provides visualizations of feature distributions
- Identifies potential issues in the cleaned data

## Usage
1. Run the module:
```bash
python src/data_cleaner_investigation.py
```

2. The module will:
   - Load cleaned data from `data/processed/`
   - Analyze cleaning results for both datasets
   - Show distributions of all features
   - Provide recommendations for next steps

3. Next Steps:
   - Review the analysis output
   - If satisfied, proceed with `data_features.py`
   - If issues are found, adjust `data_cleaner.py`

## Analysis Components
1. Cleaning Results:
   - Missing value analysis
   - Temporal feature verification
   - Categorical feature analysis
   - Numerical feature statistics

2. Feature Distributions:
   - Histograms for numerical features
   - Bar charts for categorical features
   - Temporal feature patterns

## Output
The module provides:
- Detailed statistics for each feature
- Visualizations of feature distributions
- Identification of remaining issues
- Recommendations for improvements

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn

## Notes
- Run after `data_cleaner.py`
- Review all visualizations carefully
- Check for unexpected patterns
- Ensure data quality before feature engineering 