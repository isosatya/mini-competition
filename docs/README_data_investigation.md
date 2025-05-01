# Data Investigation Module

## Overview
The `data_investigation.py` module is the first step in the data processing pipeline. It focuses on understanding the raw data structure, identifying patterns, and suggesting cleaning strategies without modifying the data.

## Key Features
- Loads raw training and test data
- Analyzes data structure and basic statistics
- Identifies missing values and suggests handling strategies
- Provides visualizations for data understanding
- Prepares recommendations for data cleaning

## Usage
1. Run the module:
```bash
python src/data_investigation.py
```

2. The module will:
   - Load raw data from `data/raw/`
   - Analyze data structure for all datasets
   - Identify missing values
   - Provide cleaning recommendations
   - Show visualizations of data patterns

3. Next Steps:
   - Review the analysis output
   - Proceed with `data_clean.py` to implement cleaning strategies
   - Use `data_features.py` to add new features

## Output
The module provides:
- Basic statistics for each dataset
- Missing value analysis
- Data structure insights
- Visualizations of data patterns
- Recommendations for data cleaning

## Dependencies
- pandas
- numpy
- matplotlib
- seaborn

## Notes
- This module does not modify any data
- All analysis is performed on raw data
- Results are printed to console and displayed as plots
- Recommendations should be reviewed before proceeding with cleaning 