# Dengue Fever Prediction

This project aims to predict dengue fever cases using machine learning techniques.

## Project Structure

```
.
├── data/
│   ├── raw/           # Original data files
│   └── processed/     # Processed and cleaned data
├── src/
│   ├── investigate_data.py    # Data exploration and analysis
│   ├── train_xgboost.py       # XGBoost model training and prediction
│   └── utils.py               # Utility functions
├── notebooks/         # Jupyter notebooks for analysis
├── environment.yml    # Conda environment specification
└── README.md         # Project documentation
```

## Setup

1. Create and activate the conda environment:
```bash
conda env create -f environment.yml
conda activate mini-competition
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

1. Run data investigation:
```bash
python src/investigate_data.py
```

2. Train and predict with XGBoost:
```bash
python src/train_xgboost.py
```

## Data Processing

The data processing pipeline includes:
- Loading and merging training and test data
- Handling missing values
- Feature engineering
- Temporal and weather feature analysis
- Correlation analysis
- Time series visualization

## Model Training

The XGBoost model is trained with the following features:
- Temporal features (year, week of year)
- Weather features (temperature, precipitation, etc.)
- City-specific features

The model uses time series cross-validation for evaluation and includes feature importance analysis.