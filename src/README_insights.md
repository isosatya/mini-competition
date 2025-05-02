# XGBoost Timeframes Analysis Tools

This directory contains scripts to generate advanced insights and visualizations from the XGBoost timeframes model results.

## Available Scripts

### 1. `analyze_xgboost_results.py`

This script analyzes the results from the XGBoost timeframes model, focusing on:
- Feature importance comparison between San Juan and Iquitos
- Error analysis across time periods
- Climate pattern visualization
- Performance metrics summarization

**Usage:**
```
python src/insights/analyze_xgboost_results.py
```

### 2. `visualize_model_results.py`

This script creates comprehensive visualizations from the XGBoost timeframes model results, focusing on:
- Prediction accuracy across cities
- Feature importance comparison
- Submission analysis
- City comparison

**Usage:**
```
python src/insights/visualize_model_results.py
```

### 3. `xgboost_insights_dashboard.py`

This script creates an interactive Streamlit dashboard for advanced analysis of the XGBoost timeframes model results.

**Usage:**
```
streamlit run src/insights/xgboost_insights_dashboard.py
```

**Note:** This requires Streamlit to be installed (`pip install streamlit`).

## Key Insights

The analysis tools generate the following key insights:

1. **Model Performance Analysis**:
   - Compares RMSE, MAE, R², and outbreak detection metrics for both cities
   - Analyzes precision and recall for outbreak detection
   - Identifies systematic overestimation in both models

2. **Feature Importance Analysis**:
   - Visualizes top predictive features for each city
   - Compares feature importance across cities
   - Analyzes climate variable importance by category

3. **Time Period Analysis**:
   - Compares prediction accuracy across testing periods
   - Analyzes seasonal pattern differences
   - Identifies concept drift between training and testing periods

4. **Climate Pattern Analysis**:
   - Analyzes lag effects of climate variables on dengue cases
   - Compares climate variable distributions during outbreak vs. non-outbreak periods
   - Identifies optimal lag periods for early warning systems

5. **Improvement Recommendations**:
   - Provides data handling improvements
   - Suggests feature engineering enhancements
   - Recommends model architecture refinements
   - Offers training approach optimizations

## Output Files

All output files are saved to the `results/insights/` directory. Key files include:

- **Performance Metrics**: `model_performance_summary.csv`, `performance_dashboard.png`
- **Feature Importance**: `feature_importance_comparison.png`, `category_importance_comparison.png`
- **Climate Analysis**: `climate_lag_importance.png`
- **Summary Reports**: `XGBOOST_TIMEFRAMES_INSIGHTS.md`, `xgboost_results_summary.html`

## Implementation Plan

Based on the insights, an implementation plan is provided with:

1. **Short-term fixes** (1-2 weeks) focusing on prediction stability
2. **Medium-term improvements** (2-4 weeks) enhancing features and models
3. **Long-term strategy** (1-3 months) implementing comprehensive solutions