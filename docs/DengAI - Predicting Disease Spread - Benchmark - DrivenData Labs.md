Sure! Here’s the complete summary in English, formatted as a clean and structured Markdown file:

# DengAI: Predicting Disease Spread – Benchmark Summary

**Author**: Peter Bull  
**Date**: December 23, 2016  
**Source**: [DrivenData Competition](https://www.drivendata.org/competitions/44/)

---

## 🦟 Goal of the Challenge

Predict weekly **dengue fever cases** based on **climate variables** in two cities:
- **San Juan, Puerto Rico**
- **Iquitos, Peru**

Dengue is a mosquito-borne disease, and its spread is influenced by weather patterns such as humidity and temperature.

---

## 🔍 Initial Data Exploration

### Splitting by City
```python
sj = df.loc['sj']
iq = df.loc['iq']

Missing Values

Forward-fill missing values due to time series gaps:

sj_train_features.fillna(method='ffill', inplace=True)



⸻

🎯 Target Variable: total_cases

Distribution Stats
    •   San Juan:
    •   Mean: 34.18
    •   Variance: 2640.05
    •   Iquitos:
    •   Mean: 7.57
    •   Variance: 115.90

🟡 Note: Variance >> Mean → Justifies use of Negative Binomial Regression

⸻

🧠 Feature Engineering

Correlation Insights

Only weak direct correlations to total_cases, but some patterns are evident:

Selected features:
    •   reanalysis_specific_humidity_g_per_kg
    •   reanalysis_dew_point_temp_k
    •   station_avg_temp_c
    •   station_min_temp_c

⸻

🧼 Preprocessing Function

def preprocess_data(data_path, labels_path=None):
    ...
    return sj, iq

Prepares both training and test datasets with selected features, filled missing values, and split by city.

⸻

🧪 Modeling

Model Formula

"total_cases ~ 1 + reanalysis_specific_humidity_g_per_kg + reanalysis_dew_point_temp_k + station_min_temp_c + station_avg_temp_c"

Model Type
    •   Negative Binomial Regression
    •   Hyperparameter tuning for alpha using Grid Search
    •   Evaluation Metric: Mean Absolute Error

⸻

⏳ Train/Test Splits

Time-based split to simulate forecasting:

San Juan:
    •   Training: first 800 weeks
    •   Testing: remaining 136 weeks

Iquitos:
    •   Training: first 400 weeks
    •   Testing: remaining 120 weeks

⸻

📉 Results
    •   Best Alpha: 1e-08
    •   Best Scores:
    •   San Juan: MAE ≈ 22.08
    •   Iquitos: MAE ≈ 6.47

Visual Analysis
    •   Captures seasonality, but
    •   Misses large outbreak spikes
    •   Potential fix: include lag features and autoregressive components

⸻

📤 Submission Code

submission.total_cases = np.concatenate([sj_predictions, iq_predictions])
submission.to_csv("../data-processed/benchmark.csv")



⸻

🧭 Takeaways

This is a baseline benchmark model, but:

💡 Opportunities for improvement:
    •   Include lag features to capture mosquito lifecycle and incubation time
    •   Model contagion dynamics across weeks
    •   Improve feature engineering with domain knowledge

⸻

🔗 Useful Links
    •   📄 Competition Page
    •   🧪 Benchmark GitHub Repo

---
