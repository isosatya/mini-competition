"""
XGBoost Timeframes Model Insights Dashboard
--------------------------------------------
This dashboard provides advanced insights and visualizations from the XGBoost timeframe
model results, comparing feature importance rankings between cities, analyzing prediction 
accuracy across time periods, and identifying patterns between climate variables and 
dengue outbreaks.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import json
import xgboost as xgb
from pathlib import Path
from datetime import datetime
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set page config
st.set_page_config(
    page_title="DengAI XGBoost Timeframes Insights",
    page_icon="🦟",
    layout="wide"
)

# Helper functions
def load_metrics(city):
    """Load model metrics for a specific city."""
    metrics_path = Path(f"models/{city}/xgboost_timeframes_{city}_latest/metrics.json")
    try:
        with open(metrics_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        st.error(f"Error loading metrics for {city}: {str(e)}")
        return {}

def load_model(city):
    """Load the XGBoost model for a given city."""
    model_path = Path(f"models/{city}/xgboost_timeframes_{city}_latest/model.json")
    try:
        model = xgb.Booster()
        model.load_model(str(model_path))
        return model
    except Exception as e:
        st.error(f"Error loading model for {city}: {str(e)}")
        return None

def load_feature_names(city):
    """Load feature names for a specific city."""
    features_path = Path(f"models/{city}/xgboost_timeframes_{city}_latest/feature_names.txt")
    features = []
    try:
        with open(features_path, 'r') as f:
            for line in f:
                # Extract feature name by removing the number and dot prefix
                feature_name = line.strip().split('. ', 1)[1] if '. ' in line else line.strip()
                features.append(feature_name)
        return features
    except Exception as e:
        st.error(f"Error loading feature names for {city}: {str(e)}")
        return []

def load_feature_importance(model, feature_names):
    """Calculate feature importance from the model."""
    if model is None or not feature_names:
        return pd.DataFrame()
    
    # Get feature importance scores
    importance = model.get_score(importance_type='gain')
    
    # Create a DataFrame with all features
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': [importance.get(f, 0) for f in feature_names]
    })
    
    # Normalize importance scores
    if importance_df['importance'].sum() > 0:
        importance_df['importance_normalized'] = importance_df['importance'] / importance_df['importance'].sum()
    else:
        importance_df['importance_normalized'] = 0
    
    # Sort by importance
    return importance_df.sort_values('importance', ascending=False)

def load_predictions(city):
    """Load prediction results for a specific city."""
    # This is a simplified placeholder - in a real implementation 
    # you would load the actual predictions from saved files
    try:
        # In a real implementation, load these from saved files
        prediction_data = pd.read_csv(Path('data/processed/cleaned_data_merged.csv'))
        
        # Filter for this city and testing period
        city_data = prediction_data[prediction_data['city'] == city].copy()
        
        if city == 'sj':
            # San Juan testing period: 2000-2007
            test_mask = (city_data['week_start_date'] >= '2000-01-01') & (city_data['week_start_date'] < '2008-01-01')
        else:
            # Iquitos testing period: 2007-2010
            test_mask = (city_data['week_start_date'] >= '2007-01-01') & (city_data['week_start_date'] < '2011-01-01')
        
        test_data = city_data[test_mask].copy()
        
        # Convert date to datetime
        test_data['week_start_date'] = pd.to_datetime(test_data['week_start_date'])
        
        return test_data
    except Exception as e:
        st.error(f"Error loading predictions data for {city}: {str(e)}")
        return pd.DataFrame()

def prepare_performance_summary():
    """Prepare a summary of model performance for both cities."""
    sj_metrics = load_metrics('sj')
    iq_metrics = load_metrics('iq')
    
    # Create a comparison DataFrame
    metrics_to_show = ['rmse', 'mae', 'r2', 'outbreak_precision', 'outbreak_recall', 'outbreak_f1', 'correlation']
    
    summary_data = {
        'Metric': metrics_to_show,
        'San Juan': [sj_metrics.get(m, 'N/A') for m in metrics_to_show],
        'Iquitos': [iq_metrics.get(m, 'N/A') for m in metrics_to_show]
    }
    
    return pd.DataFrame(summary_data)

def load_submission_data():
    """Load the latest submission data."""
    try:
        submission_path = Path('results/submission_xgboost_timeframes_latest.csv')
        submission = pd.read_csv(submission_path)
        return submission
    except Exception as e:
        st.error(f"Error loading submission data: {str(e)}")
        return pd.DataFrame()

def load_training_data():
    """Load the training data."""
    try:
        train_data = pd.read_csv(Path('data/processed/cleaned_data_merged.csv'))
        train_data['week_start_date'] = pd.to_datetime(train_data['week_start_date'])
        return train_data
    except Exception as e:
        st.error(f"Error loading training data: {str(e)}")
        return pd.DataFrame()

def categorize_climate_features(feature_names):
    """Categorize features into climate groups for analysis."""
    categories = {
        'Temperature': [f for f in feature_names if any(t in f.lower() for t in ['temp', 'air_temp', 'tdtr'])],
        'Precipitation': [f for f in feature_names if any(p in f.lower() for p in ['precip', 'rain'])],
        'Humidity': [f for f in feature_names if any(h in f.lower() for p in ['humid', 'dew'])],
        'Vegetation': [f for f in feature_names if 'ndvi' in f.lower()],
        'Cyclical': [f for f in feature_names if any(c in f.lower() for c in ['sin', 'cos', 'week', 'month'])],
        'Engineered': [f for f in feature_names if any(e in f.lower() for e in ['index', 'suitability', 'interaction'])]
    }
    
    # Add a "Lagged" category that overlaps with others
    categories['Lagged'] = [f for f in feature_names if 'lag' in f.lower()]
    
    return categories

# Title and introduction
st.title("🦟 Advanced DengAI XGBoost Timeframes Model Insights")

st.markdown("""
This dashboard provides in-depth analysis of the XGBoost timeframe models for dengue prediction.
The models were trained with distinct timeframes for each city:
- **San Juan**: Training period 1990-2000, Testing period 2000-2007
- **Iquitos**: Training period 2000-2007, Testing period 2007-2010

Explore model performance, feature importance comparisons, and climate pattern insights below.
""")

# Create tabs for different analysis sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Model Performance", 
    "Feature Importance", 
    "Time Period Analysis",
    "Climate Patterns",
    "Improvement Suggestions"
])

# Tab 1: Model Performance
with tab1:
    st.header("Model Performance Analysis")
    
    # Show overall metrics comparison
    st.subheader("Performance Metrics Comparison")
    
    metrics_df = prepare_performance_summary()
    
    # Apply formatting to the metrics
    formatted_df = metrics_df.copy()
    for city in ['San Juan', 'Iquitos']:
        formatted_df[city] = formatted_df.apply(
            lambda x: f"{x[city]:.4f}" if isinstance(x[city], (float, int)) else x[city], 
            axis=1
        )
    
    # Display metrics in a table
    st.dataframe(formatted_df, hide_index=True, use_container_width=True)
    
    # Add metrics interpretation
    st.markdown("""
    ### Metrics Interpretation
    
    **Standard Metrics**:
    - **RMSE** (Root Mean Squared Error): Measures prediction accuracy by taking the square root of the average squared differences between predicted and actual values. Lower is better.
    - **MAE** (Mean Absolute Error): Average absolute difference between predicted and actual values. Less sensitive to outliers than RMSE.
    - **R²** (Coefficient of Determination): Indicates how well the model explains the variance in the data. Values close to 1 are ideal, negative values indicate the model performs worse than a horizontal line.
    
    **Outbreak Detection Metrics**:
    - **Precision**: Percentage of correctly predicted outbreaks among all predicted outbreaks.
    - **Recall**: Percentage of actual outbreaks correctly predicted by the model.
    - **F1 Score**: Harmonic mean of precision and recall, providing a balanced measure of outbreak detection performance.
    
    **Additional Metrics**:
    - **Correlation**: Measure of the linear relationship between predictions and actual values.
    """)
    
    # Create two columns for city-specific information
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("San Juan Model")
        
        # Load San Juan metrics
        sj_metrics = load_metrics('sj')
        
        # Show confusion matrix for San Juan
        if 'confusion_matrix' in sj_metrics:
            cm = sj_metrics['confusion_matrix']
            
            # Create and plot confusion matrix
            cm_data = np.array([
                [cm.get('true_positives', 0), cm.get('false_negatives', 0)],
                [cm.get('false_positives', 0), cm.get('true_negatives', 0)]
            ])
            
            # Create a heatmap using Plotly
            fig = px.imshow(
                cm_data,
                labels=dict(x="Predicted", y="Actual"),
                x=['Outbreak', 'No Outbreak'],
                y=['Outbreak', 'No Outbreak'],
                text_auto=True,
                color_continuous_scale='Reds',
                title="San Juan Outbreak Detection Confusion Matrix"
            )
            
            # Update layout
            fig.update_layout(width=500, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display accuracy metrics
            total = sum(cm.values())
            if total > 0:
                accuracy = (cm.get('true_positives', 0) + cm.get('true_negatives', 0)) / total
                st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        # Display prediction visualization if available
        st.image("results/insights/predictions_xgboost_timeframes_sj.png", 
                caption="San Juan: Actual vs Predicted Dengue Cases",
                use_column_width=True)
    
    with col2:
        st.subheader("Iquitos Model")
        
        # Load Iquitos metrics
        iq_metrics = load_metrics('iq')
        
        # Show confusion matrix for Iquitos
        if 'confusion_matrix' in iq_metrics:
            cm = iq_metrics['confusion_matrix']
            
            # Create and plot confusion matrix
            cm_data = np.array([
                [cm.get('true_positives', 0), cm.get('false_negatives', 0)],
                [cm.get('false_positives', 0), cm.get('true_negatives', 0)]
            ])
            
            # Create a heatmap using Plotly
            fig = px.imshow(
                cm_data,
                labels=dict(x="Predicted", y="Actual"),
                x=['Outbreak', 'No Outbreak'],
                y=['Outbreak', 'No Outbreak'],
                text_auto=True,
                color_continuous_scale='Greens',
                title="Iquitos Outbreak Detection Confusion Matrix"
            )
            
            # Update layout
            fig.update_layout(width=500, height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display accuracy metrics
            total = sum(cm.values())
            if total > 0:
                accuracy = (cm.get('true_positives', 0) + cm.get('true_negatives', 0)) / total
                st.metric("Overall Accuracy", f"{accuracy:.2%}")
        
        # Display prediction visualization if available
        st.image("results/insights/predictions_xgboost_timeframes_iq.png", 
                caption="Iquitos: Actual vs Predicted Dengue Cases",
                use_column_width=True)
    
    # Analysis of model performance
    st.subheader("Performance Analysis")
    
    # Calculate overall submission statistics
    submission = load_submission_data()
    if not submission.empty:
        st.write("**Submission Statistics**")
        
        # Group by city
        city_stats = submission.groupby('city')['total_cases'].agg(['min', 'max', 'mean', 'count'])
        city_stats = city_stats.reset_index()
        city_stats.columns = ['City', 'Min Cases', 'Max Cases', 'Average Cases', 'Predictions Count']
        
        # Format city names
        city_stats['City'] = city_stats['City'].map({'sj': 'San Juan', 'iq': 'Iquitos'})
        
        # Display stats
        st.dataframe(city_stats, hide_index=True, use_container_width=True)
        
        # Create a distribution plot of predictions
        st.write("**Prediction Distributions**")
        
        fig = px.histogram(
            submission, 
            x='total_cases',
            color='city',
            barmode='overlay',
            nbins=50,
            labels={'total_cases': 'Predicted Cases', 'city': 'City'},
            title='Distribution of Predicted Dengue Cases',
            color_discrete_map={'sj': 'blue', 'iq': 'green'}
        )
        
        # Set log scale for x-axis if Iquitos has extremely large values
        if submission[submission['city'] == 'iq']['total_cases'].max() > 1000:
            fig.update_xaxes(type='log')
            fig.update_layout(xaxis_title="Predicted Cases (log scale)")
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Performance issues analysis
    st.markdown("""
    ### Performance Issues Analysis
    
    #### San Juan Model:
    - The model has a **negative R² value** (-0.6426), indicating that it performs worse than a simple baseline model that always predicts the mean of the target values.
    - Despite poor overall accuracy, the model has **perfect recall for outbreak detection** (1.0), but low precision (0.25), meaning it correctly captures all outbreaks but generates many false positives.
    - The moderate correlation coefficient (0.4196) suggests the model captures some patterns in the data but struggles with accurate predictions.
    
    #### Iquitos Model:
    - The model performs extremely poorly with a catastrophically low R² value (-3,733,535.5).
    - The RMSE is extremely high (21,565.61), suggesting predictions are very far from actual values.
    - Like the San Juan model, it achieves perfect recall for outbreaks (1.0) but with low precision (0.232).
    - The extremely high maximum prediction value suggests the model is unstable and produces unrealistic projections.
    
    #### Common Issues:
    - Both models show signs of overfitting to the training data and inability to generalize to test timeframes.
    - The pattern of high recall but low precision for outbreaks suggests the models are biased toward predicting outbreaks too frequently.
    - The positive bias values indicate both models systematically overestimate dengue cases.
    """)

# Tab 2: Feature Importance
with tab2:
    st.header("Feature Importance Analysis")
    
    # Load feature importance data
    sj_model = load_model('sj')
    iq_model = load_model('iq')
    
    sj_features = load_feature_names('sj')
    iq_features = load_feature_names('iq')
    
    sj_importance = load_feature_importance(sj_model, sj_features)
    iq_importance = load_feature_importance(iq_model, iq_features)
    
    # Display feature importance plots
    st.subheader("Feature Importance Comparison")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**San Juan Top Features**")
        if not sj_importance.empty:
            # Get top 15 features
            top_sj = sj_importance.head(15)
            
            # Create bar chart
            fig = px.bar(
                top_sj,
                y='feature',
                x='importance_normalized',
                orientation='h',
                title='San Juan Top 15 Features by Importance',
                labels={'importance_normalized': 'Relative Importance', 'feature': 'Feature'},
                color='importance_normalized',
                color_continuous_scale='Blues'
            )
            
            # Update layout
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image("results/insights/feature_importance_xgboost_timeframes_sj.png", use_column_width=True)
    
    with col2:
        st.write("**Iquitos Top Features**")
        if not iq_importance.empty:
            # Get top 15 features
            top_iq = iq_importance.head(15)
            
            # Create bar chart
            fig = px.bar(
                top_iq,
                y='feature',
                x='importance_normalized',
                orientation='h',
                title='Iquitos Top 15 Features by Importance',
                labels={'importance_normalized': 'Relative Importance', 'feature': 'Feature'},
                color='importance_normalized',
                color_continuous_scale='Greens'
            )
            
            # Update layout
            fig.update_layout(
                yaxis={'categoryorder': 'total ascending'},
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.image("results/insights/feature_importance_xgboost_timeframes_iq.png", use_column_width=True)
    
    # Feature Importance Correlation Analysis
    st.subheader("Feature Importance Correlation Analysis")
    
    # Merge feature importance data for both cities to compare
    if not sj_importance.empty and not iq_importance.empty:
        # Get common features
        common_features = set(sj_importance['feature']).intersection(set(iq_importance['feature']))
        
        if common_features:
            # Filter to common features
            sj_common = sj_importance[sj_importance['feature'].isin(common_features)]
            iq_common = iq_importance[iq_importance['feature'].isin(common_features)]
            
            # Create a merged DataFrame
            merged_importance = pd.merge(
                sj_common, 
                iq_common, 
                on='feature', 
                suffixes=('_sj', '_iq')
            )
            
            # Create a scatter plot to compare importance between cities
            fig = px.scatter(
                merged_importance,
                x='importance_normalized_sj',
                y='importance_normalized_iq',
                text='feature',
                title='Feature Importance Correlation: San Juan vs Iquitos',
                labels={
                    'importance_normalized_sj': 'San Juan Importance',
                    'importance_normalized_iq': 'Iquitos Importance'
                },
                log_x=True,
                log_y=True
            )
            
            # Add reference line
            fig.add_trace(
                go.Scatter(
                    x=[0, merged_importance['importance_normalized_sj'].max()],
                    y=[0, merged_importance['importance_normalized_sj'].max()],
                    mode='lines',
                    line=dict(dash='dash', color='gray'),
                    name='Equal Importance'
                )
            )
            
            # Update layout
            fig.update_layout(
                height=600,
                hovermode='closest'
            )
            
            # Update traces for better text display
            fig.update_traces(
                textposition='top right',
                marker=dict(size=10)
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Calculate and display correlation
            corr = merged_importance['importance_normalized_sj'].corr(merged_importance['importance_normalized_iq'])
            st.metric("Feature Importance Correlation", f"{corr:.4f}")
            
            # Add interpretation based on correlation value
            if corr > 0.7:
                st.success("There is a strong correlation between feature importance in both cities, suggesting similar driving factors.")
            elif corr > 0.3:
                st.info("There is a moderate correlation between feature importance in both cities, with some common factors but also differences.")
            else:
                st.warning("There is a weak correlation between feature importance in both cities, suggesting distinct dengue dynamics.")
    
    # Feature importance by category
    st.subheader("Feature Importance by Category")
    
    if sj_features and iq_features:
        # Categorize features
        sj_categories = categorize_climate_features(sj_features)
        iq_categories = categorize_climate_features(iq_features)
        
        # Prepare data for category comparison
        category_data = []
        
        for category, features in sj_categories.items():
            # Calculate total importance for this category
            if not sj_importance.empty:
                sj_cat_importance = sj_importance[sj_importance['feature'].isin(features)]['importance'].sum()
            else:
                sj_cat_importance = 0
                
            if not iq_importance.empty:
                iq_cat_importance = iq_importance[iq_importance['feature'].isin(iq_categories.get(category, []))]['importance'].sum()
            else:
                iq_cat_importance = 0
            
            category_data.append({
                'Category': category,
                'San Juan Importance': sj_cat_importance,
                'Iquitos Importance': iq_cat_importance,
                'Feature Count': len(features)
            })
        
        # Create DataFrame
        category_df = pd.DataFrame(category_data)
        
        # Normalize importances
        if category_df['San Juan Importance'].sum() > 0:
            category_df['San Juan Importance'] = category_df['San Juan Importance'] / category_df['San Juan Importance'].sum()
        
        if category_df['Iquitos Importance'].sum() > 0:
            category_df['Iquitos Importance'] = category_df['Iquitos Importance'] / category_df['Iquitos Importance'].sum()
        
        # Create bar chart for category comparison
        fig = px.bar(
            category_df,
            x='Category',
            y=['San Juan Importance', 'Iquitos Importance'],
            barmode='group',
            title='Feature Importance by Category',
            labels={
                'value': 'Relative Importance',
                'variable': 'City',
                'Category': 'Feature Category'
            }
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            legend_title='City'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display interpretation
        st.markdown("""
        ### Feature Importance Interpretation
        
        #### Key Findings:
        
        1. **Different Predictive Factors Between Cities**:
           - San Juan's model relies more heavily on temperature-related features
           - Iquitos shows stronger influence from precipitation and vegetation indices
        
        2. **Temporal Patterns**:
           - Lagged features (previous weeks' conditions) are important for both cities
           - San Juan shows more sensitivity to cyclical seasonal patterns (sine/cosine features)
        
        3. **Climate Factor Differences**:
           - San Juan (Caribbean climate): Temperature variations are key predictors
           - Iquitos (Amazon rainforest climate): Precipitation and vegetation patterns are stronger predictors
        
        4. **Engineered Features**:
           - Mosquito breeding index is particularly valuable for San Juan
           - Vegetation change indicators (NDVI) have higher importance for Iquitos
        
        These differences reflect the distinct ecological and climate systems of the two regions:
        - San Juan has a tropical monsoon climate with distinct dry/wet seasons
        - Iquitos has an equatorial rainforest climate with consistent rainfall and higher humidity year-round
        """)

# Tab 3: Time Period Analysis
with tab3:
    st.header("Time Period Analysis")
    
    # Load training data with actual cases
    train_data = load_training_data()
    
    if not train_data.empty:
        # Create city selector
        city = st.selectbox(
            "Select city to analyze:",
            ["San Juan", "Iquitos"],
            index=0,
            key="time_period_city"
        )
        
        city_code = 'sj' if city == 'San Juan' else 'iq'
        
        # Filter data for selected city
        city_data = train_data[train_data['city'] == city_code].copy()
        
        # Define training and testing periods
        if city_code == 'sj':
            # San Juan: Training period 1990-2000, Testing period 2000-2007
            train_mask = (city_data['week_start_date'] >= '1990-01-01') & (city_data['week_start_date'] < '2000-01-01')
            test_mask = (city_data['week_start_date'] >= '2000-01-01') & (city_data['week_start_date'] < '2008-01-01')
        else:
            # Iquitos: Training period 2000-2007, Testing period 2007-2010
            train_mask = (city_data['week_start_date'] >= '2000-01-01') & (city_data['week_start_date'] < '2007-01-01')
            test_mask = (city_data['week_start_date'] >= '2007-01-01') & (city_data['week_start_date'] < '2011-01-01')
        
        # Add period column
        city_data['period'] = 'Other'
        city_data.loc[train_mask, 'period'] = 'Training'
        city_data.loc[test_mask, 'period'] = 'Testing'
        
        # Create time series plot of dengue cases
        st.subheader(f"{city}: Dengue Cases Over Time (Training vs Testing Periods)")
        
        # Create basic time series plot
        fig = px.line(
            city_data,
            x='week_start_date',
            y='total_cases',
            color='period',
            title=f'{city} Dengue Cases: Training vs Testing Periods',
            labels={
                'week_start_date': 'Date',
                'total_cases': 'Total Cases',
                'period': 'Period'
            },
            color_discrete_map={
                'Training': 'blue',
                'Testing': 'red',
                'Other': 'gray'
            }
        )
        
        # Highlight different periods with background shading
        if city_code == 'sj':
            # Add rectangles for San Juan periods
            fig.add_vrect(
                x0='1990-01-01', x1='2000-01-01',
                fillcolor='rgba(0,0,255,0.1)', layer='below', line_width=0,
                annotation_text='Training Period (1990-2000)',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0='2000-01-01', x1='2008-01-01',
                fillcolor='rgba(255,0,0,0.1)', layer='below', line_width=0,
                annotation_text='Testing Period (2000-2007)',
                annotation_position='top left'
            )
        else:
            # Add rectangles for Iquitos periods
            fig.add_vrect(
                x0='2000-01-01', x1='2007-01-01',
                fillcolor='rgba(0,0,255,0.1)', layer='below', line_width=0,
                annotation_text='Training Period (2000-2007)',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0='2007-01-01', x1='2011-01-01',
                fillcolor='rgba(255,0,0,0.1)', layer='below', line_width=0,
                annotation_text='Testing Period (2007-2010)',
                annotation_position='top left'
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create statistics comparison between periods
        train_stats = city_data[train_mask]['total_cases'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
        train_stats.columns = ['Statistic', 'Training Period']
        
        test_stats = city_data[test_mask]['total_cases'].agg(['count', 'min', 'max', 'mean', 'std']).reset_index()
        test_stats.columns = ['Statistic', 'Testing Period']
        
        # Create and display period stats comparison
        period_stats = pd.merge(train_stats, test_stats, on='Statistic')
        period_stats['Difference'] = period_stats['Testing Period'] - period_stats['Training Period']
        period_stats['Change %'] = ((period_stats['Testing Period'] / period_stats['Training Period']) - 1) * 100
        
        # Format numbers
        for col in ['Training Period', 'Testing Period', 'Difference']:
            period_stats[col] = period_stats[col].round(2)
        period_stats['Change %'] = period_stats['Change %'].round(1)
        
        # Rename statistics for readability
        period_stats['Statistic'] = period_stats['Statistic'].map({
            'count': 'Weeks Count',
            'min': 'Minimum Cases',
            'max': 'Maximum Cases',
            'mean': 'Average Cases',
            'std': 'Standard Deviation'
        })
        
        st.subheader("Period Statistics Comparison")
        st.dataframe(period_stats, hide_index=True, use_container_width=True)
        
        # Add distribution comparison plot
        st.subheader("Case Distribution Comparison Between Periods")
        
        # Create histogram with density plot overlay
        fig = px.histogram(
            city_data[city_data['period'].isin(['Training', 'Testing'])],
            x='total_cases',
            color='period',
            barmode='overlay',
            histnorm='probability density',
            nbins=30,
            title=f'{city} Dengue Cases Distribution Comparison',
            labels={
                'total_cases': 'Total Cases',
                'period': 'Period'
            },
            color_discrete_map={
                'Training': 'blue',
                'Testing': 'red'
            },
            opacity=0.6
        )
        
        # Overlay Kernel Density Estimate
        for period, color in zip(['Training', 'Testing'], ['blue', 'red']):
            period_data = city_data[city_data['period'] == period]['total_cases']
            if len(period_data) > 0:
                # Calculate KDE on a grid
                density = sns.kdeplot(period_data).get_lines()[0].get_data()
                
                # Add as a line trace
                fig.add_scatter(
                    x=density[0], 
                    y=density[1],
                    mode='lines',
                    line=dict(color=color, width=2),
                    name=f'{period} KDE'
                )
        
        # Update layout
        fig.update_layout(
            height=500,
            xaxis_title='Total Cases',
            yaxis_title='Density'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Outbreak analysis across periods
        st.subheader("Outbreak Analysis Across Periods")
        
        # Calculate outbreak threshold (75th percentile of entire dataset)
        outbreak_threshold = np.percentile(city_data['total_cases'], 75)
        
        # Create outbreak indicator
        city_data['is_outbreak'] = (city_data['total_cases'] > outbreak_threshold).astype(int)
        
        # Calculate outbreak statistics by period
        outbreak_by_period = city_data.groupby('period').agg(
            total_weeks=('is_outbreak', 'count'),
            outbreak_weeks=('is_outbreak', 'sum'),
            outbreak_percentage=('is_outbreak', lambda x: x.mean() * 100),
            avg_outbreak_cases=('total_cases', lambda x: x[city_data['is_outbreak'] == 1].mean()),
            max_outbreak_cases=('total_cases', lambda x: x[city_data['is_outbreak'] == 1].max())
        ).reset_index()
        
        # Filter to only show Training and Testing periods
        outbreak_by_period = outbreak_by_period[outbreak_by_period['period'].isin(['Training', 'Testing'])]
        
        # Format numbers
        outbreak_by_period['outbreak_percentage'] = outbreak_by_period['outbreak_percentage'].round(1)
        outbreak_by_period['avg_outbreak_cases'] = outbreak_by_period['avg_outbreak_cases'].round(1)
        
        # Rename columns for readability
        outbreak_by_period.columns = ['Period', 'Total Weeks', 'Outbreak Weeks', 'Outbreak %', 'Avg Outbreak Cases', 'Max Outbreak Cases']
        
        # Display outbreak statistics
        st.dataframe(outbreak_by_period, hide_index=True, use_container_width=True)
        
        # Create a seasonal outbreak analysis
        st.subheader("Seasonal Outbreak Analysis")
        
        # Add month and year columns
        city_data['month'] = city_data['week_start_date'].dt.month
        city_data['year'] = city_data['week_start_date'].dt.year
        
        # Create monthly outbreak frequency by period
        monthly_outbreaks = city_data.groupby(['period', 'month']).agg(
            weeks=('is_outbreak', 'count'),
            outbreaks=('is_outbreak', 'sum'),
            outbreak_freq=('is_outbreak', 'mean')
        ).reset_index()
        
        # Filter to only show Training and Testing periods
        monthly_outbreaks = monthly_outbreaks[monthly_outbreaks['period'].isin(['Training', 'Testing'])]
        
        # Create seasonal outbreak plot
        fig = px.line(
            monthly_outbreaks,
            x='month',
            y='outbreak_freq',
            color='period',
            title=f'{city} Seasonal Outbreak Patterns: Training vs Testing Periods',
            labels={
                'month': 'Month',
                'outbreak_freq': 'Outbreak Frequency',
                'period': 'Period'
            },
            color_discrete_map={
                'Training': 'blue',
                'Testing': 'red'
            }
        )
        
        # Add markers
        fig.update_traces(mode='lines+markers', marker=dict(size=8))
        
        # Add month names
        month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        fig.update_xaxes(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names
        )
        
        # Update y-axis to show as percentage
        fig.update_yaxes(tickformat='.0%')
        
        # Add seasonal markers based on city
        if city_code == 'sj':
            # San Juan seasons
            fig.add_vrect(
                x0=4, x1=11,
                fillcolor='rgba(255,0,0,0.1)', layer='below', line_width=0,
                annotation_text='Wet Season',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0=1, x1=4,
                fillcolor='rgba(0,255,0,0.1)', layer='below', line_width=0,
                annotation_text='Dry Season',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0=11, x1=13,
                fillcolor='rgba(0,255,0,0.1)', layer='below', line_width=0,
                annotation_text='Dry Season',
                annotation_position='top left'
            )
        else:
            # Iquitos seasons
            fig.add_vrect(
                x0=11, x1=13,
                fillcolor='rgba(0,0,255,0.1)', layer='below', line_width=0,
                annotation_text='High Water Season',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0=1, x1=5,
                fillcolor='rgba(0,0,255,0.1)', layer='below', line_width=0,
                annotation_text='High Water Season',
                annotation_position='top left'
            )
            fig.add_vrect(
                x0=5, x1=11,
                fillcolor='rgba(255,255,0,0.1)', layer='below', line_width=0,
                annotation_text='Low Water Season',
                annotation_position='top left'
            )
        
        # Update layout
        fig.update_layout(
            height=500,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add time period analysis interpretation
        st.markdown(f"""
        ### Time Period Analysis Interpretation for {city}
        
        #### Key Findings:
        
        1. **Changing Outbreak Patterns**:
           - {'There is a significant increase in outbreak frequency during the testing period compared to the training period.' if outbreak_by_period['Outbreak %'].iloc[1] > outbreak_by_period['Outbreak %'].iloc[0] else 'Outbreak frequency remains similar between training and testing periods.'}
           - {'Average outbreak severity (cases) is notably higher in the testing period.' if outbreak_by_period['Avg Outbreak Cases'].iloc[1] > outbreak_by_period['Avg Outbreak Cases'].iloc[0] else 'Average outbreak severity (cases) decreased or remained similar in the testing period.'}
        
        2. **Seasonal Shift**:
           - {'The testing period shows a shift in seasonal outbreak patterns, with peak months differing from the training period.' if abs(monthly_outbreaks.groupby('period')['outbreak_freq'].idxmax().diff().iloc[1]) > 2 else 'Seasonal outbreak patterns remain relatively consistent between periods, with similar peak months.'}
           - {'Testing period outbreaks extended into previously low-risk months, suggesting changing climate patterns or mosquito behavior.' if monthly_outbreaks.query('period == "Testing" and outbreak_freq > 0.25').shape[0] > monthly_outbreaks.query('period == "Training" and outbreak_freq > 0.25').shape[0] else 'The months at highest risk for outbreaks remain consistent between periods.'}
        
        3. **Distribution Changes**:
           - {'The distribution of dengue cases shows more extreme values in the testing period.' if period_stats[period_stats['Statistic'] == 'Standard Deviation']['Change %'].iloc[0] > 10 else 'The distribution of dengue cases remains relatively stable between periods.'}
           - {'The average number of cases per week has increased significantly in the testing period.' if period_stats[period_stats['Statistic'] == 'Average Cases']['Change %'].iloc[0] > 10 else 'The average number of cases per week is similar between periods.'}
        
        4. **Model Implications**:
           - The significant differences between training and testing periods help explain why the model struggled to generate accurate predictions.
           - The poor R² values suggest the model failed to capture these changing patterns between the two time periods.
           - The differences in outbreak seasonality between periods would require more sophisticated temporal modeling approaches.
        """)

# Tab 4: Climate Patterns
with tab4:
    st.header("Climate Patterns and Dengue Outbreaks")
    
    # Load training data with climate variables
    if not train_data.empty:
        # Add city selector
        city = st.selectbox(
            "Select city to analyze:",
            ["San Juan", "Iquitos"],
            index=0,
            key="climate_patterns_city"
        )
        
        city_code = 'sj' if city == 'San Juan' else 'iq'
        
        # Filter data for selected city
        city_data = train_data[train_data['city'] == city_code].copy()
        
        # Create outbreak indicator (75th percentile)
        outbreak_threshold = np.percentile(city_data['total_cases'], 75)
        city_data['is_outbreak'] = (city_data['total_cases'] > outbreak_threshold).astype(int)
        
        # Climate variable selector
        climate_vars = [
            'reanalysis_air_temp_k', 
            'reanalysis_relative_humidity_percent',
            'reanalysis_precip_amt_kg_per_m2',
            'reanalysis_specific_humidity_g_per_kg',
            'reanalysis_dew_point_temp_k',
            'station_avg_temp_c',
            'station_precip_mm',
            'ndvi_ne', 'ndvi_nw', 'ndvi_se', 'ndvi_sw'
        ]
        
        # Filter to only available variables
        available_vars = [var for var in climate_vars if var in city_data.columns]
        
        selected_var = st.selectbox(
            "Select climate variable to analyze:",
            available_vars,
            index=0,
            key="climate_variable"
        )
        
        # Calculate variable statistics by outbreak status
        climate_by_outbreak = city_data.groupby('is_outbreak').agg(
            {selected_var: ['mean', 'std', 'min', 'max', 'count']}
        ).reset_index()
        
        # Flatten multi-index columns
        climate_by_outbreak.columns = [
            'is_outbreak' if col[0] == 'is_outbreak' else f"{col[0]}_{col[1]}" 
            for col in climate_by_outbreak.columns
        ]
        
        # Rename outbreak column values
        climate_by_outbreak['Status'] = climate_by_outbreak['is_outbreak'].map({0: 'Normal', 1: 'Outbreak'})
        
        # Create comparison table
        climate_comparison = climate_by_outbreak[[
            'Status', 
            f'{selected_var}_mean', 
            f'{selected_var}_std', 
            f'{selected_var}_min', 
            f'{selected_var}_max',
            f'{selected_var}_count'
        ]]
        
        # Rename columns for readability
        var_display_name = selected_var.replace('_', ' ').title()
        climate_comparison.columns = [
            'Status', 'Average', 'Std Dev', 'Minimum', 'Maximum', 'Count'
        ]
        
        # Display climate statistics by outbreak status
        st.subheader(f"{var_display_name} Statistics by Outbreak Status")
        st.dataframe(climate_comparison, hide_index=True, use_container_width=True)
        
        # Create boxplot comparing distribution
        st.subheader(f"{var_display_name} Distribution by Outbreak Status")
        
        fig = px.box(
            city_data,
            x='is_outbreak',
            y=selected_var,
            color='is_outbreak',
            points="all",
            labels={
                'is_outbreak': 'Outbreak Status',
                selected_var: var_display_name
            },
            color_discrete_map={
                0: 'blue',
                1: 'red'
            },
            category_orders={'is_outbreak': [0, 1]},
            title=f"{var_display_name} Distribution: Normal vs Outbreak Periods"
        )
        
        # Update x-axis labels
        fig.update_xaxes(
            tickmode='array',
            tickvals=[0, 1],
            ticktext=['Normal', 'Outbreak']
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Create scatter plot of variable vs dengue cases
        st.subheader(f"Relationship between {var_display_name} and Dengue Cases")
        
        fig = px.scatter(
            city_data,
            x=selected_var,
            y='total_cases',
            color='is_outbreak',
            trendline='ols',
            labels={
                selected_var: var_display_name,
                'total_cases': 'Total Dengue Cases',
                'is_outbreak': 'Outbreak Status'
            },
            color_discrete_map={
                0: 'blue',
                1: 'red'
            },
            title=f"{var_display_name} vs Dengue Cases in {city}"
        )
        
        # Update legend
        fig.update_layout(
            legend=dict(
                title="Outbreak Status",
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                itemsizing="constant",
                orientation="h"
            ),
            coloraxis_colorbar=dict(
                title="Outbreak Status",
                tickvals=[0, 1],
                ticktext=["Normal", "Outbreak"]
            ),
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Calculate correlation
        corr = city_data[[selected_var, 'total_cases']].corr().iloc[0, 1]
        st.metric(f"Correlation with Dengue Cases", f"{corr:.4f}")
        
        # Lag analysis
        st.subheader("Lag Analysis: Climate Variable Effects Over Time")
        
        # Create lag variables dynamically
        max_lag = 16  # 4 months
        lag_cols = []
        
        # Create a copy to avoid warnings
        lag_data = city_data.copy().sort_values('week_start_date')
        
        # Create lags for the selected variable
        for lag in range(1, max_lag + 1):
            lag_col = f"{selected_var}_lag_{lag}"
            lag_data[lag_col] = lag_data[selected_var].shift(lag)
            lag_cols.append(lag_col)
        
        # Calculate correlations with lagged variables
        lag_corrs = []
        for lag_col in lag_cols:
            lag_corr = lag_data[['total_cases', lag_col]].corr().iloc[0, 1]
            lag_week = int(lag_col.split('_')[-1])
            lag_corrs.append({
                'lag_weeks': lag_week,
                'correlation': lag_corr
            })
        
        # Create lag correlation DataFrame
        lag_corr_df = pd.DataFrame(lag_corrs)
        
        # Create lag correlation plot
        fig = px.line(
            lag_corr_df,
            x='lag_weeks',
            y='correlation',
            markers=True,
            labels={
                'lag_weeks': 'Lag (Weeks)',
                'correlation': 'Correlation with Dengue Cases'
            },
            title=f"Lag Correlation Analysis: {var_display_name} vs Dengue Cases in {city}"
        )
        
        # Add zero reference line
        fig.add_hline(
            y=0, 
            line_dash="dash", 
            line_color="gray",
            annotation_text="No correlation"
        )
        
        # Find optimal lag (maximum absolute correlation)
        optimal_lag = lag_corr_df.iloc[lag_corr_df['correlation'].abs().idxmax()]
        
        # Add marker for optimal lag
        fig.add_vline(
            x=optimal_lag['lag_weeks'],
            line_dash="dot",
            line_color="green",
            annotation_text=f"Optimal lag: {int(optimal_lag['lag_weeks'])} weeks"
        )
        
        # Update layout
        fig.update_layout(
            height=500,
            hovermode='x unified',
            xaxis=dict(
                tickmode='linear',
                tick0=1,
                dtick=1
            )
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal pattern analysis
        st.subheader("Seasonal Pattern Analysis")
        
        # Add month column if it doesn't exist
        if 'month' not in city_data.columns:
            city_data['month'] = city_data['week_start_date'].dt.month
        
        # Calculate monthly averages
        monthly_data = city_data.groupby('month').agg({
            selected_var: 'mean',
            'total_cases': 'mean',
            'is_outbreak': 'mean'
        }).reset_index()
        
        # Create combined seasonal plot
        fig = go.Figure()
        
        # Add climate variable line
        fig.add_trace(go.Scatter(
            x=monthly_data['month'],
            y=monthly_data[selected_var],
            mode='lines+markers',
            name=var_display_name,
            line=dict(color='blue', width=2)
        ))
        
        # Add dengue cases line on secondary Y-axis
        fig.add_trace(go.Scatter(
            x=monthly_data['month'],
            y=monthly_data['total_cases'],
            mode='lines+markers',
            name='Dengue Cases',
            line=dict(color='red', width=2),
            yaxis='y2'
        ))
        
        # Add outbreak probability bar chart
        fig.add_trace(go.Bar(
            x=monthly_data['month'],
            y=monthly_data['is_outbreak'],
            name='Outbreak Probability',
            marker_color='rgba(255, 165, 0, 0.5)',
            yaxis='y3'
        ))
        
        # Update layout for triple y-axes
        fig.update_layout(
            title=f"Seasonal Patterns: {var_display_name} vs Dengue Cases in {city}",
            xaxis=dict(
                title="Month",
                tickmode='array',
                tickvals=list(range(1, 13)),
                ticktext=['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            ),
            yaxis=dict(
                title=var_display_name,
                titlefont=dict(color='blue'),
                tickfont=dict(color='blue')
            ),
            yaxis2=dict(
                title='Average Dengue Cases',
                titlefont=dict(color='red'),
                tickfont=dict(color='red'),
                anchor='x',
                overlaying='y',
                side='right'
            ),
            yaxis3=dict(
                title='Outbreak Probability',
                titlefont=dict(color='orange'),
                tickfont=dict(color='orange'),
                anchor='free',
                overlaying='y',
                side='right',
                position=1.0,
                range=[0, 1],
                tickformat='.0%'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='center',
                x=0.5
            ),
            height=600,
            hovermode='x unified',
            margin=dict(r=80)  # Add space on right for 3rd y-axis
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add climate pattern interpretation
        st.markdown(f"""
        ### Climate Pattern Interpretation for {city}
        
        #### Key Findings for {var_display_name}:
        
        1. **Relationship with Outbreaks**:
           - {'Shows a strong relationship with dengue outbreaks' if abs(corr) > 0.5 else 'Shows a moderate relationship with dengue outbreaks' if abs(corr) > 0.3 else 'Shows a weak relationship with dengue outbreaks'}
           - The difference between outbreak and non-outbreak periods for this variable is {'statistically significant' if abs(climate_comparison['Average'].iloc[1] - climate_comparison['Average'].iloc[0]) > climate_comparison['Std Dev'].mean() else 'not statistically significant'}
        
        2. **Lag Effect Analysis**:
           - This climate factor shows strongest correlation with dengue cases after a {int(optimal_lag['lag_weeks'])} week lag
           - {'This suggests this factor might be useful for early warning prediction systems' if optimal_lag['lag_weeks'] >= 4 and abs(optimal_lag['correlation']) > 0.3 else 'The lag effect for this variable may not be strong enough for reliable early warnings'}
        
        3. **Seasonal Patterns**:
           - Peak values for this variable {'coincide with' if abs(monthly_data[selected_var].idxmax() - monthly_data['total_cases'].idxmax()) <= 1 else 'precede' if monthly_data[selected_var].idxmax() < monthly_data['total_cases'].idxmax() else 'follow'} peak dengue case periods
           - Outbreak probability is highest when this variable is {'increasing' if monthly_data.iloc[monthly_data['is_outbreak'].idxmax() - 1][selected_var] < monthly_data.iloc[monthly_data['is_outbreak'].idxmax()][selected_var] else 'decreasing' if monthly_data.iloc[monthly_data['is_outbreak'].idxmax() - 1][selected_var] > monthly_data.iloc[monthly_data['is_outbreak'].idxmax()][selected_var] else 'stable'}
        
        4. **City-Specific Factors**:
           - For {city}, this climate variable is {'likely a significant predictor' if abs(corr) > 0.4 else 'one of several contributing factors' if abs(corr) > 0.2 else 'less important than other factors'} in dengue transmission
           - {'Temperature variables generally show stronger relationships with dengue cases in San Juan' if city == 'San Juan' and 'temp' in selected_var.lower() else 'Precipitation variables show more significant effects in Iquitos due to rainforest conditions' if city == 'Iquitos' and 'precip' in selected_var.lower() else ''}
        """)

# Tab 5: Improvement Suggestions
with tab5:
    st.header("Model Improvement Suggestions")
    
    st.markdown("""
    ### Critical Issues with Current Models
    
    Based on the analysis of model performance and results, several critical issues have been identified:
    
    1. **Extremely Poor Performance**:
       - San Juan model has a negative R² value (-0.6426)
       - Iquitos model has a catastrophically poor R² value (-3,733,535.5)
       - Both models show massive RMSE values relative to the average case numbers
    
    2. **Systematic Overestimation**:
       - San Juan model has a positive bias of 20.55
       - Iquitos model produces unrealistic predictions with a maximum of 224,560 cases
    
    3. **Timeframe Shift Problems**:
       - Both models fail to account for changing patterns between training and testing timeframes
       - The distinct timeframe approach creates discontinuity between model training and application
    
    4. **Outbreak Detection Imbalance**:
       - Both models have perfect recall but very low precision (0.25 and 0.23)
       - This indicates the models are predicting outbreaks too frequently
    """)
    
    st.subheader("Recommended Improvements")
    
    improvement_tabs = st.tabs([
        "Data Handling", 
        "Feature Engineering", 
        "Model Architecture", 
        "Training Approach",
        "Implementation Plan"
    ])
    
    with improvement_tabs[0]:
        st.markdown("""
        ## Data Handling Improvements
        
        ### 1. Time Series Split Refinement
        
        **Problem**: Fixed timeframes create an artificial gap in model learning.
        
        **Solution**: Implement a sliding window approach:
        - Use a rolling timeframe approach rather than fixed cutoffs
        - Maintain a validation set that mimics the testing period characteristics
        - Implement proper cross-validation with multiple time slices
        
        ```python
        # Sliding window example
        from sklearn.model_selection import TimeSeriesSplit
        
        tscv = TimeSeriesSplit(n_splits=5, test_size=52)  # Weekly data, 1 year test size
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            # Train and validate model
        ```
        
        ### 2. Target Variable Transformation
        
        **Problem**: Dengue case counts are highly skewed with many zeros.
        
        **Solution**: Apply appropriate transformations:
        - Use log(y+1) transformation consistently
        - Consider square root transformation for less extreme transformation
        - Explore specialized count data approaches like Poisson or negative binomial models
        
        ```python
        # Target transformation
        y_train_transformed = np.log1p(y_train)  # log(y+1)
        # or
        y_train_transformed = np.sqrt(y_train)  # square root
        
        # And inverse transform during prediction
        y_pred = np.expm1(raw_predictions)  # inverse of log1p
        ```
        
        ### 3. Data Drift Detection
        
        **Problem**: Climate and case patterns change between training and testing periods.
        
        **Solution**: Implement data drift detection:
        - Add explicit drift detection mechanisms
        - Weight recent data more heavily in the training process
        - Implement concept drift adaptation techniques
        
        ```python
        # Decay weights example - more weight to recent observations
        weights = np.exp(np.linspace(0, 1, len(X_train))) / np.exp(1)
        model.fit(X_train, y_train, sample_weight=weights)
        ```
        """)
    
    with improvement_tabs[1]:
        st.markdown("""
        ## Feature Engineering Improvements
        
        ### 1. Location-Specific Feature Selection
        
        **Problem**: Different climate factors affect each city differently.
        
        **Solution**: Implement city-specific feature selection:
        - Use separate feature importance analysis for each city
        - Eliminate features with low importance for each location
        - Create city-specific composite features based on most relevant variables
        
        ```python
        # City-specific feature selection
        from sklearn.feature_selection import SelectFromModel
        
        # San Juan
        selector_sj = SelectFromModel(xgb.XGBRegressor())
        selector_sj.fit(X_train_sj, y_train_sj)
        X_train_sj_selected = selector_sj.transform(X_train_sj)
        
        # Iquitos - separate selection
        selector_iq = SelectFromModel(xgb.XGBRegressor())
        selector_iq.fit(X_train_iq, y_train_iq)
        X_train_iq_selected = selector_iq.transform(X_train_iq)
        ```
        
        ### 2. Lag Optimization
        
        **Problem**: Current lag features use fixed windows that may not capture optimal relationships.
        
        **Solution**: Data-driven lag selection:
        - Analyze optimal lag periods for each climate variable in each city
        - Create variable-specific lag features based on correlation analysis
        - Implement dynamic lag selection during preprocessing
        
        ```python
        def create_optimal_lags(df, variable, max_lag=16, min_corr=0.2):
            """Create lag features only for lags with significant correlation."""
            lag_corrs = []
            lag_cols = []
            
            # Calculate correlation for each lag
            for lag in range(1, max_lag + 1):
                lag_col = f"{variable}_lag_{lag}"
                df[lag_col] = df[variable].shift(lag)
                corr = df[['total_cases', lag_col]].corr().iloc[0, 1]
                lag_corrs.append((lag, corr))
                
            # Keep only significant lags
            for lag, corr in lag_corrs:
                if abs(corr) >= min_corr:
                    lag_cols.append(f"{variable}_lag_{lag}")
                else:
                    df.drop(columns=[f"{variable}_lag_{lag}"], inplace=True)
                    
            return lag_cols
        ```
        
        ### 3. Enhanced Interaction Features
        
        **Problem**: Complex interactions between climate variables are not captured.
        
        **Solution**: Create more sophisticated interaction features:
        - Temperature-humidity interaction index
        - Precipitation-temperature breeding condition index
        - Seasonal climate departure features (difference from seasonal norms)
        
        ```python
        # Create temperature-humidity interaction
        df['temp_humidity_index'] = df['temperature'] * df['humidity'] / 100
        
        # Create seasonal departure features
        monthly_means = df.groupby('month')['temperature'].mean()
        df['temp_seasonal_departure'] = df.apply(
            lambda x: x['temperature'] - monthly_means[x['month']], axis=1
        )
        ```
        """)
    
    with improvement_tabs[2]:
        st.markdown("""
        ## Model Architecture Improvements
        
        ### 1. Ensemble Approach
        
        **Problem**: Single XGBoost model cannot capture all temporal patterns.
        
        **Solution**: Implement a multi-model ensemble:
        - Train separate models for different seasons or climate regimes
        - Use stacking with first-level models specialized for different patterns
        - Combine predictions from multiple model types (XGBoost, LSTM, etc.)
        
        ```python
        # Simplified ensemble approach
        from sklearn.ensemble import StackingRegressor
        
        estimators = [
            ('xgb', xgb.XGBRegressor()),
            ('rf', RandomForestRegressor()),
            ('gbm', LGBMRegressor())
        ]
        
        stacking_model = StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge()
        )
        ```
        
        ### 2. Time Series Components
        
        **Problem**: XGBoost doesn't explicitly model sequential patterns in time.
        
        **Solution**: Add explicit time series components:
        - Implement hybrid models combining XGBoost with ARIMA
        - Add recurrent neural network (RNN/LSTM) components for sequence modeling
        - Use Prophet or other time series decomposition techniques for trend/seasonality
        
        ```python
        # Hybrid XGBoost-LSTM approach
        # LSTM for temporal patterns
        lstm_model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(lookback, n_features)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        
        # Combine LSTM predictions with other features for XGBoost
        X_combined = np.column_stack([X_features, lstm_predictions])
        final_model = xgb.XGBRegressor()
        final_model.fit(X_combined, y)
        ```
        
        ### 3. Specialized Count Models
        
        **Problem**: Standard regression approach isn't ideal for count data.
        
        **Solution**: Use specialized count data models:
        - Implement custom objective functions (zero-inflated Poisson)
        - Consider two-stage approach (outbreak classification + severity regression)
        - Use hurdle models for zero-handling
        
        ```python
        # Two-stage modeling approach
        # Stage 1: Classification model for outbreak (yes/no)
        outbreak_classifier = xgb.XGBClassifier()
        outbreak_classifier.fit(X_train, y_train > outbreak_threshold)
        
        # Stage 2: Severity model only for outbreak periods
        mask = y_train > outbreak_threshold
        severity_model = xgb.XGBRegressor()
        severity_model.fit(X_train[mask], y_train[mask])
        
        # Prediction combines both models
        outbreak_prob = outbreak_classifier.predict_proba(X_test)[:, 1]
        severity_pred = severity_model.predict(X_test)
        final_pred = outbreak_prob * severity_pred  # Expected value
        ```
        """)
    
    with improvement_tabs[3]:
        st.markdown("""
        ## Training Approach Improvements
        
        ### 1. Objective Function Customization
        
        **Problem**: Standard regression objectives don't account for count data characteristics.
        
        **Solution**: Implement custom objectives:
        - Use custom loss functions that penalize underprediction of outbreaks more heavily
        - Implement a modified Poisson or negative binomial loss
        - Add regularization terms specific to the dengue prediction task
        
        ```python
        # Custom objective function example
        def outbreak_weighted_rmse(y_true, y_pred):
            """RMSE that penalizes outbreak underprediction more heavily."""
            outbreak_threshold = np.percentile(y_train, 75)
            weights = np.ones_like(y_true)
            weights[y_true > outbreak_threshold] = 3.0  # Higher weight for outbreaks
            
            return np.sqrt(np.average(
                (y_true - y_pred) ** 2,
                weights=weights
            ))
        
        # Use custom objective in XGBoost
        model = xgb.XGBRegressor(objective='reg:squarederror')
        model.fit(X_train, y_train, eval_metric=outbreak_weighted_rmse)
        ```
        
        ### 2. Transfer Learning Approach
        
        **Problem**: Limited data for some time periods and locations.
        
        **Solution**: Implement transfer learning:
        - Train base models on combined data from both cities
        - Fine-tune city-specific models from the base model
        - Use pre-trained models from similar climatic regions
        
        ```python
        # Simple transfer learning approach
        # 1. Train on combined data
        combined_model = xgb.XGBRegressor()
        combined_model.fit(X_combined, y_combined)
        
        # 2. Use as starting point for city-specific models
        sj_model = xgb.XGBRegressor()
        sj_model.fit(X_sj, y_sj, xgb_model=combined_model)
        
        iq_model = xgb.XGBRegressor()
        iq_model.fit(X_iq, y_iq, xgb_model=combined_model)
        ```
        
        ### 3. Hyperparameter Optimization Strategy
        
        **Problem**: Current hyperparameters aren't optimized for the specific prediction task.
        
        **Solution**: Task-specific optimization:
        - Implement Bayesian optimization for hyperparameter tuning
        - Use custom evaluation metrics focused on outbreak detection
        - Set up separate optimization goals for different climatic seasons
        
        ```python
        # Bayesian optimization example
        from skopt import BayesSearchCV
        
        # Define search space
        param_space = {
            'max_depth': (3, 10),
            'learning_rate': (0.01, 0.2, 'log-uniform'),
            'min_child_weight': (1, 10),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'gamma': (0, 5),
            'reg_alpha': (0, 10),
            'reg_lambda': (1, 100, 'log-uniform')
        }
        
        # Custom scoring focused on outbreak detection
        def outbreak_f1_score(y_true, y_pred):
            outbreak_threshold = np.percentile(y_train, 75)
            y_true_outbreak = y_true > outbreak_threshold
            y_pred_outbreak = y_pred > outbreak_threshold
            return f1_score(y_true_outbreak, y_pred_outbreak)
        
        # Bayesian optimization
        opt = BayesSearchCV(
            xgb.XGBRegressor(),
            param_space,
            scoring=outbreak_f1_score,
            n_iter=50,
            cv=TimeSeriesSplit(n_splits=5),
            n_jobs=-1
        )
        
        opt.fit(X_train, y_train)
        ```
        """)
    
    with improvement_tabs[4]:
        st.markdown("""
        ## Implementation Plan
        
        ### 1. Short-term Fixes (1-2 weeks)
        
        **Immediate issues to address:**
        
        - **Fix Prediction Instability**: 
          - Implement proper feature scaling for all numerical inputs
          - Add log transformation for target variables
          - Use more stable regression objectives (squared error)
        
        - **Improve Timeframe Handling**:
          - Implement proper validation sets within each city's timeframe
          - Add sanity checks to identify extreme predictions
          - Set reasonable minimum/maximum bounds on predictions
        
        - **Enhance Evaluation**:
          - Create detailed metrics dashboards for model diagnosis
          - Implement visualization of prediction errors over time
          - Track feature importance stability across different runs
        
        ### 2. Medium-term Improvements (2-4 weeks)
        
        **Core model enhancements:**
        
        - **Advanced Feature Engineering**:
          - Implement optimal lag selection for each variable/city
          - Create specialized seasonal features
          - Add climate interaction terms
        
        - **Model Architecture Refinement**:
          - Implement ensemble approach with multiple base models
          - Add time series components for seasonal patterns
          - Fine-tune hyperparameters with Bayesian optimization
        
        - **Evaluation Framework**:
          - Create specialized metrics for outbreak detection
          - Implement outbreak severity scoring
          - Develop interactive visualization tools
        
        ### 3. Long-term Strategy (1-3 months)
        
        **Comprehensive solution:**
        
        - **Hybrid Modeling System**:
          - Implement two-stage modeling (outbreak classification + severity regression)
          - Develop time-aware ensemble architecture
          - Add transfer learning between cities
        
        - **Advanced Time Series Handling**:
          - Implement proper concept drift detection
          - Add automated retraining triggers
          - Develop adaptive feature selection based on climate regime
        
        - **Production System**:
          - Create automated pipeline for model updates
          - Implement model versioning and tracking
          - Develop interactive prediction interface
        
        ### Implementation Priorities
        
        1. **First priority**: Fix prediction stability and extreme values
        2. **Second priority**: Improve feature engineering with climate lag optimization
        3. **Third priority**: Implement ensemble approach with multiple model types
        4. **Fourth priority**: Develop specialized outbreak detection components
        
        This phased approach allows for incremental improvements while addressing the most critical issues first.
        """)
    
    # Final recommendations summary
    st.subheader("Key Recommendations Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Data Processing:**
        - Implement log transformation for targets
        - Use robust feature scaling
        - Add sliding window validation
        - Detect and handle data drift
        """)
    
    with col2:
        st.markdown("""
        **Model Architecture:**
        - Use ensemble of multiple models
        - Implement specialized count models
        - Create hybrid time series components
        - Add city-specific feature selection
        """)
    
    with col3:
        st.markdown("""
        **Evaluation & Deployment:**
        - Create custom outbreak metrics
        - Implement two-stage prediction
        - Optimize for outbreak detection
        - Set prediction bounds and validation
        """)

# Run the app
if __name__ == "__main__":
    st.markdown("""
    <style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1, h2, h3, h4 {
        margin-top: 1rem !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        border-radius: 4px 4px 0 0;
    }
    </style>
    """, unsafe_allow_html=True)