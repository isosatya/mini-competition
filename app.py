import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stMetric {
        background-color: white;
        padding: 10px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Load data
@st.cache_data
def load_data():
    # Load predictions
    predictions = pd.read_csv('data/processed/submission_xgboost.csv')
    
    # Load training features and labels
    features = pd.read_csv('data/raw/Training_Data_Features.csv')
    labels = pd.read_csv('data/raw/Training_Data_Labels.csv')
    
    # Merge features and labels for historical data
    historical_data = pd.merge(features, labels, on=['city', 'year', 'weekofyear'])
    
    # Create year-week column for better time series visualization
    historical_data['date'] = pd.to_datetime(historical_data['week_start_date'])
    predictions['year_week'] = predictions['year'].astype(str) + '-W' + predictions['weekofyear'].astype(str).str.zfill(2)
    
    return predictions, historical_data

def create_correlation_heatmap(data, city):
    # Select relevant numerical columns
    numerical_cols = ['total_cases', 'precipitation_amt_mm', 'reanalysis_air_temp_k',
                     'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg']
    
    # Calculate correlation matrix
    corr_matrix = data[data['city'] == city][numerical_cols].corr()
    
    # Create heatmap
    fig = go.Figure(data=go.Heatmap(
        z=corr_matrix,
        x=corr_matrix.columns,
        y=corr_matrix.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title=f'Correlation Heatmap for {city.upper()}',
        height=500
    )
    
    return fig

def main():
    st.title("🦟 Dengue Prediction Dashboard")
    
    # Load data
    predictions, historical_data = load_data()
    
    # Sidebar
    st.sidebar.header("Filters")
    city = st.sidebar.selectbox(
        "Select City",
        ["sj", "iq"],
        format_func=lambda x: "San Juan" if x == "sj" else "Iquitos"
    )
    
    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🔍 Detailed Analysis",
        "🎯 Predictions",
        "📈 Model Performance"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        city_data = historical_data[historical_data['city'] == city]
        
        with col1:
            st.metric("Average Cases", f"{city_data['total_cases'].mean():.1f}")
        with col2:
            st.metric("Max Cases", int(city_data['total_cases'].max()))
        with col3:
            st.metric("Total Historical Cases", int(city_data['total_cases'].sum()))
        with col4:
            st.metric("Years of Data", len(city_data['year'].unique()))
        
        # Historical trends
        st.subheader("Historical Trends")
        fig = px.line(city_data, x='date', y='total_cases',
                     title=f'Historical Dengue Cases in {"San Juan" if city == "sj" else "Iquitos"}',
                     labels={'date': 'Date', 'total_cases': 'Number of Cases'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal patterns
        st.subheader("Seasonal Patterns")
        seasonal_data = city_data.groupby('weekofyear')['total_cases'].mean().reset_index()
        fig_seasonal = px.line(seasonal_data, x='weekofyear', y='total_cases',
                             title='Average Cases by Week of Year',
                             labels={'weekofyear': 'Week of Year', 'total_cases': 'Average Cases'})
        st.plotly_chart(fig_seasonal, use_container_width=True)
    
    with tab2:
        st.header("Detailed Analysis")
        
        # Correlation heatmap
        st.subheader("Feature Correlations")
        fig_corr = create_correlation_heatmap(historical_data, city)
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Environmental factors analysis
        st.subheader("Environmental Factors")
        col1, col2 = st.columns(2)
        
        with col1:
            env_factors = ['precipitation_amt_mm', 'reanalysis_air_temp_k',
                         'reanalysis_relative_humidity_percent', 'reanalysis_specific_humidity_g_per_kg']
            selected_factor = st.selectbox("Select Environmental Factor", env_factors,
                                         format_func=lambda x: x.replace('_', ' ').title())
            
            fig_env = px.scatter(city_data, x=selected_factor, y='total_cases',
                               title=f'{selected_factor.replace("_", " ").title()} vs Cases',
                               trendline="ols")
            st.plotly_chart(fig_env, use_container_width=True)
        
        with col2:
            # Monthly distribution
            monthly_data = city_data.copy()
            monthly_data['month'] = monthly_data['date'].dt.month
            monthly_avg = monthly_data.groupby('month')['total_cases'].mean().reset_index()
            
            fig_monthly = px.bar(monthly_avg, x='month', y='total_cases',
                               title='Average Cases by Month',
                               labels={'month': 'Month', 'total_cases': 'Average Cases'})
            st.plotly_chart(fig_monthly, use_container_width=True)
    
    with tab3:
        st.header("Predictions")
        
        # Predictions plot
        city_predictions = predictions[predictions['city'] == city]
        
        st.subheader("Predicted Cases")
        fig_pred = px.line(city_predictions, x='weekofyear', y='total_cases',
                          title=f'Predicted Dengue Cases for {"San Juan" if city == "sj" else "Iquitos"}',
                          labels={'weekofyear': 'Week of Year', 'total_cases': 'Predicted Cases'})
        st.plotly_chart(fig_pred, use_container_width=True)
        
        # Risk assessment
        st.subheader("Risk Assessment")
        col1, col2 = st.columns(2)
        
        with col1:
            risk_threshold = st.slider("Risk Threshold (Cases per Week)",
                                     min_value=0,
                                     max_value=int(city_predictions['total_cases'].max() * 1.2),
                                     value=int(city_predictions['total_cases'].mean() * 2))
            
            high_risk_weeks = city_predictions[city_predictions['total_cases'] > risk_threshold]
            st.write(f"Number of high-risk weeks: {len(high_risk_weeks)}")
            
            # Risk distribution
            fig_risk = px.histogram(city_predictions, x='total_cases',
                                  title='Distribution of Predicted Cases',
                                  labels={'total_cases': 'Number of Cases'})
            fig_risk.add_vline(x=risk_threshold, line_dash="dash", line_color="red",
                             annotation_text="Risk Threshold")
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab4:
        st.header("Model Performance")
        
        st.markdown("""
        ### Model Details
        - **Algorithm**: XGBoost (eXtreme Gradient Boosting)
        - **Features**: Environmental and climate variables
        - **Target**: Weekly dengue cases
        
        ### Key Features
        1. Temperature
        2. Precipitation
        3. Relative Humidity
        4. Specific Humidity
        
        ### Model Strengths
        - Captures seasonal patterns
        - Accounts for environmental factors
        - Provides weekly predictions
        - Identifies high-risk periods
        """)

if __name__ == "__main__":
    main() 