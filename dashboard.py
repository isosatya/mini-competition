import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(
    page_title="Dengue Prediction Dashboard",
    page_icon="🦟",
    layout="wide"
)

# Load data
@st.cache_data
def load_data():
    df = pd.read_csv('data/processed/cleaned_train_data.csv')
    df['week_start_date'] = pd.to_datetime(df['week_start_date'])
    return df

# Load the data
df = load_data()

# Create tabs
tab1, tab2, tab3 = st.tabs(["Introduction", "Training Data", "Correlation Analysis"])

with tab1:
    st.title("Dengue Prediction Dashboard")
    st.markdown("""
    ## Project Overview
    
    This dashboard presents a dengue fever prediction system that analyzes historical data to forecast potential outbreaks. 
    The project focuses on two cities and uses various environmental and meteorological factors to predict dengue cases.
    
    ### Key Features:
    - Historical dengue case analysis
    - Environmental factor correlation
    - Predictive modeling
    - Interactive data visualization
    
    The system aims to help public health officials and researchers better understand and predict dengue outbreaks, 
    enabling more effective prevention and response strategies.
    """)

with tab2:
    st.title("Training Data Analysis")
    
    # City selection
    cities = st.multiselect(
        "Select cities to display",
        options=['sj', 'iq'],
        default=['sj', 'iq']
    )
    
    if not cities:
        st.warning("Please select at least one city to display the data.")
    else:
        # Filter data for selected cities
        filtered_df = df[df['city'].isin(cities)]
        
        # Create a single line plot for all selected cities
        fig = px.line(
            filtered_df,
            x='week_start_date',
            y='total_cases',
            color='city',
            title='Dengue Cases Over Time',
            labels={
                'week_start_date': 'Date',
                'total_cases': 'Number of Cases',
                'city': 'City'
            }
        )
        
        # Update layout
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Cases",
            hovermode='x unified',
            height=500
        )
        
        # Customize colors for better distinction
        fig.update_traces(
            line=dict(width=2)
        )
        
        # Display the plot
        st.plotly_chart(fig, use_container_width=True)
        
        # Add population comparison and summary statistics side by side
        st.subheader("City Population and Statistics")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # Population data
            population_data = {
                'City': ['San Juan', 'Iquitos'],
                'Population': [2508000, 318000],
                'City Code': ['sj', 'iq']
            }
            pop_df = pd.DataFrame(population_data)
            pop_df = pop_df[pop_df['City Code'].isin(cities)]  # Filter based on selected cities
            
            # Create population bar chart
            pop_fig = px.bar(
                pop_df,
                x='City',
                y='Population',
                title='Population Size Comparison (Year 2000)',
                color='City',
                text='Population',
                labels={
                    'Population': 'Population Size',
                    'City': 'City'
                }
            )
            
            # Update layout
            pop_fig.update_layout(
                yaxis_title="Population Size",
                height=400,
                showlegend=False,
                margin=dict(t=50, b=50, l=50, r=50),  # Add margins
                yaxis=dict(
                    range=[0, max(pop_df['Population']) * 1.1]  # Add 10% padding to top
                )
            )
            
            # Format population numbers with commas and rotate text
            pop_fig.update_traces(
                texttemplate='%{text:,}',
                textposition='outside',
                textangle=0,  # Horizontal text
                textfont=dict(size=12)  # Adjust font size
            )
            
            # Display the population plot
            st.plotly_chart(pop_fig, use_container_width=True)
        
        with col2:
            # Show combined summary statistics
            st.subheader("Dengue Cases Statistics")
            stats_list = []
            for city in cities:
                city_df = df[df['city'] == city]
                stats = city_df['total_cases'].agg(['mean', 'max', 'min', 'sum']).round(2)
                stats_list.append({
                    'City': city.upper(),
                    'Average Cases': stats['mean'],
                    'Maximum Cases': stats['max'],
                    'Minimum Cases': stats['min'],
                    'Total Cases': stats['sum']
                })
            
            stats_df = pd.DataFrame(stats_list)
            st.dataframe(stats_df, hide_index=True)
        
        # Add seasonality analysis
        st.subheader("Seasonality Analysis")
        
        # Create two columns for seasonality plots
        season_col1, season_col2 = st.columns([1, 1])
        
        with season_col1:
            if 'sj' in cities:
                # San Juan seasonality
                sj_df = df[df['city'] == 'sj']
                sj_monthly = sj_df.groupby('month')['total_cases'].mean().reset_index()
                
                sj_fig = px.bar(
                    sj_monthly,
                    x='month',
                    y='total_cases',
                    title='San Juan: Monthly Average Dengue Cases',
                    labels={
                        'month': 'Month',
                        'total_cases': 'Average Cases'
                    }
                )
                
                # Add seasonal information
                sj_fig.add_vrect(
                    x0=4, x1=11,
                    fillcolor="red", opacity=0.1,
                    line_width=0,
                    annotation_text="Wet Season",
                    annotation_position="top left",
                    annotation_y=1.1  # Move text above the plot
                )
                
                sj_fig.add_vrect(
                    x0=11, x1=4,
                    fillcolor="green", opacity=0.1,
                    line_width=0,
                    annotation_text="Dry Season",
                    annotation_position="top right",
                    annotation_y=1.1  # Move text above the plot
                )
                
                sj_fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(t=100)  # Add more top margin for annotations
                )
                
                st.plotly_chart(sj_fig, use_container_width=True)
        
        with season_col2:
            if 'iq' in cities:
                # Iquitos seasonality
                iq_df = df[df['city'] == 'iq']
                iq_monthly = iq_df.groupby('month')['total_cases'].mean().reset_index()
                
                iq_fig = px.bar(
                    iq_monthly,
                    x='month',
                    y='total_cases',
                    title='Iquitos: Monthly Average Dengue Cases',
                    labels={
                        'month': 'Month',
                        'total_cases': 'Average Cases'
                    }
                )
                
                # Add seasonal information
                iq_fig.add_vrect(
                    x0=11, x1=5,
                    fillcolor="blue", opacity=0.1,
                    line_width=0,
                    annotation_text="High Water",
                    annotation_position="top left",
                    annotation_y=1.1  # Move text above the plot
                )
                
                iq_fig.add_vrect(
                    x0=5, x1=11,
                    fillcolor="yellow", opacity=0.1,
                    line_width=0,
                    annotation_text="Low Water",
                    annotation_position="top right",
                    annotation_y=1.1  # Move text above the plot
                )
                
                iq_fig.update_layout(
                    height=400,
                    showlegend=False,
                    margin=dict(t=100)  # Add more top margin for annotations
                )
                
                st.plotly_chart(iq_fig, use_container_width=True)
        
        # Add temperature and precipitation analysis
        st.subheader("Temperature and Precipitation Analysis")
        
        # Create two columns for the plots
        temp_precip_col1, temp_precip_col2 = st.columns([1, 1])
        
        with temp_precip_col1:
            if 'sj' in cities:
                # San Juan temperature and precipitation
                sj_df = df[df['city'] == 'sj']
                sj_monthly = sj_df.groupby('month').agg({
                    'station_avg_temp_c': 'mean',
                    'station_precip_mm': 'sum'
                }).reset_index()
                
                # Create figure with secondary y-axis
                sj_temp_precip_fig = go.Figure()
                
                # Add temperature line
                sj_temp_precip_fig.add_trace(
                    go.Scatter(
                        x=sj_monthly['month'],
                        y=sj_monthly['station_avg_temp_c'],
                        name='Temperature (°C)',
                        line=dict(color='red', width=2)
                    )
                )
                
                # Add precipitation bars
                sj_temp_precip_fig.add_trace(
                    go.Bar(
                        x=sj_monthly['month'],
                        y=sj_monthly['station_precip_mm'],
                        name='Precipitation (mm)',
                        yaxis='y2',
                        marker_color='blue',
                        opacity=0.5
                    )
                )
                
                # Update layout
                sj_temp_precip_fig.update_layout(
                    title='San Juan: Monthly Temperature and Precipitation',
                    xaxis_title='Month',
                    yaxis_title='Temperature (°C)',
                    yaxis2=dict(
                        title='Precipitation (mm)',
                        overlaying='y',
                        side='right'
                    ),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(sj_temp_precip_fig, use_container_width=True)
        
        with temp_precip_col2:
            if 'iq' in cities:
                # Iquitos temperature and precipitation
                iq_df = df[df['city'] == 'iq']
                iq_monthly = iq_df.groupby('month').agg({
                    'station_avg_temp_c': 'mean',
                    'station_precip_mm': 'sum'
                }).reset_index()
                
                # Create figure with secondary y-axis
                iq_temp_precip_fig = go.Figure()
                
                # Add temperature line
                iq_temp_precip_fig.add_trace(
                    go.Scatter(
                        x=iq_monthly['month'],
                        y=iq_monthly['station_avg_temp_c'],
                        name='Temperature (°C)',
                        line=dict(color='red', width=2)
                    )
                )
                
                # Add precipitation bars
                iq_temp_precip_fig.add_trace(
                    go.Bar(
                        x=iq_monthly['month'],
                        y=iq_monthly['station_precip_mm'],
                        name='Precipitation (mm)',
                        yaxis='y2',
                        marker_color='blue',
                        opacity=0.5
                    )
                )
                
                # Update layout
                iq_temp_precip_fig.update_layout(
                    title='Iquitos: Monthly Temperature and Precipitation',
                    xaxis_title='Month',
                    yaxis_title='Temperature (°C)',
                    yaxis2=dict(
                        title='Precipitation (mm)',
                        overlaying='y',
                        side='right'
                    ),
                    height=400,
                    showlegend=True,
                    legend=dict(
                        yanchor="top",
                        y=0.99,
                        xanchor="left",
                        x=0.01
                    )
                )
                
                st.plotly_chart(iq_temp_precip_fig, use_container_width=True)
        
        # Add NDVI analysis
        st.subheader("Vegetation Index (NDVI) Analysis")
        
        # Create two columns for the plots
        ndvi_col1, ndvi_col2 = st.columns([1, 1])
        
        with ndvi_col1:
            if 'sj' in cities:
                # San Juan NDVI
                sj_df = df[df['city'] == 'sj']
                # Calculate average NDVI
                sj_df['avg_ndvi'] = sj_df[['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']].mean(axis=1)
                sj_monthly = sj_df.groupby('month')['avg_ndvi'].mean().reset_index()
                
                # Create NDVI plot
                sj_ndvi_fig = px.line(
                    sj_monthly,
                    x='month',
                    y='avg_ndvi',
                    title='San Juan: Monthly Average NDVI',
                    labels={
                        'month': 'Month',
                        'avg_ndvi': 'Average NDVI'
                    }
                )
                
                # Update layout
                sj_ndvi_fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis=dict(
                        range=[sj_monthly['avg_ndvi'].min() * 0.95, sj_monthly['avg_ndvi'].max() * 1.05],  # Add 5% padding
                        title='NDVI Value'
                    )
                )
                
                st.plotly_chart(sj_ndvi_fig, use_container_width=True)
        
        with ndvi_col2:
            if 'iq' in cities:
                # Iquitos NDVI
                iq_df = df[df['city'] == 'iq']
                # Calculate average NDVI
                iq_df['avg_ndvi'] = iq_df[['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']].mean(axis=1)
                iq_monthly = iq_df.groupby('month')['avg_ndvi'].mean().reset_index()
                
                # Create NDVI plot
                iq_ndvi_fig = px.line(
                    iq_monthly,
                    x='month',
                    y='avg_ndvi',
                    title='Iquitos: Monthly Average NDVI',
                    labels={
                        'month': 'Month',
                        'avg_ndvi': 'Average NDVI'
                    }
                )
                
                # Update layout
                iq_ndvi_fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis=dict(
                        range=[iq_monthly['avg_ndvi'].min() * 0.95, iq_monthly['avg_ndvi'].max() * 1.05],  # Add 5% padding
                        title='NDVI Value'
                    )
                )
                
                st.plotly_chart(iq_ndvi_fig, use_container_width=True)
        
        # Add relative humidity analysis
        st.subheader("Relative Humidity Analysis")
        
        # Create two columns for the plots
        humidity_col1, humidity_col2 = st.columns([1, 1])
        
        with humidity_col1:
            if 'sj' in cities:
                # San Juan relative humidity
                sj_df = df[df['city'] == 'sj']
                sj_monthly = sj_df.groupby('month')['reanalysis_relative_humidity_percent'].mean().reset_index()
                
                # Create humidity plot
                sj_humidity_fig = px.line(
                    sj_monthly,
                    x='month',
                    y='reanalysis_relative_humidity_percent',
                    title='San Juan: Monthly Average Relative Humidity',
                    labels={
                        'month': 'Month',
                        'reanalysis_relative_humidity_percent': 'Relative Humidity (%)'
                    }
                )
                
                # Update layout
                sj_humidity_fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis=dict(
                        range=[sj_monthly['reanalysis_relative_humidity_percent'].min() * 0.95, 
                              sj_monthly['reanalysis_relative_humidity_percent'].max() * 1.05],
                        title='Relative Humidity (%)'
                    )
                )
                
                st.plotly_chart(sj_humidity_fig, use_container_width=True)
        
        with humidity_col2:
            if 'iq' in cities:
                # Iquitos relative humidity
                iq_df = df[df['city'] == 'iq']
                iq_monthly = iq_df.groupby('month')['reanalysis_relative_humidity_percent'].mean().reset_index()
                
                # Create humidity plot
                iq_humidity_fig = px.line(
                    iq_monthly,
                    x='month',
                    y='reanalysis_relative_humidity_percent',
                    title='Iquitos: Monthly Average Relative Humidity',
                    labels={
                        'month': 'Month',
                        'reanalysis_relative_humidity_percent': 'Relative Humidity (%)'
                    }
                )
                
                # Update layout
                iq_humidity_fig.update_layout(
                    height=400,
                    showlegend=False,
                    yaxis=dict(
                        range=[iq_monthly['reanalysis_relative_humidity_percent'].min() * 0.95, 
                              iq_monthly['reanalysis_relative_humidity_percent'].max() * 1.05],
                        title='Relative Humidity (%)'
                    )
                )
                
                st.plotly_chart(iq_humidity_fig, use_container_width=True)

with tab3:
    st.title("Correlation Analysis")
    
    # Define columns for correlation analysis
    corr_columns = [
        'total_cases',
        'station_avg_temp_c',
        'station_precip_mm',
        'avg_ndvi',
        'reanalysis_relative_humidity_percent'
    ]
    
    # Create two columns for the correlation matrices
    corr_col1, corr_col2 = st.columns([1, 1])
    
    with corr_col1:
        # San Juan correlation
        sj_corr_df = df[df['city'] == 'sj']
        sj_corr_df['avg_ndvi'] = sj_corr_df[['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']].mean(axis=1)
        
        # Create correlation matrix
        sj_correlation_matrix = sj_corr_df[corr_columns].corr().round(2)
        
        # Rename columns for better readability
        sj_correlation_matrix.columns = [
            'Total Cases',
            'Temperature (°C)',
            'Precipitation (mm)',
            'NDVI',
            'Relative Humidity (%)'
        ]
        sj_correlation_matrix.index = sj_correlation_matrix.columns
        
        # Create heatmap
        sj_fig = px.imshow(
            sj_correlation_matrix,
            text_auto='.2f',
            aspect="auto",
            title='Correlation Matrix - San Juan',
            color_continuous_scale=[
                [0, 'darkblue'],
                [0.5, 'white'],
                [1, 'darkred']
            ],
            zmin=-1,
            zmax=1
        )
        
        # Update layout
        sj_fig.update_layout(
            height=600,
            xaxis_title="Variables",
            yaxis_title="Variables",
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.00", "-0.50", "0.00", "0.50", "1.00"]
            )
        )
        
        st.plotly_chart(sj_fig, use_container_width=True)
    
    with corr_col2:
        # Iquitos correlation
        iq_corr_df = df[df['city'] == 'iq']
        iq_corr_df['avg_ndvi'] = iq_corr_df[['ndvi_se', 'ndvi_sw', 'ndvi_ne', 'ndvi_nw']].mean(axis=1)
        
        # Create correlation matrix
        iq_correlation_matrix = iq_corr_df[corr_columns].corr().round(2)
        
        # Rename columns for better readability
        iq_correlation_matrix.columns = [
            'Total Cases',
            'Temperature (°C)',
            'Precipitation (mm)',
            'NDVI',
            'Relative Humidity (%)'
        ]
        iq_correlation_matrix.index = iq_correlation_matrix.columns
        
        # Create heatmap
        iq_fig = px.imshow(
            iq_correlation_matrix,
            text_auto='.2f',
            aspect="auto",
            title='Correlation Matrix - Iquitos',
            color_continuous_scale=[
                [0, 'darkblue'],
                [0.5, 'white'],
                [1, 'darkred']
            ],
            zmin=-1,
            zmax=1
        )
        
        # Update layout
        iq_fig.update_layout(
            height=600,
            xaxis_title="Variables",
            yaxis_title="Variables",
            coloraxis_colorbar=dict(
                title="Correlation",
                tickvals=[-1, -0.5, 0, 0.5, 1],
                ticktext=["-1.00", "-0.50", "0.00", "0.50", "1.00"]
            )
        )
        
        st.plotly_chart(iq_fig, use_container_width=True)
    
    # Add interpretation section below both plots
    st.subheader("Correlation Interpretation")
    st.markdown("""
    The correlation matrices show the relationship between different variables:
    - Values close to 1 indicate strong positive correlation
    - Values close to -1 indicate strong negative correlation
    - Values close to 0 indicate weak or no correlation
    
    Key relationships to observe:
    - How dengue cases correlate with environmental factors
    - Relationships between different environmental factors
    - Seasonal patterns in the correlations
    - Differences between the two cities
    """) 