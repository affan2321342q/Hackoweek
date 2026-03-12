import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Admin Building Energy Dashboard", layout="wide")

st.title("Admin Building Energy Usage Analysis")
st.markdown("### Weekend Dip Detection & Forecasting")

# Load data and results
@st.cache_data
def load_data():
    df = pd.read_csv('admin_usage_with_clusters.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    with open('model_results.pkl', 'rb') as f:
        results = pickle.load(f)
    return df, results

df, results = load_data()

# --- Sidebar ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Overview & Forecast", "Clustering Analysis", "Savings Potential"])

# --- Helper for Forecast ---
def get_forecast(df, results):
    regressor = results['regressor']
    last_date = df['timestamp'].max()
    next_7_days = [last_date + timedelta(hours=x) for x in range(1, 24 * 7 + 1)]
    
    forecast_df = pd.DataFrame({'timestamp': next_7_days})
    forecast_df['hour'] = forecast_df['timestamp'].dt.hour
    forecast_df['day_of_week'] = forecast_df['timestamp'].dt.weekday
    forecast_df['is_weekend'] = (forecast_df['day_of_week'] >= 5).astype(int)
    
    # Estimate cluster (simplified: assume typical cluster for day of week)
    cluster_map = df.groupby('is_weekend')['cluster'].agg(lambda x: x.value_counts().index[0]).to_dict()
    forecast_df['cluster'] = forecast_df['is_weekend'].map(cluster_map)
    
    X = forecast_df[['hour', 'day_of_week', 'cluster', 'is_weekend']]
    forecast_df['usage_kwh'] = regressor.predict(X)
    return forecast_df

if page == "Overview & Forecast":
    st.header("Energy Usage & Forecast")
    
    # Actual vs Forecast (for last 14 days)
    last_14_days = df[df['timestamp'] > df['timestamp'].max() - pd.Timedelta(days=14)]
    
    forecast_df = get_forecast(df, results)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=last_14_days['timestamp'], y=last_14_days['usage_kwh'], name='Actual Usage', line=dict(color='#1f77b4', width=2)))
    fig.add_trace(go.Scatter(x=forecast_df['timestamp'], y=forecast_df['usage_kwh'], name='Forecasted Usage', line=dict(color='#ff7f0e', dash='dash', width=2)))
    
    fig.update_layout(
        title="Last 14 Days Actual vs Next 7 Days Forecast",
        xaxis_title="Time",
        yaxis_title="Usage (kWh)",
        template="plotly_dark",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Model MAE", f"{results['mae']:.2f} kWh")
    with col2:
        st.metric("Peak Usage (L30D)", f"{df.tail(24*30)['usage_kwh'].max():.1f} kWh")
    with col3:
        st.metric("Total Usage (L30D)", f"{df.tail(24*30)['usage_kwh'].sum():.0f} kWh")

elif page == "Clustering Analysis":
    st.header("Daily Usage Profiles (K-Means Clusters)")
    
    daily_profiles = results['daily_profiles']
    clusters = results['clusters']
    
    profiles_with_clusters = daily_profiles.copy()
    profiles_with_clusters['cluster'] = clusters
    
    avg_profiles = profiles_with_clusters.groupby('cluster').mean()
    
    fig = go.Figure()
    colors = ['#00CC96', '#EF553B', '#636EFA']
    for i, cluster_id in enumerate(avg_profiles.index):
        mean_val = avg_profiles.loc[cluster_id].mean()
        if mean_val > 30: name = "Peak Pattern (Weekday)"
        elif mean_val < 15: name = "Dipped Pattern (Weekend)"
        else: name = "Baseline/Mixed Pattern"
        
        fig.add_trace(go.Scatter(x=avg_profiles.columns, y=avg_profiles.loc[cluster_id], 
                                 name=f"Cluster {cluster_id}: {name}",
                                 line=dict(color=colors[i % len(colors)], width=3)))
        
    fig.update_layout(
        title="Avg. Hourly Profile per Cluster",
        xaxis_title="Hour of Day",
        yaxis_title="Usage (kWh)",
        template="plotly_dark",
        xaxis=dict(tickmode='linear', tick0=0, dtick=2)
    )
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("#### Cluster Occurrence")
    cluster_counts = pd.Series(clusters).value_counts().reset_index()
    cluster_counts.columns = ['Cluster ID', 'Occurrence (Days)']
    st.table(cluster_counts)

elif page == "Savings Potential":
    st.header("Energy Savings Potential")
    
    # Calculation
    weekend_usage = df[df['is_weekend'] == 1]['usage_kwh'].sum()
    weekday_usage = df[df['is_weekend'] == 0]['usage_kwh'].sum()
    
    # Potential savings: Assume we can reduce weekend idle load by 40%
    potential_savings = weekend_usage * 0.4
    essential_load = (weekend_usage - potential_savings) + weekday_usage
    
    col1, col2 = st.columns(2)
    
    with col1:
        pie_df = pd.DataFrame({
            'Category': ['Essential Load', 'Optimization Potential'],
            'kWh': [essential_load, potential_savings]
        })
        fig = px.pie(pie_df, values='kWh', names='Category', 
                     title="Usage vs Optimization Potential",
                     color_discrete_sequence=['#2ca02c', '#d62728'],
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
        
    with col2:
        # Pie chart by Cluster
        usage_by_cluster = df.groupby('cluster')['usage_kwh'].sum().reset_index()
        usage_by_cluster['Label'] = usage_by_cluster['cluster'].apply(lambda x: f"Profile Type {x}")
        
        fig2 = px.pie(usage_by_cluster, values='usage_kwh', names='Label', 
                      title="Energy Stake by Growth Cluster",
                      color_discrete_sequence=px.colors.qualitative.Pastel)
        st.plotly_chart(fig2, use_container_width=True)
    
    st.success(f"**Actionable Insight:** By optimizing HVAC and lighting on weekends, we can reduce total annual energy consumption by approximately **{(potential_savings / (weekend_usage + weekday_usage) * 100):.1f}%**.")
