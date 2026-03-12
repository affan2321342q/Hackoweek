import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.holtwinters import ExponentialSmoothing

st.set_page_config(page_title="Library Energy Forecast", layout="wide", page_icon="⚡")

# Generate Large Synthetic Dataset
@st.cache_data
def generate_data():
    # Load dataset
    df_raw = pd.read_csv("library_energy_data.csv")
    df_raw["Datetime"] = pd.to_datetime(df_raw["Datetime"])
    
    # Resample to daily
    df_daily = df_raw.set_index("Datetime").resample("D").agg({
        "Energy_Usage_kWh": "sum",
        "Is_Exam": "max"
    }).reset_index()
    
    df_daily.rename(columns={"Datetime": "Date"}, inplace=True)
    return df_daily

df = generate_data()

st.title("⚡ Library Energy Forecast During Exams")
st.markdown("Aggregate historical usage with event calendars; implement exponential smoothing for semester-end forecasts.")

# Forecasting with Exponential Smoothing
def train_and_forecast(df, forecast_days=30):
    # Prepare time series
    ts_data = df.set_index('Date')['Energy_Usage_kWh']
    # Add freq
    ts_data = ts_data.asfreq('D')
    
    # Fit Exponential Smoothing Model (Holt-Winters)
    # Using additive trend and seasonality (weekly = 7)
    model = ExponentialSmoothing(ts_data, trend="add", seasonal="add", seasonal_periods=7, initialization_method="estimated")
    fit_model = model.fit()
    
    forecast = fit_model.forecast(forecast_days)
    
    # Create forecast dataframe
    forecast_df = pd.DataFrame({
        "Date": forecast.index,
        "Forecast_kWh": forecast.values
    })
    return ts_data, forecast_df

with st.spinner("Training Exponential Smoothing model and generating forecasts..."):
    historical_ts, forecast_df = train_and_forecast(df, forecast_days=30)
    
# Calculate metrics for the gauge
latest_usage = historical_ts.iloc[-1]
forecast_peak = forecast_df["Forecast_kWh"].max()
forecast_avg = forecast_df["Forecast_kWh"].mean()

# Layout
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Semester-End Peak Forecast")
    # Gauge Chart for forecasted peak vs capacity
    max_capacity = 45000  # Assume 45000 kWh is max facility capacity
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = forecast_peak,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Forecasted Peak Energy (kWh)", 'font': {'size': 20}},
        delta = {'reference': latest_usage, 'increasing': {'color': "red"}, 'decreasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, max_capacity], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 25000], 'color': "lightgreen"},
                {'range': [25000, 35000], 'color': "yellow"},
                {'range': [35000, max_capacity], 'color': "salmon"}],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 40000}
        }
    ))
    fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)
    
    st.metric("Avg Forecasted Daily Usage", f"{forecast_avg:.1f} kWh")
    st.metric("Historical Daily Average", f"{df['Energy_Usage_kWh'].mean():.1f} kWh")

with col2:
    st.subheader("Historical Usage & Exponential Smoothing Forecast")
    
    # Plot historical (last 180 days for better visibility) and forecast
    plot_df = df.tail(180).copy()
    
    fig = go.Figure()
    
    # Historical Line
    fig.add_trace(go.Scatter(
        x=plot_df["Date"], 
        y=plot_df["Energy_Usage_kWh"],
        mode='lines',
        name='Historical Usage',
        line=dict(color='blue')
    ))
    
    # Highlight Exam periods in historical data
    exam_df = plot_df[plot_df["Is_Exam"] == 1]
    if not exam_df.empty:
        fig.add_trace(go.Scatter(
            x=exam_df["Date"],
            y=exam_df["Energy_Usage_kWh"],
            mode='markers',
            name='Exam Period',
            marker=dict(color='red', size=6)
        ))
        
    # Forecast Line
    fig.add_trace(go.Scatter(
        x=forecast_df["Date"],
        y=forecast_df["Forecast_kWh"],
        mode='lines',
        name='Forecast (Next 30 Days)',
        line=dict(color='orange', dash='dash')
    ))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Energy Usage (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        hovermode="x unified"
    )
    
    st.plotly_chart(fig, use_container_width=True)

st.divider()

st.subheader("Historical Dataset Sample")
st.dataframe(df.tail(20), use_container_width=True)
