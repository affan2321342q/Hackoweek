import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta

# Set page config
st.set_page_config(page_title="Classroom Electricity Forecasting", layout="wide")

st.title("Classroom Usage Forecasting Dashboard")
st.markdown("Forecasts next-hour room electricity draw using Wi-Fi occupancy logs.")

# --- 1. Data Generation ---
@st.cache_data
def generate_synthetic_data(days=7):
    """Generates synthetic hourly data for occupancy and electricity draw."""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamp index for the last 'days' + current hour
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)
    hours = int((end_time - start_time).total_seconds() / 3600)
    
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    occupancy = []
    electricity_draw = []
    
    for dt in timestamps:
        hour = dt.hour
        # Base pattern: classes during the day, empty at night
        if 8 <= hour <= 18:
            base_occ = np.random.normal(50, 15)  # Average 50 students
        elif 19 <= hour <= 22:
            base_occ = np.random.normal(15, 5)   # Evening study
        else:
            base_occ = np.random.normal(0, 1)    # Night time
            
        occ = max(0, int(base_occ)) # Occupancy can't be negative
        occupancy.append(occ)
        
        # Electricity draw: Base load + (occupancy * per_person_load) + noise
        base_load = 5.0 # kW
        active_load = occ * 0.15 # 0.15 kW per student (laptops, lights, HVAC load)
        noise = np.random.normal(0, 0.5)
        
        elec = max(0, base_load + active_load + noise)
        electricity_draw.append(elec)
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'occupancy': occupancy,
        'electricity_draw_kw': electricity_draw
    })
    df.set_index('timestamp', inplace=True)
    return df

with st.spinner("Loading data..."):
    df = generate_synthetic_data()

# Show current status
latest_time = df.index[-1]
latest_occ = df['occupancy'].iloc[-1]
latest_elec = df['electricity_draw_kw'].iloc[-1]

col1, col2, col3 = st.columns(3)
col1.metric("Latest Update", latest_time.strftime("%Y-%m-%d %H:00"))
col2.metric("Current Occupancy", f"{latest_occ} students")
col3.metric("Current Electricity Draw", f"{latest_elec:.2f} kW")

st.divider()

# --- 2. Model Training and Prediction ---
st.subheader("Next-Hour Forecast")

with st.spinner("Training ARIMAX model..."):
    # Target variable (endogenous)
    y = df['electricity_draw_kw']
    # Predictor variable (exogenous)
    X = df['occupancy']
    
    # Fit ARIMAX model (p=1, d=0, q=1 with exogenous variable)
    # Using a simple ARMA(1,1) error structure for the residual
    model = ARIMA(endog=y, exog=X, order=(1, 0, 1))
    results = model.fit()

# Define next hour's expected occupancy (can be interactive)
st.markdown("### Simulation Parameters")
expected_occupancy = st.slider(
    "Expected Occupancy for Next Hour (Simulation):", 
    min_value=0, 
    max_value=150, 
    value=int(X.iloc[-1]),
    help="Adjust this slider to see how occupancy affects the electricity forecast."
)

# Predict next hour
forecast = results.get_forecast(steps=1, exog=np.array([[expected_occupancy]]))
pred_mean = forecast.predicted_mean.iloc[0]
conf_int = forecast.conf_int(alpha=0.05) # 95% confidence interval
lower_bound = conf_int.iloc[0, 0]
upper_bound = conf_int.iloc[0, 1]

col_f1, col_f2 = st.columns(2)
col_f1.metric(
    "Forecasted Electricity Draw",
    f"{pred_mean:.2f} kW",
    delta=f"{pred_mean - latest_elec:.2f} kW vs current",
    delta_color="inverse"
)
col_f2.metric(
    "95% Confidence Interval",
    f"[{max(0, lower_bound):.2f} kW - {upper_bound:.2f} kW]"
)

st.divider()

# --- 3. Dashboard Visualizations ---
st.subheader("Historical Data & Forecast Visualization")

# Plotly chart
fig = go.Figure()

# Plot Historical Electricity
fig.add_trace(go.Scatter(
    x=df.index, 
    y=df['electricity_draw_kw'],
    mode='lines',
    name='Actual Electricity Draw (kW)',
    line=dict(color='blue')
))

# Plot Historical Occupancy (on secondary y-axis)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['occupancy'],
    mode='lines',
    name='Occupancy (Students)',
    line=dict(color='orange', dash='dot'),
    yaxis='y2'
))

# Add Forecast Point
next_hour = latest_time + timedelta(hours=1)
fig.add_trace(go.Scatter(
    x=[df.index[-1], next_hour],
    y=[df['electricity_draw_kw'].iloc[-1], pred_mean],
    mode='lines+markers',
    name='Forecast (kW)',
    line=dict(color='red', dash='dash'),
    marker=dict(size=8, symbol='star')
))

# Add Confidence Interval for Forecast
fig.add_trace(go.Scatter(
    x=[next_hour, next_hour],
    y=[max(0, lower_bound), upper_bound],
    mode='lines',
    name='95% Confidence Interval',
    line=dict(color='rgba(255, 0, 0, 0.5)', width=4)
))

# Update layout for dual axis
fig.update_layout(
    title="Electricity Draw & Occupancy Over the Past Week",
    xaxis=dict(title="Time"),
    yaxis=dict(
        title=dict(text="Electricity Draw (kW)", font=dict(color="blue")),
        tickfont=dict(color="blue")
    ),
    yaxis2=dict(
        title=dict(text="Occupancy (Count)", font=dict(color="orange")),
        tickfont=dict(color="orange"),
        anchor="x",
        overlaying="y",
        side="right"
    ),
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
)

st.plotly_chart(fig, use_container_width=True)

# Data table View
with st.expander("View Raw Data Model Resume"):
    st.dataframe(df.sort_index(ascending=False).head(24))
    st.markdown("**Model Details:**")
    st.text(results.summary())
