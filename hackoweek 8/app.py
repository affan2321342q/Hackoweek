import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import plotly.graph_objects as go
from datetime import datetime

# Set page config
st.set_page_config(page_title="Parking Lot Lighting Forecast", layout="wide")

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stAlert {
        background-color: rgba(255, 75, 75, 0.2) !important;
        border: 1px solid #ff4b4b !important;
        color: #ffffff !important;
    }
    .metric-container {
        background: rgba(255, 255, 255, 0.05);
        padding: 20px;
        border-radius: 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    h1, h2, h3 {
        color: #00d4ff !important;
    }
    </style>
""", unsafe_allow_html=True)

# Load Model and Data
@st.cache_resource
def load_assets():
    model = joblib.load('model.joblib')
    poly = joblib.load('poly_transformer.joblib')
    df = pd.read_csv('parking_data.csv')
    return model, poly, df

model, poly, df = load_assets()

# Sidebar
st.sidebar.title("Dashboard Controls")
sim_speed = st.sidebar.slider("Simulation Speed (Seconds per Step)", 0.1, 2.0, 0.5)
reset_sim = st.sidebar.button("Reset Simulation")

if 'step' not in st.session_state or reset_sim:
    st.session_state.step = 0
    st.session_state.history = []

# Main Title
st.title("🅿️ Parking Lot Lighting Forecast System")
st.markdown("### Real-time Sensor Fusion & Polynomial Forecast")

# Placeholders
metrics_placeholder = st.empty()
chart_placeholder = st.empty()
alert_placeholder = st.empty()

# Run Simulation
if st.session_state.step < len(df):
    row = df.iloc[st.session_state.step]
    
    # Predict
    vehicle_count = row['vehicle_count']
    X_poly = poly.transform([[vehicle_count]])
    predicted_usage = model.predict(X_poly)[0]
    actual_usage = row['light_usage']
    
    # Inject anomaly for demonstration (randomly)
    is_anomaly = False
    if np.random.random() < 0.05:
        actual_usage *= 1.5
        is_anomaly = True
    
    # Calculate difference
    diff = abs(actual_usage - predicted_usage)
    error_threshold = 10 # Hardcoded threshold for anomaly
    
    # Store history
    st.session_state.history.append({
        'time': row['timestamp'],
        'vehicle_count': vehicle_count,
        'predicted': predicted_usage,
        'actual': actual_usage,
        'anomaly': is_anomaly or diff > error_threshold
    })
    
    # Keep last 24 entries
    hist_df = pd.DataFrame(st.session_state.history[-24:])
    
    # 1. Metrics
    with metrics_placeholder.container():
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Current Vehicle Count", int(vehicle_count))
            st.markdown('</div>', unsafe_allow_html=True)
        with col2:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Forecasted Light Usage", f"{predicted_usage:.2f} kW")
            st.markdown('</div>', unsafe_allow_html=True)
        with col3:
            st.markdown('<div class="metric-container">', unsafe_allow_html=True)
            st.metric("Actual Light Usage", f"{actual_usage:.2f} kW", delta=f"{actual_usage-predicted_usage:.2f} kW", delta_color="inverse")
            st.markdown('</div>', unsafe_allow_html=True)

    # 2. Bar Chart
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hist_df['time'],
        y=hist_df['predicted'],
        name='Forecasted Usage',
        marker_color='#00d4ff',
        opacity=0.6
    ))
    fig.add_trace(go.Bar(
        x=hist_df['time'],
        y=hist_df['actual'],
        name='Actual Usage',
        marker_color='#ff4b4b'
    ))
    
    fig.update_layout(
        template="plotly_dark",
        margin=dict(l=20, r=20, t=40, b=20),
        title="Lighting Usage Forecast vs Actual (Last 24 Data Points)",
        xaxis_title="Time",
        yaxis_title="Power Usage (kW)",
        barmode='group',
        height=450
    )
    chart_placeholder.plotly_chart(fig, use_container_width=True)

    # 3. Alerts
    if hist_df['anomaly'].any():
        with alert_placeholder:
            st.error(f"⚠️ **Anomaly Detected!** Unusual lighting usage at {row['timestamp']}. Please check sensor health.")
    else:
        alert_placeholder.info("✅ System Operating Normally: Usage matches forecast patterns.")

    # Advance
    st.session_state.step += 1
    time.sleep(sim_speed)
    st.rerun()
else:
    st.success("Simulation Complete!")
    if st.button("Restart Simulation"):
        st.session_state.step = 0
        st.session_state.history = []
        st.rerun()
