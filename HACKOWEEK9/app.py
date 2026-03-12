import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from datetime import datetime, timedelta

# Page config
st.set_page_config(page_title="Hostel Laundry Analytics", layout="wide")

# Custom CSS for premium look
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #ffffff;
    }
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #00f2fe 0%, #4facfe 100%);
    }
    .metric-card {
        background-color: #1e2130;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #3e4150;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    model_nb = joblib.load('model_nb.pkl')
    model_prophet = joblib.load('model_prophet.pkl')
    return model_nb, model_prophet

@st.cache_data
def load_data():
    df = pd.read_csv('laundry_data.csv')
    df['ds'] = pd.to_datetime(df['ds'])
    return df

st.title("🧺 Hostel Laundry Peak Prediction Dashboard")
st.markdown("---")

# Load models and data
try:
    nb_model, prophet_model = load_models()
    data = load_data()
except Exception as e:
    st.error(f"Error loading models or data: {e}. Please ensure you ran the training scripts.")
    st.stop()

# Sidebar for What-If Scenarios
st.sidebar.header("🕹️ What-If Scenarios")
days_to_forecast = st.sidebar.slider("Forecast Timeline (Days)", 1, 30, 7)
load_multiplier = st.sidebar.slider("Occupancy / Usage Multiplier", 0.5, 2.0, 1.0, 0.1)

# Generate Forecast
future = prophet_model.make_future_dataframe(periods=days_to_forecast * 24, freq='H')
forecast = prophet_model.predict(future)

# Apply load multiplier to future values
mask = forecast['ds'] > data['ds'].max()
forecast.loc[mask, 'yhat'] *= load_multiplier
forecast.loc[mask, 'yhat_lower'] *= load_multiplier
forecast.loc[mask, 'yhat_upper'] *= load_multiplier

# Classification using Naive Bayes for future data
future_features = pd.DataFrame({
    'hour': forecast['ds'].dt.hour,
    'day_of_week': forecast['ds'].dt.dayofweek
})
forecast['predicted_category'] = nb_model.predict(future_features)

# Layout
col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📈 Laundry Load Forecast")
    fig = go.Figure()
    
    # Actuals (last 7 days of data)
    recent_actuals = data[data['ds'] > data['ds'].max() - pd.Timedelta(days=7)]
    fig.add_trace(go.Scatter(x=recent_actuals['ds'], y=recent_actuals['load_kg'], 
                             name="Recent Actuals", line=dict(color='#00f2fe', width=2)))
    
    # Forecast
    future_forecast = forecast[forecast['ds'] > data['ds'].max()]
    fig.add_trace(go.Scatter(x=future_forecast['ds'], y=future_forecast['yhat'], 
                             name="Forecasted Load", line=dict(color='#ff007f', width=3, dash='dash')))
    
    # Confidence Intervals
    fig.add_trace(go.Scatter(x=future_forecast['ds'].tolist() + future_forecast['ds'].tolist()[::-1],
                             y=future_forecast['yhat_upper'].tolist() + future_forecast['yhat_lower'].tolist()[::-1],
                             fill='toself', fillcolor='rgba(255, 0, 127, 0.2)',
                             line=dict(color='rgba(255, 255, 255, 0)'),
                             hoverinfo="skip", showlegend=False))
                             
    fig.update_layout(template="plotly_dark", height=500, margin=dict(l=20, r=20, t=50, b=20),
                      xaxis_title="Timeline", yaxis_title="Load (kg)")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🚨 Peak Heatmap")
    peak_counts = future_forecast['predicted_category'].value_counts().reset_index()
    peak_counts.columns = ['Category', 'Hours']
    
    fig_pie = px.pie(peak_counts, values='Hours', names='Category', 
                     color='Category', color_discrete_map={'High': '#ff0000', 'Medium': '#ffa500', 'Low': '#00ff00'},
                     hole=0.4)
    fig_pie.update_layout(template="plotly_dark", height=300, showlegend=True, margin=dict(l=0, r=0, t=30, b=0))
    st.plotly_chart(fig_pie, use_container_width=True)
    
    # Info metrics
    st.markdown("---")
    max_load = future_forecast['yhat'].max()
    st.metric("Predicted Max Load", f"{max_load:.2f} kg", delta=f"{((load_multiplier-1)*100):.1f}%")
    
    avg_load = future_forecast['yhat'].mean()
    st.metric("Average Load", f"{avg_load:.2f} kg")

# Detailed hourly breakdown table
if st.checkbox("Show Hourly Forecast Data"):
    st.dataframe(future_forecast[['ds', 'yhat', 'predicted_category']].rename(
        columns={'ds': 'Time', 'yhat': 'Predicted Load (kg)', 'predicted_category': 'Category'}
    ).tail(24))

st.info("The Naive Bayes model categorizes usage into Low/Medium/High, while Prophet provides the numerical forecast. The slider allows you to simulate capacity changes.")
