import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import joblib
import os

st.set_page_config(page_title="HVAC Optimization Dashboard", layout="wide")

# Load model and feature columns
@st.cache_resource
def load_assets():
    model = joblib.load('model.pkl')
    feature_cols = joblib.load('feature_cols.pkl')
    return model, feature_cols

try:
    model, feature_cols = load_assets()
    model_loaded = True
except Exception as e:
    st.error(f"Error loading model: {e}. Please run data generation and model training scripts first.")
    model_loaded = False

st.title("🌡️ Lab HVAC Optimization Dashboard")
st.markdown("Forecast cooling needs for different laboratory zones using a Decision Tree model trained on occupancy and temperature data.")

if model_loaded:
    st.sidebar.header("Input Conditions")
    st.sidebar.markdown("Adjust conditions below to see real-time cooling predictions.")
    
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    
    # Let user set input for each zone or simulate random current conditions
    input_mode = st.sidebar.radio("Input Mode", ["Manual Entry", "Simulate Live Data"])
    
    input_data = []
    
    if input_mode == "Manual Entry":
        for zone in zones:
            st.sidebar.subheader(zone)
            col1, col2 = st.sidebar.columns(2)
            temp = col1.slider(f"Temp (°C)", 15.0, 35.0, 25.0, key=f"temp_{zone}")
            occ = col2.slider(f"Occupancy", 0, 50, 15, key=f"occ_{zone}")
            input_data.append({'Zone': zone, 'Temperature': temp, 'Occupancy': occ})
    else:
        st.sidebar.markdown("Generating random live data...")
        if st.sidebar.button("Refresh Data"):
            pass # Reruns script
        
        for zone in zones:
            temp = np.random.uniform(18.0, 32.0)
            occ = int(np.random.poisson(15))
            occ = min(max(occ, 0), 50)
            input_data.append({'Zone': zone, 'Temperature': temp, 'Occupancy': occ})

    df_input = pd.DataFrame(input_data)
    
    # Prepare data for prediction
    # Ensure one-hot encoding matches the training features
    df_encoded = pd.get_dummies(df_input, columns=['Zone'])
    
    # Add missing columns with 0
    for col in feature_cols:
        if col not in df_encoded.columns:
            df_encoded[col] = 0
            
    # Reorder columns to match training data
    X_pred = df_encoded[feature_cols]
    
    # Make Prediction
    predictions = model.predict(X_pred)
    df_input['Predicted_Cooling'] = predictions
    
    # Main Dashboard Display
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Current Settings & Predictions")
        st.dataframe(df_input.style.highlight_max(axis=0, subset=['Predicted_Cooling'], color='#ff4b4b'), use_container_width=True)
        
        total_cooling = df_input['Predicted_Cooling'].sum()
        st.metric("Total Facility Cooling Demand", f"{total_cooling:.2f} kW")
        
    with col2:
        st.subheader("Cooling Demand Heatmap")
        
        # We'll create a dummy spatial representation for the zones
        # Mapping zones to mock X, Y coordinates
        coords = {
            'Zone_A': (1, 1),
            'Zone_B': (1, 2),
            'Zone_C': (2, 1),
            'Zone_D': (2, 2),
            'Zone_E': (1.5, 1.5)
        }
        
        df_input['X'] = df_input['Zone'].map(lambda z: coords[z][0])
        df_input['Y'] = df_input['Zone'].map(lambda z: coords[z][1])
        
        # Use a scatter plot with square markers to simulate a heatmap of a floor plan
        fig = px.scatter(
            df_input, 
            x='X', 
            y='Y', 
            color='Predicted_Cooling',
            size='Predicted_Cooling',
            text='Zone',
            hover_data=['Temperature', 'Occupancy', 'Predicted_Cooling'],
            color_continuous_scale='RdYlBu_r',
            range_x=[0.5, 2.5],
            range_y=[0.5, 2.5],
            title="Lab Floor Plan - Cooling Requirements"
        )
        
        fig.update_traces(marker=dict(symbol='square'), textposition='top center')
        fig.update_layout(
            xaxis=dict(showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            plot_bgcolor='rgba(0,0,0,0)'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Model Insights")
    st.info("The Decision Tree model heavily weights the `Zone` and `Temperature` features to determine cooling needs. High occupancy adds secondary load. Adjusting the sliders to lower temperatures or lower occupancies immediately reduces the predicted cooling requirements.")
