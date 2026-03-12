import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
import os

st.set_page_config(page_title="Facility Night Usage", layout="wide")

@st.cache_resource
def load_assets():
    with open('scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('lstm_emulator.pkl', 'rb') as f:
        model = pickle.load(f)
    return model, scaler

@st.cache_data
def load_data():
    df = pd.read_csv('facility_data.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    return df

def main():
    st.title("⚡ Sports Facility Post-Event Usage Predictor")
    
    if not os.path.exists('facility_data.csv') and os.path.exists('sports_facility_lstm/facility_data.csv'):
        os.chdir('sports_facility_lstm')

    try:
        model, scaler = load_assets()
        df = load_data()
    except Exception as e:
        st.error(f"Error loading models or data. Please run data generator and training scripts first. {e}")
        st.stop()

    st.sidebar.header("Filters")
    day_types = df['day_type'].unique()
    selected_day_type = st.sidebar.selectbox("Select Day Type", day_types)

    filtered_df = df[df['day_type'] == selected_day_type].copy()
    filtered_df = filtered_df.reset_index(drop=True)
    
    filtered_df['event_shifted'] = filtered_df['event'].shift(1)
    event_end_rows = filtered_df[(filtered_df['event_shifted'] == 1) & (filtered_df['event'] == 0)]
    event_ends = event_end_rows['datetime'].tolist()

    if not event_ends:
        st.warning(f"No events found for {selected_day_type}.")
        st.stop()
        
    event_ends.sort(reverse=True)
    selected_event_end = st.sidebar.selectbox("Select Event End Time", event_ends[:100])

    st.write(f"### Analyzing usage for event ending at: {selected_event_end}")

    end_idx = df[df['datetime'] == selected_event_end].index[0]
    seq_length = 24
    forecast_length = 12

    if end_idx < seq_length:
        st.warning("Not enough historical data before this event.")
        st.stop()

    historical_df = df.iloc[end_idx-seq_length : end_idx]
    future_df = df.iloc[end_idx : end_idx+forecast_length] if end_idx+forecast_length < len(df) else pd.DataFrame()

    features = ['electricity_usage', 'event', 'hour']
    current_seq = historical_df[features].values
    scaled_seq = scaler.transform(current_seq)

    predictions = []
    current_input = scaled_seq.copy()

    for i in range(forecast_length):
        input_flat = current_input.flatten().reshape(1, -1)
        pred_scaled = model.predict(input_flat)[0]
        
        dummy_row = np.zeros((1, len(features)))
        dummy_row[0, 0] = pred_scaled
        pred_usage = scaler.inverse_transform(dummy_row)[0, 0]
        predictions.append(pred_usage)
        
        next_hour = (int(historical_df.iloc[-1]['hour']) + i + 1) % 24
        next_row = np.array([[pred_usage, 0, next_hour]])
        next_scaled_row = scaler.transform(next_row)
        
        current_input = np.vstack([current_input[1:], next_scaled_row])

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=historical_df['datetime'], 
        y=historical_df['electricity_usage'],
        mode='lines+markers',
        name='Historical Usage',
        line=dict(color='#1f77b4', width=3)
    ))

    if not future_df.empty:
        future_datetimes = [historical_df['datetime'].iloc[-1]] + future_df['datetime'].tolist()
        future_usages = [historical_df['electricity_usage'].iloc[-1]] + future_df['electricity_usage'].tolist()
        fig.add_trace(go.Scatter(
            x=future_datetimes, 
            y=future_usages,
            mode='lines+markers',
            name='Actual Post-Event Usage',
            line=dict(color='gray', dash='dash', width=2)
        ))

    pred_datetimes = [historical_df['datetime'].iloc[-1] + pd.Timedelta(hours=i) for i in range(forecast_length + 1)]
    pred_plot_usages = [historical_df['electricity_usage'].iloc[-1]] + predictions

    fig.add_trace(go.Scatter(
        x=pred_datetimes, 
        y=pred_plot_usages,
        mode='lines+markers',
        name='Predicted Usage (Emulated LSTM)',
        line=dict(color='#ff7f0e', width=3)
    ))
    
    event_start_time = historical_df[historical_df['event'] == 1]['datetime'].min()
    if pd.isna(event_start_time):
        event_start_time = selected_event_end - pd.Timedelta(hours=4)

    fig.add_vrect(
        x0=event_start_time, x1=selected_event_end,
        fillcolor="rgba(255, 0, 0, 0.1)", layer="below", line_width=0,
        annotation_text="Event Duration", annotation_position="top left"
    )

    fig.add_vline(x=selected_event_end, line_width=2, line_dash="dash", line_color="green", annotation_text="Event End")

    fig.update_layout(
        title="Post-Event Electricity Usage Forecast",
        xaxis_title="Time",
        yaxis_title="Electricity Usage (kWh)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("### Model Info")
        st.warning("⚠️ **Note:** Scikit-learn MLP Regression was used to emulate LSTM behavior. Both TensorFlow and PyTorch encountered DLL load errors relating to missing Windows C++ redistributables on this system.")
        st.write("- **Architecture:** Flat Sequence -> `Dense` (64) -> `Dense` (32) -> Output (1)")
        st.write("- **Features:** Past 24 hours of [Electricity, Event State, Hour]")
        st.write("- **Forecast Horizon:** 12 hours.")
    
    with col2:
        st.write("### Usage Summary")
        max_pred = max(predictions)
        min_pred = min(predictions)
        st.metric("Peak Post-Event Load Expected", f"{max_pred:.2f} kWh")
        st.metric("Minimum Expected Load", f"{min_pred:.2f} kWh")

if __name__ == '__main__':
    main()
