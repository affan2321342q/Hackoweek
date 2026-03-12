import asyncio
import json
import pickle
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Attempt to load the pre-trained model on startup
try:
    with open('model.pkl', 'rb') as f:
        model_data = pickle.load(f)
        model_pipeline = model_data['pipeline']
        model_features = model_data['features']
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model (it might not exist yet): {e}")
    model_pipeline = None
    model_features = []

@app.get("/", response_class=HTMLResponse)
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    # We will simulate time moving faster for the demo purposes.
    # Start from current time
    current_sim_time = datetime.now()
    
    # Generate some historical data to populate the chart initially (past 12 steps)
    historical_data = []
    
    for i in range(12, 0, -1):
        hist_time = current_sim_time - timedelta(minutes=5 * i)
        
        day_of_week = hist_time.weekday()
        hour = hist_time.hour + (hist_time.minute / 60)
        is_weekend = 1 if day_of_week >= 5 else 0
        temperature = 22 + 5 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 1)
        precipitation = 0
        
        features_dict = {
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'hour_of_day': [hour],
            'temperature': [temperature],
            'precipitation': [precipitation]
        }
        
        if model_pipeline:
            X_pred = pd.DataFrame(features_dict)
            pred_load = max(0, int(model_pipeline.predict(X_pred)[0]))
        else:
            pred_load = 50 # Fallback
            
        actual_load = max(0, int(pred_load + np.random.normal(0, 15)))
            
        historical_data.append({
            'timestamp': hist_time.strftime("%H:%M"),
            'actual_load': actual_load,
            'predicted_load': pred_load,
            'temperature': round(temperature, 1)
        })
        
    await websocket.send_text(json.dumps({'type': 'history', 'data': historical_data}))

    # Stream real-time data
    while True:
        current_sim_time += timedelta(minutes=5) # Advance time by 5 minutes per tick
        
        day_of_week = current_sim_time.weekday()
        hour = current_sim_time.hour + (current_sim_time.minute / 60)
        is_weekend = 1 if day_of_week >= 5 else 0
        
        temperature = 22 + 5 * np.sin((hour - 6) * np.pi / 12) + np.random.normal(0, 1)
        precipitation = 0
        if np.random.random() > 0.95:
            precipitation = np.random.uniform(0.5, 5)
            
        features_dict = {
            'day_of_week': [day_of_week],
            'is_weekend': [is_weekend],
            'hour_of_day': [hour],
            'temperature': [temperature],
            'precipitation': [precipitation]
        }
        
        if model_pipeline:
            X_pred = pd.DataFrame(features_dict)
            pred_load = max(0, int(model_pipeline.predict(X_pred)[0]))
        else:
            pred_load = 50
            
        # Add some noise to the predicted value to simulate "actual" load
        actual_load = max(0, int(pred_load + np.random.normal(0, 10)))
            
        data_point = {
            'type': 'update',
            'data': {
                'timestamp': current_sim_time.strftime("%H:%M"),
                'actual_load': actual_load,
                'predicted_load': pred_load,
                'temperature': round(temperature, 1)
            }
        }
        
        await websocket.send_text(json.dumps(data_point))
        await asyncio.sleep(2) # Send real-time update every 2 seconds
