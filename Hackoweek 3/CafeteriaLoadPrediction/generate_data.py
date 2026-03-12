import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_cafeteria_data(num_days=180, output_file='cafeteria_data.csv'):
    np.random.seed(42)
    
    # 180 days, hourly data (180 * 24 = 4320 samples + some extra to easily exceed 5000)
    # Actually let's do 250 days to get 6000 records.
    num_days = 250
    num_samples = num_days * 24
    
    # Start date 250 days ago
    start_date = datetime.now() - timedelta(days=num_days)
    
    dates = [start_date + timedelta(hours=i) for i in range(num_samples)]
    
    # Extract time features
    day_of_week = np.array([d.weekday() for d in dates])  # 0: Monday, 6: Sunday
    hour_of_day = np.array([d.hour for d in dates])
    
    # Weather features
    # Temperature: seasonal variation + daily variation + noise
    base_temp = 20 + 10 * np.sin(np.linspace(0, 4*np.pi, num_samples))
    daily_temp = 5 * np.sin((hour_of_day - 6) * np.pi / 12)
    temperature = base_temp + daily_temp + np.random.normal(0, 2, num_samples)
    
    # Precipitation: sporadic rain, mostly 0, occasionally up to 10mm
    precipitation = np.random.exponential(1, num_samples) * (np.random.rand(num_samples) > 0.8)
    # Clip precipitation to 0 when it's supposed to be dry
    precipitation = np.where(precipitation < 0.5, 0, precipitation)
    
    # Cafeteria load (base)
    # Higher on weekdays, lower on weekends
    is_weekend = (day_of_week >= 5).astype(int)
    weekend_penalty = is_weekend * 70
    
    # Hourly load pattern
    hourly_base_load = np.zeros(num_samples)
    for i, h in enumerate(hour_of_day):
        if 11 <= h <= 14: # Lunch rush
            hourly_base_load[i] = np.random.normal(250, 40)
        elif 17 <= h <= 19: # Dinner rush
            hourly_base_load[i] = np.random.normal(120, 25)
        elif 8 <= h <= 10: # Breakfast/Morning
            hourly_base_load[i] = np.random.normal(60, 15)
        elif 15 <= h <= 16: # Afternoon lull
            hourly_base_load[i] = np.random.normal(40, 10)
        else: # Late night / early morning
            hourly_base_load[i] = np.random.normal(5, 5)
            
    # Weather impact
    # Precipitation -> MORE people stay in cafeteria
    weather_impact = (precipitation * 8) - (temperature - 20) * 0.3
    
    # Combine features
    load = hourly_base_load + weather_impact - weekend_penalty + np.random.normal(0, 10, num_samples)
    
    # Clean values: no negative load
    load = np.maximum(0, np.round(load)).astype(int)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'temperature': np.round(temperature, 1),
        'precipitation': np.round(precipitation, 2),
        'load': load
    })
    
    df.to_csv(output_file, index=False)
    print(f"Dataset generated successfully with {len(df)} rows: {output_file}")

if __name__ == "__main__":
    generate_cafeteria_data()
