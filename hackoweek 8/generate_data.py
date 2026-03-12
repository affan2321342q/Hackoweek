import pandas as pd
import numpy as np
import datetime

def generate_parking_data(num_days=30):
    np.random.seed(42)
    start_date = datetime.datetime(2026, 1, 1)
    data = []

    for day in range(num_days):
        current_day = start_date + datetime.timedelta(days=day)
        is_weekend = current_day.weekday() >= 5
        
        for hour in range(24):
            # Base vehicle count pattern (diurnal)
            if 7 <= hour <= 9: # Morning rush
                base_count = np.random.normal(80, 10)
            elif 16 <= hour <= 19: # Evening rush
                base_count = np.random.normal(90, 15)
            elif 10 <= hour <= 15: # Mid-day
                base_count = np.random.normal(40, 10)
            else: # Night/Early morning
                base_count = np.random.normal(10, 5)
            
            if is_weekend:
                base_count *= 0.4 # Less traffic on weekends
            
            vehicle_count = max(0, int(base_count))
            
            # Light usage: Polynomial relationship with vehicle count + base lighting
            # y = a*x^2 + b*x + c + noise
            # Higher vehicle count -> more lights (safety/visibility)
            # Base lighting is higher at night
            base_light = 20 if (hour < 6 or hour > 18) else 5
            light_usage = 0.005 * (vehicle_count**2) + 0.2 * vehicle_count + base_light + np.random.normal(0, 2)
            light_usage = max(0, light_usage)
            
            data.append({
                'timestamp': current_day.replace(hour=hour),
                'hour': hour,
                'vehicle_count': vehicle_count,
                'light_usage': light_usage,
                'is_weekend': int(is_weekend)
            })

    df = pd.DataFrame(data)
    df.to_csv('parking_data.csv', index=False)
    print(f"Generated {len(df)} records in parking_data.csv")

if __name__ == "__main__":
    generate_parking_data()
