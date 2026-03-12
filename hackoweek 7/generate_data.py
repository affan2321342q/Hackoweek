import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_usage_data(days=365):
    start_date = datetime(2025, 1, 1)
    date_list = [start_date + timedelta(hours=x) for x in range(days * 24)]
    
    data = []
    for dt in date_list:
        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        is_weekend = day_of_week >= 5
        
        # Base usage: Admin buildings have higher usage during work hours (8 AM - 6 PM)
        if 8 <= hour <= 18:
            base = 50 + np.random.normal(0, 5)
        else:
            base = 10 + np.random.normal(0, 2)
            
        # Weekend dip: Usage is significantly lower on weekends
        if is_weekend:
            usage = base * 0.2 + np.random.normal(0, 1) # 80% reduction
        else:
            usage = base
            
        # Seasonal variation (optional, but makes it realistic)
        # Higher usage in summer (AC) and winter (Heating)
        month = dt.month
        if month in [6, 7, 8]: # Summer
            usage *= 1.3
        elif month in [12, 1, 2]: # Winter
            usage *= 1.2
            
        data.append({
            'timestamp': dt,
            'usage_kwh': max(0, usage),
            'hour': hour,
            'day_of_week': day_of_week,
            'is_weekend': int(is_weekend)
        })
        
    df = pd.DataFrame(data)
    df.to_csv('admin_usage_data.csv', index=False)
    print("Data generated and saved to admin_usage_data.csv")
    return df

if __name__ == "__main__":
    generate_usage_data()
