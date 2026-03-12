import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_data(days=365):
    start_date = datetime(2025, 1, 1)
    date_list = [start_date + timedelta(hours=x) for x in range(days * 24)]
    
    df = pd.DataFrame(date_list, columns=['ds'])
    df['hour'] = df['ds'].dt.hour
    df['day_of_week'] = df['ds'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    # Base load varies by hour (morning and evening peaks)
    def get_base_load(hour):
        if 7 <= hour <= 10: return 20 # Morning rush
        if 18 <= hour <= 22: return 25 # Evening rush
        if 23 <= hour or hour <= 6: return 2 # Night lull
        return 10 # Afternoon steady
    
    df['base_load'] = df['hour'].apply(get_base_load)
    
    # Add noise and weekend multiplier
    np.random.seed(42)
    df['load_kg'] = df['base_load'] * (1 + 0.5 * df['is_weekend']) + np.random.normal(0, 3, len(df))
    df['load_kg'] = df['load_kg'].clip(lower=0)
    
    # Number of machines active (roughly proportional to load)
    df['machines_active'] = (df['load_kg'] / 5).round().astype(int).clip(upper=10)
    
    # Usage Category for Naive Bayes
    def categorize(load):
        if load < 8: return 'Low'
        if load < 18: return 'Medium'
        return 'High'
    
    df['usage_category'] = df['load_kg'].apply(categorize)
    
    # Drop helper column
    df = df.drop(columns=['base_load'])
    
    df.to_csv('laundry_data.csv', index=False)
    print("Data generated and saved to laundry_data.csv")

if __name__ == "__main__":
    generate_data()
