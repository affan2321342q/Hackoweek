import pandas as pd
import numpy as np
import os

def generate_data(start_date, end_date):
    date_range = pd.date_range(start=start_date, end=end_date, freq='H')
    n_samples = len(date_range)
    
    df = pd.DataFrame({'datetime': date_range})
    
    # Day type: 0=Weekday, 1=Weekend
    df['day_of_week'] = df['datetime'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # We can add holidays randomly or use a library, let's just randomly assign some days as holidays
    np.random.seed(42)
    days = pd.Series(df['datetime'].dt.date.unique())
    n_holidays = int(len(days) * 0.03) # 3% holidays
    holidays = np.random.choice(days, n_holidays, replace=False)
    
    df['is_holiday'] = df['datetime'].dt.date.isin(holidays).astype(int)
    
    df['day_type'] = 'Weekday'
    df.loc[df['is_weekend'] == 1, 'day_type'] = 'Weekend'
    df.loc[df['is_holiday'] == 1, 'day_type'] = 'Holiday'
    
    # Events: typically happen in evenings on weekends or some weekdays
    # Let's say events happen between 18:00 and 22:00
    df['event'] = 0
    df['hour'] = df['datetime'].dt.hour
    
    # Randomly select event days
    event_prob = 0.4 # 40% chance of an event on a given day
    event_days = np.random.choice(days, int(len(days) * event_prob), replace=False)
    
    df.loc[(df['datetime'].dt.date.isin(event_days)) & (df['hour'] >= 18) & (df['hour'] <= 22), 'event'] = 1
    
    # Base electricity usage
    # Higher during day, lower at night
    base_usage = np.sin((df['hour'] - 6) * np.pi / 12) * 50 + 100
    base_usage = np.maximum(base_usage, 40) # min usage 40
    
    # Weekend / holiday multiplier
    multiplier = np.ones(n_samples)
    multiplier[df['day_type'] == 'Weekend'] = 1.2
    multiplier[df['day_type'] == 'Holiday'] = 0.8
    
    usage = base_usage * multiplier
    
    # Add noise
    usage += np.random.normal(0, 10, n_samples)
    
    # Event bump: Huge spike during event
    usage[df['event'] == 1] += 200 + np.random.normal(0, 30, sum(df['event'] == 1))
    
    # Post-Event electricity usage (next 3 hours after event ends)
    # We want to predict this. Let's make it depend on the event intensity implicitly
    for i in range(1, len(df)):
        if df.loc[i-1, 'event'] == 1 and df.loc[i, 'event'] == 0:
            # Event just ended
            # Add tail usage
            usage[i] += 100 + np.random.normal(0, 10)
            if i+1 < len(df):
                usage[i+1] += 50 + np.random.normal(0, 10)
            if i+2 < len(df):
                usage[i+2] += 20 + np.random.normal(0, 5)
    
    df['electricity_usage'] = np.maximum(usage, 0)
    
    return df

if __name__ == '__main__':
    print("Generating synthetic data...")
    # Generate 5 years of hourly data
    start_date = '2019-01-01'
    end_date = '2024-01-01'
    df = generate_data(start_date, end_date)
    os.makedirs('sports_facility_lstm', exist_ok=True)
    df.to_csv('sports_facility_lstm/facility_data.csv', index=False)
    print(f"Dataset generated with {len(df)} records.")
    print(df.head())
