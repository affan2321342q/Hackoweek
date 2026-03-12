import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_synthetic_data(days=7, filename="classroom_electricity_data.csv"):
    """Generates synthetic hourly data for occupancy and electricity draw and saves to CSV."""
    np.random.seed(42)  # For reproducibility
    
    # Generate timestamp index for the last 'days' + current hour
    end_time = datetime.now().replace(minute=0, second=0, microsecond=0)
    start_time = end_time - timedelta(days=days)
    hours = int((end_time - start_time).total_seconds() / 3600)
    
    timestamps = [start_time + timedelta(hours=i) for i in range(hours)]
    
    occupancy = []
    electricity_draw = []
    
    for dt in timestamps:
        hour = dt.hour
        # Base pattern: classes during the day, empty at night
        if 8 <= hour <= 18:
            base_occ = np.random.normal(50, 15)  # Average 50 students
        elif 19 <= hour <= 22:
            base_occ = np.random.normal(15, 5)   # Evening study
        else:
            base_occ = np.random.normal(0, 1)    # Night time
            
        occ = max(0, int(base_occ)) # Occupancy can't be negative
        occupancy.append(occ)
        
        # Electricity draw: Base load + (occupancy * per_person_load) + noise
        base_load = 5.0 # kW
        active_load = occ * 0.15 # 0.15 kW per student (laptops, lights, HVAC load)
        noise = np.random.normal(0, 0.5)
        
        elec = max(0, base_load + active_load + noise)
        electricity_draw.append(round(elec, 2))
        
    df = pd.DataFrame({
        'timestamp': timestamps,
        'occupancy': occupancy,
        'electricity_draw_kw': electricity_draw
    })
    
    df.to_csv(filename, index=False)
    print(f"Dataset successfully saved to {filename}")

if __name__ == "__main__":
    generate_synthetic_data()
