import pandas as pd
import numpy as np
import os

def generate_hvac_data(num_samples=500000, output_file='hvac_data.csv'):
    np.random.seed(42)
    
    # 5 different lab zones
    zones = ['Zone_A', 'Zone_B', 'Zone_C', 'Zone_D', 'Zone_E']
    
    # Generate zone ids
    zone_data = np.random.choice(zones, size=num_samples)
    
    # Occupancy (number of people in the zone) - typically 0 to 50
    occupancy_data = np.random.poisson(lam=15, size=num_samples)
    occupancy_data = np.clip(occupancy_data, 0, 50)
    
    # Ambient Temperature (°C) - typically 15 to 35
    temperature_data = np.random.normal(loc=25, scale=5, size=num_samples)
    temperature_data = np.clip(temperature_data, 15, 35)

    # Base cooling requirement depends on zone
    zone_base_cooling = {
        'Zone_A': 10,
        'Zone_B': 15,
        'Zone_C': 20,
        'Zone_D': 12,
        'Zone_E': 18
    }
    
    base_cooling_arr = np.array([zone_base_cooling[z] for z in zone_data])
    
    # Calculate cooling requirement: 
    # More people -> more cooling
    # Higher temperature -> more cooling
    # Adding some noise
    cooling_requirement = (
        base_cooling_arr + 
        (occupancy_data * 1.5) + 
        ((temperature_data - 20) * 2.0) +
        np.random.normal(loc=0, scale=3, size=num_samples)
    )
    
    # Cooling requirement cannot be negative
    cooling_requirement = np.clip(cooling_requirement, 0, None)
    
    # Create DataFrame
    df = pd.DataFrame({
        'Zone': zone_data,
        'Occupancy': occupancy_data,
        'Temperature': temperature_data,
        'Cooling_Requirement': cooling_requirement
    })
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Successfully generated {num_samples} samples and saved to {output_file}")

if __name__ == '__main__':
    generate_hvac_data()
