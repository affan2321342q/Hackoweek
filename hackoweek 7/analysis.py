import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import pickle

def perform_analysis():
    # 1. Load Data
    df = pd.read_csv('admin_usage_data.csv')
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 2. Preprocess for Clustering (Daily Profiles)
    # Pivot so each row is a day and columns are hours 0-23
    daily_profiles = df.pivot_table(index=df['timestamp'].dt.date, columns='hour', values='usage_kwh')
    
    # 3. K-Means Clustering
    # We'll use 3 clusters: High (Weekday), Low (Weekend), and maybe Medium (Transitional/Holiday)
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(daily_profiles)
    
    # Map clusters back to the original dataframe
    daily_clusters = pd.DataFrame({'date': daily_profiles.index, 'cluster': clusters})
    df['date'] = df['timestamp'].dt.date
    df = df.merge(daily_clusters, on='date')
    
    # 4. Regression for Forecasting
    # Features: hour, day_of_week, cluster
    X = df[['hour', 'day_of_week', 'cluster', 'is_weekend']]
    y = df['usage_kwh']
    
    # Split into train and test (last 30 days for testing)
    split_date = df['timestamp'].max() - pd.Timedelta(days=30)
    train_mask = df['timestamp'] <= split_date
    test_mask = df['timestamp'] > split_date
    
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    regressor = RandomForestRegressor(n_estimators=100, random_state=42)
    regressor.fit(X_train, y_train)
    
    predictions = regressor.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)
    print(f"Regression MAE: {mae:.2f} kWh")
    
    # 5. Save Results and Models
    results = {
        'kmeans': kmeans,
        'regressor': regressor,
        'daily_profiles': daily_profiles,
        'clusters': clusters,
        'mae': mae
    }
    with open('model_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    # Save the dataframe with clusters for the dashboard
    df.to_csv('admin_usage_with_clusters.csv', index=False)
    print("Analysis complete. Models and results saved.")

if __name__ == "__main__":
    perform_analysis()
