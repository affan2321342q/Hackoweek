import pandas as pd
from prophet import Prophet
import joblib

def train_prophet():
    df = pd.read_csv('laundry_data.csv')
    
    # Prophet requires 'ds' and 'y' columns
    prophet_df = df[['ds', 'load_kg']].rename(columns={'load_kg': 'y'})
    
    model = Prophet(yearly_seasonality=True, daily_seasonality=True)
    model.fit(prophet_df)
    
    # Save the model
    joblib.dump(model, 'model_prophet.pkl')
    print("Prophet model saved as model_prophet.pkl")

if __name__ == "__main__":
    train_prophet()
