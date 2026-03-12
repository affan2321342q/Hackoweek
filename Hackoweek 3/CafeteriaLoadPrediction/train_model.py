import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pickle

def train_and_save_model(data_path='cafeteria_data.csv', model_path='model.pkl'):
    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please run generate_data.py first.")
        return
        
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    features = ['day_of_week', 'is_weekend', 'hour_of_day', 'temperature', 'precipitation']
    target = 'load'
    
    X = df[features]
    y = df[target]
    
    print("Training Linear Regression model...")
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', LinearRegression())
    ])
    
    pipeline.fit(X, y)
    
    score = pipeline.score(X, y)
    print(f"Model R^2 score: {score:.4f}")
    
    with open(model_path, 'wb') as f:
        pickle.dump({
            'pipeline': pipeline,
            'features': features
        }, f)
        
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    train_and_save_model()
