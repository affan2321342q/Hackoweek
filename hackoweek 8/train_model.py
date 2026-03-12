import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_model():
    # Load data
    df = pd.read_csv('parking_data.csv')
    
    X = df[['vehicle_count']].values
    y = df['light_usage'].values
    
    # Polynomial features (Degree 2)
    poly = PolynomialFeatures(degree=2)
    X_poly = poly.fit_transform(X)
    
    # Fit model
    model = LinearRegression()
    model.fit(X_poly, y)
    
    # Evaluate
    y_pred = model.predict(X_poly)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)
    
    print(f"Model trained. MSE: {mse:.4f}, R2: {r2:.4f}")
    
    # Save model and transformer
    joblib.dump(model, 'model.joblib')
    joblib.dump(poly, 'poly_transformer.joblib')
    print("Model and transformer saved.")

if __name__ == "__main__":
    train_model()
