import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

def create_sequences(data, seq_length):
    xs = []
    ys = []
    for i in range(len(data)-seq_length):
        x = data[i:(i+seq_length)].flatten() # Flatten for MLP to emulate sequence reading
        y = data[i+seq_length, 0] 
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def train_model():
    print("Loading data...")
    print("NOTE: TensorFlow and PyTorch are failing on this machine due to missing Windows C++ Redistributable DLLs.")
    print("Falling back to scikit-learn MLP regression to emulate LSTM (acting on flattened sequence windows).")
    
    data_path = 'facility_data.csv'
    if not os.path.exists(data_path):
        data_path = 'sports_facility_lstm/facility_data.csv'
    df = pd.read_csv(data_path)
    
    features = ['electricity_usage', 'event', 'hour']
    data = df[features].values
    
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    save_dir = 'sports_facility_lstm' if os.path.exists('sports_facility_lstm/facility_data.csv') else '.'
    with open(os.path.join(save_dir, 'scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    
    seq_length = 24
    
    print("Creating sequences...")
    # Train on a smaller subset to make sklearn MLP training fast
    subset_data = scaled_data[-8000:] 
    X, y = create_sequences(subset_data, seq_length)
    
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    
    print("Training model... (using MLP to emulate simple LSTM)")
    # Emulate the complexity of a small LSTM
    model = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', max_iter=20, random_state=42)
    model.fit(X_train, y_train)
    
    print(f"Training R^2 score: {model.score(X_train, y_train):.4f}")
    print(f"Test R^2 score: {model.score(X_test, y_test):.4f}")
    
    with open(os.path.join(save_dir, 'lstm_emulator.pkl'), 'wb') as f:
        pickle.dump(model, f)
    print(f"Model saved to {os.path.join(save_dir, 'lstm_emulator.pkl')}")

if __name__ == '__main__':
    train_model()
