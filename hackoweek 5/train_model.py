import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def train_hvac_model(data_file='hvac_data.csv', model_file='model.pkl'):
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    # One-hot encode the categorical Zone feature
    df_encoded = pd.get_dummies(df, columns=['Zone'], drop_first=False)
    
    # Feature columns and target column
    X = df_encoded.drop('Cooling_Requirement', axis=1)
    y = df_encoded['Cooling_Requirement']
    
    # Save the feature columns for later inference
    feature_cols = X.columns.tolist()
    print("Features used:", feature_cols)
    print("Saving feature columns to 'feature_cols.pkl'...")
    joblib.dump(feature_cols, 'feature_cols.pkl')
    
    print("Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("Training Decision Tree Regressor...")
    # Add max depth to prevent severe overfitting and keep the model size reasonable
    model = DecisionTreeRegressor(max_depth=15, random_state=42)
    model.fit(X_train, y_train)
    
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    
    print(f"Saving model to {model_file}...")
    joblib.dump(model, model_file)
    print("Model training complete.")

if __name__ == '__main__':
    train_hvac_model()
