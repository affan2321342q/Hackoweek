import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def train_nb():
    df = pd.read_csv('laundry_data.csv')
    
    # Feature engineering for NB
    df['hour'] = pd.to_datetime(df['ds']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['ds']).dt.dayofweek
    
    X = df[['hour', 'day_of_week']]
    y = df['usage_category']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = GaussianNB()
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("Naive Bayes Classification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, 'model_nb.pkl')
    print("Naive Bayes model saved as model_nb.pkl")

if __name__ == "__main__":
    train_nb()
