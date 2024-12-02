# modelt.py

import pandas as pd
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def train_model():
    try:
        # Load dataset
        data = pd.read_csv('/workspaces/FlaskDemo/agri_yield_prediction/data/crop_data.csv')
        
        # Check if data is loaded correctly
        print(f"Data loaded successfully, shape of data: {data.shape}")
        
        # Preprocessing (example)
        features = data[['rainfall', 'temperature', 'soil_quality']]
        target = data['yield']
        
        # Check for missing values
        if features.isnull().sum().any() or target.isnull().sum() > 0:
            raise ValueError("There are missing values in the data. Please handle them.")
        
        print(f"Features: {features.head()}")
        print(f"Target: {target.head()}")
        
        # Feature scaling
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        print(f"Features scaled successfully.")
        
        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(features_scaled, target)
        print("Model training complete.")
        
        # Save the trained model as a pickle file
        model_filename = '/workspaces/FlaskDemo/agri_yield_prediction/models/crop_yield_model.pkl'
        joblib.dump(model, model_filename)
        print(f"Model saved at {model_filename}")
        
    except Exception as e:
        print(f"An error occurred: {e}")

# Call the train_model function to train and save the model
if __name__ == "__main__":
    print("Starting model training...")
    train_model()
