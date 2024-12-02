import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def load_data(self):
        """
        Load dataset from a fixed path: 'data/crop_data.csv'.
        """
        file_path = '/workspaces/FlaskDemo/agri_yield_prediction/data/crop_data.csv'
        try:
            data = pd.read_csv(file_path)
            print("Data loaded successfully from 'data/crop_data.csv'.")
            return data
        except Exception as e:
            print(f"Error loading data from {file_path}: {e}")
            return None

    def clean_data(self, data):
        """
        Handle missing values and remove outliers.
        """
        # Handle missing values (if any)
        if data.isnull().sum().sum() > 0:
            data = data.fillna(data.mean())
            print("Missing values filled with column means.")

        # Remove outliers (optional, based on your needs)
        # Remove rows where any value is > 3 standard deviations
        z_scores = np.abs((data - data.mean()) / data.std())
        data = data[(z_scores < 3).all(axis=1)]
        print("Outliers removed.")
        
        return data

    def scale_features(self, features):
        """
        Standardize numerical features using StandardScaler.
        """
        scaled_features = self.scaler.fit_transform(features)
        print("Features scaled successfully.")
        return scaled_features

    def preprocess(self):
        """
        Full preprocessing pipeline for 'data/crop_data.csv'.
        """
        # Step 1: Load data
        data = self.load_data()
        if data is None:
            return None, None

        # Step 2: Clean data
        data = self.clean_data(data)

        # Step 3: Separate features and target
        features = data[['rainfall', 'temperature', 'soil_quality']]
        target = data['yield']

        # Step 4: Scale features
        features_scaled = self.scale_features(features)

        return features_scaled, target


# Example usage
if __name__ == "__main__":
    preprocessor = Preprocessor()
    features, target = preprocessor.preprocess()

    if features is not None and target is not None:
        print("Processed Features (first 5 rows):\n", features[:5])
        print("Target (first 5 rows):\n", target[:5])
