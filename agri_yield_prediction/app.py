# app.py
from flask import Flask, request, jsonify
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Function to load the saved model
def load_model():
    try:
        # Load the trained model from the pickle file
        model = joblib.load('/workspaces/FlaskDemo/agri_yield_prediction/models/crop_yield_model.pkl')
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# Endpoint for predictions
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        model = load_model()
        if model is None:
            return jsonify({"error": "Model not found. Please ensure the model is trained and saved."}), 400
        
        # Get input data from the request (expecting JSON)
        data = request.get_json()
        
        # Validate the input data
        if not all(feature in data for feature in ['rainfall', 'temperature', 'soil_quality']):
            return jsonify({"error": "Missing required features. Ensure input includes 'rainfall', 'temperature', and 'soil_quality'."}), 400
        
        # Extract features from the input data
        input_data = [data['rainfall'], data['temperature'], data['soil_quality']]
        
        # Convert input data into a DataFrame for scaling
        input_df = pd.DataFrame([input_data], columns=['rainfall', 'temperature', 'soil_quality'])
        
        # Scale the features using StandardScaler (same as during training)
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)
        
        # Make a prediction using the model
        prediction = model.predict(input_scaled)
        
        # Return the prediction
        return jsonify({"prediction": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Main route to check if the server is running
@app.route('/', methods=['GET'])
def index():
    return jsonify({"message": "Welcome to the Crop Yield Prediction API!"})

if __name__ == '__main__':
    app.run(debug=True)
