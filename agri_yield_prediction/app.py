from flask import Flask, request, render_template, redirect, url_for
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

# Route to display the index page (the form)
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the trained model
        model = load_model()
        if model is None:
            return "Model not found. Please ensure the model is trained and saved.", 500
        
        # Get input data from the form
        rainfall = float(request.form['rainfall'])
        temperature = float(request.form['temperature'])
        soil_quality = float(request.form['soil_quality'])
        
        # Prepare input data for prediction
        input_data = [rainfall, temperature, soil_quality]
        
        # Convert input data into a DataFrame for scaling
        input_df = pd.DataFrame([input_data], columns=['rainfall', 'temperature', 'soil_quality'])
        
        # Scale the features using StandardScaler (same as during training)
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_df)
        
        # Make a prediction using the model
        prediction = model.predict(input_scaled)[0]
        
        # Render the result page with the prediction
        return render_template('result.html', yield_prediction=prediction)
    
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    app.run(debug=True)
