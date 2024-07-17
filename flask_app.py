import logging

import joblib
import pandas as pd
from flask import Flask, request, jsonify
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the model at the start
try:
    model = joblib.load('best_rf_model.pkl')
    logging.info("Model loaded successfully.")
except Exception as e:
    model = None
    logging.error(f"Failed to load model: {e}")


# Define a function to preprocess the input data
def preprocess_data(features):
    features = pd.DataFrame([features])  # Create a DataFrame from the input list
    features.columns = ['heart_rate', 'heart_rate_variability', 'electrodermal_activity', 'blood_oxygen_level',
                        'accelerometer', 'gyroscope', 'skin_temperature']
    # Preprocessing (Scaling)
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    return scaled_features


@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"message": "Model is not loaded.", "predictions": None}), 500

    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({"message": "Missing 'features' in request.", "predictions": None}), 400

        features = data['features']
        if not isinstance(features, list):
            return jsonify({"message": "'features' should be a list.", "predictions": None}), 400

        # Preprocess the input features
        scaled_features = preprocess_data(features)
        predictions = model.predict(scaled_features)
        return jsonify({"message": "Prediction successful", "predictions": predictions.tolist()}), 200
    except Exception as error:
        logging.error(f"Error during prediction: {error}")
        return jsonify({"message": "Error processing the request.", "predictions": None}), 500


if __name__ == '__main__':
    app.run(debug=True)
