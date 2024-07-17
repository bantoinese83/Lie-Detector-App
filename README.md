# Lie Detector App 🕵️‍♀️

This project implements a Streamlit web application for simulating a lie detector interrogation. The app uses a machine learning model to analyze physiological data from a simulated "subject" and predict whether they are telling the truth or lying.

## Project Structure

The project is structured into the following modules:

- **`synthetic_dataset_generator_v1.py`:** This module generates synthetic data for training and testing the lie detector model.
- **`data_display.py`:** This module provides functions for displaying the synthetic data.
- - **`train.py`:** This module contains functions for training and evaluating the machine learning model.
    - **`load_data(train_path, val_path, test_path)`:** Loads data from parquet files.
    - **`preprocess_data(data, target_column='is_lying')`:** Separates features and target variables from the DataFrame.
    - **`scale_data(train_features, val_features, test_features)`:** Applies standardization to the features.
    - **`feature_selection(train_features, train_target, feature_names)`:** Performs feature selection using a Random Forest model.
    - **`filter_selected_features(features, selected_features)`:** Filters features based on feature selection results.
    - **`display_data_info(training_data, validation_data, testing_data)`:** Displays information about the data, including shapes and summaries.
    - **`train_model(train_features, train_target)`:** Trains a Random Forest model with GridSearchCV for hyperparameter tuning.
    - **`evaluate_model(model, val_features, val_target, test_features, test_target)`:** Evaluates the model on the validation and test sets, calculating metrics such as accuracy, precision, recall, and F1-score.
    - **`save_model(model, filename)`:** Saves the trained model to disk.
- **`human_subject.py`:** This module defines the `HumanSubject` class, which simulates a person being interrogated.
    - **`HumanSubject(update_interval: int = 1, lie_probability: float = 0.3, stress_sensitivity: float = 1.0, baseline_noise: float = 0.1, feature_ranges: dict = None)`:**  Initializes the `HumanSubject` with parameters for controlling its physiological responses and  behavior.
    - **`simulate_response(self, is_lying: bool)`:** Simulates a physiological response based on whether the subject is lying.
    - **`update_features(self)`:** Updates the features with random fluctuations and noise, mimicking real-time sensor data.
    - **`get_features(self)`:** Returns the current physiological features.
    - **`simulate(self, duration: int)`:** Runs the simulation for the specified duration, simulating responses to a series of questions.
- **`streamlit_app.py`:** This module contains the Streamlit web application.
    - It provides a user interface for controlling the simulation, displaying the real-time physiological data, and presenting the results.
- **`flask_app.py`:** This module contains a Flask API for making predictions using the trained model.

## Usage

1. **Data Generation:** Run `synthetic_dataset_generator_v1.py` to generate synthetic data for training and testing the lie detector model.
2. **Model Training:** Run `train.py` to train and save the model.
3. **Run the App:** Execute `streamlit run streamlit_app.py` to run the Streamlit app.
4. **Flask API:** Run `flask_app.py` to start the Flask API server.

## Features

- **Interactive Simulation:**  Control the lie probability, stress sensitivity, and baseline noise of the simulated subject to observe how they affect the physiological responses.
- **Real-time Data Visualization:**  Visualize the heart rate, heart rate variability, and electrodermal activity in real-time as the interrogation progresses.
- **Lie Detection Predictions:**  View the model's prediction of the subject's truthfulness in real-time.
- **Session Summary:**  Get a summary of the interrogation, including average heart rate, HRV, and EDA values.
- **Histograms:**  Explore the distribution of the physiological data with interactive histograms.
- **Flask API:**  A Flask API is available for making predictions using the trained model.

## Requirements

- Python 3.x
- Streamlit
- Flask
- scikit-learn
- pandas
- matplotlib
- scipy

## Installation

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
   
2. Run the Streamlit app:
   ```bash
    streamlit run streamlit_app.py
    ```
      
3. Start the Flask API:
4. ```bash
     python flask_app.py
     ```


## Acknowledgements
- Streamlit
- Flask
- scikit-learn
- pandas
- matplotlib
- scipy
- NumPy
