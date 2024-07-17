import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_display import display_data_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_lie_detection_data(n_samples=10000, random_state=42):
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    np.random.seed(random_state)

    logging.info("Generating synthetic features...")

    # Generating synthetic features based on Apple Watch sensors
    heart_rate = np.random.normal(70, 10, size=n_samples)  # Normal heart rate
    heart_rate_variability = np.random.normal(50, 15, size=n_samples)  # HRV in ms
    electrodermal_activity = np.random.normal(1, 0.3, size=n_samples)  # EDA in microsiemens
    blood_oxygen_level = np.random.normal(95, 2, size=n_samples)  # SpO2 in percentage
    accelerometer = np.random.normal(0, 1, size=n_samples)  # Accelerometer magnitude
    gyroscope = np.random.normal(0, 1, size=n_samples)  # Gyroscope magnitude
    skin_temperature = np.random.normal(32, 1, size=n_samples)  # Skin temperature in Celsius

    logging.info("Generating synthetic labels...")

    # Generating synthetic labels
    is_lying = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # Assuming 30% lying and 70% truth

    logging.info("Adding noise to the features...")

    # Adding some noise to the features based on the label
    heart_rate[is_lying == 1] += np.random.normal(10, 5, size=is_lying.sum())
    heart_rate_variability[is_lying == 1] -= np.random.normal(10, 5, size=is_lying.sum())
    electrodermal_activity[is_lying == 1] += np.random.normal(0.5, 0.2, size=is_lying.sum())
    blood_oxygen_level[is_lying == 1] -= np.random.normal(2, 1, size=is_lying.sum())
    accelerometer[is_lying == 1] += np.random.normal(1, 0.5, size=is_lying.sum())
    gyroscope[is_lying == 1] += np.random.normal(1, 0.5, size=is_lying.sum())
    skin_temperature[is_lying == 1] += np.random.normal(1, 0.5, size=is_lying.sum())

    logging.info("Creating DataFrame...")

    # Creating a DataFrame
    data = pd.DataFrame({
        'heart_rate': heart_rate,
        'heart_rate_variability': heart_rate_variability,
        'electrodermal_activity': electrodermal_activity,
        'blood_oxygen_level': blood_oxygen_level,
        'accelerometer': accelerometer,
        'gyroscope': gyroscope,
        'skin_temperature': skin_temperature,
        'is_lying': is_lying
    })

    logging.info(f"Generated {n_samples} samples of synthetic data")
    return data


def split_data(data, test_size=0.2, validation_size=0.2, random_state=42):
    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1")
    if not (0 < validation_size < 1):
        raise ValueError("validation_size must be between 0 and 1")

    logging.info("Splitting data into training and testing sets...")

    # Splitting the data into training and testing sets
    training_data, testing_data = train_test_split(data, test_size=test_size, random_state=random_state)

    logging.info("Further splitting training data into training and validation sets...")

    # Further splitting the training data into training and validation sets
    training_data, validation_data = train_test_split(training_data, test_size=validation_size,
                                                      random_state=random_state)

    logging.info(
        f"Split data into training ({training_data.shape}), validation ({validation_data.shape}), and testing ({testing_data.shape}) sets")
    return training_data, validation_data, testing_data


def save_data_to_parquet(data, file_path):
    try:
        logging.info(f"Saving data to {file_path}...")
        data.to_parquet(file_path, index=False)
        logging.info(f"Data saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving data to {file_path}: {e}")
        raise


def main(n_samples=10000, random_state=42, test_size=0.2, validation_size=0.2, output_dir="."):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Generate synthetic data
    data = generate_lie_detection_data(n_samples=n_samples, random_state=random_state)

    # Split data into train, validation, and test sets
    training_data, validation_data, testing_data = split_data(data, test_size=test_size,
                                                              validation_size=validation_size,
                                                              random_state=random_state)

    # Save the data to disk
    train_data_path = os.path.join(output_dir, "train_data.parquet")
    val_data_path = os.path.join(output_dir, "val_data.parquet")
    test_data_path = os.path.join(output_dir, "test_data.parquet")

    save_data_to_parquet(training_data, train_data_path)
    save_data_to_parquet(validation_data, val_data_path)
    save_data_to_parquet(testing_data, test_data_path)

    # Return the data
    training_data = pd.read_parquet(train_data_path)
    validation_data = pd.read_parquet(val_data_path)
    testing_data = pd.read_parquet(test_data_path)

    return training_data, validation_data, testing_data


if __name__ == "__main__":
    n_samples = 10000
    random_state = 42
    test_size = 0.2
    validation_size = 0.2
    output_dir = "data"

    training_data, validation_data, testing_data = main(n_samples=n_samples, random_state=random_state,
                                                        test_size=test_size, validation_size=validation_size,
                                                        output_dir=output_dir)

    display_data_info(training_data, validation_data, testing_data)
