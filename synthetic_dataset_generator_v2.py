import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data_display import display_data_info

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def generate_loan_approval_data(n_samples=10000, random_state=42):
    if n_samples <= 0:
        raise ValueError("n_samples must be a positive integer")

    np.random.seed(random_state)

    logging.info("Generating synthetic features for loan approval prediction...")

    # Generating synthetic features for loan approval prediction
    credit_score = np.random.normal(700, 50, size=n_samples)  # Credit score
    annual_income = np.random.normal(50000, 10000, size=n_samples)  # Annual income in USD
    loan_amount = np.random.normal(200000, 50000, size=n_samples)  # Loan amount requested in USD
    years_in_job = np.random.randint(1, 30, size=n_samples)  # Years in a current job
    age = np.random.randint(21, 70, size=n_samples)  # Age of the applicant
    own_home = np.random.choice([0, 1], size=n_samples, p=[0.6, 0.4])  # Owning a home or not

    # Additional features
    employment_type = np.random.choice(['Full-time', 'Part-time', 'Self-employed', 'Unemployed'], size=n_samples)
    education_level = np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], size=n_samples)
    marital_status = np.random.choice(['Single', 'Married', 'Divorced', 'Widowed'], size=n_samples)
    previous_loans_count = np.random.randint(0, 5, size=n_samples)  # Number of previous loans taken

    logging.info("Generating synthetic labels for loan approval...")

    # Generating synthetic labels for loan approval (assuming binary classification)
    approved_loan = np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])  # 30% approved, 70% not approved

    logging.info("Adding noise to the features...")

    # Adding noise to the features based on the label
    credit_score[approved_loan == 1] += np.random.normal(50, 10, size=approved_loan.sum())
    annual_income[approved_loan == 1] += np.random.normal(10000, 5000, size=approved_loan.sum())
    loan_amount[approved_loan == 1] += np.random.normal(50000, 10000, size=approved_loan.sum())
    years_in_job[approved_loan == 1] -= np.random.randint(1, 10, size=approved_loan.sum())
    age[approved_loan == 1] += np.random.randint(1, 5, size=approved_loan.sum())
    own_home[approved_loan == 1] += np.random.choice([0, 1], size=approved_loan.sum(), p=[0.4, 0.6])

    logging.info("Creating DataFrame for loan approval prediction...")

    # Creating a DataFrame for loan approval prediction
    data = pd.DataFrame({
        'credit_score': credit_score,
        'annual_income': annual_income,
        'loan_amount': loan_amount,
        'years_in_job': years_in_job,
        'age': age,
        'own_home': own_home,
        'employment_type': employment_type,
        'education_level': education_level,
        'marital_status': marital_status,
        'previous_loans_count': previous_loans_count,
        'approved_loan': approved_loan
    })

    logging.info(f"Generated {n_samples} samples of synthetic loan approval prediction data")
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

    # Generate synthetic loan approval prediction data
    data = generate_loan_approval_data(n_samples=n_samples, random_state=random_state)

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
    output_dir = "loan_approval_data"

    training_data, validation_data, testing_data = main(n_samples=n_samples, random_state=random_state,
                                                        test_size=test_size, validation_size=validation_size,
                                                        output_dir=output_dir)

    display_data_info(training_data, validation_data, testing_data)
