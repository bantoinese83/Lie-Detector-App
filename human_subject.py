import logging
import math
import random
import time
from typing import List, Tuple

from scipy.stats import truncnorm

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# Define a custom distribution for feature fluctuations
def truncated_normal(mean, stddev, lower, upper):
    return truncnorm.rvs((lower - mean) / stddev, (upper - mean) / stddev, loc=mean, scale=stddev)


class HumanSubject:
    def __init__(self, update_interval: int = 1, lie_probability: float = 0.3,
                 stress_sensitivity: float = 1.0, baseline_noise: float = 0.1,
                 feature_ranges: dict = None) -> None:
        self.prob_lie = lie_probability
        self.update_interval: int = update_interval
        self.stress_sensitivity = stress_sensitivity
        self.baseline_noise = baseline_noise

        # Define feature ranges (default or user-specified)
        self.feature_ranges = {
            'heart_rate': (60, 100),  # Adjusted range for heart rate
            'hrv': (30, 70),
            'eda': (0.5, 1.5),
            'spo2': (90, 100),
            'accelerometer': (-1, 1),
            'gyroscope': (-1, 1),
            'skin_temperature': (31, 33)
        }
        if feature_ranges is not None:
            self.feature_ranges.update(feature_ranges)

        # Initialize features with random values within their ranges
        self.features: dict = {feature: random.uniform(low, high)
                               for feature, (low, high) in self.feature_ranges.items()}

        self.time_elapsed: float = 0.0
        self.questions_answers: List[Tuple[str, bool]] = [
            ("Did you visit Jamaica Queens on the night of the robbery?", False),
            ("Have you ever owned a black hoodie?", True),
            ("Were you aware of the robbery before it was reported?", False),
            ("Do you know anyone who lives in Jamaica Queens?", True),
            ("Have you ever been convicted of a felony?", False),
            ("Do you own any firearms?", True),
            ("Have you ever been involved in a physical altercation?", False),
            ("Do you have a criminal record?", True),
            ("Were you involved in any illegal activity last month?", False),
            ("Do you use drugs or alcohol regularly?", True)
        ]
        self.answers: List[bool] = []

    def simulate_response(self, is_lying: bool) -> None:
        # Determine lie/truth based on lie probability
        if random.random() < self.prob_lie:
            is_lying = True

        if is_lying:
            # Increase physiological stress indicators based on stress sensitivity
            # **Important Correction: ** Apply truncated_normal to the CURRENT value of heart_rate,
            # not the stress sensitivity directly.
            self.features['heart_rate'] = truncated_normal(mean=self.features['heart_rate'],
                                                           stddev=self.stress_sensitivity * 3,
                                                           lower=self.feature_ranges['heart_rate'][0],
                                                           upper=self.feature_ranges['heart_rate'][1])
            self.features['eda'] += truncated_normal(mean=self.stress_sensitivity * 0.2, stddev=0.05,
                                                     lower=self.feature_ranges['eda'][0],
                                                     upper=self.feature_ranges['eda'][1])
            self.features['hrv'] -= truncated_normal(mean=self.stress_sensitivity * 5, stddev=2,
                                                     lower=self.feature_ranges['hrv'][0],
                                                     upper=self.feature_ranges['hrv'][1])
        else:
            # Introduce random noise to simulate calmness
            self.features['heart_rate'] += truncated_normal(mean=self.baseline_noise * 2, stddev=1,
                                                            lower=self.feature_ranges['heart_rate'][0],
                                                            upper=self.feature_ranges['heart_rate'][1])
            self.features['eda'] += truncated_normal(mean=self.baseline_noise * 0.05, stddev=0.02,
                                                     lower=self.feature_ranges['eda'][0],
                                                     upper=self.feature_ranges['eda'][1])
            self.features['hrv'] += truncated_normal(mean=self.baseline_noise * 2, stddev=1,
                                                     lower=self.feature_ranges['hrv'][0],
                                                     upper=self.feature_ranges['hrv'][1])

    def update_features(self) -> None:
        try:
            # Update features with noise and random fluctuations
            for feature, (low, high) in self.feature_ranges.items():
                if feature == 'heart_rate' or feature == 'hrv' or feature == 'eda':
                    # These features have already been adjusted during simulate_response
                    continue
                elif feature == 'accelerometer':
                    # Sinusoidal pattern for accelerometer
                    self.features['accelerometer'] = math.sin(self.time_elapsed) * 0.5
                elif feature == 'gyroscope':
                    # Cosine pattern for gyroscope
                    self.features['gyroscope'] = math.cos(self.time_elapsed) * 0.5
                else:
                    # Apply noise and random fluctuations to other features
                    self.features[feature] = truncated_normal(mean=self.features[feature], stddev=self.baseline_noise,
                                                              lower=low, upper=high)
            self.time_elapsed += self.update_interval
        except Exception as e:
            logging.error(f"Error updating features: {e}")

    def get_features(self) -> List[float]:
        self.update_features()
        return [self.features[feature] for feature in [
            'heart_rate', 'hrv', 'eda', 'spo2', 'accelerometer', 'gyroscope', 'skin_temperature'
        ]]

    def simulate(self, duration: int) -> None:
        if duration <= 0:
            logging.error("Duration must be positive")
            return

        start_time = time.time()
        try:
            while time.time() - start_time < duration:
                for question, is_truth in self.questions_answers:
                    logging.info(f"Question: {question}")
                    self.simulate_response(is_truth)
                    current_features = self.get_features()
                    logging.info(f"Current Features: {current_features}")
                    time.sleep(self.update_interval)
        except Exception as e:
            logging.error(f"Error during simulation: {e}")
