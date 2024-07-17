import logging

import joblib
import pandas as pd
import progressbar
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(train_path, val_path, test_path):
    logger.info("Loading data from parquet files.")
    train_data = pd.read_parquet(train_path)
    val_data = pd.read_parquet(val_path)
    test_data = pd.read_parquet(test_path)
    return train_data, val_data, test_data


def preprocess_data(data, target_column='is_lying'):
    if target_column not in data.columns:
        raise ValueError(f"Target column '{target_column}' not found in data")

    features = data.drop([target_column], axis=1)
    target = data[target_column]
    return features, target


def scale_data(train_features, val_features, test_features):
    logger.info("Scaling data.")
    scaler = StandardScaler()
    scaled_train_features = scaler.fit_transform(train_features)
    scaled_val_features = scaler.transform(val_features)
    scaled_test_features = scaler.transform(test_features)
    return scaled_train_features, scaled_val_features, scaled_test_features


def feature_selection(train_features, train_target, feature_names):
    logger.info("Performing feature selection.")
    rf_feature_selector = RandomForestClassifier(random_state=42)
    rf_feature_selector.fit(train_features, train_target)
    feature_importances = rf_feature_selector.feature_importances_

    # Use feature names to filter selected features
    selected_features = [feature_names[i] for i, importance in enumerate(feature_importances) if importance > 0.01]
    return selected_features


def filter_selected_features(features, selected_features):
    logger.info("Filtering selected features.")
    return features[selected_features]


def train_model(train_features, train_target):
    logger.info("Training model.")
    rf = RandomForestClassifier(random_state=42)
    rf.fit(train_features, train_target)

    logger.info("Performing hyperparameter tuning with GridSearchCV.")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(train_features, train_target)

    best_rf = grid_search.best_estimator_
    return best_rf


def evaluate_model(model, val_features, val_target, test_features, test_target):
    logger.info("Evaluating model on validation set.")
    val_preds = model.predict(val_features)
    val_accuracy = accuracy_score(val_target, val_preds)
    val_precision = precision_score(val_target, val_preds)
    val_recall = recall_score(val_target, val_preds)
    val_f1 = f1_score(val_target, val_preds)

    logger.info(f"Validation Accuracy: {val_accuracy:.4f}")
    logger.info(f"Validation Precision: {val_precision:.4f}")
    logger.info(f"Validation Recall: {val_recall:.4f}")
    logger.info(f"Validation F1-score: {val_f1:.4f}")

    logger.info("Evaluating model on test set.")
    test_preds = model.predict(test_features)
    test_accuracy = accuracy_score(test_target, test_preds)
    test_precision = precision_score(test_target, test_preds)
    test_recall = recall_score(test_target, test_preds)
    test_f1 = f1_score(test_target, test_preds)

    logger.info(f"Test Accuracy: {test_accuracy:.4f}")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1-score: {test_f1:.4f}")


def save_model(model, filename):
    logger.info("Saving model.")
    joblib.dump(model, filename)


def main():
    with progressbar.ProgressBar(max_value=10) as bar:
        bar.update(1)
        train_data, val_data, test_data = load_data('data/train_data.parquet', 'data/val_data.parquet',
                                                    'data/test_data.parquet')
        bar.update(2)

        train_features, train_target = preprocess_data(train_data, 'is_lying')
        val_features, val_target = preprocess_data(val_data, 'is_lying')
        test_features, test_target = preprocess_data(test_data, 'is_lying')
        bar.update(3)

        # Extract feature names before scaling
        feature_names = train_features.columns.tolist()

        scaled_train_features, scaled_val_features, scaled_test_features = scale_data(train_features, val_features,
                                                                                      test_features)
        bar.update(4)

        # Pass feature names to feature_selection
        selected_features = feature_selection(scaled_train_features, train_target, feature_names)
        bar.update(5)

        train_features_selected = filter_selected_features(train_features, selected_features)
        val_features_selected = filter_selected_features(val_features, selected_features)
        test_features_selected = filter_selected_features(test_features, selected_features)
        bar.update(6)

        scaled_train_features_selected, scaled_val_features_selected, scaled_test_features_selected = \
            scale_data(train_features_selected, val_features_selected, test_features_selected)
        bar.update(7)

        best_rf = train_model(scaled_train_features_selected, train_target)
        bar.update(8)

        evaluate_model(best_rf, scaled_val_features_selected, val_target, scaled_test_features_selected, test_target)
        bar.update(9)

        save_model(best_rf, 'best_rf_model.pkl')
        bar.update(10)


if __name__ == '__main__':
    main()
