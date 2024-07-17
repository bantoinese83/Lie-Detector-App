import io

from loguru import logger


def display_data_info(train_data, val_data, test_data):
    buffer = io.StringIO()

    # Correctly capturing and logging DataFrame information
    train_data.info(buf=buffer)
    logger.info("\nTraining Data Info:\n{}", buffer.getvalue())
    buffer.truncate(0)
    buffer.seek(0)

    val_data.info(buf=buffer)
    logger.info("\nValidation Data Info:\n{}", buffer.getvalue())
    buffer.truncate(0)
    buffer.seek(0)

    test_data.info(buf=buffer)
    logger.info("\nTesting Data Info:\n{}", buffer.getvalue())
    buffer.truncate(0)
    buffer.seek(0)

    # Logging other information without attempting to use logger as a buffer for DataFrame.info()
    logger.info("\nTraining Data Head:\n{}", train_data.head())
    logger.info("\nValidation Data Head:\n{}", val_data.head())
    logger.info("\nTesting Data Head:\n{}", test_data.head())
    logger.info("\nData Shapes:\nTrain: {}, Validation: {}, Test: {}", train_data.shape, val_data.shape,
                test_data.shape)
    logger.info("\nLabel Distribution in Training Data:\n{}", train_data.is_lying.value_counts())
    logger.info("\nLabel Distribution in Validation Data:\n{}", val_data.is_lying.value_counts())
    logger.info("\nLabel Distribution in Testing Data:\n{}", test_data.is_lying.value_counts())
    logger.info("\nStatistics Summary of Training Data:\n{}", train_data.describe())
    logger.info("\nStatistics Summary of Validation Data:\n{}", val_data.describe())
    logger.info("\nStatistics Summary of Testing Data:\n{}", test_data.describe())
    logger.info("\nMissing Values in Training Data:\n{}", train_data.isnull().sum())
    logger.info("\nMissing Values in Validation Data:\n{}", val_data.isnull().sum())
    logger.info("\nMissing Values in Testing Data:\n{}", test_data.isnull().sum())
    logger.info("\nData Types in Training Data:\n{}", train_data.dtypes)
    logger.info("\nData Types in Validation Data:\n{}", val_data.dtypes)
    logger.info("\nData Types in Testing Data:\n{}", test_data.dtypes)
    logger.info("\nColumn Names in Training Data:\n{}", train_data.columns)
    logger.info("\nColumn Names in Validation Data:\n{}", val_data.columns)
    logger.info("\nColumn Names in Testing Data:\n{}", test_data.columns)
    logger.info("\nTraining Data Memory Usage:\n{}", train_data.memory_usage(deep=True).sum())
    logger.info("\nValidation Data Memory Usage:\n{}", val_data.memory_usage(deep=True).sum())
    logger.info("\nTesting Data Memory Usage:\n{}", test_data.memory_usage(deep=True).sum())
