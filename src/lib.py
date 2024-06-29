import os
import uuid
import json
import pickle
import logging

import pandas as pd
import numpy as np

from typing import List, Optional, Tuple
from pydantic import BaseModel, Field
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from src.utils import Config, default_config

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DataFrameValidator(BaseModel):
    carat: List[float] = Field(...)
    cut: List[str] = Field(...)
    color: List[str] = Field(...)
    clarity: List[str] = Field(...)
    x: List[float] = Field(...)
    y: List[float] = Field(...)
    z: List[float] = Field(...)
    price: List[int] = Field(...)

    @classmethod
    def validate_dataframe(cls, df: pd.DataFrame) -> None:
        """
        Validates if the DataFrame has the expected columns with the correct types.

        Args:
            df (pd.DataFrame): The DataFrame to validate.

        Raises:
            ValueError: If the DataFrame does not have the expected columns or has unexpected columns.
            ValidationError: If the DataFrame columns do not have the expected types.
        """
        logger.info("Validating input DataFrame.")
        expected_columns = cls.__annotations__.keys()

        df_columns = set(df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = expected_columns_set - df_columns
        unexpected_columns = df_columns - expected_columns_set

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        if unexpected_columns:
            logger.error(f"Unexpected columns: {unexpected_columns}")
            raise ValueError(f"Unexpected columns: {unexpected_columns}")

        for column, expected_type in cls.__annotations__.items():
            python_type = expected_type.__args__[0]
            if not all(isinstance(value, python_type) for value in df[column]):
                logger.error(f"Column '{column}' does not have all values of type '{python_type.__name__}'")
                raise ValueError(f"Column '{column}' does not have all values of type '{python_type.__name__}'")

        data = df.to_dict(orient='list')
        cls(**data)
        logger.info("Input DataFrame validation completed successfully.")


class Data:
    def __init__(self, data: pd.DataFrame, config: Config = default_config):
        """
        Initializes the Data object with input data and configuration.

        Args:
            data (pd.DataFrame): The input data for training.
            config (Config): The configuration object.
        """
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None

        self.data = data
        self.config = config

        self._input_validation()
        self._load_validation_data(self.config.get('validation_x_path'), self.config.get('validation_y_path'))

    def _input_validation(self) -> None:
        """
        Validates the input DataFrame.
        """
        logger.info("Starting input validation.")
        DataFrameValidator.validate_dataframe(self.data)
        logger.info("Input validation completed.")

    def _load_validation_data(self, validation_X_path: str, validation_y_path: str) -> None:
        """
        Loads validation data from the specified paths.

        Args:
            validation_X_path (str): Path to the validation features CSV file.
            validation_y_path (str): Path to the validation target CSV file.
        """
        logger.info(f"Loading validation data from {validation_X_path} and {validation_y_path}.")
        self.X_val = pd.read_csv(validation_X_path, sep=';', index_col=0)
        self.y_val = pd.read_csv(validation_y_path, sep=';', index_col=0)
        logger.info("Validation data loaded successfully.")

    def preprocess(self, get_dummies: bool = True, log_y: bool = True) -> None:
        """
        Preprocesses the training data.

        Args:
            get_dummies (bool): Whether to convert categorical variables to dummy/indicator variables.
            log_y (bool): Whether to apply logarithmic transformation to the target variable.
        """
        logger.info("Starting preprocessing of training data.")
        self.data = self.data[self.data.x * self.data.y * self.data.z != 0]
        self.data = self.data[self.data.price > 0]
        self.y_train = self.data['price']

        if get_dummies:
            logger.info("Applying get_dummies to categorical columns.")
            self.data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'], drop_first=True)

        if log_y:
            logger.info("Applying logarithmic transformation to the target variable.")
            self.y_train = np.log(self.y_train)

        self.X_train = self.data[self.config.get('training_features')]
        logger.info("Preprocessing completed.")


class Model:
    def __init__(self):
        """
        Initializes the Model object.
        """
        self.model = LinearRegression()
        self.model_name = type(self.model).__name__
        self.model_params = self.model.get_params()

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
        """
        logger.info("Starting model training.")
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: pd.DataFrame, log_y: bool = True) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame): The features for prediction.
            log_y (bool): Whether the target variable was logarithmically transformed during training.

        Returns:
            np.ndarray: The predicted values.
        """
        logger.info("Making predictions.")
        if log_y:
            return np.exp(self.model.predict(X))
        else:
            return self.model.predict(X)

    def evaluate(self, y: pd.Series, y_predicted: np.ndarray) -> Tuple[float, float]:
        """
        Evaluates the model performance.

        Args:
            y (pd.Series): The true values.
            y_predicted (np.ndarray): The predicted values.

        Returns:
            Tuple[float, float]: The R2 score and mean absolute error of the predictions.
        """
        logger.info("Evaluating model performance.")
        r2 = round(r2_score(y, y_predicted), 4)
        mae = round(mean_absolute_error(y, y_predicted), 2)
        logger.info(f"Model evaluation completed. R2: {r2}, MAE: {mae}")
        return r2, mae


class TrainingPipeline:
    def __init__(self, input_data: pd.DataFrame, config: Config = default_config):
        """
        Initializes the TrainingPipeline object with input data and configuration.

        Args:
            input_data (pd.DataFrame): The input data for training.
            config (Config): The configuration object.
        """
        self.config = config
        self.data = Data(input_data, config)
        self.data.preprocess(log_y=self.config.get('log_y'))

        self.model = Model()
        os.makedirs(self.config.get("experiment_directory"), exist_ok=True)

    def train(self) -> Tuple[str, float, float]:
        """
        Trains the model and logs the experiment.

        Returns:
            Tuple[str, float, float]: The experiment ID, R2 score, and mean absolute error.
        """
        logger.info("Starting training pipeline.")
        self.model.train(self.data.X_train, self.data.y_train)
        validation_preds = self.model.predict(self.data.X_val, log_y=self.config.get('log_y'))
        r2, mae = self.model.evaluate(self.data.y_val, validation_preds)

        experiment_id = self._log_experiment(r2, mae)
        self._save_model(experiment_id)

        logger.info("Training pipeline completed.")
        return experiment_id, r2, mae

    def _log_experiment(self, r2: float, mae: float) -> str:
        """
        Logs the experiment results.

        Args:
            r2 (float): The R2 score.
            mae (float): The mean absolute error.

        Returns:
            str: The experiment ID.
        """
        logger.info("Logging experiment results.")
        experiment_id = str(uuid.uuid4())
        timestamp = datetime.now()
        experiment_record = {
            "uuid": experiment_id,
            "timestamp": timestamp.isoformat(),
            "date": timestamp.strftime("%Y-%m-%d %H:%M"),
            "model_name": self.model.model_name,
            "model_params": self.model.model_params,
            "results": {
                "r2": r2,
                "mae": mae
            }
        }

        experiment_file = self.config.get("experiment_file")
        if os.path.exists(experiment_file):
            with open(experiment_file, "r") as file:
                experiments = json.load(file)
        else:
            experiments = []

        experiments.append(experiment_record)

        with open(experiment_file, "w") as file:
            json.dump(experiments, file, indent=4)

        logger.info("Experiment results logged.")
        return experiment_id

    def _save_model(self, experiment_id: str) -> None:
        """
        Saves the trained model to a file.

        Args:
            experiment_id (str): The experiment ID.
        """
        logger.info(f"Saving model to directory {experiment_id}.")
        model_dir = os.path.join(self.config.get("experiment_directory"), experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)
        logger.info("Model saved successfully.")