import os
import uuid
import json
import pickle
import logging

import pandas as pd
import numpy as np
import xgboost as xgb

from abc import ABC, abstractmethod
from typing import Tuple
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

from src.data import Data
from src.utils import Config, default_config

base_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BaseModel(ABC):
    model = None
    config: Config = default_config

    def __init__(self, model_name: str, load_trained_model: bool = True, config: Config = default_config) -> None:
        """
        Initializes the Model object.
        """
        self.model_name = model_name

        self.config = config
        self.model_config = self.config.get_model_config(self.model_name)

        if load_trained_model:
            self.model = self.load_best_model(self.model_name)
            self.model_params = self.model.get_params()
        else:
            self.model = None
            self.model_params = None

    @abstractmethod
    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        pass

    @abstractmethod
    def predict(self, X: pd.DataFrame, log_y: bool = True) -> np.ndarray:
        pass

    @staticmethod
    @abstractmethod
    def evaluate(y: pd.Series, y_predicted: np.ndarray) -> Tuple[float, float]:
        pass

    def load_best_model(self, model_name: str):
        experiments_tracking_file = json.load(open(os.path.join(base_dir, "..", self.config.get('experiment_file')), 'r'))
        model_class = self._load_model(self._find_best_experiment(experiments_tracking_file, model_name))
        return model_class.model

    @staticmethod
    def _find_best_experiment(data: list, model_name: str) -> dict:
        best_experiment = None
        best_mae = float('inf')

        for experiment in data:
            if experiment['model_name'] == model_name:
                mae = experiment['results']['mae']
                if mae < best_mae:
                    best_mae = mae
                    best_experiment = experiment
        return best_experiment

    @staticmethod
    def _load_model(experiment: dict):
        uuid = experiment['uuid']
        model_path = os.path.join(base_dir, '..', 'experiments', uuid, 'model.pkl')
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        with open(model_path, 'rb') as f:
            model = pickle.load(f)

        logger.info(f"Loaded model from {model_path}")

        return model


class ModelFactory:
    @staticmethod
    def get_model(model_type: str, load_trained_model: bool = False) -> BaseModel:
        if model_type == 'LinearRegressionModel':
            return LinearRegressionModel(load_trained_model)
        elif model_type == 'XGBoostModel':
            return XGBoostModel(load_trained_model)
        else:
            raise ValueError(f"Model type '{model_type}' is not supported.")


class LinearRegressionModel(BaseModel):
    def __init__(self, load_trained_model: bool = True, config: Config = default_config) -> None:
        """
        Initializes the Model object.
        """
        self.model_name = self.__class__.__name__
        super().__init__(model_name=self.model_name, load_trained_model=load_trained_model, config=config)

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Trains the model.

        Args:
            X (pd.DataFrame): The training features.
            y (pd.Series): The training target.
        """
        logger.info("Starting model training.")
        self.model = LinearRegression()
        self.model_params = self.model.get_params()
        self.model.fit(X, y)
        logger.info("Model training completed.")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Makes predictions using the trained model.

        Args:
            X (pd.DataFrame): The features for prediction.
            log_y (bool): Whether the target variable was logarithmically transformed during training.

        Returns:
            np.ndarray: The predicted values.
        """
        logger.info("Making predictions.")
        if self.model_config.get('log_y'):
            logger.info("returning exponential predictions")
            return np.exp(self.model.predict(X))
        else:
            return self.model.predict(X)

    @staticmethod
    def evaluate(y: pd.Series, y_predicted: np.ndarray) -> Tuple[float, float]:
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


class DiamondsPipeline:
    def __init__(self, input_data: pd.DataFrame, config: Config = default_config, model_name: str = 'regression', load_trained_model: bool = False):
        """
        Initializes the TrainingPipeline object with input data and configuration.

        Args:
            input_data (pd.DataFrame): The input data for training.
            config (Config): The configuration object.
            model_name (str, optional): Name of the model. Defaults to 'regression'.
        """
        self.config = config
        self.model_config = self.config.get_model_config(model_name)

        self.data = Data(input_data, config=default_config, model_name=model_name)
        self.data.preprocess()

        self.model = ModelFactory.get_model(model_name, load_trained_model)
        os.makedirs(self.config.get("experiment_directory"), exist_ok=True)

    def train(self) -> Tuple[str, float, float]:
        """
        Trains the model and logs the experiment.

        Returns:
            Tuple[str, float, float]: The experiment ID, R2 score, and mean absolute error.
        """
        logger.info("Starting training pipeline.")
        self.model.train(self.data.X_train, self.data.y_train)
        validation_preds = self.model.predict(self.data.X_val)
        r2, mae = self.model.evaluate(self.data.y_val, validation_preds)

        experiment_id = self._log_experiment(r2, mae)
        self._save_model(experiment_id)

        logger.info("Training pipeline completed.")
        return experiment_id, r2, mae

    def predict(self) -> np.ndarray:
        return self.model.predict(self.data.X_train)

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


class XGBoostModel(BaseModel):
    def __init__(self, load_trained_model: bool = True, config: Config = default_config):
        self.model_name = self.__class__.__name__
        super().__init__(model_name=self.model_name, load_trained_model=load_trained_model, config=config)


    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        logger.info("Starting XGBoost model training.")
        self.model = xgb.XGBRegressor(enable_categorical=True, random_state=42)
        self.model_params = self.model.get_params()
        self.model.fit(X, y)
        logger.info("XGBoost model training completed.")

    def predict(self, X: pd.DataFrame, log_y: bool = False) -> np.ndarray:
        logger.info("Making predictions with XGBoost model.")
        return self.model.predict(X)

    @staticmethod
    def evaluate(y: pd.Series, y_predicted: np.ndarray) -> Tuple[float, float]:
        logger.info("Evaluating XGBoost model performance.")
        r2 = round(r2_score(y, y_predicted), 4)
        mae = round(mean_absolute_error(y, y_predicted), 2)
        logger.info(f"XGBoost model evaluation completed. R2: {r2}, MAE: {mae}")
        return r2, mae
