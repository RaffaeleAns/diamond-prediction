import os
import uuid
import json
import pickle

import pandas as pd
import numpy as np

from src.utils import Config, default_config

from typing import List
from pydantic import BaseModel, Field
from datetime import datetime

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error


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
        # Get expected columns from the Pydantic model annotations
        expected_columns = cls.__annotations__.keys()

        # Check for missing or unexpected columns
        df_columns = set(df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = expected_columns_set - df_columns
        unexpected_columns = df_columns - expected_columns_set

        if missing_columns:
            raise ValueError(f"Missing columns: {missing_columns}")

        if unexpected_columns:
            raise ValueError(f"Unexpected columns: {unexpected_columns}")

        # Check for column types
        for column, expected_type in cls.__annotations__.items():
            python_type = expected_type.__args__[0]
            if not all(isinstance(value, python_type) for value in df[column]):
                raise ValueError(f"Column '{column}' does not have all values of type '{python_type.__name__}'")

        # Convert DataFrame to dictionary of lists for Pydantic validation
        data = df.to_dict(orient='list')

        # Validate using Pydantic model
        cls(**data)


class Data:
    def __init__(self, data: pd.DataFrame, config: Config = default_config):
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None

        self.data = data
        self.config = config

        self._input_validation()
        self._load_validation_data(self.config.get('validation_x_path'), self.config.get('validation_y_path'))

    def _input_validation(self):
        DataFrameValidator.validate_dataframe(self.data)

    def _load_validation_data(self, validation_X_path, validation_y_path):
        self.X_val = pd.read_csv(validation_X_path, sep=';', index_col=0)
        self.y_val = pd.read_csv(validation_y_path, sep=';', index_col=0)

    def preprocess(self, get_dummies: bool = True, log_y: bool = True):
        # Remove records with 0 dimension
        self.data = self.data[self.data.x * self.data.y * self.data.z != 0]

        # Remove records with negative price
        self.data = self.data[self.data.price > 0]

        self.y_train = self.data['price']

        if get_dummies:
            self.data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'], drop_first=True)

        if log_y:
            self.y_train = np.log(self.y_train)

        self.X_train = self.data[self.config.get('training_features')]


class Model:
    def __init__(self):
        self.model = LinearRegression()
        self.model_name = type(self.model).__name__
        self.model_params = self.model.get_params()

    def train(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame, log_y=True):
        if log_y:
            return np.exp(self.model.predict(X))
        else:
            return self.model.predict(X)

    def evaluate(self, y: pd.Series, y_predicted: pd.Series):
        r2 = round(r2_score(y, y_predicted), 4)
        mae = round(mean_absolute_error(y, y_predicted), 2)
        return r2, mae


class TrainingPipeline:
    def __init__(self, input_data: pd.DataFrame, config: Config = default_config):
        self.config = config
        self.data = Data(input_data, config)
        self.data.preprocess(log_y=self.config.get('log_y'))

        self.model = Model()
        os.makedirs(self.config.get("experiment_directory"), exist_ok=True)

    def train(self):
        self.model.train(self.data.X_train, self.data.y_train)
        validation_preds = self.model.predict(self.data.X_val, log_y=self.config.get('log_y'))
        r2, mae = self.model.evaluate(self.data.y_val, validation_preds)

        experiment_id = self._log_experiment(r2, mae)
        self._save_model(experiment_id)

        return experiment_id, r2, mae

    def _log_experiment(self, r2, mae):
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

        return experiment_id

    def _save_model(self, experiment_id):
        model_dir = os.path.join(self.config.get("experiment_directory"), experiment_id)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pkl")
        with open(model_path, "wb") as file:
            pickle.dump(self.model, file)
