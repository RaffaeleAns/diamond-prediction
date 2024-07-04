import logging
import os
import typing

import numpy as np
import pandas as pd

from typing import Optional
from pydantic import BaseModel

from src.payloads import Features, Target
from src.utils import Config, default_config

base_dir = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class DataFrameValidator(BaseModel):

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
        mandatory_column = set([k for k, v in Features.__annotations__.items() if v != typing.Optional[float]])
        expected_columns = list(Features.__annotations__.keys()) + list(Target.__annotations__.keys())

        df_columns = set(df.columns)
        expected_columns_set = set(expected_columns)

        missing_columns = mandatory_column - df_columns
        unexpected_columns = df_columns - expected_columns_set

        if missing_columns:
            logger.error(f"Missing columns: {missing_columns}")
            raise ValueError(f"Missing columns: {missing_columns}")

        if unexpected_columns:
            logger.error(f"Unexpected columns: {unexpected_columns}")
            raise ValueError(f"Unexpected columns: {unexpected_columns}")

        for column, expected_type in Features.__annotations__.items():
            if column in df_columns:
                python_type = expected_type
                if not all(isinstance(value, python_type) for value in df[column]):
                    logger.error(f"Column '{column}' does not have all values of type '{python_type.__name__}'")
                    raise ValueError(f"Column '{column}' does not have all values of type '{python_type.__name__}'")

        data = df.to_dict(orient='list')
        cls(**data)
        logger.info("Input DataFrame validation completed successfully.")


class Data:
    def __init__(self, data: pd.DataFrame, model_name: str, config: Config = default_config):
        """
        Initializes the Data object with input data and configuration.

        Args:
            data (pd.DataFrame): The input data for training.
            model_name (str): The name of the model.
            config (Config): The configuration object.
        """
        self.X_train: Optional[pd.DataFrame] = None
        self.y_train: Optional[pd.Series] = None
        self.X_val: Optional[pd.DataFrame] = None
        self.y_val: Optional[pd.Series] = None

        self.data = data
        self.config = config
        self.model_config = self.config.get_model_config(model_name)

        self._input_validation()
        self._load_validation_data(self.model_config.get('validation_x_path'),
                                   self.model_config.get('validation_y_path'))

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
        self.X_val = pd.read_parquet(os.path.join(base_dir, '..', validation_X_path))
        self.y_val = pd.read_parquet(os.path.join(base_dir, '..', validation_y_path))['price']
        logger.info("Validation data loaded successfully.")

    def preprocess(self) -> None:
        """
        Preprocesses the training data. Choose the preprocessing actions based on config file
        """
        logger.info("Starting preprocessing of training data.")
        self.data = self.data[self.data.x * self.data.y * self.data.z != 0]

        self.data['cut'] = pd.Categorical(self.data['cut'],
                                          categories=['Fair', 'Good', 'Very Good', 'Ideal', 'Premium'],
                                          ordered=True)
        self.data['color'] = pd.Categorical(self.data['color'],
                                            categories=['D', 'E', 'F', 'G', 'H', 'I', 'J'],
                                            ordered=True)
        self.data['clarity'] = pd.Categorical(self.data['clarity'],
                                              categories=['IF', 'VVS1', 'VVS2', 'VS1', 'VS2', 'SI1',
                                                          'SI2', 'I1'], ordered=True)

        if self.model_config.get('get_dummies'):
            logger.info("Applying get_dummies to categorical columns.")
            self.data = pd.get_dummies(self.data, columns=['cut', 'color', 'clarity'], drop_first=False)

        if 'price' in self.data.columns:
            self.data = self.data[self.data.price > 0]
            self.y_train = self.data['price']

            if self.model_config.get('log_y'):
                logger.info("Applying logarithmic transformation to the target variable.")
                self.y_train = np.log(self.y_train)

        self.X_train = self.data[self.model_config.get('training_features')]
        logger.info("Preprocessing completed.")


