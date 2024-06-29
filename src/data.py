import logging

import numpy as np
import pandas as pd

from typing import List, Optional
from pydantic import BaseModel, Field

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
