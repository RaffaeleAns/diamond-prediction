import json
from typing import Any, Dict, Optional


class Config:
    def __init__(self, config_path: str):
        """
        Initializes the Config object by loading the configuration from a file.

        Args:
            config_path (str): The path to the configuration file.
        """
        self.config: Dict[str, Any] = self._load_config(config_path)

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Loads the configuration from a JSON file.

        Args:
            config_path (str): The path to the configuration file.

        Returns:
            Dict[str, Any]: The configuration data.
        """
        with open(config_path, "r") as file:
            config = json.load(file)
        return config

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Retrieves a value from the configuration.

        Args:
            key (str): The key to look up in the configuration.
            default (Optional[Any]): The default value to return if the key is not found.

        Returns:
            Any: The value from the configuration, or the default value if the key is not found.
        """
        return self.config.get(key, default)


# Load the default configuration
default_config = Config("config.json")
