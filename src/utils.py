import json


class Config:
    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)

    def _load_config(self, config_path: str):
        with open(config_path, "r") as file:
            config = json.load(file)
        return config

    def get(self, key: str, default=None):
        return self.config.get(key, default)


default_config = Config("config.json")
