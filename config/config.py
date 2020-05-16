import yaml


class Config:

    _config = None

    @staticmethod
    def load_config():
        with open('config/config.yaml', 'r') as f:
            Config._config = yaml.safe_load(f.read())

    @staticmethod
    def get_config(key):
        return Config._config.get(key, None)
