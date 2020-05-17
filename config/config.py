import logging

import yaml


class Config:
    _config = None

    @staticmethod
    def load_config():
        """
        Loads configuration specified in config/config.yaml file.
        Sets the class variable _config to contain configuration
        as key-value pair.
        """
        with open('config/config.yaml', 'r') as f:
            Config._config = yaml.safe_load(f.read())
        logging.info("Config loaded: \n{}".format(Config._config))

    @staticmethod
    def get_config(key):
        """
        Returns configuration value of an input configuration key.
        :param key: input configuration key.
        :return: corresponding configuration value.
        """
        return Config._config.get(key, None)
