import logging.config

from config.config import Config


class LoggingUtils:

    @staticmethod
    def setup_logger():
        """
        Load logging configuration and setup logger.
        """
        logger_config = Config.get_config("logger_config")
        logging.config.dictConfig(logger_config)
