import os

from flask import Flask

from blueprint.routing import vidhya_api
from config.config import Config
from transformer_models.language_model import LanguageModel
from util.logger import LoggingUtils
from util.model import ModelUtils

resources_dir = Config.get_config("resources_dir_key")
templates_dir = Config.get_config("templates_dir_key")
static_dir = Config.get_config("static_dir_key")
TEMPLATE_DIR = os.path.join(resources_dir, templates_dir)
STATIC_DIR = os.path.join(resources_dir, static_dir)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.register_blueprint(vidhya_api)

def vidhya_setup():
    Config.load_config()
    LoggingUtils.setup_logger()
    LanguageModel.load_model()
    ModelUtils.setup_model_utils()

if __name__ == "__main__":
    vidhya_setup()
    app.run()
