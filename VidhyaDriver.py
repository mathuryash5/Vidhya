import os

from flask import Flask

from blueprint.routing import vidhya_api
from config.config import Config
from transformer_models.language_model import LanguageModel
from util.logger import LoggingUtils
from util.model import ModelUtils

import pandas as pd

Config.load_config()

resources_dir = Config.get_config("resources_dir_key")
templates_dir = Config.get_config("templates_dir_key")
static_dir = Config.get_config("static_dir_key")
TEMPLATE_DIR = os.path.join(resources_dir, templates_dir)
STATIC_DIR = os.path.join(resources_dir, static_dir)

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)
app.register_blueprint(vidhya_api)
app.config.update(
    DEBUG=True,
    TEMPLATES_AUTO_RELOAD=True
)

def vidhya_setup():
    LoggingUtils.setup_logger()
    LanguageModel.load_model()
    ModelUtils.setup_model_utils()

if __name__ == "__main__":
    vidhya_setup()
    # df = ModelUtils.get_answer_similarity('How does Chikungunya virus evolve?', 10)
    # df.to_csv("F:\\Hackathon\\Vidhya\\resources\\dataset\\result.csv")
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)
