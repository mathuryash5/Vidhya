import os

from flask import Flask

from blueprint.routing import vidhya_api
from config.config import Config
from gremlin.azure_gremlin import MicrosoftAzureCosmosDBGremlinAPI
from transformer_models.language_model import LanguageModel
from util.logger import LoggingUtils
from util.model import ModelUtils
from storage.azure_storage_blob import MicrosoftAzureBlobStorageAPI

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
    MicrosoftAzureBlobStorageAPI.setup_storage()
    LanguageModel.load_model()
    ModelUtils.setup_model_utils()
    MicrosoftAzureCosmosDBGremlinAPI.setup_gremlin()

if __name__ == "__main__":
    vidhya_setup()
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True
    app.run(debug=True)