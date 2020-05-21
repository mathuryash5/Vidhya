import logging
import os

import pandas as pd
from flask import Flask, request, render_template

from config.config import Config
from transformer_models.language_model import LanguageModel
from util.dataset import DatasetUtils
from util.logger import LoggingUtils
from util.model import ModelUtils

# resources_dir = Config.get_config("resources_dir")
# templates_dir = Config.get_config("templates_dir")
# static_dir = Config.get_config("static_dir")
TEMPLATE_DIR = os.path.join("resources", "templates")
STATIC_DIR = os.path.join("resources", "static")

app = Flask(__name__, template_folder=TEMPLATE_DIR, static_folder=STATIC_DIR)

@app.route('/', methods=["GET"])
@app.route('/index/', methods=["GET"])
def get_homepage():
    return render_template('index.html')

@app.route('/knowledge-graph-system/', methods=["GET"])
@app.route('/knowledge-graph-system/search/', methods=["GET"])
def get_knowledge_graph_system():
    return render_template('index.html')

@app.route('/paper-recommender-system/', methods=["GET"])
@app.route('/paper-recommender-system/search/', methods=["GET"])
def get_paper_recommeder_system():
    return render_template('index.html')

@app.route('/q-a-system/', methods=["GET"])
@app.route('/q-a-system/search/', methods=["GET"])
def get_q_a_system():
    return render_template('index.html')

@app.route('/about_us/', methods=["GET"])
@app.route('/about_us/search/', methods=["GET"])
def get_about_us():
    return render_template('index.html')

@app.route("/paper-recommender-system/search", methods=["POST"])
def get_paper_recommeder_system_search_result():
    user_input = request.form
    user_input_search_string = user_input["paper-recommender-system-user-input-string"]
    user_input_number_of_results = int(user_input["paper-recommender-system-user-input-result-count"])
    user_title_weight = int(user_input["paper-recommender-system-user-input-result-weight"])/100
    app.logger.error("user_input_search_string = {}".format(user_input_search_string))
    app.logger.error("user_input_number_of_results = {}".format(user_input_number_of_results))
    app.logger.error("user_title_weight = {}".format(user_title_weight))
    # df = pd.DataFrame(data=[["87%","Coronavirus eruption in Wuhan, China","https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html"],["83.4%","Rhinovirus", "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html"]])
    # df.columns = ["Similarity", "Title", "Link"]
    df = ModelUtils.get_similar_papers_wrappers(user_input_search_string, title_weights=user_title_weight,
                                                abstract_weights=1-user_title_weight,
                                                number_of_similar_papers=user_input_number_of_results)
    paper_recommender_search_result = df.to_html(classes=["table", "table-bordered", "table-light", "table-hover"],
    index=False)
    return render_template("index.html", paper_recommender_search_result=paper_recommender_search_result)

"""
1. Generate sentence embeddings for each abstract and paper title
2. Store in db
3. User will give paper information through a Paper ID, Paper Name.
4. Calculate similarities and return top n(configurable) results.
5. Store sentence level embeddings for each sentence in the Papers.
6. User query, calculate similarity and return top n(configurable) results.

TODO: Knowledge Graph workflow.
"""


def vidhya_setup():
    Config.load_config()
    LoggingUtils.setup_logger()
    LanguageModel.load_model()


if __name__ == "__main__":
    vidhya_setup()
    # ModelUtils.generate_and_store_embeddings()
    logger = logging.getLogger(__name__)
    logger.info("Starting Vidhya ...")
    app.run()
