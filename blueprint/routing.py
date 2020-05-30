from html import unescape

from flask import Blueprint, render_template, request

from gremlin.azure_gremlin import MicrosoftAzureCosmosDBGremlinAPI
from util.format import FormatUtils
from util.model import ModelUtils

import pandas

vidhya_api = Blueprint('vidhya_api', __name__)

@vidhya_api.route('/', methods=["GET"])
@vidhya_api.route('/index/', methods=["GET"])
def get_homepage():
    return render_template('index.html')

@vidhya_api.route('/knowledge-graph-system/', methods=["GET"])
def get_knowledge_graph_system():
    graph = MicrosoftAzureCosmosDBGremlinAPI.graph
    keys = MicrosoftAzureCosmosDBGremlinAPI.keys
    return render_template('index.html', graph=graph, keys=keys)

@vidhya_api.route('/paper-recommender-system/', methods=["GET"])
@vidhya_api.route('/paper-recommender-system/search/', methods=["GET"])
def get_paper_recommeder_system():
    return render_template('index.html')

@vidhya_api.route('/q-a-system/', methods=["GET"])
@vidhya_api.route('/q-a-system/search/', methods=["GET"])
def get_q_a_system():
    return render_template('index.html')

@vidhya_api.route('/about_us/', methods=["GET"])
@vidhya_api.route('/about_us/search/', methods=["GET"])
def get_about_us():
    return render_template('index.html')

@vidhya_api.route("/paper-recommender-system/search", methods=["POST"])
def get_paper_recommeder_system_search_result():
    user_input = request.form
    user_input_search_string = user_input["paper-recommender-system-user-input-string"]
    user_input_number_of_results = int(user_input["paper-recommender-system-user-input-result-count"])
    user_title_weight = int(user_input["paper-recommender-system-user-input-result-weight"])/100
    df = ModelUtils.get_similar_papers_wrappers(user_input_search_string, title_weights=user_title_weight,
                                                 abstract_weights=1-user_title_weight,
                                                number_of_similar_papers=user_input_number_of_results)
    pandas.set_option('display.max_colwidth', 400)
    df = FormatUtils.paper_recommender_system_formatter(df)
    paper_recommender_search_result = df.to_html(table_id="table", classes=["table", "table-bordered", "table-light", "table-hover", "table-striped"],
    index=False)
    paper_recommender_search_result = unescape(paper_recommender_search_result)
    if user_input_search_string.strip() == "":
        return render_template("index.html")
    else:
        return render_template("index.html", paper_recommender_search_result=paper_recommender_search_result, paper_recommender_system_search_query=user_input_search_string.strip())

@vidhya_api.route("/q-a-system/search", methods=["POST"])
def get_q_a_system_search_result():
    user_input = request.form
    user_input_search_string = user_input["q-a-system-user-input-string"]
    user_input_number_of_results = int(user_input["q-a-system-user-input-result-count"])
    df = ModelUtils.get_answer_similarity(user_input_search_string, user_input_number_of_results)
    df = FormatUtils.paper_recommender_system_formatter(df)
    q_a_search_result = df.to_html(table_id="table-1",
                                                 classes=["table", "table-bordered", "table-light", "table-hover",
                                                          "table-striped"],
                                                 index=False)
    q_a_search_result = unescape(q_a_search_result)
    if user_input_search_string.strip() == "":
        return render_template("intex.html")
    else:
        return render_template("index.html", q_a_search_result=q_a_search_result, q_a_system_search_query=user_input_search_string.strip())