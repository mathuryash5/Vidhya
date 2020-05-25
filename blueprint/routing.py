from html import unescape

from flask import Blueprint, render_template, request

from util.format import FormatUtils
from util.model import ModelUtils

vidhya_api = Blueprint('vidhya_api', __name__)

@vidhya_api.route('/', methods=["GET"])
@vidhya_api.route('/index/', methods=["GET"])
def get_homepage():
    return render_template('index.html')

@vidhya_api.route('/knowledge-graph-system/', methods=["GET"])
@vidhya_api.route('/knowledge-graph-system/search/', methods=["GET"])
def get_knowledge_graph_system():
    return render_template('index.html')

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
    vidhya_api.logger.error("user_input_search_string = {}".format(user_input_search_string))
    vidhya_api.logger.error("user_input_number_of_results = {}".format(user_input_number_of_results))
    vidhya_api.logger.error("user_title_weight = {}".format(user_title_weight))
    # df = pd.DataFrame(data=[["87%","Coronavirus eruption in Wuhan, China","https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html"],["83.4%","Rhinovirus", "https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.to_html.html"]])
    # df.columns = ["Similarity", "Abstract", "URL"]
    df = ModelUtils.get_similar_papers_wrappers(user_input_search_string, title_weights=user_title_weight,
                                                 abstract_weights=1-user_title_weight,
                                                number_of_similar_papers=user_input_number_of_results)
    df = FormatUtils.paper_recommender_system_formatter(df)
    paper_recommender_search_result = df.to_html(classes=["table", "table-bordered", "table-light", "table-hover"],
    index=False)
    paper_recommender_search_result = unescape(paper_recommender_search_result)
    if user_input_search_string.strip() == "":
        return render_template("index.html")
    else:
        return render_template("index.html", paper_recommender_search_result=paper_recommender_search_result, paper_recommender_system_search_query=user_input_search_string)
