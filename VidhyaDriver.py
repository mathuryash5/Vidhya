# Driver utility
import logging
from multiprocessing import freeze_support

from config.config import Config
from transformer_models.language_model import LanguageModel
from util.dataset import DatasetUtils
from util.logger import LoggingUtils
from util.model import ModelUtils

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
    # LanguageModel.load_model()


if __name__ == "__main__":
    vidhya_setup()
    ModelUtils.generate_and_store_embeddings()
    # ModelUtils.get_similar_papers_by_query_id("xqhn0vbp")
    # df = DatasetUtils.get_microsoft_cord_19_dataset()
    # DatasetUtils.get_papers_text(df)
    logger = logging.getLogger(__name__)
    logger.info("Test")
