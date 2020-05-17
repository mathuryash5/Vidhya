# Driver utility
import argparse
import logging

from config.config import Config
from model.biobert import BioBERT
from util.logger import LoggingUtils
from util.dataset import DatasetUtils

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


if __name__ == "__main__":
    vidhya_setup()
    BioBERT.generate_and_store_embeddings()
    logger = logging.getLogger(__name__)
    logger.info("Test")
