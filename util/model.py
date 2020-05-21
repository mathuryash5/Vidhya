import logging
import os
import pickle

import spacy
import re
import torch

from config.config import Config
from model.language_model import LanguageModel
from util.dataset import DatasetUtils
from concurrent.futures import ProcessPoolExecutor
import concurrent

from util.logger import LoggingUtils


class ModelUtils:

    @staticmethod
    def initializer():
        Config.load_config()
        LoggingUtils.setup_logger()
        LanguageModel.load_model()

    @staticmethod
    def multiprocessing(func, args, workers):
        with ProcessPoolExecutor(initializer=ModelUtils.initializer, max_workers=workers) as ex:
            res = ex.map(func, args)
        return list(res)

    @staticmethod
    def format_abstract(string):
        abstract_pattern = re.compile(re.escape('abstract'), re.IGNORECASE)
        return abstract_pattern.sub(' Abstract ', string)


    @staticmethod
    def get_sentence_embeddings_from_dict(uid_sentence_mapping_dict):
        """
        Generates and returns sentence embeddings of multiple sentences.
        :param uid_sentence_mapping_dict: Dict with UID as key and sentence as value.
        :return: Dict with UID as key and sentence embeddings as value.
        """
        uid_sentence_embeddings_mapping_dict = dict()
        uid, text = uid_sentence_mapping_dict
        uid_sentence_embeddings_mapping_dict[uid] = ModelUtils.get_sentence_embeddings_from_paragraph(text)
        return uid_sentence_embeddings_mapping_dict

    @staticmethod
    def get_sentence_embeddings_from_paragraph(paragraph):
        """
        Generates mean sentence embeddings for a paragraph.
        :param paragraph: Input text.
        :return: Mean sentence embeddings.
        """
        if paragraph == "":
            tensor = torch.tensor((), dtype=torch.float64)
            mean_sentence_embeddings = tensor.new_zeros(Config.get_config("embedding_size"))
        else:
            sentence_list = ModelUtils.sentence_tokenizer(paragraph)
            sentence_embeddings_list = []
            for sentence in sentence_list:
                sentence_embeddings = LanguageModel.get_sentence_embeddings_from_sentence(sentence)
                sentence_embeddings_list.append(sentence_embeddings)
            mean_sentence_embeddings = torch.mean(torch.stack(sentence_embeddings_list), dim=0)
        logging.debug("Sentence embedding generated for paragraph = {} \n with tensor size = {}".
                      format(paragraph, mean_sentence_embeddings.size()))
        return mean_sentence_embeddings

    @staticmethod
    def write_to_pickle_file(filepath, dict):
        """
        Writes a dict to a pickle file.
        :param filepath: Path to store pickle file.
        :param dict: Input dict.
        """
        with open(filepath, 'wb') as handle:
            pickle.dump(dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def sentence_tokenizer(string):
        """
        Sentence level tokenization of input string.
        :param string: input paragraph.
        :return: list of sentences.
        """
        sentence_list = []
        nlp = spacy.load(Config.get_config("spacy_model_name"))
        doc = nlp(string)
        for index, token in enumerate(doc.sents):
            sentence_list.append(token.text)
        return sentence_list

    @staticmethod
    def generate_and_store_embeddings():
        """
        Generate and store sentence embedding for title + abstract.
        """
        # these functions should be in some other file
        # load the model once and not on every function call
        resources_dir = Config.get_config("resources_dir")
        embeddings_path = Config.get_config("embeddings_path")
        title_embeddings_filename = Config.get_config("title_embeddings_filename")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename")
        title_embeddings_path = os.path.join(resources_dir, embeddings_path, title_embeddings_filename)
        abstract_embeddings_path = os.path.join(resources_dir, embeddings_path, abstract_embeddings_filename)
        dataset_df = DatasetUtils.get_microsoft_cord_19_dataset()
        dataset_df['title'].map(ModelUtils.format_abstract)
        dataset_df['abstract'].map(ModelUtils.format_abstract)
        if os.path.exists(title_embeddings_path):
            logging.info("Microsoft CORD-19 dataset title embeddings have already been generated.")
        else:
            # Read UID and title from dataset
            uid_title_mapping_dict = dict()
            for index, row in dataset_df.iterrows():
                uid_title_mapping_dict[row[Config.get_config("cord_uid_key")]] = row[Config.get_config("title_key")]
            logging.info("Generating sentence embeddings for Microsoft CORD-19 dataset titles ...")
            uid_title_embedding_mapping_dict = ModelUtils.multiprocessing(ModelUtils. \
                get_sentence_embeddings_from_dict, uid_title_mapping_dict.items(), 4)
            ModelUtils.write_to_pickle_file(title_embeddings_path, uid_title_embedding_mapping_dict)
            logging.info("Generating sentence embeddings for Microsoft CORD-19 dataset titles completed...")

        if os.path.exists(abstract_embeddings_path):
            logging.info("Microsoft CORD-19 dataset abstract embeddings have already been generated.")
        else:
            # Read UID and abstract from dataset
            uid_abstract_mapping_dict = dict()
            for index, row in dataset_df.iterrows():
                uid_abstract_mapping_dict[row[Config.get_config("cord_uid_key")]] = row[
                    Config.get_config("abstract_key")]
            logging.info("Generating sentence embeddings for Microsoft CORD-19 dataset abstracts ...")
            uid_abstract_embedding_mapping_dict = ModelUtils.multiprocessing(ModelUtils. \
                get_sentence_embeddings_from_dict, uid_abstract_mapping_dict.items(), 4)
            ModelUtils.write_to_pickle_file(abstract_embeddings_path, uid_abstract_embedding_mapping_dict)
            logging.info("Generating sentence embeddings for Microsoft CORD-19 dataset abstracts completed...")
