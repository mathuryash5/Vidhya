import logging
import os
import pickle
import re
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import scipy
import spacy
import torch

from config.config import Config
from transformer_models.language_model import LanguageModel
from util.dataset import DatasetUtils
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
    def generate_sentence_embeddings_from_file(filename):
        filepath = "resources/dataset/microsoft/paper_text/" + filename
        with open(filepath, "r") as file:
            data = file.read()
        uid_sentence_embedding_mapping = dict()
        if data == "":
            logging.debug("File : {} does not contain any content. Skipping file".format(filename))
            return uid_sentence_embedding_mapping
        else:

            sentence_list = ModelUtils.sentence_tokenizer(data)
            for sentence in sentence_list:
                key = filename[:-4] + "$%%$" + sentence
                uid_sentence_embedding_mapping[key] = LanguageModel.get_sentence_embeddings_from_sentence(sentence)
        return uid_sentence_embedding_mapping


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
        sentence_embeddings_filemame = Config.get_config("sentence_embeddings_filename")
        title_embeddings_path = os.path.join(resources_dir, embeddings_path, title_embeddings_filename)
        abstract_embeddings_path = os.path.join(resources_dir, embeddings_path, abstract_embeddings_filename)
        sentence_embeddings_path = os.path.join(resources_dir, embeddings_path, sentence_embeddings_filemame)
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
                                                                          get_sentence_embeddings_from_dict,
                                                                          uid_title_mapping_dict.items(), 4)
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
                                                                             get_sentence_embeddings_from_dict,
                                                                             uid_abstract_mapping_dict.items(), 4)
            ModelUtils.write_to_pickle_file(abstract_embeddings_path, uid_abstract_embedding_mapping_dict)
            logging.info("Generating sentence embeddings for Microsoft CORD-19 dataset abstracts completed...")

        if os.path.exists(sentence_embeddings_path):
            logging.info("Microsoft CORD-19 dataset sentence embeddings have already been generated")
        else:
            logging.info("Generating text embeddings for Microsoft CORD-19 dataset ....")
            files = os.listdir("resources/dataset/microsoft/paper_text")
            uid_sentence_embedding_mapping_list = ModelUtils.multiprocessing(ModelUtils.generate_sentence_embeddings_from_file, files, 2)
            uid_sentence_embedding_mapping_list = list(filter(None, uid_sentence_embedding_mapping_list))
            ModelUtils.write_to_pickle_file(sentence_embeddings_path, uid_sentence_embedding_mapping_list)
            logging.info("Generated text embeddings for Microsoft CORD-19 dataset ....")

    @staticmethod
    def get_similar_sentences(query_embedding, corpus_embedding_dict):
        uid_distance_dict = {}

        for id, embedding in corpus_embedding_dict.items():
            uid_distance_dict[id] = \
                scipy.spatial.distance.cdist(query_embedding, [embedding.numpy()], "cosine")[0]
            uid_distance_dict[id][np.isnan(uid_distance_dict[id])] = 1

        return uid_distance_dict

    @staticmethod
    def load_generated_embeddings(path):
        with (open(path, "rb")) as file:
            data = pickle.load(file)
        result_dict = dict()
        for uid_embedding_mapping in data:
            result_dict.update(uid_embedding_mapping)
        return result_dict

    @staticmethod
    def get_query_and_corpus_embedding(query_uid, uid_embedding_mapping):
        query_embedding = uid_embedding_mapping.pop(query_uid, None)
        if query_embedding is None:
            raise Exception("Invalid CORD UID: {}. Please check if the paper exists in the dataset".format(query_uid))
        query_uid_embedding_mapping = dict()
        query_uid_embedding_mapping[query_uid] = query_embedding
        corpus_uid_embedding_mapping = []
        for uid, embedding in uid_embedding_mapping.items():
            corpus_uid_embedding_mapping.append({uid: embedding})
        return query_uid_embedding_mapping, corpus_uid_embedding_mapping

    @staticmethod
    def get_top_n_similar_papers(df, title_based_distances, abstract_based_distances, title_weights, abstract_weights,
                                 number_of_similar_papers):
        uid_average_distance_mapping = dict()
        for uid in title_based_distances.keys():
            uid_average_distance_mapping[uid] = ((title_weights * title_based_distances[uid]) + (
                    abstract_weights * abstract_based_distances[uid]))
        uid_average_distance_mapping = [(uid, distance.tolist()[0]) for uid, distance in
                                        uid_average_distance_mapping.items()]
        sorted_uid_average_distance_mapping = sorted(uid_average_distance_mapping,
                                                     key=lambda x: x[1])

        sorted_uid_average_distance_mapping = sorted_uid_average_distance_mapping[:number_of_similar_papers]
        similar_papers = []
        for uid, score in sorted_uid_average_distance_mapping:
            row = df[df['cord_uid'] == uid]
            similar_paper_metadata = [uid, row['title'].to_list()[0], row['abstract'].to_list()[0],
                                      row['journal'].to_list()[0], row['url'].to_list()[0], round((1 - score) * 100, 4)]
            similar_papers.append(similar_paper_metadata)
        headers = ["CORD UID", "Title", "Abstract", "Journal", "Match Score"]
        df_similar_papers = pd.Dataframe(similar_papers, columns=headers)
        return df_similar_papers

    @staticmethod
    def get_similar_papers_by_query(query, df, title_weights, abstract_weights, number_of_similar_papers):
        resources_folder = Config.get_config("resources_dir")
        embeddings_folder = Config.get_config("embeddings_path")
        title_embeddings_filename = Config.get_config("title_embeddings_filename")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename")
        title_embeddings_path = os.path.join(resources_folder, embeddings_folder, title_embeddings_filename)
        uid_title_embeddings_dict = ModelUtils.load_generated_embeddings(title_embeddings_path)
        query_embedding = LanguageModel.get_sentence_embeddings_from_sentence(query)
        title_based_distances = ModelUtils.get_similar_sentences([query_embedding.numpy()], uid_title_embeddings_dict)

        abstract_embeddings_path = os.path.join(resources_folder, embeddings_folder, abstract_embeddings_filename)
        uid_abstract_embeddings_dict = ModelUtils.load_generated_embeddings(abstract_embeddings_path)
        abstract_based_distances = ModelUtils.get_similar_sentences([query_embedding.numpy(),
                                                                     uid_abstract_embeddings_dict])

        df_similar_papers = ModelUtils.get_top_n_similar_papers(df, title_based_distances, abstract_based_distances,
                                                                title_weights, abstract_weights,
                                                                number_of_similar_papers)
        return df_similar_papers

    @staticmethod
    def get_similar_papers_by_query_id(query_uid, df, title_weights, abstract_weights, number_of_similar_papers):
        resources_folder = Config.get_config("resources_dir")
        embeddings_folder = Config.get_config("embeddings_path")
        title_embeddings_filename = Config.get_config("title_embeddings_filename")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename")

        title_embeddings_path = os.path.join(resources_folder, embeddings_folder, title_embeddings_filename)
        uid_title_embeddings_dict = ModelUtils.load_generated_embeddings(title_embeddings_path)
        query_uid_title_embedding_mapping, corpus_uid_title_embedding_mapping = ModelUtils. \
            get_query_and_corpus_embedding(query_uid, uid_title_embeddings_dict)
        title_based_distances = ModelUtils.get_similar_sentences([query_uid_title_embedding_mapping[query_uid].numpy()],
                                                                 uid_title_embeddings_dict)

        abstract_embeddings_path = os.path.join(resources_folder, embeddings_folder, abstract_embeddings_filename)
        uid_abstract_embeddings_dict = ModelUtils.load_generated_embeddings(abstract_embeddings_path)
        query_uid_abstract_embedding_mapping, corpus_uid_abstract_embedding_mapping = ModelUtils. \
            get_query_and_corpus_embedding(query_uid, uid_abstract_embeddings_dict)
        abstract_based_distances = ModelUtils.get_similar_sentences(
            [query_uid_abstract_embedding_mapping[query_uid].numpy()], uid_abstract_embeddings_dict)

        df_similar_papers = ModelUtils.get_top_n_similar_papers(df, title_based_distances, abstract_based_distances,
                                                                title_weights, abstract_weights,
                                                                number_of_similar_papers)
        return df_similar_papers

    @staticmethod
    def get_similar_papers_wrappers(query, title_weights=0.5, abstract_weights=0.5, number_of_similar_papers=10):
        df = DatasetUtils.get_microsoft_cord_19_dataset()
        if len(df[df['title'].str.lower() == query.lower()]) == 0:
            return ModelUtils.get_similar_papers_by_query(query, df, title_weights, abstract_weights,
                                                          number_of_similar_papers)
        else:
            row = df[df['title'].lower() == query.lower()]
            query_uid = row['cord_uid'].to_list()[0]
            return ModelUtils.get_similar_papers_by_query_id(query_uid, df, title_weights, abstract_weights,
                                                             number_of_similar_papers)

    @staticmethod
    def get_top_n_similar_answers(sentence_distances, number_of_answers):
        return
        for uid, sentence_distance_mapping in sentence_distances:
            sentence_distance_mapping_list = [(sentence, distance.tolist()[0]) for sentence, distance in
                                              sentence_distance_mapping.items()]
        sorted_uid_average_distance_mapping = sorted(uid_average_distance_mapping,
                                                     key=lambda x: x[1])

        sorted_uid_average_distance_mapping = sorted_uid_average_distance_mapping[:number_of_similar_papers]
        similar_papers = []
        for uid, score in sorted_uid_average_distance_mapping:
            row = df[df['cord_uid'] == uid]
            similar_paper_metadata = [uid, row['title'].to_list()[0], row['abstract'].to_list()[0],
                                      row['journal'].to_list()[0], row['url'].to_list()[0], round((1 - score) * 100, 4)]
            similar_papers.append(similar_paper_metadata)
        headers = ["CORD UID", "Title", "Abstract", "Journal", "Match Score"]
        df_similar_papers = pd.Dataframe(similar_papers, columns=headers)
        return df_similar_papers
        pass

    @staticmethod
    def get_answer_similarity(query):
        resources_folder = Config.get_config("resources_dir")
        embeddings_folder = Config.get_config("embeddings_path")
        sentence_embeddings_filename = Config.get_config("sentence_embedding_filename")
        sentence_embeddings_path = os.path.join(resources_folder, embeddings_folder, sentence_embeddings_filename)

        query_embedding = LanguageModel.get_sentence_embeddings_from_sentence(query)
        all_sentences_embeddings = ModelUtils.load_generated_embeddings(sentence_embeddings_path)
        all_sentence_distances = dict()
        for uid, sentence_embedding_mapping in all_sentences_embeddings.mapping():
            all_sentence_distances[uid] = ModelUtils.get_similar_sentences([query_embedding.numpy()],
                                                                           all_sentences_embeddings)

    @staticmethod
    def get_answers(query, number_of_answers=10):
        pass
