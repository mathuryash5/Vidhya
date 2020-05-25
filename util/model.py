import functools
import logging
import os
import pickle
import re
import traceback
from concurrent.futures.process import ProcessPoolExecutor

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

    nlp = None

    @staticmethod
    def initializer():
        """
        Initializer required by the multiprocessing in Windows due to spawn.
        """
        Config.load_config()
        LoggingUtils.setup_logger()
        LanguageModel.load_model()
        ModelUtils.setup_model_utils()

    @staticmethod
    def setup_model_utils():
        """
        Loads the nlp SpaCy model.
        """
        ModelUtils.nlp = spacy.load(Config.get_config("spacy_model_name_key"))
        ModelUtils.nlp.max_length = 10030000

    @staticmethod
    def multiprocessing(func, args, workers):
        with ProcessPoolExecutor(initializer=ModelUtils.initializer, max_workers=workers) as ex:
            res = ex.map(func, args)
        return list(res)

    @staticmethod
    def format_abstract(string):
        """
        Replaces abstract ->  Abstract . 
        :param string: Input string.
        :return: Formatted string.
        """
        abstract_pattern = re.compile(re.escape(Config.get_config("abstract_escape_key")), re.IGNORECASE)
        return abstract_pattern.sub(" " + Config.get_config("abstract_sub_key") + " ", string)

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
            mean_sentence_embeddings = tensor.new_zeros(Config.get_config("embedding_size_key"))
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
        doc = ModelUtils.nlp(string)
        for index, token in enumerate(doc.sents):
            sentence_list.append(token.text)
        return sentence_list

    @staticmethod
    def generate_sentence_embeddings_from_file(filename):
        """
        Generates sentence embedding dict from a file.
        key -> filename$%%$sentence
        value -> sentence embedding
        :param filename: filepath.
        :return: sentence embedding dict.
        """
        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("dataset_dir_key")
        microsoft_dir = Config.get_config("microsoft_dir_key")
        paper_sentence_text_dir = Config.get_config("paper_sentence_text_dir_key")
        dir = os.path.join(resources_dir, dataset_dir, microsoft_dir, paper_sentence_text_dir)
        filepath = os.path.join(dir, filename)
        with open(filepath, "r", encoding="utf-8") as file:
            data = file.read()
            file.close()
        sentence_embedding_dict = dict()
        if data == "":
            pass
        else:
            sentence_list = data.split("\n")
            for sentence in sentence_list:
                if sentence.strip() != "":
                    key = filename[:-4] + Config.get_config("delimiter_key") + sentence
                    sentence_embedding_dict[key] = LanguageModel.get_sentence_embeddings_from_sentence(sentence)
        return sentence_embedding_dict

    @staticmethod
    def generate_and_store_embeddings():
        """
        Generate and store sentence embedding for title + abstract.
        """
        resources_dir = Config.get_config("resources_dir_key")
        embeddings_path = Config.get_config("embeddings_path_key")
        title_embeddings_filename = Config.get_config("title_embeddings_filename_key")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename_key")
        sentence_embeddings_filename = Config.get_config("sentence_embeddings_filename_key")
        title_embeddings_path = os.path.join(resources_dir, embeddings_path, title_embeddings_filename)
        abstract_embeddings_path = os.path.join(resources_dir, embeddings_path, abstract_embeddings_filename)
        sentence_embeddings_path = os.path.join(resources_dir, embeddings_path, sentence_embeddings_filename)
        dataset_df = DatasetUtils.get_microsoft_cord_19_dataset()
        dataset_df[Config.get_config("title_key")].map(ModelUtils.format_abstract)
        dataset_df[Config.get_config("abstract_key")].map(ModelUtils.format_abstract)
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
            resources_dir = Config.get_config("resources_dir_key")
            dataset_dir = Config.get_config("dataset_dir_key")
            microsoft_dir = Config.get_config("microsoft_dir_key")
            paper_sentence_text_dir = Config.get_config("paper_sentence_text_dir_key")
            dir = os.path.join(resources_dir, dataset_dir, microsoft_dir, paper_sentence_text_dir)
            os.mkdir(sentence_embeddings_path)
            files = os.listdir(dir)
            for file in files:
                try:
                    output_pkl_filename = file.split(Config.get_config("text_extension_key"))[0] + Config.get_config("pickle_extension_key")
                    output_pkl_filepath = os.path.join(sentence_embeddings_path, output_pkl_filename)
                    if os.path.exists(output_pkl_filepath):
                        continue
                    sentence_embeddings_mapping_dict = ModelUtils.generate_sentence_embeddings_from_file(file)
                    if sentence_embeddings_mapping_dict:
                        print("Writing path = {}".format(output_pkl_filepath))
                        ModelUtils.write_to_pickle_file(output_pkl_filepath, sentence_embeddings_mapping_dict)
                except Exception as e:
                    print("Error occurred while generating sentence embeddings for File = {}. Skipping.".format(file))
                    traceback.print_exc()

# Code Refactoring ---------

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
        """
        Loads pickle file content into a dict.
        :param path: filepath.
        :return: dict.
        """
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
    def make_clickable(val):
        # target _blank to open new window
        return '<a target="_blank" href="{}">{}</a>'.format(val, val)

    @staticmethod
    def negative_yellow(val, matching_sentences):
        for sentence in matching_sentences:
            if sentence.lower() in val.lower():
                index = val.find(sentence)
                end_index = index + len(sentence) + 1
            return val[:index] + '<mask><b>' + val[index: end_index] + "</b></mask>" + val[end_index:]

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
            similar_paper_metadata = [uid, row[Config.get_config("title_key")].to_list()[0], row[Config.get_config("abstract_key")].to_list()[0],
                                      row[Config.get_config("journal_key")].to_list()[0], row[Config.get_config("url_key")].to_list()[0], round((1 - score) * 100, 4)]
            similar_papers.append(similar_paper_metadata)
        headers = ["CORD UID", "Title", "Abstract", "Journal", "URL", "Match Score"]
        df_similar_papers = pd.DataFrame(similar_papers, columns=headers)
        df_similar_papers.style.format({'URL': ModelUtils.make_clickable})
        return df_similar_papers

    @staticmethod
    def get_similar_papers_by_query(query, df, title_weights, abstract_weights, number_of_similar_papers):
        resources_folder = Config.get_config("resources_dir_key")
        embeddings_folder = Config.get_config("embeddings_path_key")
        title_embeddings_filename = Config.get_config("title_embeddings_filename_key")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename_key")
        title_embeddings_path = os.path.join(resources_folder, embeddings_folder, title_embeddings_filename)
        uid_title_embeddings_dict = ModelUtils.load_generated_embeddings(title_embeddings_path)
        query_embedding = LanguageModel.get_sentence_embeddings_from_sentence(query)
        title_based_distances = ModelUtils.get_similar_sentences([query_embedding.numpy()], uid_title_embeddings_dict)

        abstract_embeddings_path = os.path.join(resources_folder, embeddings_folder, abstract_embeddings_filename)
        uid_abstract_embeddings_dict = ModelUtils.load_generated_embeddings(abstract_embeddings_path)
        abstract_based_distances = ModelUtils.get_similar_sentences([query_embedding.numpy()],
                                                                    uid_abstract_embeddings_dict)

        df_similar_papers = ModelUtils.get_top_n_similar_papers(df, title_based_distances, abstract_based_distances,
                                                                title_weights, abstract_weights,
                                                                number_of_similar_papers)
        return df_similar_papers

    @staticmethod
    def get_similar_papers_by_query_id(query_uid, df, title_weights, abstract_weights, number_of_similar_papers):
        resources_folder = Config.get_config("resources_dir_key")
        embeddings_folder = Config.get_config("embeddings_path_key")
        title_embeddings_filename = Config.get_config("title_embeddings_filename_key")
        abstract_embeddings_filename = Config.get_config("abstract_embeddings_filename_key")

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
        if len(df[df[Config.get_config("title_key")].str.lower() == query.lower()]) == 0:
            return ModelUtils.get_similar_papers_by_query(query, df, title_weights, abstract_weights,
                                                          number_of_similar_papers)
        else:
            row = df[df[Config.get_config("title_key")].str.lower() == query.lower()]
            query_uid = row[Config.get_config("cord_uid_key")].to_list()[0]
            return ModelUtils.get_similar_papers_by_query_id(query_uid, df, title_weights, abstract_weights,
                                                             number_of_similar_papers)

    @staticmethod
    def get_answer_similarity(query, number_of_answers=10):
        resources_folder = Config.get_config("resources_dir_key")
        embeddings_folder = Config.get_config("embeddings_path_key")
        sentence_embeddings_filename = Config.get_config("sentence_embedding_filename_key")
        sentence_embeddings_path = os.path.join(resources_folder, embeddings_folder, sentence_embeddings_filename)

        query_embedding = LanguageModel.get_sentence_embeddings_from_sentence(query)
        all_sentences_embeddings = ModelUtils.load_generated_embeddings(sentence_embeddings_path)
        all_sentence_distances = ModelUtils.get_similar_sentences([query_embedding.numpy()],
                                                                  all_sentences_embeddings)
        df = ModelUtils.get_top_n_similar_answers(all_sentence_distances, number_of_answers)
        return df

    @staticmethod
    def get_top_n_similar_answers(all_sentence_distances, number_of_answers):

        all_sentence_distances_mapping = [(uid, distance.tolist()[0]) for uid, distance in
                                          all_sentence_distances.items()]
        sorted_all_sentence_distances_mapping = sorted(all_sentence_distances_mapping,
                                                       key=lambda x: x[1])

        df = DatasetUtils.get_microsoft_cord_19_dataset()
        sorted_all_sentence_distances_mapping = sorted_all_sentence_distances_mapping[:number_of_answers]
        similar_papers = []
        matching_sentences = []
        resources_dir = Config.get_config("resources_dir_key")
        dataset_dir = Config.get_config("dataset_key")
        microsoft_dir = Config.get_config("microsoft_key")
        for key, score in sorted_all_sentence_distances_mapping:
            uid_sentence = key.split(Config.get_config("delimiter_key"))
            uid = uid_sentence[0]
            sentence = uid_sentence[1]
            matching_sentences.append(sentence)
            row = df[df[Config.get_config("cord_uid")] == uid]
            with open("resources/dataset/microsoft/paper_text/" + uid + ".txt") as file:
                data = file.read()
            sentences = ModelUtils.sentence_tokenizer(data)
            sentence_index = sentences.index(sentence)
            region_range = 2
            region_of_interest = sentences[sentence_index - region_range: sentence_index + region_range + 1]
            paragraph = " ".join(region_of_interest)
            similar_paper_metadata = [uid, row['title'].to_list()[0], row['abstract'].to_list()[0], paragraph,
                                      row['journal'].to_list()[0], row['url'].to_list()[0], round((1 - score) * 100, 4)]
            similar_papers.append(similar_paper_metadata)

        headers = ["CORD UID", "Title", "Abstract", "Relevant Context", "Journal", "URL", "Match Score"]
        df_similar_papers = pd.DataFrame(similar_papers, columns=headers)
        df_similar_papers.style.format({'URL': ModelUtils.make_clickable})
        curried_formatter = functools.partial(ModelUtils.negative_yellow, matching_sentences=matching_sentences)
        df_similar_papers.style.format({'Relevant Context': curried_formatter})
        return df_similar_papers
